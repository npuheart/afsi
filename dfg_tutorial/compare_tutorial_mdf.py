#!/usr/bin/env python
"""Compare dfg_tutorial (body-fitted CN+AB2, 参考) vs multi_direct_forcing (IBM mdf).

两组求解器使用完全相同的物理/入口（rho=1, mu=0.001, Re=100, sin 入口
U=1.5·sin(πt/8)），在若干网格分辨率下各自推进到 t=T_s，然后比较：
  - 下游中心线探针速度 u_x（逐点）
  - 最终 Cd
  - 全局速度 L2 范数

用法:
  conda activate afsi-dolfinx
  python compare_tutorial_mdf.py            # 默认 3 组分辨率, T_s=1.0s
  TS=1.5 python compare_tutorial_mdf.py     # 更长
"""
import os
import numpy as np
from petsc4py import PETSc
from mpi4py import MPI

import gmsh
import dolfinx
from dolfinx.fem import (Constant, Function, functionspace, dirichletbc,
                         locate_dofs_topological, form, assemble_scalar,
                         extract_function_spaces, Expression)
from dolfinx.fem.petsc import (apply_lifting, assemble_matrix, assemble_vector,
                               create_vector, create_matrix, set_bc)
from dolfinx.mesh import CellType, GhostMode, create_rectangle
from basix.ufl import element
from ufl import (TestFunction, TrialFunction, dot, dx, inner, grad, div,
                 as_vector, lhs, rhs, nabla_grad, FacetNormal, Measure)
from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells

from afsic import IBMesh, IBInterpolation

comm = MPI.COMM_WORLD
rank = comm.rank

# ---------------- 共同物理 ----------------
L, H = 2.2, 0.41
c_x, c_y, r = 0.2, 0.2, 0.05
D = 0.1
rho, mu = 1.0, 0.001          # nu = 0.001 → Re = U_mean·D/nu = 100
nu = mu / rho
DT = float(os.environ.get("DT", "0.001"))
TS = float(os.environ.get("TS", "1.0"))
NSTEPS = int(TS / DT)
gdim = 2


class InletVelocity:
    """与教程完全相同的 sin 入口: u_x = 4·1.5·sin(πt/8)·y(H-y)/H²"""
    def __init__(self):
        self.t = 0.0
    def update(self, t):
        self.t = t
    def __call__(self, x):
        values = np.zeros((gdim, x.shape[1]), dtype=PETSc.ScalarType)
        values[0] = 4 * 1.5 * np.sin(self.t * np.pi / 8) * x[1] * (H - x[1]) / H**2
        return values


def make_ksp(A, ksp_type, pc_type, hypre=False):
    s = PETSc.KSP().create(comm)
    s.setOperators(A); s.setType(ksp_type)
    pc = s.getPC(); pc.setType(pc_type)
    if hypre:
        pc.setHYPREType("boomeramg")
    return s


# =====================================================================
# 求解器 A: dfg_tutorial (body-fitted, CN+AB2 IPCS)
# =====================================================================
def solve_tutorial(LcMax, res_min):
    gmsh.initialize()
    gmsh.model.add("dfg")
    rectangle = gmsh.model.occ.addRectangle(0, 0, 0, L, H, tag=1)
    obstacle = gmsh.model.occ.addDisk(c_x, c_y, 0, r, r)
    fluid = gmsh.model.occ.cut([(gdim, rectangle)], [(gdim, obstacle)])
    gmsh.model.occ.synchronize()
    volumes = gmsh.model.getEntities(dim=gdim)
    gmsh.model.addPhysicalGroup(volumes[0][0], [volumes[0][1]], 1)
    inlet_marker, outlet_marker, wall_marker, obstacle_marker = 2, 3, 4, 5
    inflow, outflow, walls, obs = [], [], [], []
    boundaries = gmsh.model.getBoundary(volumes, oriented=False)
    for b in boundaries:
        com = gmsh.model.occ.getCenterOfMass(b[0], b[1])
        if np.allclose(com, [0, H/2, 0]): inflow.append(b[1])
        elif np.allclose(com, [L, H/2, 0]): outflow.append(b[1])
        elif np.allclose(com, [L/2, H, 0]) or np.allclose(com, [L/2, 0, 0]): walls.append(b[1])
        else: obs.append(b[1])
    gmsh.model.addPhysicalGroup(1, walls, wall_marker)
    gmsh.model.addPhysicalGroup(1, inflow, inlet_marker)
    gmsh.model.addPhysicalGroup(1, outflow, outlet_marker)
    gmsh.model.addPhysicalGroup(1, obs, obstacle_marker)
    dfield = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(dfield, "EdgesList", obs)
    tfield = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(tfield, "IField", dfield)
    gmsh.model.mesh.field.setNumber(tfield, "LcMin", res_min)
    gmsh.model.mesh.field.setNumber(tfield, "LcMax", LcMax)
    gmsh.model.mesh.field.setNumber(tfield, "DistMin", r)
    gmsh.model.mesh.field.setNumber(tfield, "DistMax", 2 * H)
    mfield = gmsh.model.mesh.field.add("Min")
    gmsh.model.mesh.field.setNumbers(mfield, "FieldsList", [tfield])
    gmsh.model.mesh.field.setAsBackgroundMesh(mfield)
    gmsh.option.setNumber("Mesh.Algorithm", 8)
    gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)
    gmsh.option.setNumber("Mesh.RecombineAll", 1)
    gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)
    gmsh.model.mesh.generate(gdim)
    gmsh.model.mesh.setOrder(2)
    gmsh.model.mesh.optimize("Netgen")
    from dolfinx.io import gmsh as gmshio
    md = gmshio.model_to_mesh(gmsh.model, comm, 0, gdim=gdim)
    mesh = md.mesh
    ft = md.facet_tags
    gmsh.finalize()

    k = Constant(mesh, PETSc.ScalarType(DT))
    mu_c = Constant(mesh, PETSc.ScalarType(mu))
    rho_c = Constant(mesh, PETSc.ScalarType(rho))
    v_cg2 = element("Lagrange", mesh.basix_cell(), 2, shape=(gdim,))
    s_cg1 = element("Lagrange", mesh.basix_cell(), 1)
    V = functionspace(mesh, v_cg2); Q = functionspace(mesh, s_cg1)
    fdim = mesh.topology.dim - 1

    inlet = InletVelocity()
    u_inlet = Function(V); u_inlet.interpolate(inlet)
    bcu_inflow = dirichletbc(u_inlet, locate_dofs_topological(V, fdim, ft.find(2)))
    u_ns = np.array((0,) * gdim, dtype=PETSc.ScalarType)
    bcu_walls = dirichletbc(u_ns, locate_dofs_topological(V, fdim, ft.find(4)), V)
    bcu_obs = dirichletbc(u_ns, locate_dofs_topological(V, fdim, ft.find(5)), V)
    bcu = [bcu_inflow, bcu_obs, bcu_walls]
    bcp = [dirichletbc(PETSc.ScalarType(0),
                       locate_dofs_topological(Q, fdim, ft.find(3)), Q)]

    u = TrialFunction(V); v = TestFunction(V)
    u_ = Function(V); u_s = Function(V); u_n = Function(V); u_n1 = Function(V)
    p = TrialFunction(Q); q = TestFunction(Q)
    p_ = Function(Q); phi = Function(Q)
    f = Constant(mesh, PETSc.ScalarType((0, 0)))
    F1 = rho_c / k * dot(u - u_n, v) * dx
    F1 += inner(dot(1.5*u_n - 0.5*u_n1, 0.5*nabla_grad(u + u_n)), v) * dx
    F1 += 0.5*mu_c*inner(grad(u + u_n), grad(v))*dx - dot(p_, div(v))*dx
    F1 += dot(f, v) * dx
    a1 = form(lhs(F1)); L1 = form(rhs(F1))
    A1 = create_matrix(a1); b1 = create_vector(extract_function_spaces(L1))
    a2 = form(dot(grad(p), grad(q))*dx)
    L2 = form(-rho_c/k*dot(div(u_s), q)*dx)
    A2 = assemble_matrix(a2, bcs=bcp); A2.assemble()
    b2 = create_vector(extract_function_spaces(L2))
    a3 = form(rho_c*dot(u, v)*dx)
    L3 = form(rho_c*dot(u_s, v)*dx - k*dot(nabla_grad(phi), v)*dx)
    A3 = assemble_matrix(a3); A3.assemble()
    b3 = create_vector(extract_function_spaces(L3))
    s1 = make_ksp(A1, PETSc.KSP.Type.BCGS, PETSc.PC.Type.JACOBI)
    s2 = make_ksp(A2, PETSc.KSP.Type.MINRES, PETSc.PC.Type.HYPRE, hypre=True)
    s3 = make_ksp(A3, PETSc.KSP.Type.CG, PETSc.PC.Type.SOR)

    n = -FacetNormal(mesh)
    dObs = Measure("ds", domain=mesh, subdomain_data=ft, subdomain_id=5)
    u_t = inner(as_vector((n[1], -n[0])), u_)
    drag = form(2/D*(mu_c/rho_c*inner(grad(u_t), n)*n[1] - p_*n[0])*dObs)

    for i in range(NSTEPS):
        t = (i + 1) * DT
        inlet.update(t); u_inlet.interpolate(inlet)
        A1.zeroEntries(); assemble_matrix(A1, a1, bcs=bcu); A1.assemble()
        with b1.localForm() as lc: lc.set(0)
        assemble_vector(b1, L1); apply_lifting(b1, [a1], [bcu])
        b1.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        set_bc(b1, bcu); s1.solve(b1, u_s.x.petsc_vec); u_s.x.scatter_forward()
        with b2.localForm() as lc: lc.set(0)
        assemble_vector(b2, L2); apply_lifting(b2, [a2], [bcp])
        b2.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        set_bc(b2, bcp); s2.solve(b2, phi.x.petsc_vec); phi.x.scatter_forward()
        p_.x.petsc_vec.axpy(1, phi.x.petsc_vec); p_.x.scatter_forward()
        with b3.localForm() as lc: lc.set(0)
        assemble_vector(b3, L3)
        b3.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        s3.solve(b3, u_.x.petsc_vec); u_.x.scatter_forward()
        with (u_.x.petsc_vec.localForm() as l0, u_n.x.petsc_vec.localForm() as l1,
              u_n1.x.petsc_vec.localForm() as l2_):
            l1.copy(l2_); l0.copy(l1)
    Cd = comm.allreduce(assemble_scalar(drag), op=MPI.SUM)
    uL2 = comm.allreduce(assemble_scalar(form(dot(u_, u_)*dx)), op=MPI.SUM)
    return mesh, u_, p_, Cd, uL2


# =====================================================================
# 求解器 B: multi_direct_forcing (IBM mdf) — 共同物理/入口
# =====================================================================
def solve_mdf(Nx, Ny):
    h = np.sqrt((L/Nx)*(H/Ny))
    mesh = dolfinx.mesh.create_rectangle(comm, ((0, 0), (L, H)), (Nx, Ny),
                                         cell_type=CellType.quadrilateral,
                                         ghost_mode=GhostMode.shared_facet)
    # 边界标记 (与教程一致: 2=inlet 3=outlet 4=wall)
    from dolfinx.mesh import locate_entities, meshtags
    mesh.topology.create_connectivity(1, 2)
    fdim = mesh.topology.dim - 1
    def mk(tag, f):
        e = locate_entities(mesh, fdim, f)
        return e, np.full_like(e, tag)
    ei, ti = mk(2, lambda x: np.isclose(x[0], 0))
    eo, to = mk(3, lambda x: np.isclose(x[0], L))
    ew, tw = mk(4, lambda x: np.isclose(x[1], 0) | np.isclose(x[1], H))
    e = np.hstack([ei, eo, ew]).astype(np.int32)
    t = np.hstack([ti, to, tw]).astype(np.int32)
    ft = meshtags(mesh, fdim, e, t)

    v_cg2 = element("Lagrange", mesh.topology.cell_name(), 2, shape=(gdim,))
    s_cg1 = element("Lagrange", mesh.topology.cell_name(), 1)
    V = functionspace(mesh, v_cg2); Q = functionspace(mesh, s_cg1)

    inlet = InletVelocity()
    u_inlet = Function(V); u_inlet.interpolate(inlet)
    bcu_inflow = dirichletbc(u_inlet, locate_dofs_topological(V, fdim, ft.find(2)))
    u_ns = np.zeros(gdim, dtype=PETSc.ScalarType)
    bcu_walls = dirichletbc(u_ns, locate_dofs_topological(V, fdim, ft.find(4)), V)
    bcu = [bcu_inflow, bcu_walls]
    bcp = [dirichletbc(PETSc.ScalarType(0),
                       locate_dofs_topological(Q, fdim, ft.find(3)), Q)]

    u = Function(V); u_n = Function(V); u_nm1 = Function(V); u_star = Function(V)
    p = Function(Q); p_n = Function(Q)
    f_ibm = Function(V); tU = Function(V); grad_p = Function(V)
    grad_p_expr = Expression(grad(p_n), V.element.interpolation_points)

    # IBM: 边界环标记 (固定圆柱, mask 保证内部实体)
    # 标记数随 h 缩放，保证 Δs ≈ h/2（细网格更多标记，扩散权重一致）
    n_markers = max(32, int(np.ceil(2*np.pi*r/(h/2.0))))
    th = 2*np.pi*np.arange(n_markers)/n_markers
    mx = c_x + r*np.cos(th); my = c_y + r*np.sin(th)
    dsm = 2*np.pi*r/n_markers
    dV = dsm*h
    from dolfinx.mesh import create_interval
    smesh = create_interval(comm, n_markers-1, [0.0, 1.0])
    Vs = functionspace(smesh, element("Lagrange", smesh.topology.cell_name(), 1, shape=(2,)))
    scoords = Function(Vs)
    scoords.x.array[:] = np.ravel(np.column_stack([mx, my]))
    scoords.x.scatter_forward()
    ibmesh = IBMesh(0.0, L, 0.0, H, Nx, Ny, 2)
    ib_interp = IBInterpolation(ibmesh)
    cbg = Function(V); cbg.interpolate(lambda x: np.array([x[0], x[1]]))
    ibmesh.build_map(cbg._cpp_object)
    ib_interp.evaluate_current_points(scoords._cpp_object)
    svel = Function(Vs); sforce = Function(Vs)

    # 内部掩码
    dof_coords = V.tabulate_dof_coordinates()
    bs = V.dofmap.index_map_bs
    interior_dofs = np.nonzero(
        (dof_coords[:, 0]-c_x)**2 + (dof_coords[:, 1]-c_y)**2 < r**2)[0]
    def mask(fun):
        arr = fun.x.array
        for d in interior_dofs:
            arr[d*bs] = 0.0; arr[d*bs+1] = 0.0
        fun.x.scatter_forward()

    v = TestFunction(V); ut = TrialFunction(V); q = TestFunction(Q); pt = TrialFunction(Q)
    a_pred = form(inner(ut, v)*dx + 1.5*DT*nu*inner(grad(ut), grad(v))*dx)
    A_pred = assemble_matrix(a_pred, bcs=bcu); A_pred.assemble(); b_pred = create_vector(V)
    a_p = form(inner(grad(pt), grad(q))*dx)
    A_p = assemble_matrix(a_p, bcs=bcp); A_p.assemble(); b_p = create_vector(Q)
    a_proj = form(inner(ut, v)*dx)
    A_proj = assemble_matrix(a_proj, bcs=bcu); A_proj.assemble(); b_proj = create_vector(V)
    sp = make_ksp(A_pred, PETSc.KSP.Type.BCGS, PETSc.PC.Type.HYPRE, hypre=True)
    sp2 = make_ksp(A_p, PETSc.KSP.Type.BCGS, PETSc.PC.Type.HYPRE, hypre=True)
    sp3 = make_ksp(A_proj, PETSc.KSP.Type.CG, PETSc.PC.Type.SOR)

    for i in range(NSTEPS):
        t = (i+1)*DT
        inlet.update(t); u_inlet.interpolate(inlet)
        L_pred = form(inner(u_n, v)*dx
                      - 1.5*DT*inner(dot(grad(u_n), u_n), v)*dx
                      + 0.5*DT*inner(dot(grad(u_nm1), u_nm1), v)*dx
                      - 0.5*DT*nu*inner(grad(u_n), grad(v))*dx
                      + 0.5*DT*inner(grad(p_n), v)*dx)
        with b_pred.localForm() as lc: lc.set(0)
        assemble_vector(b_pred, L_pred); apply_lifting(b_pred, [a_pred], [bcu])
        b_pred.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        set_bc(b_pred, bcu); sp.solve(b_pred, u_star.x.petsc_vec); u_star.x.scatter_forward()
        grad_p.interpolate(grad_p_expr); grad_p.x.scatter_forward()
        f_ibm.x.array[:] = 0.0; f_ibm.x.scatter_forward()
        for _ in range(10):
            tU.x.array[:] = u_star.x.array + DT*f_ibm.x.array - 1.5*DT*grad_p.x.array
            tU.x.scatter_forward()
            ib_interp.fluid_to_solid(tU._cpp_object, svel._cpp_object)
            svel.x.scatter_forward()
            sv = svel.x.array; sf = sforce.x.array
            for k in range(n_markers):
                sf[k*2] = (0.0-sv[k*2])/DT*dV
                sf[k*2+1] = (0.0-sv[k*2+1])/DT*dV
            sforce.x.scatter_forward()
            ib_interp.solid_to_fluid(f_ibm._cpp_object, sforce._cpp_object)
            f_ibm.x.scatter_forward()
        u.x.array[:] = u_star.x.array + DT*f_ibm.x.array
        u.x.scatter_forward(); mask(u)
        L_p = form(-(2.0/(3.0*DT))*inner(div(u), q)*dx)
        with b_p.localForm() as lc: lc.set(0)
        assemble_vector(b_p, L_p); apply_lifting(b_p, [a_p], [bcp])
        b_p.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        set_bc(b_p, bcp); sp2.solve(b_p, p.x.petsc_vec); p.x.scatter_forward()
        L_proj = form(inner(u, v)*dx - 1.5*DT*inner(grad(p), v)*dx)
        with b_proj.localForm() as lc: lc.set(0)
        assemble_vector(b_proj, L_proj); apply_lifting(b_proj, [a_proj], [bcu])
        b_proj.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        set_bc(b_proj, bcu); sp3.solve(b_proj, u.x.petsc_vec); u.x.scatter_forward()
        mask(u)
        u_nm1.x.array[:] = u_n.x.array[:]
        u_n.x.array[:] = u.x.array[:]
        p_n.x.array[:] = p.x.array[:]
    Fx = comm.allreduce(assemble_scalar(form(inner(f_ibm, as_vector((1.0, 0.0)))*dx)), op=MPI.SUM)
    Cd = -2.0*Fx/(1.0**2*D)   # U_mean=1 归一化
    uL2 = comm.allreduce(assemble_scalar(form(dot(u, u)*dx)), op=MPI.SUM)
    return mesh, u, p, Cd, uL2


# =====================================================================
def probe_ux(mesh, u, xs):
    tree = bb_tree(mesh, mesh.topology.dim)
    out = []
    for x0 in xs:
        got = np.nan
        for dy in (0.0, 1e-7, -1e-7, 2e-7, -2e-7):
            q = np.array([[x0, 0.2+dy, 0.0]], dtype=np.float64)
            cand = compute_collisions_points(tree, q)
            col = compute_colliding_cells(mesh, cand, q)
            links = col.links(0)
            if len(links) > 0:
                vu = np.ravel(u.eval(q, np.array([links[0]], dtype=np.int32)))
                got = float(vu[0]); break
        out.append(got)
    return np.array(out)


if __name__ == "__main__":
    levels = [
        dict(Nx=110, Ny=21,  LcMax=0.08, res_min=0.010),   # h≈0.02
        dict(Nx=220, Ny=41,  LcMax=0.04, res_min=0.005),   # h≈0.01
        dict(Nx=440, Ny=82,  LcMax=0.02, res_min=0.0025),  # h≈0.005
    ]
    xs = [0.3, 0.4, 0.5, 0.7, 1.0, 1.5]
    if rank == 0:
        print(f"对比 tutorial vs mdf | rho={rho} mu={mu} dt={DT} T_s={TS} (N={NSTEPS}步)")
        print(f"{'级别':<5}{'Nx':>5}{'LcMax':>8}{'mdfCd':>8}{'tutCd':>8}{'Cd差%':>8}"
              f"{'mdf_uL2':>10}{'tut_uL2':>10}")
    rows = []
    for i, lv in enumerate(levels):
        if rank == 0:
            print(f"--- 级别 {i+1}: mdf {lv['Nx']}×{lv['Ny']} | tut LcMax={lv['LcMax']} res_min={lv['res_min']}", flush=True)
        mesh_t, u_t, p_t, Cd_t, uL2_t = solve_tutorial(lv["LcMax"], lv["res_min"])
        mesh_m, u_m, p_m, Cd_m, uL2_m = solve_mdf(lv["Nx"], lv["Ny"])
        ux_t = probe_ux(mesh_t, u_t, xs)
        ux_m = probe_ux(mesh_m, u_m, xs)
        cd_err = 100.0 * (Cd_m - Cd_t) / abs(Cd_t) if abs(Cd_t) > 1e-9 else np.nan
        if rank == 0:
            print(f"{i+1:<5}{lv['Nx']:>5}{lv['LcMax']:>8.3f}{Cd_m:>8.3f}{Cd_t:>8.3f}"
                  f"{cd_err:>8.1f}{uL2_m:>10.5f}{uL2_t:>10.5f}", flush=True)
            print("  下游中心线 ux 探针:  x |  mdf   |  tut  | 差%")
            for j, x0 in enumerate(xs):
                if not np.isnan(ux_m[j]) and not np.isnan(ux_t[j]) and abs(ux_t[j]) > 1e-6:
                    e = 100.0*(ux_m[j]-ux_t[j])/abs(ux_t[j])
                else:
                    e = np.nan
                print(f"    {x0:.2f} | {ux_m[j]:6.4f} | {ux_t[j]:6.4f} | {e:6.1f}")
        rows.append((i+1, lv["Nx"], Cd_m, Cd_t, cd_err, uL2_m, uL2_t))
    if rank == 0:
        print("\n=== 汇总 ===")
        print(f"{'级别':<5}{'Nx':>5}{'mdfCd':>8}{'tutCd':>8}{'Cd差%':>8}{'uL2差%':>8}")
        for i, Nx, cdm, cdt, cde, u2m, u2t in rows:
            ue = 100.0*(u2m-u2t)/u2t
            print(f"{i:<5}{Nx:>5}{cdm:>8.3f}{cdt:>8.3f}{cde:>8.1f}{ue:>8.1f}")
