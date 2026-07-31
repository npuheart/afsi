#!/usr/bin/env python
"""Compare multi-direct forcing (新 demo) vs body_fitted — 同参数、同 t。

两种方法各自推进 N 步（统一参数），在相同时刻比较：
  - 全场 u_L2 / p_L2 范数
  - 下游中心线探针 (y=0.2) 速度与压力
  - Cd：body_fitted 用表面应力积分；multi-direct forcing 用体积力积分
    (F = -ρ∫IBMf_x dV)，另加 DFIBMFoam 式两点压力探针 Cd

用法:
  conda activate afsi-dolfinx
  STEPS=2500 python compare_mdf_bf.py
"""
import os
import numpy as np
from petsc4py import PETSc
from mpi4py import MPI

import dolfinx
from dolfinx.fem import (Function, functionspace, dirichletbc,
                         locate_dofs_topological, form, assemble_scalar,
                         Expression)
from dolfinx.fem.petsc import (assemble_matrix, assemble_vector,
                               apply_lifting, set_bc, create_vector)
from dolfinx.mesh import CellType, GhostMode, create_interval
from basix.ufl import element
from ufl import (TestFunction, TrialFunction, dot, dx, inner, grad, div,
                 as_vector, FacetNormal, Measure)

from afsic import IBMesh, IBInterpolation
from afsic.common import (tag_boundaries, rectangle_boundaries, TurekInlet,
                          MARKER_LEFT, MARKER_RIGHT, MARKER_BOTTOM, MARKER_TOP)
import dolfinx.geometry as dg

comm = MPI.COMM_WORLD
rank = comm.rank
N = int(os.environ.get("STEPS", "2500"))

P = dict(Um=1.0, rho=1000.0, mu=1.0, Lx=2.2, Ly=0.41,
         Nx=220, Ny=41, T=10.0, dt=0.001, D=0.1,
         cx=0.2, cy=0.2, r=0.05)
BASE = os.path.dirname(os.path.abspath(__file__))
n_iter = 10


def make_solver(A, ksp_type, pc_type):
    s = PETSc.KSP().create(comm)
    s.setOperators(A); s.setType(ksp_type)
    pc = s.getPC(); pc.setType(pc_type)
    if pc_type == PETSc.PC.Type.HYPRE:
        pc.setHYPREType("boomeramg")
    return s


# ---------------------------------------------------------------------------
# body_fitted
# ---------------------------------------------------------------------------
def run_body_fitted():
    from dolfinx.io import gmsh as gmshio
    md = gmshio.read_from_msh(os.path.join(BASE, "body_fitted", "channel_hole.msh"),
                              comm, gdim=2)
    mesh, _, facet_tags = md[0], md[1], md[2]
    mesh.topology.create_connectivity(1, 2)
    fdim = mesh.topology.dim - 1
    V = functionspace(mesh, element("Lagrange", mesh.topology.cell_name(), 2, shape=(2,)))
    Q = functionspace(mesh, element("Lagrange", mesh.topology.cell_name(), 1))
    inlet = TurekInlet(Um=P["Um"], Ly=P["Ly"])
    ui = Function(V); ui.interpolate(inlet)
    bci = dirichletbc(ui, locate_dofs_topological(V, fdim, facet_tags.find(11)))
    u0 = Function(V); u0.x.array[:] = 0.0
    bcb = dirichletbc(u0, locate_dofs_topological(V, fdim, facet_tags.find(13)))
    bct = dirichletbc(u0, locate_dofs_topological(V, fdim, facet_tags.find(14)))
    bcc = dirichletbc(u0, locate_dofs_topological(V, fdim, facet_tags.find(15)))
    bcp = dirichletbc(PETSc.ScalarType(0.0),
                      locate_dofs_topological(Q, fdim, facet_tags.find(12)), Q)
    from afsic import ChorinSolver
    sol = ChorinSolver(V, Q, [bci, bcb, bct, bcc], [bcp], P["dt"], P["rho"], P["mu"])
    for k in range(N):
        t = k * P["dt"]; inlet.update(t); ui.interpolate(inlet)
        sol.solve_one_step()
    n = FacetNormal(mesh); e_x = as_vector((1.0, 0.0))
    dsM = Measure("ds", domain=mesh, subdomain_data=facet_tags)
    traction = -sol.p_ * n + P["mu"] * dot(grad(sol.u_) + grad(sol.u_).T, n)
    fd = assemble_scalar(form(dot(traction, e_x) * dsM(15)))
    fd = mesh.comm.allreduce(fd, op=MPI.SUM)
    Cd = -2.0 * fd / (P["rho"] * P["Um"]**2 * P["D"])
    return mesh, sol, dict(Cd=Cd)


# ---------------------------------------------------------------------------
# multi-direct forcing（DFIBMFoam 移植）
# ---------------------------------------------------------------------------
def run_mdf():
    Lx, Ly = P["Lx"], P["Ly"]; Nx, Ny = P["Nx"], P["Ny"]
    rho, mu, Um, D = P["rho"], P["mu"], P["Um"], P["D"]
    dt = P["dt"]; nu = mu / rho
    cx, cy, r = P["cx"], P["cy"], P["r"]
    h = np.sqrt((Lx / Nx) * (Ly / Ny))

    mesh = dolfinx.mesh.create_rectangle(comm, ((0, 0), (Lx, Ly)), (Nx, Ny),
                                         cell_type=CellType.quadrilateral,
                                         ghost_mode=GhostMode.shared_facet)
    ftag = tag_boundaries(mesh, rectangle_boundaries(Lx, Ly))
    fdim = mesh.topology.dim - 1
    V = functionspace(mesh, element("Lagrange", mesh.topology.cell_name(), 2, shape=(2,)))
    Q = functionspace(mesh, element("Lagrange", mesh.topology.cell_name(), 1))

    inlet = TurekInlet(Um=Um, Ly=Ly)
    ui = Function(V); ui.interpolate(inlet)
    bci = dirichletbc(ui, locate_dofs_topological(V, fdim, ftag.find(MARKER_LEFT)))
    u0 = Function(V); u0.x.array[:] = 0.0
    bcb = dirichletbc(u0, locate_dofs_topological(V, fdim, ftag.find(MARKER_BOTTOM)))
    bct = dirichletbc(u0, locate_dofs_topological(V, fdim, ftag.find(MARKER_TOP)))
    bcp = dirichletbc(PETSc.ScalarType(0.0),
                      locate_dofs_topological(Q, fdim, ftag.find(MARKER_RIGHT)), Q)
    bcu = [bci, bcb, bct]; bcp_ = [bcp]

    u = Function(V); u_n = Function(V); u_nm1 = Function(V); u_star = Function(V)
    p = Function(Q); p_n = Function(Q)
    f_ibm = Function(V); tU = Function(V); grad_p = Function(V)
    grad_p_expr = Expression(grad(p_n), V.element.interpolation_points)

    # IB：填充圆盘的均匀格点（disk 模式，内部也强制 u=0）
    ibmesh = IBMesh(0.0, Lx, 0.0, Ly, Nx, Ny, 2)
    ib_interp = IBInterpolation(ibmesh)
    cbg = Function(V); cbg.interpolate(lambda x: np.array([x[0], x[1]]))
    ibmesh.build_map(cbg._cpp_object)
    a_m = 0.5 * h
    n_ax = max(2, int(np.ceil(2 * r / a_m)))
    pax = np.linspace(cx - r, cx + r, n_ax + 1)
    X, Y = np.meshgrid(pax, pax)
    keep = (X - cx) ** 2 + (Y - cy) ** 2 < r ** 2
    mx = X[keep]; my = Y[keep]
    n_markers = len(mx)
    dVm = a_m * a_m * np.ones(n_markers)
    smesh = create_interval(comm, n_markers - 1, [0.0, 1.0])
    Vs = functionspace(smesh, element("Lagrange", smesh.topology.cell_name(), 1, shape=(2,)))
    scoords = Function(Vs)
    scoords.x.array[:] = np.ravel(np.column_stack([mx, my]))
    scoords.x.scatter_forward()
    ib_interp.evaluate_current_points(scoords._cpp_object)
    svel = Function(Vs); sforce = Function(Vs)

    # matrices
    v = TestFunction(V); ut = TrialFunction(V); q = TestFunction(Q); pt = TrialFunction(Q)
    a_pred = form(inner(ut, v) * dx + 1.5 * dt * nu * inner(grad(ut), grad(v)) * dx)
    A_pred = assemble_matrix(a_pred, bcs=bcu); A_pred.assemble(); b_pred = create_vector(V)
    a_p = form(inner(grad(pt), grad(q)) * dx)
    A_p = assemble_matrix(a_p, bcs=bcp_); A_p.assemble(); b_p = create_vector(Q)
    a_proj = form(inner(ut, v) * dx)
    A_proj = assemble_matrix(a_proj, bcs=bcu); A_proj.assemble(); b_proj = create_vector(V)
    s_pred = make_solver(A_pred, PETSc.KSP.Type.BCGS, PETSc.PC.Type.HYPRE)
    s_p = make_solver(A_p, PETSc.KSP.Type.BCGS, PETSc.PC.Type.HYPRE)
    s_proj = make_solver(A_proj, PETSc.KSP.Type.CG, PETSc.PC.Type.SOR)

    for k in range(N):
        t = k * dt; inlet.update(t); ui.interpolate(inlet)
        L_pred = form(inner(u_n, v) * dx
                      - 1.5 * dt * inner(dot(grad(u_n), u_n), v) * dx
                      + 0.5 * dt * inner(dot(grad(u_nm1), u_nm1), v) * dx
                      - 0.5 * dt * nu * inner(grad(u_n), grad(v)) * dx
                      + 0.5 * dt * inner(grad(p_n), v) * dx)
        with b_pred.localForm() as lc: lc.set(0)
        assemble_vector(b_pred, L_pred)
        apply_lifting(b_pred, [a_pred], [bcu])
        b_pred.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        set_bc(b_pred, bcu)
        s_pred.solve(b_pred, u_star.x.petsc_vec); u_star.x.scatter_forward()

        grad_p.interpolate(grad_p_expr); grad_p.x.scatter_forward()
        f_ibm.x.array[:] = 0.0; f_ibm.x.scatter_forward()
        for _ in range(n_iter):
            tU.x.array[:] = u_star.x.array + dt * f_ibm.x.array - 1.5 * dt * grad_p.x.array
            tU.x.scatter_forward()
            ib_interp.fluid_to_solid(tU._cpp_object, svel._cpp_object)
            svel.x.scatter_forward()
            sv = svel.x.array; sf = sforce.x.array
            for m in range(n_markers):
                sf[m*2] = (0.0 - sv[m*2]) / dt * dVm[m]
                sf[m*2+1] = (0.0 - sv[m*2+1]) / dt * dVm[m]
            sforce.x.scatter_forward()
            ib_interp.solid_to_fluid(f_ibm._cpp_object, sforce._cpp_object)
            f_ibm.x.scatter_forward()
        u.x.array[:] = u_star.x.array + dt * f_ibm.x.array
        u.x.scatter_forward()

        L_p = form(-(2.0 / (3.0 * dt)) * inner(div(u), q) * dx)
        with b_p.localForm() as lc: lc.set(0)
        assemble_vector(b_p, L_p)
        apply_lifting(b_p, [a_p], [bcp_])
        b_p.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        set_bc(b_p, bcp_)
        s_p.solve(b_p, p.x.petsc_vec); p.x.scatter_forward()

        L_proj = form(inner(u, v) * dx - 1.5 * dt * inner(grad(p), v) * dx)
        with b_proj.localForm() as lc: lc.set(0)
        assemble_vector(b_proj, L_proj)
        apply_lifting(b_proj, [a_proj], [bcu])
        b_proj.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        set_bc(b_proj, bcu)
        s_proj.solve(b_proj, u.x.petsc_vec); u.x.scatter_forward()

        u_nm1.x.array[:] = u_n.x.array[:]
        u_n.x.array[:] = u.x.array[:]
        p_n.x.array[:] = p.x.array[:]

    form_Fx = form(inner(f_ibm, as_vector((1.0, 0.0))) * dx)
    Fx = comm.allreduce(assemble_scalar(form_Fx), op=MPI.SUM)
    Cd_body = -2.0 * Fx / (Um**2 * D)
    return mesh, (u, p), dict(Cd_bodyforce=Cd_body, f_ibm=f_ibm)


# ---------------------------------------------------------------------------
def probes(mesh, u, p):
    pts = np.array([(0.18, 0.2), (0.25, 0.2), (0.3, 0.2), (0.4, 0.2),
                    (0.5, 0.2), (0.7, 0.2), (1.0, 0.2), (1.5, 0.2)], dtype=np.float64)
    tree = dg.bb_tree(mesh, mesh.topology.dim)
    out = []
    for i in range(len(pts)):
        x0, y0 = pts[i]; found = False
        for dy in (0.0, 1e-7, -1e-7, 2e-7, -2e-7):
            q = np.array([[x0, y0 + dy, 0.0]], dtype=np.float64)
            cand = dg.compute_collisions_points(tree, q)
            col = dg.compute_colliding_cells(mesh, cand, q)
            links = col.links(0)
            if len(links) > 0:
                cell = links[0]
                vu = np.ravel(u.eval(q, np.array([cell], dtype=np.int32)))
                vp = np.ravel(p.eval(q, np.array([cell], dtype=np.int32)))
                out.append((float(x0), float(vu[0]), float(vu[1]), float(vp[0])))
                found = True; break
        if not found:
            out.append((float(x0), np.nan, np.nan, np.nan))
    return out


if __name__ == "__main__":
    t_end = N * P["dt"]
    if rank == 0:
        print(f"比较 multi-direct forcing vs body_fitted，N={N} 步，t_end={t_end:.2f}s\n")

    mesh_b, sol_b, res_b = run_body_fitted()
    if rank == 0:
        print(f"[body_fitted] Cd(表面应力) = {res_b['Cd']:.4f}")

    mesh_d, (u_d, p_d), res_d = run_mdf()
    if rank == 0:
        print(f"[mdf] Cd(体积力积分) = {res_d['Cd_bodyforce']:.4f}")

    fu = form(dot(sol_b.u_, sol_b.u_) * dx)
    fp = form(dot(sol_b.p_ / P["rho"], sol_b.p_ / P["rho"]) * dx)  # 运动学压力
    uL2_b = comm.allreduce(assemble_scalar(fu), op=MPI.SUM)
    pL2_b = comm.allreduce(assemble_scalar(fp), op=MPI.SUM)
    fu = form(dot(u_d, u_d) * dx); fp = form(dot(p_d, p_d) * dx)
    uL2_d = comm.allreduce(assemble_scalar(fu), op=MPI.SUM)
    pL2_d = comm.allreduce(assemble_scalar(fp), op=MPI.SUM)

    pb_b = probes(mesh_b, sol_b.u_, sol_b.p_)
    pb_d = probes(mesh_d, u_d, p_d)

    if rank == 0:
        print("\n" + "=" * 80)
        print(f"流场范数 @ t={t_end:.2f}s")
        print(f"  {'':<18}{'body_fitted':>14}{'mdf':>14}")
        print(f"  {'u_L2':<18}{uL2_b:>14.6f}{uL2_d:>14.6f}")
        print(f"  {'p_L2':<18}{pL2_b:>14.4f}{pL2_d:>14.4f}")
        print("\n下游中心线探针 (y=0.2):  (u_x, u_y, p_kin)")
        print(f"  {'x':>6} | {'body: ux':>10}{'uy':>10}{'p/rho':>10} | "
              f"{'mdf: ux':>10}{'uy':>10}{'p_kin':>10}")
        for r_b, r_d in zip(pb_b, pb_d):
            xb, uxb, uyb, pbv = r_b
            xd, uxd, uyd, pdv = r_d
            # body_fitted 的 Chorin 压力含 rho/dt 缩放 → p_kin = p/rho
            pb_kin = pbv / P["rho"]
            print(f"  {xb:>6.2f} | {uxb:>10.5f}{uyb:>10.5f}{pb_kin:>10.3f} | "
                  f"{uxd:>10.5f}{uyd:>10.5f}{pdv:>10.3f}")
        print("\n阻力:")
        print(f"  body_fitted    Cd(表面应力)   = {res_b['Cd']:.4f}")
        print(f"  multi-direct   Cd(体积力积分) = {res_d['Cd_bodyforce']:.4f}")
        print(f"  (DFG 2D-3 Re=100 参考: Cd≈5.57, Cl幅值≈0.0106)")
        print("=" * 80)
