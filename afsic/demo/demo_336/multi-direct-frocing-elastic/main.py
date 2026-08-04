"""demo_336 方腔驱动圆盘 — multi-direct-frocing-elastic：弹性固体版求解。

在 multi-direct-frocing（mdf 版，afsic/demo/demo_336/multi-direct-frocing）基础上
**增加固体的推进方程**：圆盘为**弹性固体**（可变形、有本构），而非刚体。

流体（与 mdf 版相同，AB2 预测 + 压力泊松 + L2 投影）:
  (U* - U^n)/dt + 1.5·(U^n·∇)U^n - 0.5·(U^{n-1}·∇)U^{n-1}
      = 1.5·ν∇²U* - 0.5·ν∇²U^n + 0.5·∇p^n + f_ibm
  ∇²p^{n+1} = (2/(3dt))·∇·U,   U^{n+1} = U - 1.5dt·∇p

固体（弹性，总拉格朗日）:
  推进方程（运动学平流）:
    X_s^{n+1} = X_s^n + V_s^n·dt,      V_s^n = 流体速度插值到固体节点
  本构（可压缩 neo-Hookean 型）:
    P(F) = μ_s(F - F^{-T}) + λ_s·ln(det F)·F^{-T},   F = ∇X_s
  节点力（弱形式）:
    F_solid = -∫ P(F) : ∇δv dx    （对参考构型积分）

耦合（每步一次，分区显式）:
  fluid_to_solid(u*, V_s) → 推进固体 → 组装弹性力 →
  solid_to_fluid(f_ibm, F_solid) → u = u* + dt·f_ibm → 投影

关于"固体与流体是否相对解耦"：是。本方案是**分区显式（staggered）耦合**——
每步依次求解流体方程、固体方程，只通过界面插值/力扩散交换一次信息，
不构成把两者联立成一个方程组的 monolithic 求解，故两者"相对解耦"。
（代价：显式耦合对"重固体"（ρ_s>>ρ_f）存在 added-mass 失稳风险；本 demo
圆盘质量忽略、密度量级与流体相当，稳定。）

输出：output/velocity.xdmf、pressure.xdmf、solid.xdmf（参考网格+位移场 u）、
      solid_force.xdmf、forces.csv。
"""

import os
import numpy as np
from petsc4py import PETSc
from mpi4py import MPI

import dolfinx
from dolfinx import log
from dolfinx.fem import (Function, functionspace, dirichletbc,
                         locate_dofs_topological, form, assemble_scalar,
                         Expression)
from dolfinx.fem.petsc import (assemble_matrix, assemble_vector,
                               apply_lifting, set_bc, create_vector)
from dolfinx.mesh import CellType, GhostMode
import ufl
from basix.ufl import element
from ufl import (TestFunction, TrialFunction, dot, dx, inner, grad, div,
                 as_vector, SpatialCoordinate, inv, ln, det)

from afsic import IBMesh, IBInterpolation
from afsic.common import tag_boundaries, MARKER_LEFT, MARKER_RIGHT, \
    MARKER_BOTTOM, MARKER_TOP

from configuration import config

comm = MPI.COMM_WORLD
rank = comm.rank

# ---------------------------------------------------------------------------
# 参数
# ---------------------------------------------------------------------------
x0, y0 = config["x0"], config["y0"]
Lx, Ly = config["Lx"], config["Ly"]
Nx, Ny = config["Nx"], config["Ny"]
U_lid, rho, mu = config["U_lid"], config["rho"], config["mu"]
T, dt = config["T"], config["dt"]
cx0, cy0, r = config["cx"], config["cy"], config["r"]
D = config["D"]
mu_s, lambda_s = config["mu_s"], config["lambda_s"]
nu = mu / rho
h = np.sqrt((Lx / Nx) * (Ly / Ny))

# ---------------------------------------------------------------------------
# 流体网格与函数空间（方腔，顶盖滑动，闭合腔）
# ---------------------------------------------------------------------------
mesh = dolfinx.mesh.create_rectangle(
    comm, ((x0, y0), (x0 + Lx, y0 + Ly)), (Nx, Ny),
    cell_type=CellType.quadrilateral, ghost_mode=GhostMode.shared_facet)
boundaries = [
    (MARKER_LEFT,   lambda x: np.isclose(x[0], x0)),
    (MARKER_RIGHT,  lambda x: np.isclose(x[0], x0 + Lx)),
    (MARKER_BOTTOM, lambda x: np.isclose(x[1], y0)),
    (MARKER_TOP,    lambda x: np.isclose(x[1], y0 + Ly)),
]
facet_tag = tag_boundaries(mesh, boundaries)
fdim = mesh.topology.dim - 1

v_cg2 = element("Lagrange", mesh.topology.cell_name(), 2, shape=(2,))
s_cg1 = element("Lagrange", mesh.topology.cell_name(), 1)
V = functionspace(mesh, v_cg2)
Q = functionspace(mesh, s_cg1)

# ---------------------------------------------------------------------------
# 边界条件：顶盖滑动 + 三壁无滑移 + 压力钉角点
# ---------------------------------------------------------------------------
u_lid = Function(V)
u_lid.interpolate(lambda x: np.array([np.full(x.shape[1], U_lid),
                                      np.zeros(x.shape[1])]))
bcu_lid = dirichletbc(u_lid, locate_dofs_topological(
    V, fdim, facet_tag.find(MARKER_TOP)))
u_zero = Function(V); u_zero.x.array[:] = 0.0
bcu = [
    bcu_lid,
    dirichletbc(u_zero, locate_dofs_topological(V, fdim, facet_tag.find(MARKER_LEFT))),
    dirichletbc(u_zero, locate_dofs_topological(V, fdim, facet_tag.find(MARKER_RIGHT))),
    dirichletbc(u_zero, locate_dofs_topological(V, fdim, facet_tag.find(MARKER_BOTTOM))),
]
q_coords = Q.tabulate_dof_coordinates()
corner = np.array([x0, y0, 0.0])
dof_pin = int(np.argmin(np.linalg.norm(q_coords - corner, axis=1)))
bcp = [dirichletbc(PETSc.ScalarType(0.0), np.array([dof_pin], dtype=np.int32), Q)]

# ---------------------------------------------------------------------------
# 流体场
# ---------------------------------------------------------------------------
u = Function(V)        # U^{n+1}
u_n = Function(V)      # U^n
u_nm1 = Function(V)    # U^{n-1}
u_star = Function(V)   # U*
p = Function(Q)        # p^{n+1}
p_n = Function(Q)      # p^n
f_ibm = Function(V)    # 固体弹性力扩散到流体的体积力场
grad_p = Function(V)
grad_p_expr = Expression(grad(p_n), V.element.interpolation_points)

# ---------------------------------------------------------------------------
# 弹性固体：三角化圆盘网格（参考构型），P2 求解
# ---------------------------------------------------------------------------
def make_disk_mesh(r, n_rings, n_circ):
    """生成三角化圆盘网格（体坐标，相对圆心），返回 (节点坐标, 三角形连接)。"""
    nodes = [(0.0, 0.0)]
    for ring in range(1, n_rings + 1):
        rr = r * ring / n_rings
        for k in range(n_circ):
            ang = 2.0 * np.pi * k / n_circ
            nodes.append((rr * np.cos(ang), rr * np.sin(ang)))
    cells = []
    for k in range(n_circ):
        k2 = (k + 1) % n_circ
        cells.append((0, 1 + k, 1 + k2))
    for ring in range(1, n_rings):
        b1 = 1 + (ring - 1) * n_circ
        b2 = 1 + ring * n_circ
        for k in range(n_circ):
            k2 = (k + 1) % n_circ
            a, b, c, d = b1 + k, b1 + k2, b2 + k, b2 + k2
            cells.append((a, b, c))
            cells.append((c, b, d))
    return np.array(nodes, dtype=np.float64), np.array(cells, dtype=np.int64)


disk_base, disk_cells = make_disk_mesh(r, config["solid_rings"], config["solid_circ"])
solid_mesh = dolfinx.mesh.create_mesh(
    comm, disk_cells, ufl.Mesh(element("Lagrange", "triangle", 1, shape=(2,))),
    disk_base + np.array([cx0, cy0]))

v_s2 = element("Lagrange", solid_mesh.topology.cell_name(), 2, shape=(2,))
v_s1 = element("Lagrange", solid_mesh.topology.cell_name(), 1, shape=(2,))
Vs = functionspace(solid_mesh, v_s2)
Vs_io = functionspace(solid_mesh, v_s1)

solid_coords = Function(Vs, name="solid_coords")
solid_coords.interpolate(lambda x: np.array([x[0], x[1]]))
solid_coords.x.scatter_forward()
solid_velocity = Function(Vs)   # 插值自流体的节点速度
solid_force = Function(Vs)      # 弹性节点力（供扩散回流体）
ref_coords = Vs.tabulate_dof_coordinates()   # 参考节点坐标（位移 = 当前 - 参考）

# ---------------------------------------------------------------------------
# IBM（标记 = 固体节点；Peskin 4点核，afsic 内置）
# ---------------------------------------------------------------------------
ibmesh = IBMesh(x0, x0 + Lx, y0, y0 + Ly, Nx, Ny, config["velocity_order"])
ib_interp = IBInterpolation(ibmesh)
coords_bg = Function(V)
coords_bg.interpolate(lambda x: np.array([x[0], x[1]]))
ibmesh.build_map(coords_bg._cpp_object)
ib_interp.evaluate_current_points(solid_coords._cpp_object)

# --- 弹性力弱形式（总拉格朗日，参考构型积分） ---
dVs = TestFunction(Vs)
FF = grad(solid_coords)
P = mu_s * (FF - inv(FF).T) + lambda_s * ln(det(FF)) * inv(FF).T
L_solid = form(-inner(P, grad(dVs)) * dx)
b_solid = create_vector(Vs)
form_volume = form(det(FF) * dx)

# ---------------------------------------------------------------------------
# 流体矩阵（组装一次）
# ---------------------------------------------------------------------------
v = TestFunction(V)
u_trial = TrialFunction(V)
q = TestFunction(Q)
p_trial = TrialFunction(Q)

a_pred = form(inner(u_trial, v) * dx
              + 1.5 * dt * nu * inner(grad(u_trial), grad(v)) * dx)
A_pred = assemble_matrix(a_pred, bcs=bcu); A_pred.assemble()
b_pred = create_vector(V)

a_p = form(inner(grad(p_trial), grad(q)) * dx)
A_p = assemble_matrix(a_p, bcs=bcp); A_p.assemble()
b_p = create_vector(Q)

a_proj = form(inner(u_trial, v) * dx)
A_proj = assemble_matrix(a_proj, bcs=bcu); A_proj.assemble()
b_proj = create_vector(V)


def make_solver(A, ksp_type, pc_type):
    s = PETSc.KSP().create(mesh.comm)
    s.setOperators(A)
    s.setType(ksp_type)
    pc = s.getPC()
    pc.setType(pc_type)
    if pc_type == PETSc.PC.Type.HYPRE:
        pc.setHYPREType("boomeramg")
    return s


solver_pred = make_solver(A_pred, PETSc.KSP.Type.BCGS, PETSc.PC.Type.HYPRE)
solver_p = make_solver(A_p, PETSc.KSP.Type.BCGS, PETSc.PC.Type.HYPRE)
solver_proj = make_solver(A_proj, PETSc.KSP.Type.CG, PETSc.PC.Type.SOR)

# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------
u_io = Function(functionspace(mesh, element(
    "Lagrange", mesh.topology.cell_name(), 1, shape=(2,))))
p_io = Function(Q)
os.makedirs(config["output_path"], exist_ok=True) if rank == 0 else None
file_vel = dolfinx.io.XDMFFile(mesh.comm, config["output_path"] + "velocity.xdmf", "w")
file_pre = dolfinx.io.XDMFFile(mesh.comm, config["output_path"] + "pressure.xdmf", "w")
file_vel.write_mesh(mesh); file_pre.write_mesh(mesh)

solid_disp_io = Function(Vs_io)     # 位移场（参考→当前，ParaView Warp by vector）
solid_force_io = Function(Vs_io)    # 弹性力场
file_solid = file_sf = None
if config["write_solid"]:
    file_solid = dolfinx.io.XDMFFile(mesh.comm, config["output_path"] + "solid.xdmf", "w")
    file_solid.write_mesh(solid_mesh)
    file_sf = dolfinx.io.XDMFFile(mesh.comm, config["output_path"] + "solid_force.xdmf", "w")
    file_sf.write_mesh(solid_mesh)

forces_path = config["output_path"] + "forces.csv"
if rank == 0:
    with open(forces_path, "w") as fh:
        fh.write("t,Fx,Fy,disp_max,volume\n")

form_u_L2 = form(dot(u, u) * dx)
form_p_L2 = form(dot(p, p) * dx)
form_Fx = form(inner(f_ibm, as_vector((1.0, 0.0))) * dx)
form_Fy = form(inner(f_ibm, as_vector((0.0, 1.0))) * dx)

log.set_log_level(log.LogLevel.INFO)
if rank == 0:
    Re = rho * U_lid * Lx / mu
    print(f"Lid-driven cavity + ELASTIC disk (multi-direct-frocing-elastic): "
          f"{Nx}×{Ny}, dt={dt}, Re≈{Re:.0f}")
    print(f"  solid: P2 on {len(disk_base)}-node disk mesh, "
          f"mu_s={mu_s}, lambda_s={lambda_s}, h={h*1e3:.2f}mm")

num_steps = config["num_steps"]
out_interval = config["out_interval"]


def _chk(tag, f):
    """NaN/Inf 检测：发现即打印阶段（诊断显式耦合失稳用）。"""
    if not np.isfinite(f.x.array).all():
        if rank == 0:
            n = int(np.isnan(f.x.array).sum())
            print(f"  [NaN] {tag} at t={step*dt:.4f}: {n} NaN", flush=True)
        return True
    return False

failed = False
fail_t = None

# ---------------------------------------------------------------------------
# 时间循环（分区显式：流体 → 固体 → 力交换 → 投影）
# ---------------------------------------------------------------------------
for step in range(num_steps):
    t = step * dt
    tt = t + dt

    # ---- 1) 预测步 (AB2，无固体力) ----
    L_pred = (
        inner(u_n, v) * dx
        - 1.5 * dt * inner(dot(grad(u_n), u_n), v) * dx
        + 0.5 * dt * inner(dot(grad(u_nm1), u_nm1), v) * dx
        - 0.5 * dt * nu * inner(grad(u_n), grad(v)) * dx
        + 0.5 * dt * inner(grad(p_n), v) * dx
    )
    L_pred = form(L_pred)
    with b_pred.localForm() as lc:
        lc.set(0)
    assemble_vector(b_pred, L_pred)
    apply_lifting(b_pred, [a_pred], [bcu])
    b_pred.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                       mode=PETSc.ScatterMode.REVERSE)
    set_bc(b_pred, bcu)
    solver_pred.solve(b_pred, u_star.x.petsc_vec)
    u_star.x.scatter_forward()
    if _chk("u_star", u_star): failed, fail_t = True, tt; break

    # ---- 2) 固体推进 + 弹性力耦合（每步一次，显式） ----
    #    a) 流体速度 → 固体节点
    ib_interp.fluid_to_solid(u_star._cpp_object, solid_velocity._cpp_object)
    solid_velocity.x.scatter_forward()
    if _chk("solid_vel", solid_velocity): failed, fail_t = True, tt; break
    #    b) 推进方程: X_s += V_s·dt
    solid_coords.x.array[:] += solid_velocity.x.array[:] * dt
    solid_coords.x.scatter_forward()
    #    c) 更新标记并组装弹性力
    ib_interp.evaluate_current_points(solid_coords._cpp_object)
    with b_solid.localForm() as loc:
        loc.set(0)
    assemble_vector(b_solid, L_solid)
    b_solid.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                        mode=PETSc.ScatterMode.REVERSE)
    solid_force.x.array[:] = b_solid.array[:]
    solid_force.x.scatter_forward()
    if _chk("solid_force", solid_force): failed, fail_t = True, tt; break
    #    d) 弹性力扩散回流体（C++ assign_dofs 为"替换"，单次调用即正确）
    ib_interp.solid_to_fluid(f_ibm._cpp_object, solid_force._cpp_object)
    f_ibm.x.scatter_forward()
    if _chk("f_ibm", f_ibm): failed, fail_t = True, tt; break
    #    e) 流体获得弹性冲量: U = U* + dt·f_ibm
    u.x.array[:] = u_star.x.array + dt * f_ibm.x.array
    u.x.scatter_forward()
    if _chk("u_mid", u): failed, fail_t = True, tt; break

    # ---- 3) 压力泊松: ∇²p = (2/(3dt))∇·U ----
    L_p = form(-(2.0 / (3.0 * dt)) * inner(div(u), q) * dx)
    with b_p.localForm() as lc:
        lc.set(0)
    assemble_vector(b_p, L_p)
    apply_lifting(b_p, [a_p], [bcp])
    b_p.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                    mode=PETSc.ScatterMode.REVERSE)
    set_bc(b_p, bcp)
    solver_p.solve(b_p, p.x.petsc_vec)
    p.x.scatter_forward()
    if _chk("p", p): failed, fail_t = True, tt; break

    # ---- 4) 速度修正 (L2 投影): U^{n+1} = U - 1.5dt·∇p ----
    L_proj = form(inner(u, v) * dx - 1.5 * dt * inner(grad(p), v) * dx)
    with b_proj.localForm() as lc:
        lc.set(0)
    assemble_vector(b_proj, L_proj)
    apply_lifting(b_proj, [a_proj], [bcu])
    b_proj.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                       mode=PETSc.ScatterMode.REVERSE)
    set_bc(b_proj, bcu)
    solver_proj.solve(b_proj, u.x.petsc_vec)
    u.x.scatter_forward()
    if _chk("u_final", u): failed, fail_t = True, tt; break

    # ---- 5) 更新历史场 ----
    u_nm1.x.array[:] = u_n.x.array[:]
    u_n.x.array[:] = u.x.array[:]
    p_n.x.array[:] = p.x.array[:]

    # ---- 监控 / 输出 ----
    u_L2 = comm.allreduce(assemble_scalar(form_u_L2), op=MPI.SUM)
    p_L2 = comm.allreduce(assemble_scalar(form_p_L2), op=MPI.SUM)
    F_x = comm.allreduce(assemble_scalar(form_Fx), op=MPI.SUM)
    F_y = comm.allreduce(assemble_scalar(form_Fy), op=MPI.SUM)
    volume = comm.allreduce(assemble_scalar(form_volume), op=MPI.SUM)
    disp = solid_coords.x.array - np.ravel(ref_coords[:, :2])
    disp_max = float(np.max(np.linalg.norm(disp.reshape(-1, 2), axis=1)))

    if step % out_interval == 0 or step == num_steps - 1:
        u_io.interpolate(u)
        p_io.interpolate(p)
        file_vel.write_function(u_io, tt)
        file_pre.write_function(p_io, tt)
        if file_solid is not None:
            # 位移场 u = X_s - X_ref（参考网格不变，ParaView Warp by vector 查看）
            disp2 = Function(Vs)
            disp2.x.array[:] = solid_coords.x.array - np.ravel(ref_coords[:, :2])
            disp2.x.scatter_forward()
            solid_disp_io.interpolate(disp2)
            file_solid.write_function(solid_disp_io, tt)
            solid_force_io.interpolate(solid_force)
            file_sf.write_function(solid_force_io, tt)
        if rank == 0:
            with open(forces_path, "a") as fh:
                fh.write(f"{tt:.5f},{F_x:.6e},{F_y:.6e},{disp_max:.6e},{volume:.6e}\n")
            print(f"Step {step+1}/{num_steps}, t={tt:.3f}s, "
                  f"u_L2={u_L2:.4f}, p_L2={p_L2:.1f}, "
                  f"Fx={F_x:.4f}, Fy={F_y:+.4f}, "
                  f"|disp|max={disp_max:.5f}, vol={volume:.4f}", flush=True)

file_vel.close(); file_pre.close()
if file_solid is not None:
    file_solid.close(); file_sf.close()
if rank == 0:
    if failed:
        print(f"\n[注意] 显式弹性耦合在 t={fail_t:.3f}s 处固体网格翻转 "
              f"(det F→0) 导致 NaN，提前终止。")
        print("       这是\"无质量弹性固体 + 分区显式耦合\"的已知固有失稳（见 readme）：")
        print("       可调软本构（mu_s/lambda_s 调小）或缩短 T 以延长稳定窗口。")
    print(f"\nDone. Output: {config['output_path']}")
