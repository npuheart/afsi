"""Multi-direct forcing（迭代直接力法）圆柱绕流 — DFIBMFoam 算法的 FEniCSx 移植。

参照 MsureCFD/DFIBMFoam (Mi et al. 2025)：
  预测步（AB2 对流 + 3/2-1/2 半隐式扩散）:
    (U* - U^n)/dt + 1.5*(U^n·∇)U^n - 0.5*(U^{n-1}·∇)U^{n-1}
        = 1.5·ν∇²U* - 0.5·ν∇²U^n + 0.5·∇p^n

  multi-direct forcing（每步 n_iter 次迭代，IBMf 累加）:
    tU   = U* - 1.5·dt·∇p^n + dt·IBMf
    U_l  = Σ_ij δ(x_l - x_ij)·tU_ij                      (Peskin 4点核插值)
    F_l  = (U_l^d - U_l)/dt
    IBMf = IBMf + Σ_l δ(x_l - x_ij)·F_l·ΔV_l/(dx·dy)     (扩散累加, ΔV_l=Δs_l·h)

  投影:
    U      = U* + dt·IBMf
    ∇²p^{n+1} = (2/(3dt))·∇·U
    U^{n+1} = U - 1.5·dt·∇p^{n+1}                         (L2 投影, 保持 BC)

几何/物理参数与 demo_339 其余实现统一：DFG 2D-3, Re=100, 固定圆柱。
复用 afsic 的 IBMesh/IBInterpolation（Peskin 4点核已内置）。
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
from dolfinx.mesh import CellType, GhostMode, create_interval
from basix.ufl import element
from ufl import (TestFunction, TrialFunction, dot, dx, inner, grad, div,
                 as_vector)

from afsic import IBMesh, IBInterpolation
from afsic.common import (tag_boundaries, rectangle_boundaries, TurekInlet,
                          MARKER_LEFT, MARKER_RIGHT, MARKER_BOTTOM, MARKER_TOP)
from configuration import config

comm = MPI.COMM_WORLD
rank = comm.rank

# ---------------------------------------------------------------------------
# 参数
# ---------------------------------------------------------------------------
Lx, Ly = config["Lx"], config["Ly"]
Nx, Ny = config["Nx"], config["Ny"]
Um, rho, mu = config["Um"], config["rho"], config["mu"]
T, dt = config["T"], config["dt"]
cx, cy, r = config["cylinder_cx"], config["cylinder_cy"], config["cylinder_r"]
D = config["D"]
n_markers = config["n_markers"]
n_iter = config["n_iter"]
nu = mu / rho                       # 运动粘度 [m^2/s]
h = np.sqrt((Lx / Nx) * (Ly / Ny))  # 特征网格尺寸

# ---------------------------------------------------------------------------
# 流体网格与函数空间
# ---------------------------------------------------------------------------
mesh = dolfinx.mesh.create_rectangle(
    comm, ((0.0, 0.0), (Lx, Ly)), (Nx, Ny),
    cell_type=CellType.quadrilateral, ghost_mode=GhostMode.shared_facet)
facet_tag = tag_boundaries(mesh, rectangle_boundaries(Lx, Ly))
fdim = mesh.topology.dim - 1

v_cg2 = element("Lagrange", mesh.topology.cell_name(), 2, shape=(2,))
s_cg1 = element("Lagrange", mesh.topology.cell_name(), 1)
V = functionspace(mesh, v_cg2)
Q = functionspace(mesh, s_cg1)

# ---------------------------------------------------------------------------
# 边界条件（入口抛物线 + 上下无滑移 + 出口 p=0）
# ---------------------------------------------------------------------------
inlet = TurekInlet(Um=Um, Ly=Ly)
u_inlet = Function(V); u_inlet.interpolate(inlet)
bcu_inlet = dirichletbc(u_inlet, locate_dofs_topological(
    V, fdim, facet_tag.find(MARKER_LEFT)))
u_zero = Function(V); u_zero.x.array[:] = 0.0
bcu_bottom = dirichletbc(u_zero, locate_dofs_topological(
    V, fdim, facet_tag.find(MARKER_BOTTOM)))
bcu_top = dirichletbc(u_zero, locate_dofs_topological(
    V, fdim, facet_tag.find(MARKER_TOP)))
bcp_outlet = dirichletbc(PETSc.ScalarType(0.0), locate_dofs_topological(
    Q, fdim, facet_tag.find(MARKER_RIGHT)), Q)
bcu = [bcu_inlet, bcu_bottom, bcu_top]
bcp = [bcp_outlet]

# ---------------------------------------------------------------------------
# 场
# ---------------------------------------------------------------------------
u = Function(V)        # U^{n+1}
u_n = Function(V)      # U^n
u_nm1 = Function(V)    # U^{n-1}
u_star = Function(V)   # U*
p = Function(Q)        # p^{n+1}
p_n = Function(Q)      # p^n
f_ibm = Function(V)    # IBMf（迭代累加的体积力场, 运动学 [m/s^2]）
tU = Function(V)       # 插值用辅助速度
grad_p = Function(V)   # 每步插值一次的 ∇p^n
# grad(p_n) 用 Expression 插值进 V（P1 的梯度是 DG0，需逐点求值）
grad_p_expr = Expression(grad(p_n), V.element.interpolation_points)

# ---------------------------------------------------------------------------
# IBM：圆柱标记点（复用 afsic 的 Peskin 4点核机制）
#   - "disk" 模式：均匀格点填充圆盘（固定实体，内部也强制 u=0）
#   - "boundary" 模式：仅边界环（DFIBMFoam 原版；对固定粗圆柱会"漏"）
# ---------------------------------------------------------------------------
ibmesh = IBMesh(0.0, Lx, 0.0, Ly, Nx, Ny, config["velocity_order"])
ib_interp = IBInterpolation(ibmesh)
coords_bg = Function(V)
coords_bg.interpolate(lambda x: np.array([x[0], x[1]]))
ibmesh.build_map(coords_bg._cpp_object)

if config["marker_mode"] == "disk":
    a_m = config["marker_spacing_h"] * h          # 标记间距
    n_ax = max(2, int(np.ceil(2.0 * r / a_m)))
    pts_ax = np.linspace(cx - r, cx + r, n_ax + 1)
    X, Y = np.meshgrid(pts_ax, pts_ax)
    keep = (X - cx) ** 2 + (Y - cy) ** 2 < r ** 2  # 严格圆内
    dV_marker = a_m * a_m                           # 每标记面积 ΔV_l
else:  # boundary（DFIBMFoam 原版）
    th = 2.0 * np.pi * np.arange(config["n_markers"]) / config["n_markers"]
    X = cx + r * np.cos(th)
    Y = cy + r * np.sin(th)
    keep = np.ones_like(X, dtype=bool)
    dV_marker = (2.0 * np.pi * r / config["n_markers"]) * h

mx = X[keep]
my = Y[keep]
n_markers = len(mx)
normals = np.column_stack([np.cos(np.arctan2(my - cy, mx - cx)),
                           np.sin(np.arctan2(my - cy, mx - cx))])  # 圆柱外法线

# 标记点承载在 1D interval 网格的 P1 向量函数上（值 = 坐标）
struct_mesh = create_interval(comm, n_markers - 1, [0.0, 1.0])
v_s = element("Lagrange", struct_mesh.topology.cell_name(), 1, shape=(2,))
Vs = functionspace(struct_mesh, v_s)
solid_coords = Function(Vs)
solid_coords.x.array[:] = np.ravel(np.column_stack([mx, my]))
solid_coords.x.scatter_forward()
ib_interp.evaluate_current_points(solid_coords._cpp_object)

solid_vel = Function(Vs)    # 插值得到的标记速度 U_l
solid_force = Function(Vs)  # 标记力（已乘 ΔV_l，供扩散）

# --- 内部掩码：固定圆柱内部 DOF（保证实体固体，避免漏流破坏卡门涡街） ---
dof_coords = V.tabulate_dof_coordinates()
bs = V.dofmap.index_map_bs
interior_dofs = np.nonzero(
    (dof_coords[:, 0] - cx) ** 2 + (dof_coords[:, 1] - cy) ** 2 < r ** 2)[0]


def apply_interior_mask(fun):
    """把圆柱内部速度硬置零（掩码）。"""
    if not config["mask_interior"]:
        return
    arr = fun.x.array
    for d in interior_dofs:
        arr[d * bs] = 0.0
        arr[d * bs + 1] = 0.0
    fun.x.scatter_forward()

# ---------------------------------------------------------------------------
# 三个分步的矩阵（都只组装一次）
# ---------------------------------------------------------------------------
v = TestFunction(V)
u_trial = TrialFunction(V)
q = TestFunction(Q)
p_trial = TrialFunction(Q)

# 1) 预测步:  ∫u·v + 1.5dt·ν∫∇u·∇v   (LHS)
a_pred = form(inner(u_trial, v) * dx + 1.5 * dt * nu * inner(grad(u_trial), grad(v)) * dx)
A_pred = assemble_matrix(a_pred, bcs=bcu); A_pred.assemble()
b_pred = create_vector(V)

# 2) 压力泊松:  ∫∇p·∇q
a_p = form(inner(grad(p_trial), grad(q)) * dx)
A_p = assemble_matrix(a_p, bcs=bcp); A_p.assemble()
b_p = create_vector(Q)

# 3) 速度修正 (L2 投影):  ∫u·v
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

form_u_L2 = form(dot(u, u) * dx)
form_p_L2 = form(dot(p, p) * dx)
form_Fx = form(inner(f_ibm, as_vector((1.0, 0.0))) * dx)
form_Fy = form(inner(f_ibm, as_vector((0.0, 1.0))) * dx)

log.set_log_level(log.LogLevel.INFO)
if rank == 0:
    Re = rho * Um * D / mu
    print(f"Multi-direct forcing (DFIBMFoam 移植): {Nx}×{Ny}, dt={dt}, Re≈{Re:.0f}")
    print(f"  marker_mode={config['marker_mode']}, n_markers={n_markers}, "
          f"n_iter={n_iter}, ΔV_l={dV_marker*1e6:.2f}mm², h={h*1e3:.2f}mm")

num_steps = config["num_steps"]
out_interval = max(1, num_steps // 200)

# ---------------------------------------------------------------------------
# 时间循环
# ---------------------------------------------------------------------------
for step in range(num_steps):
    t = step * dt
    inlet.update(t)
    u_inlet.interpolate(inlet)

    # ---- 1) 预测步 (AB2) ----
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

    # ---- 2) multi-direct forcing（迭代累加 IBMf） ----
    grad_p.interpolate(grad_p_expr)
    grad_p.x.scatter_forward()
    f_ibm.x.array[:] = 0.0
    f_ibm.x.scatter_forward()
    for _ in range(n_iter):
        # tU = U* - 1.5dt·∇p^n + dt·IBMf
        tU.x.array[:] = (u_star.x.array
                         + dt * f_ibm.x.array
                         - 1.5 * dt * grad_p.x.array)
        tU.x.scatter_forward()
        # 插值 tU → 标记点
        ib_interp.fluid_to_solid(tU._cpp_object, solid_vel._cpp_object)
        solid_vel.x.scatter_forward()
        # F_l = (U^d - U_l)/dt，固定圆柱 U^d=0；乘 ΔV_l 供扩散
        sv = solid_vel.x.array
        sf = solid_force.x.array
        for k in range(n_markers):
            sf[k * 2] = (0.0 - sv[k * 2]) / dt * dV_marker
            sf[k * 2 + 1] = (0.0 - sv[k * 2 + 1]) / dt * dV_marker
        solid_force.x.scatter_forward()
        # 扩散累加进 IBMf（afsic 的 solid_to_fluid 是累加）
        ib_interp.solid_to_fluid(f_ibm._cpp_object, solid_force._cpp_object)
        f_ibm.x.scatter_forward()

    # u^{n+1/2} = U* + dt·IBMf
    u.x.array[:] = u_star.x.array + dt * f_ibm.x.array
    u.x.scatter_forward()
    apply_interior_mask(u)   # 压力泊松前保证固体内部 u=0（压力能看到实体）

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
    apply_interior_mask(u)   # 投影可能扰动内部，再次置零

    # ---- 5) 更新历史场 ----
    u_nm1.x.array[:] = u_n.x.array[:]
    u_n.x.array[:] = u.x.array[:]
    p_n.x.array[:] = p.x.array[:]

    # ---- 监控 / 输出 ----
    u_L2 = comm.allreduce(assemble_scalar(form_u_L2), op=MPI.SUM)
    p_L2 = comm.allreduce(assemble_scalar(form_p_L2), op=MPI.SUM)
    F_x = comm.allreduce(assemble_scalar(form_Fx), op=MPI.SUM)  # ∫IBMf_x dV
    F_y = comm.allreduce(assemble_scalar(form_Fy), op=MPI.SUM)  # ∫IBMf_y dV
    Cd = -2.0 * F_x / (Um**2 * D)   # F= -ρ∫IBMf_x dV → Cd=2F/(ρUm²D)
    Cl = -2.0 * F_y / (Um**2 * D)   # 升力系数（涡街振荡的直接表征）

    if step % out_interval == 0 or step == num_steps - 1:
        u_io.interpolate(u)
        p_io.interpolate(p)
        file_vel.write_function(u_io, t)
        file_pre.write_function(p_io, t)
        if rank == 0:
            print(f"Step {step+1}/{num_steps}, t={t:.3f}s, "
                  f"u_L2={u_L2:.4f}, p_L2={p_L2:.1f}, "
                  f"Cd={Cd:.4f}, Cl={Cl:+.4f}", flush=True)

file_vel.close(); file_pre.close()
if rank == 0:
    print(f"\nDone. Output: {config['output_path']}")
