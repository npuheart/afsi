"""demo_336 方腔驱动圆盘 — multi-direct forcing（迭代直接力法）求解。

求解器结构沿用 afsic/demo/demo_421/main.py（DFIBMFoam multi-direct forcing 的
FEniCSx 移植），但把"游动鱼体 (U^d=dX/dt)"换回 demo_336 原始算例的**圆盘**，
流体域改为**方腔驱动 (lid-driven cavity)**：顶盖以 U=(1,0) 匀速滑动驱动，
其余三壁无滑移，压力在角落固定一个 DOF。

算法（DFIBMFoam multi-direct forcing）:

  预测步（AB2 对流 + 3/2-1/2 半隐式扩散）:
    (U* - U^n)/dt + 1.5·(U^n·∇)U^n - 0.5·(U^{n-1}·∇)U^{n-1}
        = 1.5·ν∇²U* - 0.5·ν∇²U^n + 0.5·∇p^n

  multi-direct forcing（每步 n_iter 次迭代，IBMf 累加）:
    tU   = U* - 1.5·dt·∇p^n + dt·IBMf
    U_l  = Σ_ij δ(x_l - x_ij)·tU_ij                     (Peskin 4点核插值)
    F_l  = (U^d_l - U_l)/dt · ΔV_l
    IBMf = IBMf + Σ_l δ(x_l - x_ij)·F_l/(dx·dy)         (Python 层累加)

  投影:
    U      = U* + dt·IBMf
    ∇²p^{n+1} = (2/(3dt))·∇·U
    U^{n+1} = U - 1.5·dt·∇p^{n+1}                       (L2 投影, 保持 BC)

圆盘期望速度 U^d：
  - "fixed"：U^d = 0（固定圆盘，demo_339 圆柱风格，可加内部掩码）
  - "free"： 圆盘为**刚性体随流驱动** —— 每步由标记处流体速度求刚体速度
             V_c = ⟨U_l⟩、ω = ⟨(r×U_l)/r²⟩（仅用边界环标记），
             U^d(X) = V_c + ω×(X - X_c)，再更新圆心/转角并重设标记坐标
             （同 demo_421 的移动体机制，等价于"方腔驱动圆盘"）。

输出：output/velocity.xdmf、output/pressure.xdmf、output/forces.csv、
      output/disk.xdmf（固体圆盘参考网格 + 刚体位移场，ParaView Warp by vector
      查看圆盘运动），以及（free 模式）output/disk_trace.csv：圆心/转角轨迹。
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
import ufl
from basix.ufl import element
from ufl import (TestFunction, TrialFunction, dot, dx, inner, grad, div,
                 as_vector, SpatialCoordinate)

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
n_iter = config["n_iter"]
disk_motion = config["disk_motion"]
marker_mode = config["marker_mode"]
nu = mu / rho
h = np.sqrt((Lx / Nx) * (Ly / Ny))   # 特征网格尺寸（均匀）

# ---------------------------------------------------------------------------
# 流体网格与函数空间（方腔 [0,Lx]×[0,Ly]，顶盖滑动，闭合腔）
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
# 边界条件：顶盖滑动 U=(U_lid,0)，其余三壁无滑移 + 压力钉一角点
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
# 场
# ---------------------------------------------------------------------------
u = Function(V)        # U^{n+1}
u_n = Function(V)      # U^n
u_nm1 = Function(V)    # U^{n-1}
u_star = Function(V)   # U*
p = Function(Q)        # p^{n+1}
p_n = Function(Q)      # p^n
f_ibm = Function(V)    # IBMf（迭代累加的体积力场 [m/s^2]）
tU = Function(V)       # 插值用辅助速度
grad_p = Function(V)
grad_p_expr = Expression(grad(p_n), V.element.interpolation_points)

# ---------------------------------------------------------------------------
# IBM：圆盘标记（Peskin 4点核，afsic 内置）
#   标记按 体坐标（相对圆心、初始转角）存储于 mxr/myr，每步由圆心+转角重建绝对坐标。
#   边界环标记在最前（用于 free 模式刚体运动学）。
# ---------------------------------------------------------------------------
ibmesh = IBMesh(x0, x0 + Lx, y0, y0 + Ly, Nx, Ny, config["velocity_order"])
ib_interp = IBInterpolation(ibmesh)
coords_bg = Function(V)
coords_bg.interpolate(lambda x: np.array([x[0], x[1]]))
ibmesh.build_map(coords_bg._cpp_object)

n_rim = max(64, int(np.ceil(2.0 * np.pi * r / (config["marker_ds_h"] * h))))
th = 2.0 * np.pi * np.arange(n_rim) / n_rim
rx_rim = r * np.cos(th)
ry_rim = r * np.sin(th)
dV_rim = (2.0 * np.pi * r / n_rim) * h          # ΔV_l = Δs_l·h（DFIBMFoam: IbpDs·h）

if marker_mode == "disk":
    a_m = config["marker_spacing_h"] * h
    n_ax = max(2, int(np.ceil(2.0 * r / a_m)))
    pts = np.linspace(-r, r, n_ax + 1)
    X, Y = np.meshgrid(pts, pts)
    keep = (X ** 2 + Y ** 2) < (r - 0.5 * a_m) ** 2   # 圆内（留 margin 与环互补）
    rx_int = X[keep]
    ry_int = Y[keep]
    dV_int = a_m * a_m
    mxr = np.concatenate([rx_rim, rx_int])
    myr = np.concatenate([ry_rim, ry_int])
    dV = np.concatenate([np.full(n_rim, dV_rim), np.full(len(rx_int), dV_int)])
else:  # boundary
    mxr = rx_rim
    myr = ry_rim
    dV = np.full(n_rim, dV_rim)

n_markers = len(mxr)


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


# 圆盘当前位姿（free 模式每步更新；fixed 模式不变）
cx, cy, theta = cx0, cy0, 0.0


def disk_coordinates():
    """由当前圆心/转角重建标记绝对坐标（列向量 [x, y]）。"""
    c, s = np.cos(theta), np.sin(theta)
    rx = c * mxr - s * myr
    ry = s * mxr + c * myr
    return cx + rx, cy + ry


struct_mesh = create_interval(comm, n_markers - 1, [0.0, 1.0])
v_s = element("Lagrange", struct_mesh.topology.cell_name(), 1, shape=(2,))
Vs = functionspace(struct_mesh, v_s)
solid_coords = Function(Vs)
solid_vel = Function(Vs)    # 插值得到的标记速度 U_l
solid_force = Function(Vs)  # 标记力（已乘 ΔV_l，供扩散）
f_spread = Function(V)      # 单次迭代扩散后的力（C++ assign_dofs 是"替换"，
                            # 故在 Python 层累加回 f_ibm —— demo_421 关键修复）


def set_markers_and_reeval():
    """把当前标记坐标写入 solid_coords 并重调 evaluate_current_points。"""
    mx, my = disk_coordinates()
    solid_coords.x.array[:] = np.ravel(np.column_stack([mx, my]))
    solid_coords.x.scatter_forward()
    ib_interp.evaluate_current_points(solid_coords._cpp_object)


set_markers_and_reeval()

# --- 内部掩码（仅 "fixed" 模式有效）：圆盘内部速度硬置零 ---
dof_coords = V.tabulate_dof_coordinates()
bs = V.dofmap.index_map_bs


def apply_interior_mask(fun):
    if disk_motion != "fixed" or not config["mask_interior"]:
        return
    arr = fun.x.array
    inside = ((dof_coords[:, 0] - cx) ** 2 + (dof_coords[:, 1] - cy) ** 2) < r ** 2
    for d in np.nonzero(inside)[0]:
        arr[d * bs] = 0.0
        arr[d * bs + 1] = 0.0
    fun.x.scatter_forward()

# ---------------------------------------------------------------------------
# 三个分步的矩阵（组装一次）
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

# 固体（圆盘）可视化：三角化参考圆盘网格（固定初始位姿）+ 逐时间步位移场
# （原始 demo_336 用 circle_20 固体网格输出 solid_coords 位移场；mdf 版圆盘为
#   刚体、无固体网格，故用生成的参考圆盘网格承载位移场 u）
# 注意：dolfinx 0.10 XDMF 时间序列只写一次几何（各步 xi:include 引用），
# 不支持移动网格 —— 故输出位移场，在 ParaView 用 "Warp by vector" 查看刚体运动。
disk_base, disk_cells = make_disk_mesh(r, config["solid_rings"], config["solid_circ"])
disk_mesh = dolfinx.mesh.create_mesh(
    comm, disk_cells, ufl.Mesh(element("Lagrange", "triangle", 1, shape=(2,))),
    disk_base + np.array([cx0, cy0]))
Vsolid = functionspace(disk_mesh, element("Lagrange", "triangle", 1, shape=(2,)))
solid_disp_io = Function(Vsolid, name="u")   # 位移场：参考节点 → 当前位姿
file_solid = None
if config["write_solid"]:
    file_solid = dolfinx.io.XDMFFile(mesh.comm, config["output_path"] + "disk.xdmf", "w")
    file_solid.write_mesh(disk_mesh)

forces_path = config["output_path"] + "forces.csv"
if rank == 0:
    with open(forces_path, "w") as fh:
        fh.write("t,cx,cy,theta,Fx,Fy,Mz,Cx,Cy,Cm\n")

if disk_motion == "free" and rank == 0:
    trace_path = config["output_path"] + "disk_trace.csv"
    with open(trace_path, "w") as fh:
        fh.write("t,cx,cy,theta,Vc_x,Vc_y,omega\n")

form_u_L2 = form(dot(u, u) * dx)
form_p_L2 = form(dot(p, p) * dx)
form_Fx = form(inner(f_ibm, as_vector((1.0, 0.0))) * dx)
form_Fy = form(inner(f_ibm, as_vector((0.0, 1.0))) * dx)
xc = SpatialCoordinate(mesh)
form_Mz = form(((xc[0] - cx0) * f_ibm[1] - (xc[1] - cy0) * f_ibm[0]) * dx)

log.set_log_level(log.LogLevel.INFO)
if rank == 0:
    Re = rho * U_lid * Lx / mu
    print(f"Lid-driven cavity + disk (multi-direct forcing, demo_421 移植): "
          f"{Nx}×{Ny}, dt={dt}, Re≈{Re:.0f}")
    print(f"  disk motion={disk_motion}, marker_mode={marker_mode}, "
          f"n_markers={n_markers} (rim={n_rim}), n_iter={n_iter}, "
          f"h={h*1e3:.2f}mm, ΔV_rim={dV_rim*1e6:.3f}mm²")

num_steps = config["num_steps"]
out_interval = config["out_interval"]

# ---------------------------------------------------------------------------
# 时间循环
# ---------------------------------------------------------------------------
for step in range(num_steps):
    t = step * dt
    tt = t + dt            # 本步终点时刻

    # ---- 0) free 模式：由流体速度更新圆盘刚体位姿 ----
    if disk_motion == "free":
        # 用 U* - 1.5dt·∇p^n 处的流体速度确定刚体速度（一次/步）
        grad_p.interpolate(grad_p_expr)
        grad_p.x.scatter_forward()
        tU.x.array[:] = u_star.x.array - 1.5 * dt * grad_p.x.array
        tU.x.scatter_forward()
        ib_interp.fluid_to_solid(tU._cpp_object, solid_vel._cpp_object)
        solid_vel.x.scatter_forward()
        sv = solid_vel.x.array
        vx = sv[0:2 * n_rim:2]
        vy = sv[1:2 * n_rim:2]
        Vc_x, Vc_y = float(vx.mean()), float(vy.mean())
        omega = float(np.mean((rx_rim * vy - ry_rim * vx) / r ** 2))
        # 更新位姿（显式一步）
        cx += Vc_x * dt
        cy += Vc_y * dt
        theta += omega * dt
        # 安全钳位：防止圆盘漂出流体域（标记出域会导致插值越界）
        margin = r + 2.0 * h
        if cx < margin or cx > Lx - margin or cy < margin or cy > Ly - margin:
            cx = min(max(cx, margin), Lx - margin)
            cy = min(max(cy, margin), Ly - margin)
            if rank == 0:
                print(f"  [warn] disk clamped to wall margin at t={tt:.3f}s "
                      f"({cx:.3f},{cy:.3f})")
        set_markers_and_reeval()
    else:
        Vc_x = Vc_y = omega = 0.0

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
        tU.x.array[:] = (u_star.x.array
                         + dt * f_ibm.x.array
                         - 1.5 * dt * grad_p.x.array)
        tU.x.scatter_forward()
        ib_interp.fluid_to_solid(tU._cpp_object, solid_vel._cpp_object)
        solid_vel.x.scatter_forward()
        # U^d：free → 刚体速度（圆心/转角已更新）；fixed → 0
        sv = solid_vel.x.array
        sf = solid_force.x.array
        if disk_motion == "free":
            # 标记体坐标（当前转角下）：(X - Xc) = R(theta)·(mxr,myr)
            c, s = np.cos(theta), np.sin(theta)
            rel_x = c * mxr - s * myr
            rel_y = s * mxr + c * myr
            Ud_x = Vc_x - omega * rel_y
            Ud_y = Vc_y + omega * rel_x
            for k in range(n_markers):
                sf[2 * k] = (Ud_x[k] - sv[2 * k]) / dt * dV[k]
                sf[2 * k + 1] = (Ud_y[k] - sv[2 * k + 1]) / dt * dV[k]
        else:
            for k in range(n_markers):
                sf[2 * k] = (0.0 - sv[2 * k]) / dt * dV[k]
                sf[2 * k + 1] = (0.0 - sv[2 * k + 1]) / dt * dV[k]
        solid_force.x.scatter_forward()
        # C++ solid_to_fluid 的 assign_dofs 是"替换"，故先扩散到临时场再累加
        f_spread.x.array[:] = 0.0
        f_spread.x.scatter_forward()
        ib_interp.solid_to_fluid(f_spread._cpp_object, solid_force._cpp_object)
        f_spread.x.scatter_forward()
        f_ibm.x.array[:] += f_spread.x.array
        f_ibm.x.scatter_forward()

    # u^{n+1/2} = U* + dt·IBMf
    u.x.array[:] = u_star.x.array + dt * f_ibm.x.array
    u.x.scatter_forward()
    apply_interior_mask(u)   # 压力泊松前保证圆盘内部 u=0（仅 fixed）

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
    apply_interior_mask(u)   # 投影可能扰动内部，再次置零（仅 fixed）

    # ---- 5) 更新历史场 ----
    u_nm1.x.array[:] = u_n.x.array[:]
    u_n.x.array[:] = u.x.array[:]
    p_n.x.array[:] = p.x.array[:]

    # ---- 监控 / 输出 ----
    u_L2 = comm.allreduce(assemble_scalar(form_u_L2), op=MPI.SUM)
    p_L2 = comm.allreduce(assemble_scalar(form_p_L2), op=MPI.SUM)
    F_x = comm.allreduce(assemble_scalar(form_Fx), op=MPI.SUM)   # ∫IBMf_x dV
    F_y = comm.allreduce(assemble_scalar(form_Fy), op=MPI.SUM)   # ∫IBMf_y dV
    M_z = comm.allreduce(assemble_scalar(form_Mz), op=MPI.SUM)   # ∫(r×IBMf) dV
    Cx = -2.0 * F_x / (U_lid ** 2 * D)    # F= -ρ∫IBMf dV → C=2F/(ρU²D)（ρ 消去）
    Cy = -2.0 * F_y / (U_lid ** 2 * D)
    Cm = -2.0 * M_z / (U_lid ** 2 * D ** 2)
    u_max = float(np.max(np.abs(u.x.array)))
    f_max = float(np.max(np.abs(f_ibm.x.array)))

    if step % out_interval == 0 or step == num_steps - 1:
        u_io.interpolate(u)
        p_io.interpolate(p)
        file_vel.write_function(u_io, tt)
        file_pre.write_function(p_io, tt)
        if file_solid is not None:
            # 刚体位移 u(X_ref) = (Xc-Xc0) + (R(θ)-I)·(X_ref-Xc0)
            c, s = np.cos(theta), np.sin(theta)

            def _solid_disp(x):
                rx = x[0] - cx0
                ry = x[1] - cy0
                return np.array([(cx - cx0) + (c - 1.0) * rx - s * ry,
                                 (cy - cy0) + s * rx + (c - 1.0) * ry])

            solid_disp_io.interpolate(_solid_disp)
            file_solid.write_function(solid_disp_io, tt)
        if rank == 0:
            with open(forces_path, "a") as fh:
                fh.write(f"{tt:.5f},{cx:.6f},{cy:.6f},{theta:.6f},"
                         f"{F_x:.6e},{F_y:.6e},{M_z:.6e},"
                         f"{Cx:.4f},{Cy:.4f},{Cm:.4f}\n")
            if disk_motion == "free":
                with open(trace_path, "a") as fh:
                    fh.write(f"{tt:.5f},{cx:.6f},{cy:.6f},{theta:.6f},"
                             f"{Vc_x:.6e},{Vc_y:.6e},{omega:.6e}\n")
            print(f"Step {step+1}/{num_steps}, t={tt:.3f}s, "
                  f"u_L2={u_L2:.4f}, |u|max={u_max:.4f}, "
                  f"Cx={Cx:.4f}, Cy={Cy:+.4f}, Cm={Cm:+.4f}, "
                  f"|f|max={f_max:.2f}", flush=True)

file_vel.close(); file_pre.close()
if file_solid is not None:
    file_solid.close()
if rank == 0:
    print(f"\nDone. Output: {config['output_path']}")
