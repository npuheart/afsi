"""demo_421 鱼游动 — DFIBMFoam CircularFishSwimming 的 FEniCSx 移植。

算法 = demo_339/multi_direct_forcing 的 multi-direct forcing（迭代直接力），
但把"固定圆柱 (U^d=0)"换成"游动鱼体 (U^d = dX/dt 由摆动运动学给出)"：

  预测步（AB2 对流 + 3/2-1/2 半隐式扩散）:
    (U* - U^n)/dt + 1.5·(U^n·∇)U^n - 0.5·(U^{n-1}·∇)U^{n-1}
        = 1.5·ν∇²U* - 0.5·ν∇²U^n + 0.5·∇p^n

  multi-direct forcing（每步 n_iter 次迭代，IBMf 累加）:
    tU   = U* - 1.5·dt·∇p^n + dt·IBMf
    U_l  = Σ_ij δ(x_l - x_ij)·tU_ij                     (Peskin 4点核插值)
    F_l  = (U^d_l - U_l)/dt,  U^d_l = (X_l(t)-X_l(t-dt))/dt   ← 游动速度
    IBMf = IBMf + Σ_l δ(x_l - x_ij)·F_l·ΔV_l/(dx·dy)

  投影:
    U      = U* + dt·IBMf
    ∇²p^{n+1} = (2/(3dt))·∇·U
    U^{n+1} = U - 1.5·dt·∇p^{n+1}

鱼体几何/运动学见 fish_geometry.py（忠实 DFIBMFoam：
NACA 鱼身 + 行波摆动 + 圆周游动）。流体域为闭合水槽（四壁无滑移），
压力在角落固定一个 DOF。

输出：output/velocity.xdmf、output/pressure.xdmf、output/fish_trace.csv。
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
from afsic.common import tag_boundaries, MARKER_LEFT, MARKER_RIGHT, \
    MARKER_BOTTOM, MARKER_TOP

import fish_geometry as fish
from configuration import config

comm = MPI.COMM_WORLD
rank = comm.rank

# ---------------------------------------------------------------------------
# 参数
# ---------------------------------------------------------------------------
x0, y0 = config["x0"], config["y0"]
Lx, Ly = config["Lx"], config["Ly"]
Nx, Ny = config["Nx"], config["Ny"]
rho, mu = config["rho"], config["mu"]
T, dt = config["T"], config["dt"]
n_markers = 2 * config["n_sections"]
n_iter = config["n_iter"]
nu = mu / rho
h = np.sqrt((Lx / Nx) * (Ly / Ny))   # 特征网格尺寸（均匀）

# ---------------------------------------------------------------------------
# 流体网格与函数空间（闭合水槽，域 [x0, x0+Lx]×[y0, y0+Ly]）
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
# 边界条件：闭合水槽（四壁无滑移）+ 压力钉一个角点（消去零模态）
# ---------------------------------------------------------------------------
u_zero = Function(V); u_zero.x.array[:] = 0.0
bcu = [
    dirichletbc(u_zero, locate_dofs_topological(V, fdim, facet_tag.find(m)))
    for m in (MARKER_LEFT, MARKER_RIGHT, MARKER_BOTTOM, MARKER_TOP)
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
# IBM：游动鱼体的表面标记（上/下表面，DFIBMFoam 顺序交错）
# ---------------------------------------------------------------------------
ibmesh = IBMesh(x0, x0 + Lx, y0, y0 + Ly, Nx, Ny, config["velocity_order"])
ib_interp = IBInterpolation(ibmesh)
coords_bg = Function(V)
coords_bg.interpolate(lambda x: np.array([x[0], x[1]]))
ibmesh.build_map(coords_bg._cpp_object)

# 初始鱼体标记（t=0）
mx0, my0, _, _, ds0 = fish.fish_surface(config, 0.0)
dV_marker = ds0 * h          # ΔV_l = Δs_l·h（DFIBMFoam: IbpDs·sqrt(dx·dy)）

struct_mesh = create_interval(comm, n_markers - 1, [0.0, 1.0])
v_s = element("Lagrange", struct_mesh.topology.cell_name(), 1, shape=(2,))
Vs = functionspace(struct_mesh, v_s)
solid_coords = Function(Vs)
solid_coords.x.array[:] = np.ravel(np.column_stack([mx0, my0]))
solid_coords.x.scatter_forward()
ib_interp.evaluate_current_points(solid_coords._cpp_object)

solid_vel = Function(Vs)    # 插值得到的标记速度 U_l
solid_force = Function(Vs)  # 标记力（已乘 ΔV_l，供扩散）
f_spread = Function(V)      # 单次迭代扩散后的力（C++ assign_dofs 是"替换"非"累加"，
                            # 故在 Python 层累加回 f_ibm）

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

trace_path = config["output_path"] + "fish_trace.csv"
if rank == 0:
    with open(trace_path, "w") as fh:
        fh.write("t," + ",".join(f"x{k},y{k}" for k in range(n_markers)) + "\n")

form_u_L2 = form(dot(u, u) * dx)
form_p_L2 = form(dot(p, p) * dx)
form_Fx = form(inner(f_ibm, as_vector((1.0, 0.0))) * dx)
form_Fy = form(inner(f_ibm, as_vector((0.0, 1.0))) * dx)

log.set_log_level(log.LogLevel.INFO)
if rank == 0:
    print(f"Fish swimming (DFIBMFoam CircularFishSwimming 移植): {Nx}×{Ny}, "
          f"h={h*1e3:.1f}mm, dt={dt}, Re≈{rho*0.15*config['fish_length']/mu:.0f}")
    print(f"  fish L={config['fish_length']}m, n_markers={n_markers}, "
          f"n_iter={n_iter}, ΔV_l≈{dV_marker.mean()*1e6:.3f}mm²")

num_steps = config["num_steps"]
out_interval = config["out_interval"]

# 鱼体游动方向（圆周轨迹切向），用于推力分解
init_angle = config["fish_index"] * 2.0 * np.pi / config["n_fish"]


def tangent(tt):
    ang = init_angle + 2.0 * np.pi * tt / config["cycle_period"]
    return np.array([-np.sin(ang), np.cos(ang)])


# ---------------------------------------------------------------------------
# 时间循环
# ---------------------------------------------------------------------------
for step in range(num_steps):
    t = step * dt
    tt = t + dt            # 本步终点时刻（位移用本步末计算）

    # ---- 鱼体更新：标记坐标 + 期望速度 U^d = dX/dt ----
    mx, my, _, _, ds = fish.fish_surface(config, tt)
    Ud = fish.fish_desired_velocity(config, tt, dt)
    dV_marker = ds * h
    solid_coords.x.array[:] = np.ravel(np.column_stack([mx, my]))
    solid_coords.x.scatter_forward()
    ib_interp.evaluate_current_points(solid_coords._cpp_object)

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

    # ---- 2) multi-direct forcing（迭代累加 IBMf，U^d = 游动速度） ----
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
        sv = solid_vel.x.array
        sf = solid_force.x.array
        for k in range(n_markers):
            sf[2 * k] = (Ud[2 * k] - sv[2 * k]) / dt * dV_marker[k]
            sf[2 * k + 1] = (Ud[2 * k + 1] - sv[2 * k + 1]) / dt * dV_marker[k]
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

    # ---- 5) 更新历史场 ----
    u_nm1.x.array[:] = u_n.x.array[:]
    u_n.x.array[:] = u.x.array[:]
    p_n.x.array[:] = p.x.array[:]

    # ---- 监控 / 输出 ----
    u_L2 = comm.allreduce(assemble_scalar(form_u_L2), op=MPI.SUM)
    p_L2 = comm.allreduce(assemble_scalar(form_p_L2), op=MPI.SUM)
    F_x = comm.allreduce(assemble_scalar(form_Fx), op=MPI.SUM)
    F_y = comm.allreduce(assemble_scalar(form_Fy), op=MPI.SUM)
    t_hat = tangent(tt)
    F_thrust = t_hat[0] * F_x + t_hat[1] * F_y      # 沿游动方向推力
    F_lateral = -t_hat[1] * F_x + t_hat[0] * F_y    # 垂直游动方向侧向力
    u_max = float(np.max(np.abs(u.x.array)))
    f_max = float(np.max(np.abs(f_ibm.x.array)))

    if step % out_interval == 0 or step == num_steps - 1:
        u_io.interpolate(u)
        p_io.interpolate(p)
        file_vel.write_function(u_io, tt)
        file_pre.write_function(p_io, tt)
        if rank == 0:
            with open(trace_path, "a") as fh:
                row = [f"{tt:.5f}"]
                row += [f"{mx[k]:.6f},{my[k]:.6f}" for k in range(n_markers)]
                fh.write(",".join(row) + "\n")
            print(f"Step {step+1}/{num_steps}, t={tt:.3f}s, "
                  f"u_L2={u_L2:.4f}, |u|max={u_max:.4f}, "
                  f"F_thrust={F_thrust:.4f}, F_lateral={F_lateral:+.4f}, "
                  f"|f|max={f_max:.2f}", flush=True)

file_vel.close(); file_pre.close()
if rank == 0:
    print(f"\nDone. Output: {config['output_path']}")
