"""demo_336 方腔驱动圆盘 — multi-direct-frocing-elastic：弹性固体 direct-forcing 版。

在 mdf 刚体版基础上增加"固体推进方程"，并采用**真正的 direct-forcing**：
圆盘为**带惯性的弹性固体**（可压缩 neo-Hookean），流体在标记处被直接力强制匹配
固体速度（无滑移），反作用力（added-mass 项）喂回固体动量方程。

流体（同 mdf 版，AB2 预测 + 压力泊松 + L2 投影）:
  (U* - U^n)/dt + 1.5·(U^n·∇)U^n - 0.5·(U^{n-1}·∇)U^{n-1}
      = 1.5·ν∇²U* - 0.5·ν∇²U^n + 0.5·∇p^n
  ∇²p^{n+1} = (2/(3dt))·∇·U,   U^{n+1} = U - 1.5dt·∇p

固体（总拉格朗日，带惯性 + neo-Hookean）:
  动量方程:  ρ_s ∂²X/∂t² = ∇_X·P(F) + f^{fluid→solid}
  本构:      P(F) = μ_s(F - F^{-T}) + λ_s·ln(det F)·F^{-T},  F = ∇X_s
  弱形式:    ∫ρ_s a·δv dX + ∫P(F):∇δv dX = ∫f^{fluid→solid}·δv dX

每步（分区显式，但固液耦合隐式——added-mass 稳定）:
  1) 流体预测 → u*
  2) 插值 u* 到固体节点 → U_l
  3) 弹性内力 F_int = ∫P:∇δv dX
  4) 固体推进（added-mass 隐式）:
       (M_s + ρ_f·diag(V_node)) V_s^{n+1} = M_s V_s^n + ρ_f·diag(V_node)·U_l - dt·F_int
       X_s^{n+1} = X_s^n + dt·V_s^{n+1}
  5) 直接力约束（目标 = 固体速度）:
       F_IBM = (V_s^{n+1} - U_l)/dt · ΔV_l,  扩散 → f_IBM
       u = u* + dt·f_IBM
  6) 压力泊松 + L2 投影

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
from scipy.spatial import Delaunay
from ufl import (TestFunction, TrialFunction, dot, dx, inner, grad, div,
                 as_vector, SpatialCoordinate, inv, ln, det, sym)

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
rho_s = config["rho_s"]
T, dt = config["T"], config["dt"]
cx0, cy0, r = config["cx"], config["cy"], config["r"]
D = config["D"]
mu_s, lambda_s = config["mu_s"], config["lambda_s"]
mu_s_visc = config.get("mu_s_visc", 0.0)   # 固体粘性（Kelvin-Voigt），默认=流体 μ
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
    """准均匀点分布 + Delaunay 三角化圆盘网格（体坐标，相对圆心）。

    每环点数随半径缩放（保持 ~ 恒定间距），避免圆心处细长三角扇（fan）——
    那种极细三角对节点抖动极敏感，稍一扰动即翻转（det F<0）导致 NaN。
    返回 (节点坐标, 三角形连接)。
    """
    s = 2.0 * np.pi * r / n_circ          # 目标间距 ≈ 外圈周向间距
    pts = [(0.0, 0.0)]
    for ring in range(1, n_rings + 1):
        rr = r * ring / n_rings
        nk = max(6, int(np.round(2.0 * np.pi * rr / s)))
        for k in range(nk):
            ang = 2.0 * np.pi * k / nk
            pts.append((rr * np.cos(ang), rr * np.sin(ang)))
    pts = np.array(pts)
    tri = Delaunay(pts)
    cen = pts[tri.simplices].mean(axis=1)  # 三角形质心
    keep = np.linalg.norm(cen, axis=1) <= r * 0.9999
    cells = tri.simplices[keep].astype(np.int64)
    return pts, cells


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
solid_velocity = Function(Vs, name="solid_velocity")  # V_s：固体自身速度（0 起步）
solid_velocity.x.array[:] = 0.0
solid_velocity.x.scatter_forward()
fluid_at_solid = Function(Vs)   # U_l：插值自流体的标记处速度
solid_force = Function(Vs)      # 直接力 F_IBM·ΔV_l（供扩散回流体）
ref_coords = Vs.tabulate_dof_coordinates()   # 参考节点坐标（位移 = 当前 - 参考）

# --- 正定集中质量（HRZ）：M_HRZ,i = M_ii·(M_total/Σ_j M_jj)，全 > 0 ---
# 注：P2 单元的行求和集中会给出顶点对角项≈0/负（实测 769 顶点体积≈0），不能用于
# 除式；改用一致质量矩阵对角项 + 按总质量缩放（HRZ），保证每节点质量为正。
dVs = TestFunction(Vs)
u_trial_s = TrialFunction(Vs)
a_M = form(rho_s * inner(u_trial_s, dVs) * dx)
A_M = assemble_matrix(a_M); A_M.assemble()
diag_vec = create_vector(Vs)
A_M.getDiagonal(diag_vec)
_diag = diag_vec.array.copy()
_M_total = rho_s * (np.pi * r ** 2)            # 固体总质量 = ρ_s·面积
_sum = float(np.sum(_diag))                     # 单进程（IBM 限制），无需 allreduce
M_hrz = _diag * (_M_total / _sum)              # 正定集中质量（每分量）
vol_hrz = M_hrz / rho_s                         # 正节点体积（added-mass 与扩散 ΔV_l 用）
added_mass_vec = rho * vol_hrz                  # ρ_f·V_node
if rank == 0:
    print(f"  HRZ lumped: min M={M_hrz.min():.3e}, max={M_hrz.max():.3e}, "
          f"ΣM={M_hrz.sum():.4f} (target {_M_total:.4f})")

# ---------------------------------------------------------------------------
# IBM（标记 = 固体节点；Peskin 4点核，afsic 内置）
# ---------------------------------------------------------------------------
ibmesh = IBMesh(x0, x0 + Lx, y0, y0 + Ly, Nx, Ny, config["velocity_order"])
ib_interp = IBInterpolation(ibmesh)
coords_bg = Function(V)
coords_bg.interpolate(lambda x: np.array([x[0], x[1]]))
ibmesh.build_map(coords_bg._cpp_object)
ib_interp.evaluate_current_points(solid_coords._cpp_object)

# --- 弹性内力弱形式（总拉格朗日，参考构型积分）F_int = ∫P(F):∇δv dX ---
# 粘弹性（Kelvin-Voigt）：再加内部粘性应力 σ_visc = 2·μ_s_visc·sym(∇V_s)，
# 用固体当前速度（显式，V_s^n）→ 阻尼固体变形与对流动的跟随，更接近 IBFE。
dVs = TestFunction(Vs)
FF = grad(solid_coords)
P = mu_s * (FF - inv(FF).T) + lambda_s * ln(det(FF)) * inv(FF).T
L_int = form(inner(P, grad(dVs)) * dx)
b_int = create_vector(Vs)
form_volume = form(det(FF) * dx)

# --- Kelvin-Voigt 固体粘性（隐式，稳定） ---
# 显式处理粘性应力（forward Euler）会失稳：粘性项=速度的刚度，显式会放大高频分量
# → 网格翻转。改为隐式：把粘性刚度并入 LHS，每步解一个很小的 SPD 系统
# （固体 ~1700 dof）：
#   A_solid = diag(M_HRZ + ρ_f·V_node) + dt·K_visc
#   右端     = M_HRZ·V_s^n + ρ_f·V_node·U_l − dt·F_el
# 注意：mu_s_visc=0 时不能构造形式（UFL 会把乘 0 化简为零表达式、丢失网格域）。
_solid_ksp = None
b_solid = None
Vs_new_vec = None
if mu_s_visc > 0.0:
    a_visc = form(2.0 * mu_s_visc
                  * inner(sym(grad(u_trial_s)), sym(grad(dVs))) * dx)
    A_visc = assemble_matrix(a_visc); A_visc.assemble()
    A_solid = A_visc.copy(); A_solid.scale(dt)
    # 加对角质量 M_HRZ + ρ_f·V_node（用对角矩阵 + axpy 避免 setValues 形状问题）
    A_diag = A_solid.duplicate()
    A_diag.zeroEntries()
    _mass_diag = PETSc.Vec().createWithArray(
        np.ascontiguousarray(M_hrz + added_mass_vec, dtype=np.float64))
    A_diag.setDiagonal(_mass_diag)
    A_diag.assemble()
    A_solid.axpy(1.0, A_diag)
    A_solid.assemble()
    _solid_ksp = PETSc.KSP().create(solid_mesh.comm)
    _solid_ksp.setOperators(A_solid)
    _solid_ksp.setType(PETSc.KSP.Type.CG)
    _solid_ksp.getPC().setType(PETSc.PC.Type.JACOBI)
    _solid_ksp.setTolerances(rtol=1e-10, atol=1e-12)
    b_solid = create_vector(Vs)
    Vs_new_vec = Function(Vs)
    if rank == 0:
        print(f"  Kelvin-Voigt solid viscosity: mu_s_visc={mu_s_visc:.3g} "
              f"(implicit, KSP=CG)", flush=True)

# --- 诊断：每个单元 det(F) 最小值（翻转即 det F ≤ 0） ---
_solid_tdim = solid_mesh.topology.dim
_solid_mid = dolfinx.mesh.compute_midpoints(
    solid_mesh, _solid_tdim,
    np.arange(solid_mesh.topology.index_map(_solid_tdim).size_local))
det_expr = Expression(det(FF), _solid_mid[:, :2])  # 2D 点构造；eval(mesh, entities)

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
_last_clamp_pos = None   # 钳位日志节流：位置变化超过阈值才打印

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

    # ---- 2) 固体推进（带惯性，added-mass 隐式）+ 直接力约束（每步一次） ----
    #      solid_active=False 时跳过 = 纯方腔（无固体）参照
    if config["solid_active"]:
        #    a) 流体速度插值到固体节点 → U_l
        ib_interp.fluid_to_solid(u_star._cpp_object, fluid_at_solid._cpp_object)
        fluid_at_solid.x.scatter_forward()
        if _chk("U_l", fluid_at_solid): failed, fail_t = True, tt; break
        #    b) 弹性内力 F_int = ∫P(F):∇δv dX（当前变形，显式）
        with b_int.localForm() as loc:
            loc.set(0)
        assemble_vector(b_int, L_int)
        b_int.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                          mode=PETSc.ScatterMode.REVERSE)
        F_int = b_int.array.copy()
        #    c) 固体推进（added-mass 隐式稳定）:
        #       无粘性: (M_HRZ+ρ_f·V) V_s^{n+1} = M_HRZ V_s^n + ρ_f·V·U_l − dt·F_el
        #       有粘性: (diag + dt·K_visc) V_s^{n+1} = 同上右端（Kelvin-Voigt 隐式）
        if mu_s_visc > 0.0:
            b_solid.array[:] = (M_hrz * solid_velocity.x.array
                                + added_mass_vec * fluid_at_solid.x.array
                                - dt * F_int)
            b_solid.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                                mode=PETSc.ScatterMode.REVERSE)
            _solid_ksp.solve(b_solid, Vs_new_vec.x.petsc_vec)
            Vs_new_vec.x.scatter_forward()
            Vs_new = Vs_new_vec.x.array
        else:
            den = M_hrz + added_mass_vec
            Vs_new = (M_hrz * solid_velocity.x.array
                      + added_mass_vec * fluid_at_solid.x.array
                      - dt * F_int) / den
        #    d) 直接力: F_IBM = (V_s^{n+1} - U_l)/dt · ΔV_l → 扩散回流体
        solid_force.x.array[:] = (Vs_new - fluid_at_solid.x.array) / dt * vol_hrz
        solid_force.x.scatter_forward()
        if _chk("solid_force", solid_force): failed, fail_t = True, tt; break
        ib_interp.solid_to_fluid(f_ibm._cpp_object, solid_force._cpp_object)
        f_ibm.x.scatter_forward()
        if _chk("f_ibm", f_ibm): failed, fail_t = True, tt; break
        #    e) 流体获得直接力冲量: U = U* + dt·f_IBM
        u.x.array[:] = u_star.x.array + dt * f_ibm.x.array
        u.x.scatter_forward()
        if _chk("u_mid", u): failed, fail_t = True, tt; break
        #    f) 更新固体状态并重设标记
        solid_velocity.x.array[:] = Vs_new
        solid_velocity.x.scatter_forward()
        solid_coords.x.array[:] += dt * Vs_new
        solid_coords.x.scatter_forward()
        #    质心钳位（纯平动修正，不影响变形/应变）：防止圆盘被主涡带出域，
        #    顶部标记伸出 y>1 后插值得到垃圾速度 → 单元翻转（t≈4.9s 实测）。
        if config.get("clamp_solid", True):
            mgn = r + 2.0 * h
            cx_now = float(np.mean(solid_coords.x.array[0::2]))
            cy_now = float(np.mean(solid_coords.x.array[1::2]))
            dx_cl = min(max(cx_now, mgn), Lx - mgn) - cx_now
            dy_cl = min(max(cy_now, mgn), Ly - mgn) - cy_now
            if abs(dx_cl) > 1e-9 or abs(dy_cl) > 1e-9:
                solid_coords.x.array[0::2] += dx_cl
                solid_coords.x.array[1::2] += dy_cl
                solid_coords.x.scatter_forward()
                _cp = (cx_now + dx_cl, cy_now + dy_cl)
                # 节流：位置变化 > 0.02 才打印，避免每步刷屏
                if (rank == 0 and _last_clamp_pos is not None
                        and abs(_cp[0] - _last_clamp_pos[0]) > 0.02):
                    print(f"  [clamp] centroid -> ({_cp[0]:.3f},{_cp[1]:.3f})",
                          flush=True)
                if rank == 0 and _last_clamp_pos is None:
                    print(f"  [clamp] centroid -> ({_cp[0]:.3f},{_cp[1]:.3f}) "
                          f"(开始钳位)", flush=True)
                _last_clamp_pos = _cp
        #    壁面越界检查（自由体公转到壁面时优雅终止，避免域外标记插值 NaN）
        if not config.get("clamp_solid", True):
            sc_min_x = float(np.min(solid_coords.x.array[0::2]))
            sc_max_x = float(np.max(solid_coords.x.array[0::2]))
            sc_min_y = float(np.min(solid_coords.x.array[1::2]))
            sc_max_y = float(np.max(solid_coords.x.array[1::2]))
            if (sc_min_x < -0.5 * h or sc_max_x > Lx + 0.5 * h
                    or sc_min_y < -0.5 * h or sc_max_y > Ly + 0.5 * h):
                if rank == 0:
                    print(f"\n[注意] 圆盘公转触壁/出域 (t={tt:.3f}s, "
                          f"x∈[{sc_min_x:.3f},{sc_max_x:.3f}], "
                          f"y∈[{sc_min_y:.3f},{sc_max_y:.3f}])", flush=True)
                    print("       自由体沿闭合腔涡流线公转，轨道贴着壁面，必然触壁"
                          "（物理行为）。", flush=True)
                    print("       若要 5s 内不触壁：加重固体(RHO_S>1，公转慢/不上顶)"
                          " 或减弱流动(U_lid 调小) 或减小圆盘。", flush=True)
                failed, fail_t = True, tt
                break
        ib_interp.evaluate_current_points(solid_coords._cpp_object)
    else:
        # 纯方腔（无固体）参照：无直接力，流体 = 预测步结果
        f_ibm.x.array[:] = 0.0
        f_ibm.x.scatter_forward()
        u.x.array[:] = u_star.x.array[:]
        u.x.scatter_forward()
        Vs_new = solid_velocity.x.array[:]

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
    # 圆盘质心（诊断：看是否漂到顶盖/壁面等强剪切区）
    cx_s = float(np.mean(solid_coords.x.array[0::2]))
    cy_s = float(np.mean(solid_coords.x.array[1::2]))
    # det(F) 最小值 + 所在单元质心（诊断翻转区域；严重变形时 eval 可能越界，
    # 用 try 保护，避免打断优雅终止）
    det_min = 1.0
    if rank == 0:
        try:
            det_vals = det_expr.eval(solid_mesh, np.arange(len(_solid_mid)))
            det_min = float(np.min(det_vals))
            if det_min < 0.5:
                c0 = int(np.argmin(det_vals))
                print(f"  [warn] det_min={det_min:.3f} @cell {c0}, "
                      f"centroid=({_solid_mid[c0][0]:.3f},"
                      f"{_solid_mid[c0][1]:.3f})", flush=True)
        except Exception as _e:  # noqa: BLE001 诊断失败不中断主循环
            print(f"  [warn] det_eval failed: {_e}", flush=True)
    det_min = comm.allreduce(det_min, op=MPI.MIN)

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
                  f"|disp|max={disp_max:.5f}, det_min={det_min:.4f}, "
                  f"vol={volume:.4f}, C=({cx_s:.3f},{cy_s:.3f})", flush=True)

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
