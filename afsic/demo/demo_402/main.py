from petsc4py import PETSc
from mpi4py import MPI

import os
import numpy as np

import dolfinx
from dolfinx import log
from dolfinx.fem import Function, functionspace, dirichletbc, locate_dofs_topological
from dolfinx.mesh import CellType, GhostMode, locate_entities, meshtags
from basix.ufl import element

from ufl import (
    FacetNormal,
    Identity,
    Measure,
    SpatialCoordinate,
    TestFunction,
    TrialFunction,
    inv,
    ln,
    det,
    as_vector,
    div,
    dot,
    ds,
    dx,
    inner,
    lhs,
    grad,
    nabla_grad,
    rhs,
    sym,
    system,
)
from dolfinx.fem import form, assemble_scalar

from afsic import IPCSSolver, ChorinSolver, TimeManager
from afsic import swanlab_init, swanlab_upload
from dolfinx.fem.petsc import create_vector, assemble_vector, assemble_matrix
from configuration import config

swanlab_init(
    config["project_name"],
    config["experiment_name"],
    config,
    api_key="odR9FodGeQojOPlk2sir1",
)


def pressure_waveform(t, period, amp, fast_ratio, waveform="sin"):
    """Return pressure at time t.

    waveform:
        'sin'       – symmetric sine
        'fast_open' – piecewise linear two-segment:
                      [0, fast_ratio*T]         : 0 -> amp  (fast rise)
                      [fast_ratio*T, T]         : amp -> 0  (slow fall)
    """
    phase = (t % period) / period  # in [0, 1)
    if waveform == "sin":
        return amp * np.sin(2 * np.pi * phase)
    # # fast_open: slow 0→-amp, fast -amp→amp, slow amp→-amp, repeat
    # s1 = (1.0 - fast_ratio) / 2.0   # phase fraction for first slow segment
    # s2 = s1 + fast_ratio              # phase fraction at end of fast segment
    # if phase < s1:
    #     return -amp * (phase / s1)
    # elif phase < s2:
    #     return amp * (-1.0 + 2.0 * (phase - s1) / fast_ratio)
    # else:
    #     return amp * (1.0 - (phase - s2) / s1)
    # fast_open (default asymmetric)
    if phase < fast_ratio:
        return amp * (phase / fast_ratio)
    else:
        return amp * (1.0 - (phase - fast_ratio) / (1.0 - fast_ratio))


###########################################################################################################
##########################################  Fluid #########################################################
###########################################################################################################
# Create mesh
mesh = dolfinx.mesh.create_rectangle(
    comm=MPI.COMM_WORLD,
    points=((0.0, 0.0), (config["Lx"], config["Ly"])),
    n=(config["Nx"], config["Ny"]),
    cell_type=CellType.quadrilateral,
    ghost_mode=GhostMode.shared_facet,
)

# Mark the boundaries
mesh.topology.create_connectivity(1, 2)
marker_inlet, marker_outlet, marker_bottom, marker_top = 14, 12, 11, 13
boundaries = [
    (14, lambda x: np.isclose(x[0], 0)),  # inlet (left)
    (12, lambda x: np.isclose(x[0], config["Lx"])),  # outlet (right)
    (11, lambda x: np.isclose(x[1], 0)),  # bottom
    (13, lambda x: np.isclose(x[1], config["Ly"])), # top
]
facet_indices, facet_markers = [], []
fdim = mesh.topology.dim - 1
for marker, locator in boundaries:
    facets = locate_entities(mesh, fdim, locator)
    facet_indices.append(facets)
    facet_markers.append(np.full_like(facets, marker))
facet_indices = np.hstack(facet_indices).astype(np.int32)
facet_markers = np.hstack(facet_markers).astype(np.int32)
sorted_facets = np.argsort(facet_indices)
facet_tag = meshtags(
    mesh, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets]
)

v_cg2 = element("Lagrange", mesh.topology.cell_name(), 2, shape=(mesh.geometry.dim,))
v_cg1 = element("Lagrange", mesh.topology.cell_name(), 1, shape=(mesh.geometry.dim,))
s_cg1 = element("Lagrange", mesh.topology.cell_name(), 1)
V_io = functionspace(mesh, v_cg1)
V = functionspace(mesh, v_cg2)
Q = functionspace(mesh, s_cg1)

# Define boundary conditions
fdim = mesh.topology.dim - 1
gdim = mesh.geometry.dim
tdim = mesh.topology.dim


# Turek FSI parabolic inlet: u_x(y) = 1.5*Um * y*(Ly-y) / (Ly/2)^2
class InletVelocity:
    def __init__(self, Um, Ly):
        self.t = 0.0
        # 论文入口无斜坡（RAMP_T=0）；原 Turek 基准的 2 s 余弦软启动可设 RAMP_T=2
        self.t_ramp = float(os.environ.get("RAMP_T", "0.0"))
        self.Um = Um
        self.Ly = Ly
        self.scale = 0.0

    def update(self, t):
        self.t = t
        # smooth ramp-up over t_ramp seconds
        if self.t < self.t_ramp:
            self.scale = self.Um * (1.0 - np.cos(np.pi * self.t / self.t_ramp)) / 2.0
        else:
            self.scale = self.Um

    def __call__(self, x):
        values = np.zeros((gdim, x.shape[1]), dtype=PETSc.ScalarType)
        H = self.Ly
        values[0] = 1.5 * self.scale * x[1] * (H - x[1]) / (H / 2.0) ** 2
        return values


inlet_velocity = InletVelocity(config["Um"], config["Ly"])
u_inlet_func = Function(V)
u_inlet_func.interpolate(inlet_velocity)
bcu_inlet = dirichletbc(
    u_inlet_func, locate_dofs_topological(V, fdim, facet_tag.find(marker_inlet))
)

# No-slip on top/bottom (both components = 0), tags 11, 13
u_zero = Function(V)
u_zero.x.array[:] = 0.0
bcu_bottom = dirichletbc(
    u_zero, locate_dofs_topological(V, fdim, facet_tag.find(marker_bottom))
)
bcu_top = dirichletbc(
    u_zero, locate_dofs_topological(V, fdim, facet_tag.find(marker_top))
)

# Pressure outlet (right wall, tag 12)
bcp_outlet = dirichletbc(
    PETSc.ScalarType(0.0),
    locate_dofs_topological(Q, fdim, facet_tag.find(marker_outlet)),
    Q,
)

bcu = [bcu_inlet, bcu_bottom, bcu_top]
bcp = [bcp_outlet]

# 实验开关：IB 载荷处理方式（三选一，默认原始实现；不得组合使用）。
#   默认            : 铺展 f_raw=(1/V_h)J^T F 作为节点值，由弱式经质量矩阵组装（与采样不互相伴）。
#   IB_CONSISTENT=1 : 解 M f = V_h f_raw，使组装载荷 = J^T F（质量一致方案）。
#   IB_DIRECT_LOAD=1: 动量弱式去掉 ∫f·v 项，直接把 b_ib = V_h·f_stored 加到右端
#                     （J^T F 的等价直接载荷，省去质量矩阵求解）。
IB_CONSISTENT = os.environ.get("IB_CONSISTENT", "0").lower() not in ("0", "", "false", "no")
# IB_DIRECT_LOAD 默认开启：b_ib = V_h·f_stored 与采样算子互为伴随（推荐路径，
# 已验证 IPCS 稳定跑满 T=10 s）；旧的装配式载荷可用 IB_DIRECT_LOAD=0 复现。
IB_DIRECT_LOAD = os.environ.get("IB_DIRECT_LOAD", "1").lower() not in ("0", "", "false", "no")
if IB_CONSISTENT and IB_DIRECT_LOAD:
    raise ValueError("IB_CONSISTENT 与 IB_DIRECT_LOAD 不能同时开启（会重复处理 IB 载荷）")

# Define Solver
# SOLVER=chorin (默认) 或 SOLVER=ipcs，便于同一算例下的方法对比。
# 两个求解器的动量弱式现在使用同一符号约定（都含 -∫f·v 项），因此两种方法都直接
# 存放物理 IB 力 b1 的原值，无需任何 force_scale 类符号补偿（已移除）。
SOLVER = os.environ.get("SOLVER", config.get("nssolver", "chorin")).lower()
if SOLVER.startswith("ipcs"):
    ns_solver = IPCSSolver(V, Q, bcu, bcp, config["dt"], config["rho"], config["mu"],
                           ib_body_force=not IB_DIRECT_LOAD)
    if MPI.COMM_WORLD.rank == 0:
        print("solver: IPCSSolver (incremental pressure correction)")
else:
    ns_solver = ChorinSolver(V, Q, bcu, bcp, config["dt"], config["rho"], config["mu"],
                             ib_body_force=not IB_DIRECT_LOAD)
    if MPI.COMM_WORLD.rank == 0:
        print("solver: ChorinSolver (projection / fractional step)")

# 实验开关：FREEZE_SOLID=1 冻结固体坐标（约束残差≈0 → f≈0），
# 用于"纯流体、无固体反馈"对照实验。
FREEZE_SOLID = os.environ.get("FREEZE_SOLID", "0").lower() not in ("0", "", "false", "no")
# 实验开关：FREEZE_FORCE_AT=<t*>(秒)。到 t* 时冻结固体构型与 IB 力：之后固体
# 不再运动、持续施加冻结时刻的同一非零铺展力。用于区分"纯流体压力修正算子失稳"
# 与"固体运动-力更新-压力动态耦合失稳"两种机制。
FREEZE_FORCE_AT = os.environ.get("FREEZE_FORCE_AT", "")
_freeze_force_step = (
    int(round(float(FREEZE_FORCE_AT) / config["dt"])) if FREEZE_FORCE_AT else None
)

###########################################################################################################
##########################################  Structure  ####################################################
###########################################################################################################
# Turek FSI: circle (area tag 1, facet tag 3) + elastic tail (area tag 2)
turek_mesh_path = os.path.join(os.path.dirname(__file__), "./turek_mesh.xdmf")
with dolfinx.io.XDMFFile(MPI.COMM_WORLD, turek_mesh_path, "r") as xdmf:
    structure = xdmf.read_mesh(name="mesh")
    structure.topology.create_connectivity(
        structure.topology.dim, structure.topology.dim - 1
    )
    cell_tags = xdmf.read_meshtags(structure, name="cell_tags")
    facet_tags = xdmf.read_meshtags(structure, name="facet_tags")

# 固体网格已是 CGS（cm）—— turek.geo 直接用 cm 建模，此处不再做 SI->CGS 缩放。

v_cg2_s = element(
    "Lagrange",
    structure.topology.cell_name(),
    config["force_order"],
    shape=(structure.geometry.dim,),
)
v_cg1_s = element(
    "Lagrange", structure.topology.cell_name(), 1, shape=(structure.geometry.dim,)
)
Vs = functionspace(structure, v_cg2_s)
Vs_io = functionspace(structure, v_cg1_s)

solid_coords = Function(Vs, name="solid_coords")
solid_coords_io = Function(Vs_io, name="solid_coords_io")
solid_force = Function(Vs, name="solid_force")
solid_force_io = Function(Vs_io, name="solid_force_io")
solid_velocity = Function(Vs, name="solid_velocity")

dVs = TestFunction(Vs)
mu_s = config["mu_s"]
lambda_s = config["lambda_s"]
beta = config["beta"]

# 圆柱系绳（弹簧）罚参数。论文(SI)给的是 kappa_s = 5.0e4 * Δx/Δt²，其中无量纲组
#   kappa_hat = kappa_s*Δt²/(ρΔx) 与单位制无关（论文取 5e4）。
# CGS(g,cm,s) 换算：kappa_s = kappa_hat * ρ * Δx/Δt²  [dyne/cm^4]
#   ρ=1 g/cm³、Δx=1.9219 cm、Δt=5e-5 s -> κ̂=5e4 ⇔ 3.84e13 dyne/cm^4；
#   但显式无质量 IB 耦合的预算里 κ̂≳2.5 就爆（实测 2.5 爆 / 1.0 稳），
#   故默认 κ̂=1.0（=7.69e8 dyne/cm^4；圆柱漂移 ~1e-3 cm，等效刚固）。
# PENALTY_MODE=beta 时改用 config["beta"] 固定值（旧行为）。
# 论文的 rigid penalty 力为 F = kappa*(psi - chi) + eta*(V - dchi/dt)（V=0 为静止目标）。
# 本实现默认只含弹性项（ETA_HAT=0，纯弹簧，即 eta=0 的特例）；ETA_HAT>0 时加入阻尼项
#   eta = ETA_HAT * ρ * Δx/Δt  [dyne*s/cm^4]，F_damp = -eta * dchi/dt。
PENALTY_MODE = os.environ.get("PENALTY_MODE", "paper").lower()
KAPPA_HAT = float(os.environ.get("KAPPA_HAT", "1.0"))
ETA_HAT = float(os.environ.get("ETA_HAT", "0.0"))
_dx = config["Lx"] / config["Nx"]
if PENALTY_MODE.startswith("paper"):
    beta = KAPPA_HAT * config["rho"] * _dx / config["dt"] ** 2
    if MPI.COMM_WORLD.rank == 0:
        print(f"penalty(paper form): kappa_hat={KAPPA_HAT:g} -> beta={beta:.6g} "
              f"dyne/cm^4  (dx={_dx:.6g} cm, dt={config['dt']:.6g} s)")
eta = ETA_HAT * config["rho"] * _dx / config["dt"]
if ETA_HAT > 0 and MPI.COMM_WORLD.rank == 0:
    print(f"tether damping: eta_hat={ETA_HAT:g} -> eta={eta:.6g} dyne*s/cm^4")

FF = grad(solid_coords)
J = det(FF)

# Penalty: fix circle boundary (facet tag 3), same as turtle head/tail fixation
X0 = SpatialCoordinate(structure)
dss = Measure("ds", domain=structure, subdomain_data=facet_tags)
dxx = Measure("dx", domain=structure, subdomain_data=cell_tags)
x_constraint = solid_coords[0] - X0[0]
y_constraint = solid_coords[1] - X0[1]
solid_constraint = as_vector((x_constraint, y_constraint))

# 固体本构：SOLID_LAW=svk（默认，论文的 Saint Venant–Kirchhoff）或 neo_hookean（旧默认）
# 论文(SI) mu_s=1e6 Pa -> CGS 1e7 dyne/cm²；lambda_s=8e6 Pa -> 8e7 dyne/cm²
# （1 Pa = 10 dyne/cm²）—— 见 configuration.py 的默认值。
SOLID_LAW = os.environ.get("SOLID_LAW", "svk").lower()
# 实验开关 BALL_STIFFNESS_FACTOR（默认 1）：圆柱区（tag 1）弹性模量放大系数。
# 论文的圆柱是 immersed rigid structure（纯罚力、无弹性应力）；本 demo 用"弹性盘+系绳"
# 近似。实测圆柱残余变形主要来自其自身弹性（κ̂ 0.25→1.0 只降 ~20%），故提供此开关把
# 圆柱弹硬（×10 稳定、×100 会超出显式无质量耦合的刚度预算而爆）。
BALL_STIFF = float(os.environ.get("BALL_STIFFNESS_FACTOR", "1.0"))
if BALL_STIFF != 1.0 and MPI.COMM_WORLD.rank == 0:
    print(f"ball stiffness factor: {BALL_STIFF:g} (cylinder elastic moduli x{BALL_STIFF:g})")
if SOLID_LAW.startswith("svk"):
    # S = lambda_s*tr(E)*I + 2*mu_s*E,  E = (F^T F - I)/2,  P = F·S
    # （二维平面应变：E_33 = 0，故用三维 lambda_s 直接写面内分量）
    from ufl import tr as _tr
    _E = 0.5 * (dot(FF.T, FF) - Identity(gdim))
    _S = lambda_s * _tr(_E) * Identity(gdim) + 2.0 * mu_s * _E
    P_s = dot(FF, _S)
    if BALL_STIFF != 1.0:
        _S_ball = (lambda_s * BALL_STIFF) * _tr(_E) * Identity(gdim) \
            + 2.0 * (mu_s * BALL_STIFF) * _E
        P_s_ball = dot(FF, _S_ball)
    else:
        P_s_ball = P_s
else:
    # Neo-Hookean: P = mu_s*(F - F^-T) + lambda_s*ln(J)*F^-T
    I1 = inner(FF, FF)
    P_iso = mu_s * J ** (-2.0 / 2.0) * (FF - (I1 / 2.0) * inv(FF).T)
    P_vol = lambda_s * ln(J) * inv(FF).T
    P_s = P_iso + P_vol
    P_s_ball = P_s

# Circle (facet tag 3, cell tag 1) is fixed via penalty; tail (cell tag 2) deforms freely
_L_expr = (
    -inner(P_s_ball, grad(dVs)) * dxx(1)
    - inner(P_s, grad(dVs)) * dxx(2)
    - beta * inner(solid_constraint, dVs) * dxx(1)
    #  - beta*inner(circum_constraint, dVs)*dss(3)
)
if ETA_HAT > 0:
    # 论文系绳的阻尼项 -eta*dchi/dt（V=0）：solid_velocity 为本步插值得到的固体速度
    _L_expr = _L_expr - eta * inner(solid_velocity, dVs) * dxx(1)
L_hat = form(_L_expr)
# dolfinx 0.10: create_vector 需要函数空间，而不是 Form
b1 = create_vector(Vs)

###########################################################################################################
##########################################  Interaction  ##################################################
###########################################################################################################
from afsic import IBMesh, IBInterpolation

ibmesh = IBMesh(
    0.0,
    config["Lx"],
    0.0,
    config["Ly"],
    config["Nx"],
    config["Ny"],
    config["velocity_order"],
)
ib_interpolation = IBInterpolation(ibmesh)
coords_bg = Function(V)
coords_bg.interpolate(lambda x: np.array([x[0], x[1]]))
solid_coords.interpolate(lambda x: np.array([x[0], x[1]]))
ibmesh.build_map(coords_bg._cpp_object)
ib_interpolation.evaluate_current_points(solid_coords._cpp_object)

# 实验开关（IB_CONSISTENT / IB_DIRECT_LOAD 已在本文件上方、求解器创建前定义）：
#  * IB_CONSISTENT=1：解 M f = V_h f_raw（质量矩阵求解不做任何边界条件处理，全局 SPD
#    问题），使弱式组装载荷 = J^T F，与速度采样算子 J 互为伴随。
#  * IB_DIRECT_LOAD=1：动量弱式不含 ∫f·v 项（ib_body_force=False），由求解器在 lifting
#    之后、set_bc 之前把 b_ib = V_h·f_stored 加到右端 owned 自由度。
if IB_CONSISTENT or IB_DIRECT_LOAD:
    _hx = config["Lx"] / (config["velocity_order"] * config["Nx"])
    _hy = config["Ly"] / (config["velocity_order"] * config["Ny"])
    _Vh = _hx * _hy
    if MPI.COMM_WORLD.rank == 0:
        _mode = ("mass-consistent (M f = V_h f_raw)" if IB_CONSISTENT
                 else "direct load (b_ib = V_h*f_stored)")
        print(f"IB load fix: {_mode} enabled (V_h={_Vh:g})")
if IB_CONSISTENT:
    _A_mass = assemble_matrix(form(inner(TrialFunction(V), TestFunction(V)) * dx))
    _A_mass.assemble()
    _ksp_mass = PETSc.KSP().create(mesh.comm)
    _ksp_mass.setOperators(_A_mass)
    _ksp_mass.setType(PETSc.KSP.Type.CG)
    _ksp_mass.getPC().setType(PETSc.PC.Type.JACOBI)
    _ksp_mass.setTolerances(rtol=1e-10, atol=1e-300)
    _b_ib = create_vector(V)
if IB_DIRECT_LOAD:
    _b_direct = create_vector(V)
    ns_solver.ib_load = _b_direct  # 求解器在 lifting 后、set_bc 前将其加入动量右端

###########################################################################################################
##########################################  Output  #######################################################
###########################################################################################################
u_io = Function(V_io)
p_io = Function(Q)
file_velocity = dolfinx.io.XDMFFile(
    mesh.comm, config["output_path"] + "velocity.xdmf", "w"
)
file_pressure = dolfinx.io.XDMFFile(
    mesh.comm, config["output_path"] + "pressure.xdmf", "w"
)
file_solid = dolfinx.io.XDMFFile(mesh.comm, config["output_path"] + "solid.xdmf", "w")
file_velocity.write_mesh(mesh)
file_pressure.write_mesh(mesh)
file_solid.write_mesh(structure)

time_manager = TimeManager(config["T"], config["num_steps"], fps=100)

form_u_L2 = form(dot(ns_solver.u_, ns_solver.u_) * dx)
form_p_L2 = form(dot(ns_solver.p_, ns_solver.p_) * dx)
form_F_L2 = form(dot(solid_coords, solid_coords) * dx)
form_volume = form(det(grad(solid_coords)) * dx)

log.set_log_level(log.LogLevel.INFO)
for step in range(config["num_steps"]):
    current_time = step * config["dt"]
    inlet_velocity.update(current_time)
    u_inlet_func.interpolate(inlet_velocity)
    _force_frozen = (_freeze_force_step is not None) and (step >= _freeze_force_step)
    ns_solver.solve_one_step()
    ib_interpolation.fluid_to_solid(
        ns_solver.u_._cpp_object, solid_velocity._cpp_object
    )
    if not FREEZE_SOLID and not _force_frozen:
        solid_coords.x.array[:] += solid_velocity.x.array[:] * config["dt"]
        solid_coords.x.scatter_forward()
    u_L2 = mesh.comm.allreduce(assemble_scalar(form_u_L2), op=MPI.SUM)
    p_L2 = mesh.comm.allreduce(assemble_scalar(form_p_L2), op=MPI.SUM)
    F_L2 = mesh.comm.allreduce(assemble_scalar(form_F_L2), op=MPI.SUM)
    volume = mesh.comm.allreduce(assemble_scalar(form_volume), op=MPI.SUM)

    if not _force_frozen:
        ib_interpolation.evaluate_current_points(solid_coords._cpp_object)
        with b1.localForm() as loc_b:
            loc_b.set(0)
        assemble_vector(b1, L_hat)
        b1.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        with b1.getBuffer() as arr:
            solid_force.x.array[: len(arr)] = arr[:]
        ib_interpolation.solid_to_fluid(ns_solver.f._cpp_object, solid_force._cpp_object)
        ns_solver.f.x.scatter_forward()
        if IB_CONSISTENT:
            # 质量一致方案：解 M f = V_h f_raw  ⇔  组装载荷 = J^T F（采样算子的伴随）
            ns_solver.f.x.petsc_vec.copy(result=_b_ib)
            _b_ib.scale(_Vh)
            _ksp_mass.solve(_b_ib, ns_solver.f.x.petsc_vec)
            ns_solver.f.x.scatter_forward()
        elif IB_DIRECT_LOAD:
            # 直接载荷方案：b_ib = V_h·f_stored = J^T F 的等价右端量。
            ns_solver.f.x.petsc_vec.copy(result=_b_direct)
            _b_direct.scale(_Vh)

    data_log = {}
    if time_manager.should_output(step):
        u_io.interpolate(ns_solver.u_)
        p_io.interpolate(ns_solver.p_)
        file_velocity.write_function(u_io, current_time)
        file_pressure.write_function(p_io, current_time)
        solid_force_io.interpolate(solid_force)
        solid_coords_io.interpolate(solid_coords)
        file_solid.write_function(solid_force_io, current_time)
        file_solid.write_function(solid_coords_io, current_time)
        if MPI.COMM_WORLD.rank == 0:
            data_log["u_norm"] = u_L2
            data_log["p_norm"] = p_L2
            data_log["solid_force_norm"] = F_L2
            data_log["volume"] = volume
            data_log["inlet_velocity"] = inlet_velocity.scale
            print(f"Step {step+1}/{config['num_steps']}, Time: {current_time:.2f}s")
            swanlab_upload(current_time, data_log)
