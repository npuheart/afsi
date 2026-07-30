"""Direct Forcing 圆柱绕流 — DFG 2D-3 基准 (Re=100).

基于 no_cylinder 纯流道基线，增加圆柱边界 direct forcing：
  1. 读取圆柱边界网格 (cylinder_solid.xdmf 的 facet_tags)
  2. 标记流体界面 DOFs (距边界 < 1.5h)
  3. 每步求解后强制界面 u=0，反算曳力/升力

与 IBFE 对比：无需固体本构、IB 插值、Lagrangian 网格。
"""

import os
import numpy as np
from petsc4py import PETSc
from mpi4py import MPI

import dolfinx
from dolfinx import log
from dolfinx.fem import (Function, functionspace,
                         dirichletbc, locate_dofs_topological)
from dolfinx.mesh import CellType, GhostMode
from basix.ufl import element
from ufl import dot, dx
from dolfinx.fem import form, assemble_scalar

from afsic import ChorinSolver, TimeManager
# from afsic import swanlab_init, swanlab_upload  # 需要网络，跳过
from afsic.common import (tag_boundaries, rectangle_boundaries,
                          TurekInlet, MARKER_LEFT, MARKER_RIGHT,
                          MARKER_BOTTOM, MARKER_TOP)
from configuration import config

comm = MPI.COMM_WORLD
rank = comm.rank

# swanlab_init(config['project_name'], config['experiment_name'], config,
#              api_key="odR9FodGeQojOPlk2sir1")

# ==========================================================================
# Fluid mesh (same as no_cylinder — full rectangle)
# ==========================================================================
mesh = dolfinx.mesh.create_rectangle(
    comm=comm,
    points=((0.0, 0.0), (config["Lx"], config["Ly"])),
    n=(config["Nx"], config["Ny"]),
    cell_type=CellType.quadrilateral,
    ghost_mode=GhostMode.shared_facet,
)

facet_tag = tag_boundaries(mesh, rectangle_boundaries(config["Lx"], config["Ly"]))

v_cg2 = element("Lagrange", mesh.topology.cell_name(), 2,
                shape=(mesh.geometry.dim,))
v_cg1 = element("Lagrange", mesh.topology.cell_name(), 1,
                shape=(mesh.geometry.dim,))
s_cg1 = element("Lagrange", mesh.topology.cell_name(), 1)
V_io = functionspace(mesh, v_cg1)
V    = functionspace(mesh, v_cg2)
Q    = functionspace(mesh, s_cg1)

fdim = mesh.topology.dim - 1

# Boundary conditions (same as no_cylinder)
inlet_velocity = TurekInlet(Um=config["Um"], Ly=config["Ly"])
u_inlet_func = Function(V)
u_inlet_func.interpolate(inlet_velocity)
bcu_inlet = dirichletbc(u_inlet_func,
                        locate_dofs_topological(V, fdim, facet_tag.find(MARKER_LEFT)))
u_zero = Function(V)
u_zero.x.array[:] = 0.0
bcu_bottom = dirichletbc(u_zero,
                         locate_dofs_topological(V, fdim, facet_tag.find(MARKER_BOTTOM)))
bcu_top    = dirichletbc(u_zero,
                         locate_dofs_topological(V, fdim, facet_tag.find(MARKER_TOP)))
bcp_outlet = dirichletbc(PETSc.ScalarType(0.0),
                         locate_dofs_topological(Q, fdim, facet_tag.find(MARKER_RIGHT)), Q)
bcu = [bcu_inlet, bcu_bottom, bcu_top]
bcp = [bcp_outlet]

ns_solver = ChorinSolver(V, Q, bcu, bcp, config['dt'], config['rho'], config['mu'])

# ==========================================================================
# Direct Forcing: 用 IB delta 核函数计算权重 α = S[S*[1]]
#
# 原理:
#   1. 常量场 u≡1 → 插值到固体 markers: w_k = S*[1]
#   2. spread 回去: α_ij = S[w_k]
#   3. 仿真时: U_k = S*[u] → f = S[-U_k/dt] → f /= max(α, ε)
#
# 好处: 自动处理核函数平滑，无需手动标记 DOF/调 h。
# ==========================================================================
from afsic import IBMesh, IBInterpolation

# 读取固体网格 → Lagrangian markers 坐标
solid_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "cylinder_solid.xdmf")
with dolfinx.io.XDMFFile(comm, solid_path, "r") as xdmf:
    structure = xdmf.read_mesh(name="mesh")
    structure.topology.create_connectivity(
        structure.topology.dim, structure.topology.dim - 1)

# 固体 DOF 坐标作为 Lagrangian markers
v_s = element("Lagrange", structure.topology.cell_name(),
              config["velocity_order"], shape=(structure.geometry.dim,))
Vs = functionspace(structure, v_s)
solid_coords = Function(Vs)
solid_coords.interpolate(lambda x: np.array([x[0], x[1]]))

# 创建 IBMesh + IBInterpolation
ibmesh = IBMesh(0.0, config["Lx"], 0.0, config["Ly"],
                config["Nx"], config["Ny"], config["velocity_order"])
ib_interp = IBInterpolation(ibmesh)
coords_bg = Function(V)
coords_bg.interpolate(lambda x: np.array([x[0], x[1]]))
ibmesh.build_map(coords_bg._cpp_object)
ib_interp.evaluate_current_points(solid_coords._cpp_object)

# --- 计算权重场 α = S[S*[1]] ---
fluid_one = Function(V)
one_arr = fluid_one.x.array
one_arr[0::2] = 1.0   # u_x = 1
one_arr[1::2] = 1.0   # u_y = 1
fluid_one.x.scatter_forward()

solid_w = Function(Vs)
ib_interp.fluid_to_solid(fluid_one._cpp_object, solid_w._cpp_object)
solid_w.x.scatter_forward()

fluid_alpha = Function(V)
ib_interp.solid_to_fluid(fluid_alpha._cpp_object, solid_w._cpp_object)
fluid_alpha.x.scatter_forward()
alpha_arr = fluid_alpha.x.array
eps = 1e-6
# 权重场: α_ij，用于归一化 spread-back 力
bs = V.dofmap.index_map_bs

# 用于力计算的临时场
solid_vel = Function(Vs)    # U_k = S*[u]
solid_force = Function(Vs)  # F_k = -U_k/dt
fluid_force = Function(V)   # f = S[F_k] / α

if rank == 0:
    D = config["D"]
    Re = config["rho"] * config["Um"] * D / config["mu"]
    alpha_max = alpha_arr.max()
    alpha_min = alpha_arr[alpha_arr > 0].min() if np.any(alpha_arr > 0) else 0
    print(f"Direct Forcing (delta-kernel weights): "
          f"{config['Nx']}×{config['Ny']}, dt={config['dt']}, Re≈{Re:.0f}")
    print(f"  α = S[S*[1]]: max={alpha_max:.4f} min>0={alpha_min:.4f}")

# ==========================================================================
# Output
# ==========================================================================
u_io, p_io = Function(V_io), Function(Q)
file_vel = dolfinx.io.XDMFFile(mesh.comm, config["output_path"] + "velocity.xdmf", "w")
file_pre = dolfinx.io.XDMFFile(mesh.comm, config["output_path"] + "pressure.xdmf", "w")
file_vel.write_mesh(mesh)
file_pre.write_mesh(mesh)

time_manager = TimeManager(config['T'], config['num_steps'], fps=100)
form_u_L2 = form(dot(ns_solver.u_, ns_solver.u_) * dx)
form_p_L2 = form(dot(ns_solver.p_, ns_solver.p_) * dx)
log.set_log_level(log.LogLevel.INFO)

drag_history, lift_history = [], []

# ==========================================================================
# Time loop
# ==========================================================================
dV = (config["Lx"] / config["Nx"]) * (config["Ly"] / config["Ny"])

for step in range(config['num_steps']):
    t = step * config['dt']

    inlet_velocity.update(t)
    u_inlet_func.interpolate(inlet_velocity)

    # 求解流体 (f 已含上一步的固体力)
    ns_solver.solve_one_step()

    # ---- Direct Forcing via delta kernel ----
    # 1. Interpolate u_fluid → solid markers BEFORE correction
    ib_interp.fluid_to_solid(ns_solver.u_._cpp_object, solid_vel._cpp_object)
    solid_vel.x.scatter_forward()

    # 2. Force at markers: F_k = -U_k / dt
    sv_arr = solid_vel.x.array
    sf_arr = solid_force.x.array
    drag_raw, lift_raw = 0.0, 0.0
    for k in range(len(sv_arr) // bs):
        for d in range(bs):
            sf_arr[k * bs + d] = -sv_arr[k * bs + d] / config['dt']
            if d == 0:
                drag_raw += sv_arr[k * bs + d]
            else:
                lift_raw += sv_arr[k * bs + d]
    solid_force.x.scatter_forward()

    # 3. Spread force back and normalize: f = S[F] / α
    ib_interp.solid_to_fluid(fluid_force._cpp_object, solid_force._cpp_object)
    fluid_force.x.scatter_forward()
    ff_arr = fluid_force.x.array
    for i in range(len(ff_arr)):
        a = alpha_arr[i]
        if a > eps:
            ff_arr[i] /= a
    fluid_force.x.scatter_forward()

    # 4. Velocity correction: u=0 where solid (alpha > threshold)
    u_arr = ns_solver.u_.x.array
    for i in range(len(u_arr)):
        if alpha_arr[i] > 1.0:
            u_arr[i] = 0.0
    ns_solver.u_n.x.array[:] = u_arr[:]  # sync u_n for next convection

    # 5. Body force for NEXT step
    ns_solver.f.x.array[:] = ff_arr[:]
    ns_solver.f.x.scatter_forward()

    # Drag/Lift
    drag = comm.allreduce(drag_raw, op=MPI.SUM) * dV / config['dt']
    lift = comm.allreduce(lift_raw, op=MPI.SUM) * dV / config['dt']
    drag_history.append((t, drag))
    lift_history.append((t, lift))

    # Output
    u_L2 = mesh.comm.allreduce(assemble_scalar(form_u_L2), op=MPI.SUM)
    p_L2 = mesh.comm.allreduce(assemble_scalar(form_p_L2), op=MPI.SUM)

    if time_manager.should_output(step):
        u_io.interpolate(ns_solver.u_)
        p_io.interpolate(ns_solver.p_)
        file_vel.write_function(u_io, t)
        file_pre.write_function(p_io, t)
        if rank == 0:
            Cd = 2.0 * drag / (config['rho'] * config['Um']**2 * D)
            Cl = 2.0 * lift / (config['rho'] * config['Um']**2 * D)
            print(f"  t={t:.3f}  u_L2={u_L2:.2f}  Cd={Cd:+.4f}  Cl={Cl:+.4f}")
            # swanlab_upload(t, {"u_L2": u_L2, "p_L2": p_L2,
            #                    "Cd": Cd, "Cl": Cl})

# ==========================================================================
# Final summary
# ==========================================================================
if rank == 0:
    drag_arr = np.array(drag_history)
    lift_arr = np.array(lift_history)
    mask = drag_arr[:, 0] > config['T'] / 2
    Cd_mean = np.mean(2.0 * drag_arr[mask, 1] /
                      (config['rho'] * config['Um']**2 * D))
    Cl_amp = (np.max(lift_arr[mask, 1]) - np.min(lift_arr[mask, 1])) / 2
    Cl_amp = 2.0 * Cl_amp / (config['rho'] * config['Um']**2 * D)

    print(f"\n  Results (t > {config['T']/2:.1f}s, Re≈{Re:.0f}):")
    print(f"    Mean Cd  = {Cd_mean:.4f}")
    print(f"    Cl amp   = {Cl_amp:.4f}")
    print(f"  Output: {config['output_path']}")
