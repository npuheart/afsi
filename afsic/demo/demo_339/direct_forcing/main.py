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
from afsic import swanlab_init, swanlab_upload
from afsic.common import (tag_boundaries, rectangle_boundaries,
                          TurekInlet, MARKER_LEFT, MARKER_RIGHT,
                          MARKER_BOTTOM, MARKER_TOP)
from configuration import config

comm = MPI.COMM_WORLD
rank = comm.rank

swanlab_init(config['project_name'], config['experiment_name'], config,
             api_key="odR9FodGeQojOPlk2sir1")

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
# Direct Forcing: 提取圆柱边界 → 标记流体界面 DOFs
# ==========================================================================
solid_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "cylinder_solid.xdmf")
with dolfinx.io.XDMFFile(comm, solid_path, "r") as xdmf:
    structure = xdmf.read_mesh(name="mesh")
    structure.topology.create_connectivity(
        structure.topology.dim, structure.topology.dim - 1)
    facet_tags_struct = xdmf.read_meshtags(structure, name="facet_tags")

# 固体边界顶点 (facet tag 2 = 圆柱外表面)
boundary_facets = facet_tags_struct.find(2)
# 通过 facet→vertex connectivity 获取边界顶点
c_f2v = structure.topology.connectivity(structure.topology.dim - 1, 0)
verts_set = set()
for f in boundary_facets:
    for v in c_f2v.links(f):
        verts_set.add(v)
verts_arr = np.array(sorted(verts_set), dtype=np.int32)
boundary_coords = structure.geometry.x[verts_arr]

# 流体 DOF 坐标 → 标记界面处 DOFs
V_coords = V.tabulate_dof_coordinates()
h = config["Ly"] / config["Ny"]
interface_dofs = set()
for bc in boundary_coords:
    dist = np.sqrt((V_coords[:, 0] - bc[0])**2 +
                   (V_coords[:, 1] - bc[1])**2)
    interface_dofs.update(np.where(dist < 1.5 * h)[0])
interface_dofs = np.array(sorted(interface_dofs), dtype=np.int32)
bs = V.dofmap.index_map_bs  # = gdim

if rank == 0:
    D = config["D"]
    Re = config["rho"] * config["Um"] * D / config["mu"]
    print(f"Direct Forcing: {config['Nx']}×{config['Ny']}, dt={config['dt']}, Re≈{Re:.0f}")
    print(f"  Boundary vertices: {len(boundary_coords)}, interface DOFs: {len(interface_dofs)}")

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
for step in range(config['num_steps']):
    t = step * config['dt']

    inlet_velocity.update(t)
    u_inlet_func.interpolate(inlet_velocity)

    # Direct Forcing: 同一步内求解 + 修正 (solve_one_step_df 内部完成)
    force_sum = ns_solver.solve_one_step_df(interface_dofs, bs)
    drag_raw, lift_raw = force_sum

    dV = (config["Lx"] / config["Nx"]) * (config["Ly"] / config["Ny"])
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
            swanlab_upload(t, {"u_L2": u_L2, "p_L2": p_L2,
                               "Cd": Cd, "Cl": Cl})

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
