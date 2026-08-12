"""demo_343 圆盘随流体通过二维理想瓣膜 —— IB-FE FSI 求解。

流体为 2D 通道（8×1.61，正弦入口），固体为上下两片瓣膜 + 两个圆盘
（generate_mesh.py 生成 plot/mesh-343.xdmf）：
  rect1 下瓣膜 (cell 1)、rect2 上瓣膜 (cell 11)、circ1/circ2 圆盘 (cell 21/31)

材料：Neo-Hookean（PK1 = mu*(F-F^-T) + lambda*ln(J)*F^-T）。
  - 上瓣膜：mu_s（默认）
  - 下瓣膜：mu_s_down = 10×mu_s（更硬）
  - 圆盘：0.01×PK1（软，随流）

输出到 plot/（本地）。环境变量 STEPS 可覆盖步数（t 由步数决定）。

运行：
    conda activate afsi-dolfinx
    python generate_mesh.py          # 先生成固体网格
    STEPS=1000 python fsi_paralell.py
"""
import os
from mpi4py import MPI
from petsc4py import PETSc

import numpy as np

import dolfinx
from dolfinx import log
from dolfinx.fem import (Function, functionspace,
                         dirichletbc, locate_dofs_topological)
from dolfinx.mesh import CellType, GhostMode, locate_entities, meshtags
from basix.ufl import element

from ufl import (FacetNormal, Identity, Measure, TestFunction, TrialFunction, inv, ln, det,
                 as_vector, div, dot, ds, dx, inner, lhs, grad, nabla_grad, rhs, sym, system,
                 SpatialCoordinate)
from dolfinx.fem import form, assemble_scalar
from dolfinx.fem.petsc import create_vector, assemble_vector

from afsic import ChorinSolver, TimeManager
from afsic import swanlab_init, swanlab_upload
from configuration import config

_demo_dir = os.path.dirname(os.path.abspath(__file__))
_CIRCLE = int(os.environ.get("CIRCLE", "1"))  # 1=含圆盘, 0=无圆盘（对照，类比 demo_339 no_cylinder）
# 输出到本地 plot/circle<CIRCLE>/（有/无圆盘分开，便于对比）
config["output_path"] = os.path.join(_demo_dir, "plot", f"circle{_CIRCLE}") + os.sep
os.makedirs(config["output_path"], exist_ok=True)
swanlab_init(config['project_name'], config['experiment_name'], config)


###########################################################################################################
##########################################  Fluid #########################################################
###########################################################################################################
mesh = dolfinx.mesh.create_rectangle(
    comm=MPI.COMM_WORLD,
    points=((0.0, 0.0), (config["Lx"], config["Ly"])),
    n=(config["Nx"], config["Ny"]),
    cell_type=CellType.quadrilateral,
    ghost_mode=GhostMode.shared_facet,
)

mesh.topology.create_connectivity(1, 2)
marker_left, marker_right, marker_down, marker_up = 1, 2, 3, 4
boundaries = [(1, lambda x: np.isclose(x[0], 0)),
              (2, lambda x: np.isclose(x[0], config["Lx"])),
              (3, lambda x: np.isclose(x[1], 0)),
              (4, lambda x: np.isclose(x[1], config["Ly"]))]


def fixed_points(x):
    return np.logical_and(np.isclose(x[0], 0.0), np.isclose(x[1], 0.0))


point_loc = dolfinx.mesh.locate_entities_boundary(mesh, 0, fixed_points)

facet_indices, facet_markers = [], []
fdim = mesh.topology.dim - 1
for (marker, locator) in boundaries:
    facets = locate_entities(mesh, fdim, locator)
    facet_indices.append(facets)
    facet_markers.append(np.full_like(facets, marker))
facet_indices = np.hstack(facet_indices).astype(np.int32)
facet_markers = np.hstack(facet_markers).astype(np.int32)
sorted_facets = np.argsort(facet_indices)
facet_tag = meshtags(mesh, fdim, facet_indices[sorted_facets],
                     facet_markers[sorted_facets])

v_cg2 = element("Lagrange", mesh.topology.cell_name(), 2, shape=(mesh.geometry.dim,))
v_cg1 = element("Lagrange", mesh.topology.cell_name(), 1, shape=(mesh.geometry.dim,))
s_cg1 = element("Lagrange", mesh.topology.cell_name(), 1)
V_io = functionspace(mesh, v_cg1)
V = functionspace(mesh, v_cg2)
Q = functionspace(mesh, s_cg1)

fdim = mesh.topology.dim - 1
gdim = mesh.geometry.dim


class InletVelocity:
    def __init__(self, t):
        self.t = t

    def __call__(self, x):
        values = np.zeros((gdim, x.shape[1]), dtype=PETSc.ScalarType)
        values[0] = 5 * (np.sin(2 * np.pi * self.t) + 1.1) * x[1] * (1.61 - x[1])
        values[1] = 0.0
        return values


u_inlet = Function(V)
inlet_velocity = InletVelocity(0.0)
u_inlet.interpolate(inlet_velocity)
bcu_inlet = dirichletbc(u_inlet, locate_dofs_topological(
    V, fdim, facet_tag.find(marker_left)))
u_nonslip = np.array((0,) * mesh.geometry.dim, dtype=PETSc.ScalarType)
bcu_up = dirichletbc(u_nonslip, locate_dofs_topological(
    V, fdim, facet_tag.find(marker_up)), V)
bcu_down = dirichletbc(u_nonslip, locate_dofs_topological(
    V, fdim, facet_tag.find(marker_down)), V)
bcp_outlet = dirichletbc(0.0, locate_dofs_topological(
    Q, fdim, facet_tag.find(marker_right)), Q)
bcu = [bcu_inlet, bcu_up, bcu_down]
bcp = [bcp_outlet]

ns_solver = ChorinSolver(V, Q, bcu, bcp, config['dt'], config['rho'], config['mu'])

###########################################################################################################
##########################################  Structure  ####################################################
###########################################################################################################
import ufl

mesh_path = os.path.join(_demo_dir, "plot", "mesh-343.xdmf")
with dolfinx.io.XDMFFile(MPI.COMM_WORLD, mesh_path, "r",
                         encoding=dolfinx.io.XDMFFile.Encoding.HDF5) as file:
    structure = file.read_mesh(name="mesh")
    structure.topology.create_connectivity(structure.topology.dim - 1,
                                           structure.topology.dim)
    ft = file.read_meshtags(structure, "Facet markers")
    ct = file.read_meshtags(structure, "Cell markers")
    # 瓣膜尖端固定：上瓣膜边界 tag 15、下瓣膜边界 tag 4
    valve_up_facets = ft.find(15)
    valve_down_facets = ft.find(4)
    marked_facets = np.hstack([valve_up_facets, valve_down_facets])
    marked_values = np.hstack([np.full_like(valve_up_facets, 15),
                               np.full_like(valve_down_facets, 4)])
    sorted_facets = np.argsort(marked_facets)
    facet_tag = dolfinx.mesh.meshtags(structure, ft.dim,
                                      marked_facets[sorted_facets],
                                      marked_values[sorted_facets])
    facet_tag.name = ft.name
    cell_tag = ct

metadata = {"quadrature_degree": 4}
dss = ufl.Measure('ds', domain=structure, subdomain_data=facet_tag, metadata=metadata)
dxx = ufl.Measure("dx", domain=structure, subdomain_data=cell_tag, metadata=metadata)

v_cg2 = element("Lagrange", structure.topology.cell_name(),
                config["force_order"], shape=(structure.geometry.dim,))
v_cg1 = element("Lagrange", structure.topology.cell_name(),
                1, shape=(structure.geometry.dim,))
Vs = functionspace(structure, v_cg2)
Vs_io = functionspace(structure, v_cg1)

solid_coords = Function(Vs, name="solid_coords")
solid_coords_io = Function(Vs_io, name="solid_coords_io")
solid_force = Function(Vs, name="solid_force")
solid_force_io = Function(Vs_io, name="solid_force_io")
solid_velocity = Function(Vs, name="solid_velocity")

# ---- 材料弱形式（上/下瓣膜不同刚度，下瓣膜更硬） ----
# 复用 demo_340 已验证的 FRHMaterial（等容-体积分解 + 45° 纤维增强）：
#   psi = 0.5*C0*(I1bar-3) + C1*(exp(I4bar-1)-I4bar) + 0.5*kappa*(0.5*(J^2-1)-ln(J))
# 刚度对齐 demo_340（C0=2e5, C1=1e6, kappa=4e5）。下瓣膜 C0×mu_s_down_factor（更硬）。
from materials import FRHMaterial

dVs = TestFunction(Vs)
FF = grad(solid_coords)
X0 = ufl.SpatialCoordinate(structure)
x_constraint = solid_coords[0] - X0[0]
y_constraint = solid_coords[1] - X0[1]
circum_constraint = ufl.as_vector((x_constraint, y_constraint))

C0_up = config["C0"]
C1_up = config["C1"]
kappa_s = config["kappa"]
C0_down = config["mu_s_down_factor"] * C0_up      # 下瓣膜更硬
f1_u = ufl.as_vector((0.70710678118654750, -0.7071067811865475))   # 上瓣膜 45°
f1_d = ufl.as_vector((0.70710678118654750, 0.7071067811865475))    # 下瓣膜 45°

material_up = FRHMaterial(C0=C0_up, C1=C1_up, kappa_s=kappa_s, f1=f1_u)
material_down = FRHMaterial(C0=C0_down, C1=C1_up, kappa_s=kappa_s, f1=f1_d)
PK1_up = material_up.first_piola_kirchhoff_stress_v1(structure, solid_coords)
PK1_down = material_down.first_piola_kirchhoff_stress_v1(structure, solid_coords)

L_hat = -inner(PK1_up, grad(dVs)) * dxx(11)            # 上瓣膜（较软）
L_hat -= inner(PK1_down, grad(dVs)) * dxx(1)           # 下瓣膜（更硬）
if _CIRCLE:
    L_hat -= inner(0.01 * PK1_up, grad(dVs)) * dxx(21)     # 圆盘1（软，随流）
    L_hat -= inner(0.01 * PK1_up, grad(dVs)) * dxx(31)     # 圆盘2
L_hat -= config["beta"] * ufl.inner(circum_constraint, dVs) * dss(4)
L_hat -= config["beta"] * ufl.inner(circum_constraint, dVs) * dss(15)
L_hat = form(L_hat)
b1 = create_vector(Vs)  # dolfinx 0.10.0: create_vector 需函数空间而非 Form

###########################################################################################################
##########################################  Interaction  ##################################################
###########################################################################################################
from afsic import IBMesh, IBInterpolation

ibmesh = IBMesh(0.0, config["Lx"], 0.0, config["Ly"],
                config["Nx"], config["Ny"], config["velocity_order"])
ib_interpolation = IBInterpolation(ibmesh)
coords_bg = Function(V)
coords_bg.interpolate(lambda x: np.array([x[0], x[1]]))
solid_coords.interpolate(lambda x: np.array([x[0], x[1]]))
ibmesh.build_map(coords_bg._cpp_object)
ib_interpolation.evaluate_current_points(solid_coords._cpp_object)

###########################################################################################################
##########################################  Output  #######################################################
###########################################################################################################
u_io = Function(V_io)
file_velocity = dolfinx.io.XDMFFile(mesh.comm, config["output_path"] + "velocity.xdmf", "w")
file_solid = dolfinx.io.XDMFFile(mesh.comm, config["output_path"] + "solid_force.xdmf", "w")
file_velocity.write_mesh(mesh)
file_solid.write_mesh(structure)

time_manager = TimeManager(config['T'], config['num_steps'], fps=config['fps'])

form_u_L2 = form(dot(ns_solver.u_, ns_solver.u_) * dx)
form_p_L2 = form(dot(ns_solver.p_, ns_solver.p_) * dx)
form_X_L2 = form(dot(solid_coords, solid_coords) * dx)
form_F_L2 = form(dot(solid_force, solid_force) * dx)
form_volume = form(det(grad(solid_coords)) * dx)

log.set_log_level(log.LogLevel.INFO)
if MPI.COMM_WORLD.rank == 0:
    print(f"demo_343: {config['Nx']}×{config['Ny']}, dt={config['dt']}, "
          f"steps={config['num_steps']}, C0_up={C0_up:.1e}, "
          f"C0_down={C0_down:.1e}, out={config['output_path']}")

for step in range(config['num_steps']):
    current_time = step * config['dt']
    inlet_velocity.t = current_time
    u_inlet.interpolate(inlet_velocity)
    ns_solver.solve_one_step()
    ib_interpolation.fluid_to_solid(ns_solver.u_._cpp_object, solid_velocity._cpp_object)
    solid_coords.x.array[:] += solid_velocity.x.array[:] * config['dt']
    solid_coords.x.scatter_forward()
    u_L2 = mesh.comm.allreduce(assemble_scalar(form_u_L2), op=MPI.SUM)
    p_L2 = mesh.comm.allreduce(assemble_scalar(form_p_L2), op=MPI.SUM)
    F_L2 = mesh.comm.allreduce(assemble_scalar(form_F_L2), op=MPI.SUM)
    X_L2 = mesh.comm.allreduce(assemble_scalar(form_X_L2), op=MPI.SUM)
    volume = mesh.comm.allreduce(assemble_scalar(form_volume), op=MPI.SUM)
    u_max = ns_solver.u_.x.array.max()
    ib_interpolation.evaluate_current_points(solid_coords._cpp_object)
    with b1.localForm() as loc_2:
        loc_2.set(0)
    assemble_vector(b1, L_hat)
    b1.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    with b1.getBuffer() as arr:
        solid_force.x.array[:len(arr)] = arr[:]
    ib_interpolation.solid_to_fluid(ns_solver.f._cpp_object, solid_force._cpp_object)
    ns_solver.f.x.scatter_forward()

    if step == 1 or time_manager.should_output(step):
        u_io.interpolate(ns_solver.u_)
        file_velocity.write_function(u_io, current_time)
        solid_force_io.interpolate(solid_force)
        solid_coords_io.interpolate(solid_coords)
        file_solid.write_function(solid_force_io, current_time)
        file_solid.write_function(solid_coords_io, current_time)
        if MPI.COMM_WORLD.rank == 0:
            print(f"Step {step+1}/{config['num_steps']}, Time: {current_time:.3f}s, "
                  f"u_L2={u_L2:.4f}", flush=True)
            swanlab_upload(current_time, {"u_norm": u_L2, "p_norm": p_L2,
                                          "solid_force_norm": F_L2,
                                          "solid_coord_norm": X_L2,
                                          "volume": volume})

file_velocity.close()
file_solid.close()
if MPI.COMM_WORLD.rank == 0:
    print(f"\nDone. Output in: {config['output_path']}")
