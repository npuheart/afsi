"""demo_337 — idealized left ventricle FSI (diastole & systole).

Main program. IB-FE coupling of a passive Neo-Hookean LV ellipsoid
(loaded by a physiological endocardial pressure) inside a 3-D fluid
box, using the Chorin projection solver.

Method (per time step):
  1. Solve NS with the current body force f
  2. Interpolate fluid velocity to the solid mesh nodes
  3. Update solid position: X += v * dt
  4. Compute solid internal force (Neo-Hookean PK1 stress + base-ring
     penalty + endocardial pressure traction)
  5. Spread the solid force back to the fluid grid as f

Run (from this directory):
    python main.py                  # single process
    mpirun -n <N> python main.py    # parallel

All outputs are written to data/results/ (velocity, solid_force and a
metrics.csv summary).
"""

from mpi4py import MPI
from petsc4py import PETSc

import os
import json
import numpy as np

import dolfinx
from dolfinx import fem, default_scalar_type
from dolfinx import log
from dolfinx.fem import (Function, functionspace,
                         dirichletbc, locate_dofs_topological)
from dolfinx.mesh import CellType, GhostMode, locate_entities, meshtags
from basix.ufl import element

import ufl
from ufl import (FacetNormal, Measure, TestFunction, det,
                 as_vector, dot, ds, dx, grad, inner)
from dolfinx.fem import form, assemble_scalar
from dolfinx.fem.petsc import create_vector, assemble_vector

from afsic import ChorinSolver, TimeManager, IBMesh3D, IBInterpolation3D
from afsic import swanlab_init, swanlab_upload

from configuration import config

# 本构关系（同目录）
from NeoHookean import NeoHookeanMaterial
Material = NeoHookeanMaterial()

from PressureEndo import calculate_pressure_linear

# ------------------------------------------------------------------
# Output directory (data/results/) — 保证存在
# ------------------------------------------------------------------
if MPI.COMM_WORLD.rank == 0:
    os.makedirs(config["output_path"], exist_ok=True)

swanlab_init(config["project_name"], config["experiment_name"], config)

# ------------------------------------------------------------------
# 1. Fluid mesh (3-D box, hexahedra)
# ------------------------------------------------------------------
mesh = dolfinx.mesh.create_box(
    comm=MPI.COMM_WORLD,
    points=((0.0, 0.0, 0.0), (config["Lx"], config["Ly"], config["Lz"])),
    n=(config["Nx"], config["Ny"], config["Nz"]),
    cell_type=CellType.hexahedron,
    ghost_mode=GhostMode.shared_facet,
)

mesh.topology.create_connectivity(1, 2)
marker_left, marker_right, marker_down, marker_up, marker_front, marker_back = 1, 2, 3, 4, 5, 6
boundaries = [(1, lambda x: np.isclose(x[0], 0)),
              (2, lambda x: np.isclose(x[0], config["Lx"])),
              (3, lambda x: np.isclose(x[1], 0)),
              (4, lambda x: np.isclose(x[1], config["Ly"])),
              (5, lambda x: np.isclose(x[2], 0)),
              (6, lambda x: np.isclose(x[2], config["Lz"])),
              ]


def fixed_points(x):
    return np.logical_and.reduce((np.isclose(x[0], 0.0), np.isclose(x[1], 0.0), np.isclose(x[2], 0.0)))


point_loc = dolfinx.mesh.locate_entities_boundary(mesh, 0, fixed_points)

fdim = mesh.topology.dim - 1
facet_indices, facet_markers = [], []
for (marker, locator) in boundaries:
    facets = locate_entities(mesh, fdim, locator)
    facet_indices.append(facets)
    facet_markers.append(np.full_like(facets, marker))
facet_indices = np.hstack(facet_indices).astype(np.int32)
facet_markers = np.hstack(facet_markers).astype(np.int32)
sorted_facets = np.argsort(facet_indices)
facet_tag = meshtags(
    mesh, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets])

v_cg2 = element("Lagrange", mesh.topology.cell_name(),
                2, shape=(mesh.geometry.dim,))
v_cg1 = element("Lagrange", mesh.topology.cell_name(),
                1, shape=(mesh.geometry.dim,))
s_cg1 = element("Lagrange", mesh.topology.cell_name(), 1)
V_io = functionspace(mesh, v_cg1)
V = functionspace(mesh, v_cg2)
Q = functionspace(mesh, s_cg1)

gdim = mesh.geometry.dim


class UpVelocity():
    """Boundary velocity — zero everywhere (static box)."""
    def __init__(self, t):
        self.t = t

    def __call__(self, x):
        values = np.zeros((gdim, x.shape[1]), dtype=PETSc.ScalarType)
        return values


# Boundary conditions
u_up = Function(V)
up_velocity = UpVelocity(0.0)
u_up.interpolate(up_velocity)
bcu_up = dirichletbc(u_up, locate_dofs_topological(
    V, fdim, facet_tag.find(marker_up)))
u_nonslip = np.array((0,) * mesh.geometry.dim, dtype=PETSc.ScalarType)
bcu_left = dirichletbc(u_nonslip, locate_dofs_topological(
    V, fdim, facet_tag.find(marker_left)), V)
bcu_right = dirichletbc(u_nonslip, locate_dofs_topological(
    V, fdim, facet_tag.find(marker_right)), V)
bcu_down = dirichletbc(u_nonslip, locate_dofs_topological(
    V, fdim, facet_tag.find(marker_down)), V)
bcu_front = dirichletbc(u_nonslip, locate_dofs_topological(
    V, fdim, facet_tag.find(marker_front)), V)
bcu_back = dirichletbc(u_nonslip, locate_dofs_topological(
    V, fdim, facet_tag.find(marker_back)), V)
points_dofs = locate_dofs_topological(Q, 0, point_loc)
bcp_point = dirichletbc(0.0, points_dofs, Q)
bcu = [bcu_up, bcu_left, bcu_right, bcu_down, bcu_front, bcu_back]
bcp = [bcp_point]

ns_solver = ChorinSolver(V, Q, bcu, bcp, config["dt"], config["rho"], config["mu"])

# ------------------------------------------------------------------
# 2. Structure mesh (LV ellipsoid, from data/mesh/)
# ------------------------------------------------------------------
mesh_dir = config["mesh_dir"]
with open(f"{mesh_dir}/markers.json", "r", encoding="utf-8") as file:
    mesh_markers = json.load(file)

with dolfinx.io.XDMFFile(MPI.COMM_WORLD, f"{mesh_dir}/mesh.xdmf", "r",
                         encoding=dolfinx.io.XDMFFile.Encoding.HDF5) as file:
    structure = file.read_mesh(name="Mesh")
    structure.topology.create_connectivity(structure.topology.dim - 1, structure.topology.dim)
    ft = file.read_meshtags(structure, "Facet tags")
    endo_facets = ft.find(mesh_markers["ENDO"][0])
    base_facets = ft.find(mesh_markers["BASE"][0])
    epi_facets = ft.find(mesh_markers["EPI"][0])
    marked_facets = np.hstack([endo_facets, base_facets, epi_facets])
    marked_values = np.hstack([np.full_like(endo_facets, mesh_markers["ENDO"][0]),
                               np.full_like(base_facets, mesh_markers["BASE"][0]),
                               np.full_like(epi_facets, mesh_markers["EPI"][0])])
    sorted_facets = np.argsort(marked_facets)
    facet_tag = dolfinx.mesh.meshtags(structure, ft.dim,
                                      marked_facets[sorted_facets], marked_values[sorted_facets])
    facet_tag.name = ft.name

metadata = {"quadrature_degree": 5}
ds = Measure("ds", domain=structure, subdomain_data=facet_tag, metadata=metadata)

# 缩放并平移到流体域中央 (5x5x5 box)
structure._geometry._cpp_object.x[:, 0] = structure._geometry._cpp_object.x[:, 0] / 10.0 + 3.0
structure._geometry._cpp_object.x[:, 1] = structure._geometry._cpp_object.x[:, 1] / 10.0 + 2.5
structure._geometry._cpp_object.x[:, 2] = structure._geometry._cpp_object.x[:, 2] / 10.0 + 2.5

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

# 弱形式：Neo-Hookean PK1 应力 + 基底约束 + 心室内压牵引
dVs = TestFunction(Vs)
mu_s = config["mu_s"]
endo_pressure = fem.Constant(structure, default_scalar_type(0.0))
N = FacetNormal(structure)

FF = grad(solid_coords)
X0 = ufl.SpatialCoordinate(structure)

x_constraint = solid_coords[0] - X0[0]
y_constraint = solid_coords[1] - X0[1]
z_constraint = solid_coords[2] - X0[2]
circum_constraint = as_vector((x_constraint, y_constraint, z_constraint))

# First Piola-Kirchhoff stress
PK1 = Material.first_piola_kirchhoff_stress_v1(structure, solid_coords)
L_hat = -inner(PK1, grad(dVs)) * dx
L_hat -= config["beta"] * inner(circum_constraint, dVs) * ds(mesh_markers["BASE"][0])
L_hat -= inner(dVs, endo_pressure * ufl.cofac(FF) * N) * ds(mesh_markers["ENDO"][0])
L_hat = form(L_hat)
b1 = create_vector(Vs)  # dolfinx 0.10: create_vector 接受 FunctionSpace，不接受 Form

# ------------------------------------------------------------------
# 3. IB interaction
# ------------------------------------------------------------------
ibmesh = IBMesh3D(0.0, config["Lx"], 0.0, config["Ly"], 0.0, config["Lz"],
                  config["Nx"], config["Ny"], config["Nz"], config["velocity_order"])
ib_interpolation = IBInterpolation3D(ibmesh)
coords_bg = Function(V)
coords_bg.interpolate(lambda x: np.array([x[0], x[1], x[2]]))
solid_coords.interpolate(lambda x: np.array([x[0], x[1], x[2]]))
ibmesh.build_map(coords_bg._cpp_object)
ib_interpolation.evaluate_current_points(solid_coords._cpp_object)

# ------------------------------------------------------------------
# 4. Output setup (→ data/results/)
# ------------------------------------------------------------------
u_io = Function(V_io)
file_velocity = dolfinx.io.XDMFFile(mesh.comm, config["output_path"] + "velocity.xdmf", "w")
file_solid = dolfinx.io.XDMFFile(mesh.comm, config["output_path"] + "solid_force.xdmf", "w")
file_velocity.write_mesh(mesh)
file_solid.write_mesh(structure)

time_manager = TimeManager(config["T"], config["num_steps"], fps=20)

form_u_L2 = form(dot(ns_solver.u_, ns_solver.u_) * dx)
form_p_L2 = form(dot(ns_solver.p_, ns_solver.p_) * dx)
form_F_L2 = form(dot(solid_coords, solid_coords) * dx)
form_volume = form(det(grad(solid_coords)) * dx)

# metrics.csv 汇总（rank 0 追加）
metrics_csv = config["output_path"] + "metrics.csv"
if MPI.COMM_WORLD.rank == 0:
    with open(metrics_csv, "w") as f:
        f.write("step,time,u_L2,p_L2,solid_coords_L2,volume,endo_pressure\n")

log.set_log_level(log.LogLevel.INFO)
for step in range(config["num_steps"]):
    current_time = step * config["dt"]
    up_velocity.t = current_time
    endo_pressure.value = calculate_pressure_linear(
        current_time,
        diastole_pressure=config["diastole_pressure"],
        systole_pressure=config["systole_pressure"],
    )
    u_up.interpolate(up_velocity)

    # 1) 流场推进
    ns_solver.solve_one_step()

    # 2) 流体 → 固体：速度插值 + 固体位置推进
    ib_interpolation.fluid_to_solid(ns_solver.u_._cpp_object, solid_velocity._cpp_object)
    solid_coords.x.array[:] += solid_velocity.x.array[:] * config["dt"]
    solid_coords.x.scatter_forward()

    # 3) 指标
    u_L2 = mesh.comm.allreduce(assemble_scalar(form_u_L2), op=MPI.SUM)
    p_L2 = mesh.comm.allreduce(assemble_scalar(form_p_L2), op=MPI.SUM)
    F_L2 = mesh.comm.allreduce(assemble_scalar(form_F_L2), op=MPI.SUM)
    volume = mesh.comm.allreduce(assemble_scalar(form_volume), op=MPI.SUM)

    # 4) 固体内力 → 固体力
    ib_interpolation.evaluate_current_points(solid_coords._cpp_object)
    with b1.localForm() as loc_2:
        loc_2.set(0)
    assemble_vector(b1, L_hat)
    b1.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    with b1.getBuffer() as arr:
        solid_force.x.array[: len(arr)] = arr[:]

    # 5) 固体 → 流体：力扩散
    ib_interpolation.solid_to_fluid(ns_solver.f._cpp_object, solid_force._cpp_object)
    ns_solver.f.x.scatter_forward()

    # 6) 输出
    if MPI.COMM_WORLD.rank == 0:
        with open(metrics_csv, "a") as f:
            f.write(f"{step},{current_time:.6f},{u_L2:.6e},{p_L2:.6e},"
                    f"{F_L2:.6e},{volume:.6f},{endo_pressure.value:.6f}\n")

    if time_manager.should_output(step):
        u_io.interpolate(ns_solver.u_)
        file_velocity.write_function(u_io, current_time)
        solid_force_io.interpolate(solid_force)
        solid_coords_io.interpolate(solid_coords)
        file_solid.write_function(solid_force_io, current_time)
        file_solid.write_function(solid_coords_io, current_time)
        if MPI.COMM_WORLD.rank == 0:
            data_log = {
                "u_norm": u_L2,
                "p_norm": p_L2,
                "solid_force_norm": F_L2,
                "volume": volume,
                "endo_pressure": endo_pressure.value,
            }
            print(f"Step {step + 1}/{config['num_steps']}, Time: {current_time:.2f}s")
            swanlab_upload(current_time, data_log)

file_velocity.close()
file_solid.close()
