# -----------------------------------------------------------------------------
# 版权所有 (c) 2025 保留所有权利。
#
# 本文件隶属于 Poromechanics Solver 项目，主要开发者为：
#   - 马鹏飞：mapengfei@mail.nwpu.edu.cn
#   - 王璇：wangxuan2022@mail.nwpu.edu.cn
#
# 本软件仅供内部使用和学术研究之用。未经明确许可，严禁重新分发、修改或用于商业用途。
# 详细授权条款请参阅：https://www.pengfeima.cn/license-strict/
# -----------------------------------------------------------------------------

"""
3D FSI simulation of a vessel wall.
Fluid domain: box encompassing the vessel.
Solid domain: wedge-element vessel wall mesh (read from XDMF).
Boundaries are identified geometrically (no facet tags in the mesh file).
"""

from afsic import unique_filename
from mpi4py import MPI
from petsc4py import PETSc

import time
import requests
import numpy as np

import dolfinx
from dolfinx import fem, default_scalar_type
from dolfinx import log
from dolfinx.fem import (Function, functionspace,
                         dirichletbc, locate_dofs_topological)
from dolfinx.mesh import CellType, GhostMode, locate_entities, meshtags
from basix.ufl import element

from ufl import (FacetNormal, Identity, Measure, TestFunction, TrialFunction, inv, ln, det,
                 as_vector, div, dot, ds, dx, inner, lhs, grad, nabla_grad, rhs, sym, system)
from dolfinx.fem import form, assemble_scalar

from afsic import IPCSSolver, ChorinSolver, TimeManager
from afsic import swanlab_init, swanlab_upload
from dolfinx.fem.petsc import create_vector, assemble_vector

# 定义本构关系
from NeoHookean import NeoHookeanMaterial
Material = NeoHookeanMaterial()

from PressureEndo import calculate_pressure_linear, mmHg

# Define the configuration for the simulation
config = {"nssolver": "chorinsolver",
          "project_name": "demo-405",
          "tag": "vessel-wall-3d",
          "velocity_order": 2,
          "force_order": 2,
          "pressure_order": 1,
          "num_processors": MPI.COMM_WORLD.size,
          "T": 0.005,
          "dt": 1/1000,
          "rho": 1.0,
          "Lx": 8.0,
          "Ly": 8.0,
          "Lz": 20.0,
          "Nx": 32,
          "Ny": 32,
          "Nz": 40,
          "mu": 0.01,
          "mu_s": 0.1,  # Solid elasticity
          "diastole_pressure": 8.0*mmHg,
          "systole_pressure": 110.0*mmHg,
          "beta": 5e6,   # Penalty for fixing end caps
          }

config["num_steps"] = int(config['T']/config['dt'])
config["output_path"] = unique_filename(config['project_name'], config['tag']) if MPI.COMM_WORLD.rank == 0 else None
config["output_path"] = MPI.COMM_WORLD.bcast(config["output_path"], root=0)
config["experiment_name"] = requests.get(f"http://counter.pengfeima.cn/{config['project_name']}").text if MPI.COMM_WORLD.rank == 0 else None
config["experiment_name"] = MPI.COMM_WORLD.bcast(config["experiment_name"], root=0)
swanlab_init(config['project_name'], config['experiment_name'], config)


###########################################################################################################
##########################################  Fluid #########################################################
###########################################################################################################
# Create fluid mesh (box starting at origin)
mesh = dolfinx.mesh.create_box(
    comm=MPI.COMM_WORLD,
    points=((0.0, 0.0, 0.0), (config["Lx"], config["Ly"], config["Lz"])),
    n=(config["Nx"], config["Ny"], config["Nz"]),
    cell_type=CellType.hexahedron,
    ghost_mode=GhostMode.shared_facet,
)

# Mark the fluid boundaries
mesh.topology.create_connectivity(1, 2)
marker_left, marker_right, marker_down, marker_up, marker_front, marker_back = 1, 2, 3, 4, 5, 6
boundaries = [(1, lambda x: np.isclose(x[0], 0.0)),
              (2, lambda x: np.isclose(x[0], config["Lx"])),
              (3, lambda x: np.isclose(x[1], 0.0)),
              (4, lambda x: np.isclose(x[1], config["Ly"])),
              (5, lambda x: np.isclose(x[2], 0.0)),
              (6, lambda x: np.isclose(x[2], config["Lz"])),
              ]

def fixed_points(x):
    return np.logical_and.reduce((np.isclose(x[0], 0.0),
                                  np.isclose(x[1], 0.0),
                                  np.isclose(x[2], 0.0)))

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
facet_tag = meshtags(
    mesh, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets])

v_cg2 = element("Lagrange", mesh.topology.cell_name(),
                2, shape=(mesh.geometry.dim, ))
v_cg1 = element("Lagrange", mesh.topology.cell_name(),
                1, shape=(mesh.geometry.dim, ))
s_cg1 = element("Lagrange", mesh.topology.cell_name(), 1)
V_io = functionspace(mesh, v_cg1)
V = functionspace(mesh, v_cg2)
Q = functionspace(mesh, s_cg1)

# Define boundary conditions
fdim = mesh.topology.dim - 1
gdim = mesh.geometry.dim

class UpVelocity():
    def __init__(self, t):
        self.t = t
    def __call__(self, x):
        values = np.zeros((gdim, x.shape[1]), dtype=PETSc.ScalarType)
        values[2] = 0.0  # No inlet velocity by default (vessel in quiescent fluid)
        return values

# Inlet
u_up = Function(V)
up_velocity = UpVelocity(0.0)
u_up.interpolate(up_velocity)
bcu_up = dirichletbc(u_up, locate_dofs_topological(
    V, fdim, facet_tag.find(marker_down)))
# Walls - all no-slip
u_nonslip = np.array((0,) * mesh.geometry.dim, dtype=PETSc.ScalarType)
bcu_left = dirichletbc(u_nonslip, locate_dofs_topological(
    V, fdim, facet_tag.find(marker_left)), V)
bcu_right = dirichletbc(u_nonslip, locate_dofs_topological(
    V, fdim, facet_tag.find(marker_right)), V)
bcu_up_wall = dirichletbc(u_nonslip, locate_dofs_topological(
    V, fdim, facet_tag.find(marker_up)), V)
bcu_front = dirichletbc(u_nonslip, locate_dofs_topological(
    V, fdim, facet_tag.find(marker_front)), V)
bcu_back = dirichletbc(u_nonslip, locate_dofs_topological(
    V, fdim, facet_tag.find(marker_back)), V)
points_dofs = locate_dofs_topological(Q, 0, point_loc)
bcp_point = dirichletbc(0.0, points_dofs, Q)
bcu = [bcu_up, bcu_left, bcu_right, bcu_up_wall, bcu_front, bcu_back]
bcp = [bcp_point]

# Define Solver
ns_solver = ChorinSolver(V, Q, bcu, bcp, config['dt'], config['rho'], config['mu'])

###########################################################################################################
##########################################  Structure  ####################################################
###########################################################################################################
import ufl
import os

# ------------------------------------------------------------------------------
# Convert prism mesh to tetrahedra (FFCX doesn't support prism elements)
# ------------------------------------------------------------------------------
_tet_mesh_path = "vessel_wall_wall_3d_tet.xdmf"
if MPI.COMM_WORLD.rank == 0 and not os.path.exists(_tet_mesh_path):
    from basix.ufl import element as basix_element
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "vessel_wall_wall_3d.xdmf", "r",
                             encoding=dolfinx.io.XDMFFile.Encoding.HDF5) as _f:
        _prism_mesh = _f.read_mesh(name="Grid")
    _prism_mesh.topology.create_connectivity(_prism_mesh.topology.dim, 0)
    _c2v = _prism_mesh.topology.connectivity(_prism_mesh.topology.dim, 0)
    _nc = _prism_mesh.topology.index_map(_prism_mesh.topology.dim).size_local
    _tet_conn = np.zeros((_nc * 3, 4), dtype=np.int64)
    for _c in range(_nc):
        _v = _c2v.links(_c)
        _tet_conn[3*_c]   = [_v[0], _v[1], _v[2], _v[5]]
        _tet_conn[3*_c+1] = [_v[0], _v[1], _v[5], _v[4]]
        _tet_conn[3*_c+2] = [_v[0], _v[4], _v[5], _v[3]]
    _e = basix_element('Lagrange', 'tetrahedron', 1, shape=(3,))
    _tet_mesh = dolfinx.mesh.create_mesh(MPI.COMM_WORLD, _tet_conn, _e, _prism_mesh.geometry.x.copy())
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, _tet_mesh_path, "w") as _f:
        _f.write_mesh(_tet_mesh)
    print("Converted prism mesh to tetrahedra.")

MPI.COMM_WORLD.barrier()

# Read the vessel wall mesh (tetrahedral, converted from wedge)
with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "vessel_wall_wall_3d_tet.xdmf", "r",
                         encoding=dolfinx.io.XDMFFile.Encoding.HDF5) as file:
    structure = file.read_mesh(name="mesh")

# Build topology connectivity
structure.topology.create_connectivity(structure.topology.dim - 1, structure.topology.dim)
sfdim = structure.topology.dim - 1

# ------------------------------------------------------------------------------
# Geometric boundary classification (no facet tags available in mesh file)
# The vessel is a tubular wall along the z-axis. We classify facets into:
#   marker_inner (1): luminal (inner) wall
#   marker_outer (2): outer wall
#   marker_inlet (3): inlet end cap
#   marker_outlet (4): outlet end cap
# ------------------------------------------------------------------------------
boundary_facets = dolfinx.mesh.exterior_facet_indices(structure.topology)
facet_centroids = dolfinx.mesh.compute_midpoints(structure, sfdim, boundary_facets)

# Radial distance from z-axis
radial_dist = np.sqrt(facet_centroids[:, 0]**2 + facet_centroids[:, 1]**2)
median_radius = np.median(radial_dist)
inner_mask = radial_dist < median_radius
outer_mask = radial_dist >= median_radius

# Z-extrema for inlet/outlet
z_coords = facet_centroids[:, 2]
z_min, z_max = z_coords.min(), z_coords.max()
inlet_mask = np.isclose(z_coords, z_min, atol=1e-2)
outlet_mask = np.isclose(z_coords, z_max, atol=1e-2)

# Remove end caps from inner/outer classification
non_end_mask = ~(inlet_mask | outlet_mask)
inner_wall_mask = inner_mask & non_end_mask
outer_wall_mask = outer_mask & non_end_mask

marker_inner = 1
marker_outer = 2
marker_inlet = 3
marker_outlet = 4

# Build facet tags
marked_facets = []
marked_values = []

if inlet_mask.any():
    marked_facets.append(boundary_facets[inlet_mask])
    marked_values.append(np.full(inlet_mask.sum(), marker_inlet))
if outlet_mask.any():
    marked_facets.append(boundary_facets[outlet_mask])
    marked_values.append(np.full(outlet_mask.sum(), marker_outlet))
if inner_wall_mask.any():
    marked_facets.append(boundary_facets[inner_wall_mask])
    marked_values.append(np.full(inner_wall_mask.sum(), marker_inner))
if outer_wall_mask.any():
    marked_facets.append(boundary_facets[outer_wall_mask])
    marked_values.append(np.full(outer_wall_mask.sum(), marker_outer))

marked_facets = np.hstack(marked_facets).astype(np.int32)
marked_values = np.hstack(marked_values).astype(np.int32)
sorted_idx = np.argsort(marked_facets)
facet_tag_s = dolfinx.mesh.meshtags(structure, sfdim, marked_facets[sorted_idx], marked_values[sorted_idx])

# Translate solid mesh to center it within the fluid domain [0,Lx]x[0,Ly]x[0,Lz]
# Mesh original range: x~[-1.55,2.07], y~[-1.93,1.93], z~[-3.45,13.74]
# Shift to center in fluid domain
structure._geometry._cpp_object.x[:, 0] += 3.5   # x: [1.95, 5.57] in [0, 8]
structure._geometry._cpp_object.x[:, 1] += 4.0   # y: [2.07, 5.93] in [0, 8]
structure._geometry._cpp_object.x[:, 2] += 5.0   # z: [1.55, 18.74] in [0, 20]

# Marker dictionary for later use (matching demo_337 pattern)
mesh_markers = {'INNER': [marker_inner], 'OUTER': [marker_outer],
                'INLET': [marker_inlet], 'OUTLET': [marker_outlet]}

if MPI.COMM_WORLD.rank == 0:
    print(f"Facet classification: inner={inner_wall_mask.sum()}, outer={outer_wall_mask.sum()}, "
          f"inlet={inlet_mask.sum()}, outlet={outlet_mask.sum()}")

# ------------------------------------------------------------------------------
# Solid function spaces
# ------------------------------------------------------------------------------
metadata = {"quadrature_degree": 5}
ds_s = ufl.Measure("ds", domain=structure, subdomain_data=facet_tag_s, metadata=metadata)

v_cg2_s = element("Lagrange", structure.topology.cell_name(),
                  config["force_order"], shape=(structure.geometry.dim, ))
v_cg1_s = element("Lagrange", structure.topology.cell_name(),
                  1, shape=(structure.geometry.dim, ))
Vs = functionspace(structure, v_cg2_s)
Vs_io = functionspace(structure, v_cg1_s)

solid_coords = Function(Vs, name="solid_coords")
solid_coords_io = Function(Vs_io, name="solid_coords_io")
solid_force = Function(Vs, name="solid_force")
solid_force_io = Function(Vs_io, name="solid_force_io")
solid_velocity = Function(Vs, name="solid_velocity")

# 定义弱形式
dVs = TestFunction(Vs)
endo_pressure = fem.Constant(structure, default_scalar_type(0.0))
N_s = ufl.FacetNormal(structure)

FF = grad(solid_coords)

# Constrain both end caps (inlet + outlet) to be fixed in z via penalty
X0 = ufl.SpatialCoordinate(structure)
x_constraint = solid_coords[0] - X0[0]
y_constraint = solid_coords[1] - X0[1]
z_constraint = solid_coords[2] - X0[2]
displacement_constraint = ufl.as_vector((x_constraint, y_constraint, z_constraint))

# First Piola-Kirchhoff stress
PK1 = Material.first_piola_kirchhoff_stress_v1(structure, solid_coords)

L_hat = -inner(PK1, grad(dVs)) * dx
# Penalty constraints on both end caps
L_hat -= config["beta"] * ufl.inner(displacement_constraint, dVs) * ds_s(mesh_markers['INLET'][0])
L_hat -= config["beta"] * ufl.inner(displacement_constraint, dVs) * ds_s(mesh_markers['OUTLET'][0])
# Luminal pressure on inner wall
L_hat -= ufl.inner(dVs, endo_pressure * ufl.cofac(FF) * N_s) * ds_s(mesh_markers['INNER'][0])
L_hat = form(L_hat)
b1 = create_vector(L_hat)

###########################################################################################################
##########################################  Interaction  ##################################################
###########################################################################################################
from afsic import IBMesh3D, IBInterpolation3D
ibmesh = IBMesh3D(0.0, config["Lx"],
                  0.0, config["Ly"],
                  0.0, config["Lz"],
                  config["Nx"], config["Ny"], config["Nz"],
                  config["velocity_order"])
ib_interpolation = IBInterpolation3D(ibmesh)
coords_bg = Function(V)
coords_bg.interpolate(lambda x: np.array([x[0], x[1], x[2]]))
solid_coords.interpolate(lambda x: np.array([x[0], x[1], x[2]]))
ibmesh.build_map(coords_bg._cpp_object)
ib_interpolation.evaluate_current_points(solid_coords._cpp_object)


###########################################################################################################
##########################################  Output  #######################################################
###########################################################################################################

u_io = Function(V_io)
file_velocity = dolfinx.io.XDMFFile(mesh.comm, config["output_path"] + "velocity.xdmf", "w")
file_solid = dolfinx.io.XDMFFile(mesh.comm, config["output_path"] + "solid.xdmf", "w")
file_velocity.write_mesh(mesh)
file_solid.write_mesh(structure)

time_manager = TimeManager(config['T'], config['num_steps'], fps=20)

form_u_L2 = form(dot(ns_solver.u_, ns_solver.u_) * dx)
form_p_L2 = form(dot(ns_solver.p_, ns_solver.p_) * dx)
form_F_L2 = form(dot(solid_coords, solid_coords) * dx)
form_volume = form(det(grad(solid_coords)) * dx)

log.set_log_level(log.LogLevel.INFO)
for step in range(config['num_steps']):
    current_time = step * config['dt']
    endo_pressure.value = calculate_pressure_linear(current_time,
                                                     diastole_pressure=config["diastole_pressure"],
                                                     systole_pressure=config["systole_pressure"])
    ns_solver.solve_one_step()
    ib_interpolation.fluid_to_solid(ns_solver.u_._cpp_object, solid_velocity._cpp_object)
    solid_coords.x.array[:] += solid_velocity.x.array[:] * config['dt']
    solid_coords.x.scatter_forward()
    u_L2 = mesh.comm.allreduce(assemble_scalar(form_u_L2), op=MPI.SUM)
    p_L2 = mesh.comm.allreduce(assemble_scalar(form_p_L2), op=MPI.SUM)
    F_L2 = mesh.comm.allreduce(assemble_scalar(form_F_L2), op=MPI.SUM)
    volume = mesh.comm.allreduce(assemble_scalar(form_volume), op=MPI.SUM)

    ib_interpolation.evaluate_current_points(solid_coords._cpp_object)
    with b1.localForm() as loc_2:
        loc_2.set(0)
    assemble_vector(b1, L_hat)
    b1.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    with b1.getBuffer() as arr:
        solid_force.x.array[:len(arr)] = arr[:]
    ib_interpolation.solid_to_fluid(ns_solver.f._cpp_object, solid_force._cpp_object)
    ns_solver.f.x.scatter_forward()

    data_log = {}
    if time_manager.should_output(step):
        u_io.interpolate(ns_solver.u_)
        file_velocity.write_function(u_io, current_time)
        solid_force_io.interpolate(solid_force)
        solid_coords_io.interpolate(solid_coords)
        file_solid.write_function(solid_force_io, current_time)
        file_solid.write_function(solid_coords_io, current_time)
        if MPI.COMM_WORLD.rank == 0:
            data_log["u_norm"] = u_L2
            data_log["p_norm"] = p_L2
            data_log["solid_force_norm"] = F_L2
            data_log["volume"] = volume
            data_log["endo_pressure"] = endo_pressure.value
            print(f"Step {step+1}/{config['num_steps']}, Time: {current_time:.2f}s")
            swanlab_upload(current_time, data_log)

file_velocity.close()
file_solid.close()
