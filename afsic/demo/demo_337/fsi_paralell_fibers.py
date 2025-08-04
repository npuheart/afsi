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
from HolzapfelOgden import HolzapfelOgdenMaterial
Material = NeoHookeanMaterial()

from PressureEndo import  calculate_pressure, mmHg, calculate_tension

from ReadFibers import fun_fiber_v1, CoordinateDataMap



# Define the configuration for the simulation
config = {"nssolver": "chorinsolver",
          "project_name": "demo-337",
          "tag": "parallel",
          "velocity_order": 2,
          "force_order": 2,
          "pressure_order": 1,
          "num_processors": MPI.COMM_WORLD.size,
          "T": 0.3,
          "dt": 1/10000,
          "rho": 1.0,
          "Lx": 5.0,
          "Ly": 5.0,
          "Lz": 5.0,
          "Nx": 32,
          "Ny": 32,
          "Nz": 32,
          "Nl": 20,
          "mu": 0.01,
          "mu_s": 0.1,  # Solid elasticity
          "diastole_pressure": 0.1, # 8.0*mmHg/100000,
          "systole_pressure": 110.0*mmHg,
          "max_tension": 600.0*mmHg,
          "beta": 5e6,
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
# Create mesh
mesh = dolfinx.mesh.create_box(
    comm=MPI.COMM_WORLD,
    points=((0.0, 0.0, 0.0), (config["Lx"], config["Ly"], config["Lz"])),
    n=(config["Nx"], config["Ny"], config["Nz"]),
    cell_type=CellType.hexahedron,
    ghost_mode=GhostMode.shared_facet,
)

# Mark the boundaries
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
tdim = mesh.topology.dim
class UpVelocity():
    def __init__(self, t):
        self.t = t
    def __call__(self, x):
        values = np.zeros((gdim, x.shape[1]), dtype=PETSc.ScalarType)
        values[0] = 0.0
        values[1] = 0.0
        values[2] = 0.0
        return values


# Inlet
u_up = Function(V)
up_velocity = UpVelocity(0.0)
u_up.interpolate(up_velocity)
bcu_up = dirichletbc(u_up, locate_dofs_topological(
    V, fdim, facet_tag.find(marker_up)))
# Walls
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
points_dofs = locate_dofs_topological(
    Q, 0, point_loc)
bcp_point = dirichletbc(0.0, points_dofs, Q)
bcu = [bcu_up, bcu_left, bcu_right, bcu_down, bcu_front, bcu_back]
bcp = [bcp_point]


# Define Solver
ns_solver = ChorinSolver(V, Q, bcu, bcp, config['dt'], config['rho'], config['mu'])

###########################################################################################################
##########################################  Structure  ####################################################
###########################################################################################################
import json
import ufl
with open('/home/dolfinx/afsi/afsic/demo/demo_337/mesh/lv_ellipsoid/geometry/markers.json', 'r', encoding='utf-8') as file:
    mesh_markers = json.load(file)

with dolfinx.io.XDMFFile(MPI.COMM_WORLD, '/home/dolfinx/afsi/afsic/demo/demo_337/mesh/lv_ellipsoid/geometry/mesh.xdmf', "r", encoding=dolfinx.io.XDMFFile.Encoding.HDF5) as file:
    structure = file.read_mesh(name="Mesh")
    structure.topology.create_connectivity(structure.topology.dim-1, structure.topology.dim)
    ft = file.read_meshtags(structure, "Facet tags")
    endo_facets = ft.find(mesh_markers['ENDO'][0])
    base_facets = ft.find(mesh_markers['BASE'][0])
    epi_facets = ft.find(mesh_markers['EPI'][0])
    marked_facets = np.hstack([endo_facets, base_facets, epi_facets])
    marked_values = np.hstack([np.full_like(endo_facets, mesh_markers['ENDO'][0]), np.full_like(base_facets, mesh_markers['BASE'][0]), np.full_like(epi_facets, mesh_markers['EPI'][0])])
    sorted_facets = np.argsort(marked_facets)
    facet_tag = dolfinx.mesh.meshtags(structure, ft.dim, marked_facets[sorted_facets], marked_values[sorted_facets])
    facet_tag.name = ft.name

metadata = {"quadrature_degree": 5}
ds = ufl.Measure("ds", domain=structure, subdomain_data=facet_tag, metadata=metadata)

v_cg2 = element("Lagrange", structure.topology.cell_name(),
                config["force_order"], shape=(structure.geometry.dim, ))
v_cg1 = element("Lagrange", structure.topology.cell_name(),
                1, shape=(structure.geometry.dim, ))
Vs = functionspace(structure, v_cg2)
Vs_io = functionspace(structure, v_cg1)

solid_coords = Function(Vs, name="solid_coords")
solid_coords_io = Function(Vs_io, name="solid_coords_io")
solid_force = Function(Vs, name="solid_force")
solid_force_io = Function(Vs_io, name="solid_force_io")
solid_velocity = Function(Vs, name="solid_velocity")
solid_fiber = Function(Vs, name="fiber")
solid_sheet = Function(Vs, name="sheet")
solid_fiber_io = Function(Vs_io, name="fiber_io")
solid_sheet_io = Function(Vs_io, name="sheet_io")
solid_coords.interpolate(lambda x: np.array([x[0], x[1], x[2]]))

# 这里假设所有进程都能访问到同样的纤维数据，每个进程都要读取完整的纤维数据，然后各个进程根据DoFs分配来提取对应的纤维数据
dict_fibers = fun_fiber_v1(
    '/home/dolfinx/afsi/afsic/demo/demo_337/mesh/f0.txt', 
    '/home/dolfinx/afsi/afsic/demo/demo_337/mesh/s0.txt', 
    '/home/dolfinx/afsi/afsic/demo/demo_337/mesh/cdm.txt')

cdm = CoordinateDataMap()
solid_coords_np = solid_coords.x.array.reshape(-1, 3)
for i, coord in enumerate(solid_coords_np):
    coord_key = cdm.hash_floats(coord)
    if np.int64(coord_key) not in dict_fibers:
        raise KeyError(f"coord_key {coord_key} for coordinate {coord} not found in dict_fibers.")
    solid_fiber.x.array[3*i:3*i+3] = dict_fibers[coord_key][:3]
    solid_sheet.x.array[3*i:3*i+3] = dict_fibers[coord_key][3:6]

structure._geometry._cpp_object.x[:,0] = structure._geometry._cpp_object.x[:,0]/10.0 + 3.0
structure._geometry._cpp_object.x[:,1] = structure._geometry._cpp_object.x[:,1]/10.0 + 2.5
structure._geometry._cpp_object.x[:,2] = structure._geometry._cpp_object.x[:,2]/10.0 + 2.5

# 定义弱形式
dVs = TestFunction(Vs)
mu_s = config["mu_s"]
lambda_s = 10
endo_pressure = fem.Constant(structure, default_scalar_type(0.0))
active_tension = fem.Constant(structure, default_scalar_type(0.0))
Material = HolzapfelOgdenMaterial(f0=solid_fiber, s0=solid_sheet, tension=active_tension)
N = ufl.FacetNormal(structure)

FF = grad(solid_coords)

X0 = ufl.SpatialCoordinate(structure)
x_constraint = solid_coords[0] - X0[0]
y_constraint = solid_coords[1] - X0[1]
z_constraint = solid_coords[2] - X0[2]
circum_constraint = ufl.as_vector((x_constraint, y_constraint, z_constraint))

# First Piola-Kirchhoff stress
PK1 = Material.first_piola_kirchhoff_stress_v1(structure, solid_coords)
# L_hat = -inner(mu_s*(FF-inv(FF).T), grad(dVs))*dx
L_hat = -inner(PK1, grad(dVs))*dx
L_hat -= config["beta"]*ufl.inner(circum_constraint, dVs)*ds(mesh_markers['BASE'][0])
L_hat -= ufl.inner(dVs, endo_pressure * ufl.cofac(FF)* N) * ds(mesh_markers['ENDO'][0])
L_hat = form(L_hat)
b1 = create_vector(L_hat)

###########################################################################################################
##########################################  Interaction  ##################################################
###########################################################################################################
from afsic import IBMesh3D, IBInterpolation3D
ibmesh = IBMesh3D(0.0, config["Lx"], 0.0, config["Ly"], 0.0, config["Lz"], config["Nx"], config["Ny"], config["Nz"], config["velocity_order"])
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
file_velocity = dolfinx.io.XDMFFile(mesh.comm, config["output_path"]+"velocity.xdmf", "w")
file_solid = dolfinx.io.XDMFFile(mesh.comm, config["output_path"]+"solid_force.xdmf", "w")
file_velocity.write_mesh(mesh)
file_solid.write_mesh(structure)

time_manager = TimeManager(config['T'], config['num_steps'], fps=2000)

form_u_L2 = form(dot(ns_solver.u_, ns_solver.u_) * dx)
form_p_L2 = form(dot(ns_solver.p_, ns_solver.p_) * dx)
form_X_L2 = form(dot(solid_coords, solid_coords) * dx)
form_F_L2 = form(dot(solid_force, solid_force) * dx)
form_volume = form(det(grad(solid_coords)) * dx)

log.set_log_level(log.LogLevel.INFO)
for step in range(config['num_steps']):
    current_time = step * config['dt']
    up_velocity.t = current_time
    endo_pressure.value = calculate_pressure(current_time, diastole_pressure=config["diastole_pressure"], systole_pressure=config["systole_pressure"])
    active_tension.value = calculate_tension(current_time, max_tension=config["max_tension"])
    u_up.interpolate(up_velocity)
    ns_solver.solve_one_step()
    ib_interpolation.fluid_to_solid(ns_solver.u_._cpp_object, solid_velocity._cpp_object)
    solid_coords.x.array[:] += solid_velocity.x.array[:]*config['dt']
    solid_coords.x.scatter_forward()
    u_L2 = mesh.comm.allreduce(assemble_scalar(form_u_L2), op=MPI.SUM)
    p_L2 = mesh.comm.allreduce(assemble_scalar(form_p_L2), op=MPI.SUM)
    F_L2 = mesh.comm.allreduce(assemble_scalar(form_F_L2), op=MPI.SUM)
    X_L2 = mesh.comm.allreduce(assemble_scalar(form_X_L2), op=MPI.SUM)
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
    if step is 1 or time_manager.should_output(step):
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
            data_log["solid_coord_norm"] = X_L2
            data_log["volume"] = volume
            data_log["endo_pressure"] = endo_pressure.value
            data_log["active_tension"] = active_tension.value
            print(f"Step {step+1}/{config['num_steps']}, Time: {current_time:.2f}s")
            swanlab_upload(current_time, data_log)







 