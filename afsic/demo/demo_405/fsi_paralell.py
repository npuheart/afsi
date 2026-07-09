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

from petsc4py import PETSc
from mpi4py import MPI
import ufl

import os
import numpy as np

import dolfinx
from dolfinx import log
from dolfinx import fem, default_scalar_type
from dolfinx.fem import (Function, functionspace,
                         dirichletbc, locate_dofs_topological)
from dolfinx.mesh import CellType, GhostMode, locate_entities, meshtags
from basix.ufl import element

from ufl import (FacetNormal, SpatialCoordinate, TestFunction, TrialFunction, det,
                 as_vector, div, dot, ds, dx, inner, lhs, grad, nabla_grad, rhs, sym, system)
from dolfinx.fem import form, assemble_scalar

from afsic import IPCSSolver, ChorinSolver, TimeManager
from afsic import swanlab_init, swanlab_upload
from dolfinx.fem.petsc import create_vector, assemble_vector
from configuration import config
from NeoHookean import NeoHookeanMaterial

Material = NeoHookeanMaterial(E=2e6, nu=0.4)

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

# Mark the fluid boundaries
mesh.topology.create_connectivity(2, 3)
marker_left, marker_right, marker_down, marker_up, marker_front, marker_back = 1, 2, 3, 4, 5, 6
boundaries = [(1, lambda x: np.isclose(x[0], 0.0)),
              (2, lambda x: np.isclose(x[0], config["Lx"])),
              (3, lambda x: np.isclose(x[1], 0.0)),
              (4, lambda x: np.isclose(x[1], config["Ly"])),
              (5, lambda x: np.isclose(x[2], 0.0)),
              (6, lambda x: np.isclose(x[2], config["Lz"])),
              ]

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

# Inlet: sinusoidal velocity in z-direction, only inside pipe lumen
class InletVelocity():
    def __init__(self, t, U_max, freq, R_inner, cx, cy):
        self.t = t
        self.U_max = U_max
        self.freq = freq
        self.R_inner = R_inner  # inner radius of pipe
        self.cx = cx            # pipe center x
        self.cy = cy            # pipe center y
    def __call__(self, x):
        values = np.zeros((gdim, x.shape[1]), dtype=PETSc.ScalarType)
        r = np.sqrt((x[0] - self.cx)**2 + (x[1] - self.cy)**2)
        mask = r < self.R_inner
        self.scale = self.U_max * np.sin(2.0 * np.pi * self.freq * self.t)
        values[2][mask] = self.scale
        return values

inlet_velocity = InletVelocity(0.0,
    U_max=config.get("U_max", 1.0),
    freq=config.get("freq", 1.0),
    R_inner=config.get("R_inner", 1.3),
    cx=config.get("cx", 4.0),
    cy=config.get("cy", 4.0))
u_inlet_func = Function(V)
u_inlet_func.interpolate(inlet_velocity)
bcu_inlet = dirichletbc(u_inlet_func, locate_dofs_topological(V, fdim, facet_tag.find(marker_front)))

# Walls: no-slip on sides

u_nonslip = np.array((0,) * mesh.geometry.dim, dtype=PETSc.ScalarType)
bcu_left = dirichletbc(u_nonslip, locate_dofs_topological(
    V, fdim, facet_tag.find(marker_left)), V)
bcu_right = dirichletbc(u_nonslip, locate_dofs_topological(
    V, fdim, facet_tag.find(marker_right)), V)
bcu_down = dirichletbc(u_nonslip, locate_dofs_topological(
    V, fdim, facet_tag.find(marker_down)), V)
bcu_up_wall = dirichletbc(u_nonslip, locate_dofs_topological(
    V, fdim, facet_tag.find(marker_up)), V)

# Outlet (marker_back, z=Lz): zero pressure (natural BC, p=0)
outlet_dofs = locate_dofs_topological(Q, fdim, facet_tag.find(marker_back))
bcp_outlet = dirichletbc(PETSc.ScalarType(0.0), outlet_dofs, Q)

bcu = [bcu_inlet, bcu_left, bcu_right, bcu_down, bcu_up_wall]
bcp = [bcp_outlet]
# Define Solver
ns_solver = ChorinSolver(V, Q, bcu, bcp, config['dt'], config['rho'], config['mu'])

###########################################################################################################
##########################################  Structure  ####################################################
###########################################################################################################
# import os

# # ------------------------------------------------------------------------------
# # Convert prism mesh to tetrahedra (FFCX doesn't support prism elements)
# # ------------------------------------------------------------------------------
# _tet_mesh_path = "vessel_wall_wall_3d_tet.xdmf"
# if MPI.COMM_WORLD.rank == 0 and not os.path.exists(_tet_mesh_path):
#     from basix.ufl import element as basix_element
#     with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "vessel_wall_wall_3d.xdmf", "r",
#                              encoding=dolfinx.io.XDMFFile.Encoding.HDF5) as _f:
#         _prism_mesh = _f.read_mesh(name="Grid")
#     _prism_mesh.topology.create_connectivity(_prism_mesh.topology.dim, 0)
#     _c2v = _prism_mesh.topology.connectivity(_prism_mesh.topology.dim, 0)
#     _nc = _prism_mesh.topology.index_map(_prism_mesh.topology.dim).size_local
#     _tet_conn = np.zeros((_nc * 3, 4), dtype=np.int64)
#     for _c in range(_nc):
#         _v = _c2v.links(_c)
#         _tet_conn[3*_c]   = [_v[0], _v[1], _v[2], _v[5]]
#         _tet_conn[3*_c+1] = [_v[0], _v[1], _v[5], _v[4]]
#         _tet_conn[3*_c+2] = [_v[0], _v[4], _v[5], _v[3]]
#     _e = basix_element('Lagrange', 'tetrahedron', 1, shape=(3,))
#     _tet_mesh = dolfinx.mesh.create_mesh(MPI.COMM_WORLD, _tet_conn, _e, _prism_mesh.geometry.x.copy())
#     with dolfinx.io.XDMFFile(MPI.COMM_WORLD, _tet_mesh_path, "w") as _f:
#         _f.write_mesh(_tet_mesh)
#     print("Converted prism mesh to tetrahedra.")

# MPI.COMM_WORLD.barrier()

# Read the merged vessel + leaflets mesh with cell tags
# tag=1: vessel wall (fixed via penalty), tag=2: leaflets (deformable, PK1 only)
with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "combined_vessel_leaflets.xdmf", "r",
                         encoding=dolfinx.io.XDMFFile.Encoding.HDF5) as file:
    structure = file.read_mesh(name="mesh")
    cell_tags_dx = file.read_meshtags(structure, name="mesh_tags")

# Build topology connectivity
structure.topology.create_connectivity(structure.topology.dim - 1, structure.topology.dim)
sfdim = structure.topology.dim - 1

# Translate solid mesh to center it within the fluid domain [0,Lx]x[0,Ly]x[0,Lz]
structure._geometry._cpp_object.x[:, 0] += 3.5
structure._geometry._cpp_object.x[:, 1] += 4.0
structure._geometry._cpp_object.x[:, 2] += 5.0

# ------------------------------------------------------------------------------
# Solid function spaces (demo_402 pattern: PK1 stress + penalty fixation)
# ------------------------------------------------------------------------------
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

# Weak form: PK1 on leaflets (tag=2), penalty on vessel wall (tag=1)
dVs = TestFunction(Vs)
PK1 = Material.first_piola_kirchhoff_stress_v1(structure, solid_coords)

# Subdomain measure from cell tags
dxx = ufl.Measure("dx", domain=structure, subdomain_data=cell_tags_dx)

# Penalty constraint: fix vessel wall (tag=1) to original position
X0 = SpatialCoordinate(structure)
displacement_constraint = ufl.as_vector((
    solid_coords[0] - X0[0],
    solid_coords[1] - X0[1],
    solid_coords[2] - X0[2],
))
beta = config["beta"]

L_hat = form(
    -inner(PK1, grad(dVs)) * dxx(2)                               # leaflets: free to deform
    - beta * inner(displacement_constraint, dVs) * dxx(1)         # vessel: penalty-fixed
)
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
p_io = Function(Q)
file_velocity = dolfinx.io.XDMFFile(mesh.comm, config["output_path"] + "velocity.xdmf", "w")
file_pressure = dolfinx.io.XDMFFile(mesh.comm, config["output_path"] + "pressure.xdmf", "w")
file_solid = dolfinx.io.XDMFFile(mesh.comm, config["output_path"] + "solid.xdmf", "w")
file_velocity.write_mesh(mesh)
file_pressure.write_mesh(mesh)
file_solid.write_mesh(structure)

time_manager = TimeManager(config['T'], config['num_steps'], fps=100)

form_u_L2 = form(dot(ns_solver.u_, ns_solver.u_) * dx)
form_p_L2 = form(dot(ns_solver.p_, ns_solver.p_) * dx)
form_F_L2 = form(dot(solid_coords, solid_coords) * dx)
form_volume = form(det(grad(solid_coords)) * dx)

log.set_log_level(log.LogLevel.INFO)
for step in range(config['num_steps']):
    current_time = step * config['dt']

    # Update inlet velocity (sinusoidal)
    inlet_velocity.t = current_time
    u_inlet_func.interpolate(inlet_velocity)

    # Solve fluid
    ns_solver.solve_one_step()

    # Solid: interpolate fluid velocity → update position → compute forces
    ib_interpolation.fluid_to_solid(ns_solver.u_._cpp_object, solid_velocity._cpp_object)
    solid_coords.x.array[:] += solid_velocity.x.array[:] * config['dt']
    solid_coords.x.scatter_forward()

    u_L2 = mesh.comm.allreduce(assemble_scalar(form_u_L2), op=MPI.SUM)
    p_L2 = mesh.comm.allreduce(assemble_scalar(form_p_L2), op=MPI.SUM)
    F_L2 = mesh.comm.allreduce(assemble_scalar(form_F_L2), op=MPI.SUM)
    volume = mesh.comm.allreduce(assemble_scalar(form_volume), op=MPI.SUM)

    # Assemble solid force from weak form (PK1 + penalty)
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
            print(f"Step {step+1}/{config['num_steps']}, Time: {current_time:.3f}s, "
                  f"u_norm={u_L2:.4e}, F_norm={F_L2:.4e}")
            swanlab_upload(current_time, data_log)

file_velocity.close()
file_solid.close()
