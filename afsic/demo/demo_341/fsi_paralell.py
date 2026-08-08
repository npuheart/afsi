"""demo_341 三维方腔驱动圆球 —— FSI（浸没边界法）求解。

流体用 ChorinSolver（方腔 [0,1]^3，入口 u_x=1 驱动），固体为 Gmsh 圆球
（Neo-Hookean 型 P=mu_s*(F-F^-T)），通过 IBMesh3D/IBInterpolation3D 耦合。
输出速度场/固体力到 plot/fsi_N<grid>/（本地）。

环境变量：
    GRID=32   # 背景网格密度 Nx=Ny=Nz（默认 32）
    STEPS=200 # 运行步数覆盖（t=1s 时 STEPS=200）
    CASE=fsi  # 输出子目录名

运行：
    conda activate afsi-dolfinx
    python generate_mesh.py            # 先生成固体网格 plot/mesh-341.xdmf
    STEPS=200 python fsi_paralell.py
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
                 as_vector, div, dot, ds, dx, inner, lhs, grad, nabla_grad, rhs, sym, system)
from dolfinx.fem import form, assemble_scalar
from dolfinx.fem.petsc import create_vector, assemble_vector

from afsic import IBMesh3D, IBInterpolation3D, ChorinSolver, TimeManager
from afsic import swanlab_init, swanlab_upload

_demo_dir = os.path.dirname(os.path.abspath(__file__))
_GRID = int(os.environ.get("GRID", "32"))
_CASE = os.environ.get("CASE", "fsi")

# Define the configuration for the simulation
config = {"nssolver": "chorinsolver",
          "project_name": "demo-341",
          "tag": "parallel",
          "velocity_order": 2,
          "force_order": 2,
          "pressure_order": 1,
          "num_processors": MPI.COMM_WORLD.size,
          "T": 10.0,
          "dt": 1/200,
          "rho": 1.0,
          "Lx": 1.0,
          "Ly": 1.0,
          "Lz": 1.0,
          "Nx": _GRID,
          "Ny": _GRID,
          "Nz": _GRID,
          "Nl": 20,
          "mu": 0.01,
          "mu_s": 0.1,  # Solid elasticity
          }

config["num_steps"] = int(config['T']/config['dt'])
# 环境变量 STEPS 覆盖步数（如 t=1s: STEPS=200）
if os.environ.get("STEPS"):
    config["num_steps"] = int(os.environ["STEPS"])
    config["T"] = config["num_steps"] * config["dt"]
# 输出到本地 plot/<case>_N<grid>/
config["output_path"] = os.path.join(_demo_dir, "plot", f"{_CASE}_N{_GRID}") + os.sep
os.makedirs(config["output_path"], exist_ok=True)
config["experiment_name"] = f"demo-341-{_CASE}-N{_GRID}"
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


class UpVelocity:
    def __init__(self, t):
        self.t = t

    def __call__(self, x):
        values = np.zeros((gdim, x.shape[1]), dtype=PETSc.ScalarType)
        values[0] = 1.0
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
mesh_path = os.path.join(_demo_dir, "plot", "mesh-341.xdmf")
with dolfinx.io.XDMFFile(MPI.COMM_WORLD, mesh_path, "r",
                         encoding=dolfinx.io.XDMFFile.Encoding.HDF5) as file:
    structure = file.read_mesh()

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

# 定义弱形式
dVs = TestFunction(Vs)
mu_s = config["mu_s"]
lambda_s = 10

FF = grad(solid_coords)

# L_hat = form(-inner(mu_s*(FF-inv(FF).T) + lambda_s*ln(det(FF))*inv(FF).T, grad(dVs))*dx)
L_hat = form(-inner(mu_s*(FF-inv(FF).T), grad(dVs))*dx)
b1 = create_vector(Vs)  # dolfinx 0.10.0: create_vector 需函数空间而非 Form

###########################################################################################################
##########################################  Interaction  ##################################################
###########################################################################################################
ibmesh = IBMesh3D(0.0, config["Lx"], 0.0, config["Ly"], 0.0, config["Lz"],
                  config["Nx"], config["Ny"], config["Nz"], config["velocity_order"])
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
file_solid = dolfinx.io.XDMFFile(mesh.comm, config["output_path"] + "solid_force.xdmf", "w")
file_velocity.write_mesh(mesh)
file_solid.write_mesh(structure)

time_manager = TimeManager(config['T'], config['num_steps'], fps=20)

form_u_L2 = form(dot(ns_solver.u_, ns_solver.u_) * dx)
form_p_L2 = form(dot(ns_solver.p_, ns_solver.p_) * dx)
form_F_L2 = form(dot(solid_coords, solid_coords) * dx)
form_volume = form(det(grad(solid_coords)) * dx)

log.set_log_level(log.LogLevel.INFO)
if MPI.COMM_WORLD.rank == 0:
    print(f"FSI 方腔驱动圆球: {config['Nx']}^3, dt={config['dt']}, steps={config['num_steps']}, "
          f"out={config['output_path']}")

for step in range(config['num_steps']):
    current_time = step * config['dt']
    up_velocity.t = current_time
    u_up.interpolate(up_velocity)
    ns_solver.solve_one_step()
    ib_interpolation.fluid_to_solid(ns_solver.u_._cpp_object, solid_velocity._cpp_object)
    solid_coords.x.array[:] += solid_velocity.x.array[:]*config['dt']
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

    if time_manager.should_output(step):
        u_io.interpolate(ns_solver.u_)
        file_velocity.write_function(u_io, current_time)
        solid_force_io.interpolate(solid_force)
        solid_coords_io.interpolate(solid_coords)
        file_solid.write_function(solid_force_io, current_time)
        file_solid.write_function(solid_coords_io, current_time)
        if MPI.COMM_WORLD.rank == 0:
            print(f"Step {step+1}/{config['num_steps']}, Time: {current_time:.2f}s, "
                  f"u_L2={u_L2:.4f} p_L2={p_L2:.4f}", flush=True)
            swanlab_upload(current_time, {"u_norm": u_L2, "p_norm": p_L2,
                                          "solid_force_norm": F_L2, "volume": volume})

file_velocity.close()
file_solid.close()
if MPI.COMM_WORLD.rank == 0:
    print(f"\nDone. Output: {config['output_path']}")
