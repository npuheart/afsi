from petsc4py import PETSc
from mpi4py import MPI

import os
import numpy as np

import dolfinx
from dolfinx import log
from dolfinx.fem import (Function, functionspace,
                         dirichletbc, locate_dofs_topological)
from dolfinx.mesh import CellType, GhostMode, locate_entities, meshtags
from basix.ufl import element

from ufl import (FacetNormal, Identity, Measure, SpatialCoordinate, TestFunction, TrialFunction, inv, ln, det,
                 as_vector, div, dot, ds, dx, inner, lhs, grad, nabla_grad, rhs, sym, system)
from dolfinx.fem import form, assemble_scalar

from afsic import IPCSSolver,ChorinSolver, TimeManager
from afsic import swanlab_init, swanlab_upload
from dolfinx.fem.petsc import create_vector, assemble_vector
from configuration import config


swanlab_init(config['project_name'], config['experiment_name'], config, api_key="odR9FodGeQojOPlk2sir1")


def pressure_waveform(t, period, amp, fast_ratio, waveform="sin"):
    """Return pressure at time t.

    waveform:
        'sin'       – symmetric sine
        'fast_open' – piecewise linear two-segment:
                      [0, fast_ratio*T]         : 0 -> amp  (fast rise)
                      [fast_ratio*T, T]         : amp -> 0  (slow fall)
    """
    phase = (t % period) / period   # in [0, 1)
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
boundaries = [(14, lambda x: np.isclose(x[0], 0)),             # inlet (left)
              (12, lambda x: np.isclose(x[0], config["Lx"])),  # outlet (right)
              (11, lambda x: np.isclose(x[1], 0)),             # bottom
              (13, lambda x: np.isclose(x[1], config["Ly"]))]  # top

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
# Turek FSI parabolic inlet: u_x(y) = 1.5*Um * y*(Ly-y) / (Ly/2)^2
class Inlet:
    def __init__(self, Um, Ly):
        self.t = 0.0
        self.t_ramp = 2.0
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


# Inlet velocity (left wall, tag 14)
inlet = Inlet(config["Um"], config["Ly"])
u_inlet_func = Function(V)
u_inlet_func.interpolate(inlet)
bcu_inlet = dirichletbc(u_inlet_func, locate_dofs_topological(V, fdim, facet_tag.find(marker_inlet)))

# No-slip on top/bottom (both components = 0), tags 11, 13
u_zero = Function(V)
u_zero.x.array[:] = 0.0
bcu_bottom = dirichletbc(u_zero, locate_dofs_topological(V, fdim, facet_tag.find(marker_bottom)))
bcu_top = dirichletbc(u_zero, locate_dofs_topological(V, fdim, facet_tag.find(marker_top)))

# Pressure outlet (right wall, tag 12)
bcp_outlet = dirichletbc(PETSc.ScalarType(0.0),
                         locate_dofs_topological(Q, fdim, facet_tag.find(marker_outlet)), Q)

bcu = [bcu_inlet, bcu_bottom, bcu_top]
bcp = [bcp_outlet]


# Define Solver
ns_solver = ChorinSolver(V, Q, bcu, bcp, config['dt'], config['rho'], config['mu'])

###########################################################################################################
##########################################  Structure  ####################################################
###########################################################################################################
# Turek FSI: circle (area tag 1, facet tag 3) + elastic tail (area tag 2)
turek_mesh_path = os.path.join(os.path.dirname(__file__), "./turek_mesh.xdmf")
with dolfinx.io.XDMFFile(MPI.COMM_WORLD, turek_mesh_path, "r") as xdmf:
    structure = xdmf.read_mesh(name="mesh")
    structure.topology.create_connectivity(structure.topology.dim, structure.topology.dim - 1)
    cell_tags = xdmf.read_meshtags(structure, name="cell_tags")
    facet_tags = xdmf.read_meshtags(structure, name="facet_tags")

# Scale from metres to centimetres (geo is in SI units)
structure.geometry.x[:, 0] *= 100.0
structure.geometry.x[:, 1] *= 100.0

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

dVs = TestFunction(Vs)
mu_s = config["mu_s"]
lambda_s = config["lambda_s"]
beta = config["beta"]

FF = grad(solid_coords)
J = det(FF)

# Penalty: fix circle boundary (facet tag 3), same as turtle head/tail fixation
X0 = SpatialCoordinate(structure)
dss = Measure("ds", domain=structure, subdomain_data=facet_tags)
dxx = Measure("dx", domain=structure, subdomain_data=cell_tags)
x_constraint = solid_coords[0] - X0[0]
y_constraint = solid_coords[1] - X0[1]
circum_constraint = as_vector((x_constraint, y_constraint))

# Neo-Hookean: P = mu_s*(F - F^-T) + lambda_s*ln(J)*F^-T
I1 = inner(FF, FF)
P_iso = mu_s * J**(-2.0/2.0) * (FF - (I1/2.0) * inv(FF).T)
P_vol = lambda_s * ln(J) * inv(FF).T
P_s = P_iso + P_vol

# Circle (facet tag 3, cell tag 1) is fixed via penalty; tail (cell tag 2) deforms freely
L_hat = form(
    -inner(P_s, grad(dVs))*dxx(1)
    -inner(P_s, grad(dVs))*dxx(2)
    - beta*inner(circum_constraint, dVs)*dxx(1)
    #  - beta*inner(circum_constraint, dVs)*dss(3)
)
b1 = create_vector(L_hat)

###########################################################################################################
##########################################  Interaction  ##################################################
###########################################################################################################
from afsic import IBMesh, IBInterpolation
ibmesh = IBMesh(0.0, config["Lx"],0.0, config["Ly"], config["Nx"], config["Ny"], config["velocity_order"])
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
p_io = Function(Q)
file_velocity = dolfinx.io.XDMFFile(mesh.comm, config["output_path"]+"velocity.xdmf", "w")
file_pressure = dolfinx.io.XDMFFile(mesh.comm, config["output_path"]+"pressure.xdmf", "w")
file_solid = dolfinx.io.XDMFFile(mesh.comm, config["output_path"]+"solid_force.xdmf", "w")
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
    inlet.update(current_time)
    u_inlet_func.interpolate(inlet)
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
            data_log["inlet_velocity"] = inlet.scale
            print(f"Step {step+1}/{config['num_steps']}, Time: {current_time:.2f}s")
            swanlab_upload(current_time, data_log)
