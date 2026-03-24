# https://jsdokken.com/dolfinx-tutorial/chapter2/ns_code1.html
from mpi4py import MPI
from petsc4py import PETSc
import numpy as np
from dolfinx.fem import (
    Constant,
    Function,
    functionspace,
    assemble_scalar,
    dirichletbc,
    form,
    locate_dofs_geometrical,
)
from dolfinx.mesh import create_unit_square
from basix.ufl import element
from ufl import (
    dot,
    dx
)
from afsic.euler import NSCode1Solver
mesh = create_unit_square(MPI.COMM_WORLD, 10, 10)
t = 0.0
T = 10.0
num_steps = 500
dt = T / num_steps
v_cg2 = element("Lagrange", mesh.basix_cell(), 2, shape=(mesh.geometry.dim,))
s_cg1 = element("Lagrange", mesh.basix_cell(), 1)
V = functionspace(mesh, v_cg2)
Q = functionspace(mesh, s_cg1)
def walls(x):
    return np.logical_or(np.isclose(x[1], 0), np.isclose(x[1], 1))

wall_dofs = locate_dofs_geometrical(V, walls)
u_noslip = np.array((0,) * mesh.geometry.dim, dtype=PETSc.ScalarType)
bc_noslip = dirichletbc(u_noslip, wall_dofs, V)
def inflow(x):
    return np.isclose(x[0], 0)

inflow_dofs = locate_dofs_geometrical(Q, inflow)
bc_inflow = dirichletbc(PETSc.ScalarType(8), inflow_dofs, Q)
def outflow(x):
    return np.isclose(x[0], 1)

outflow_dofs = locate_dofs_geometrical(Q, outflow)
bc_outflow = dirichletbc(PETSc.ScalarType(0), outflow_dofs, Q)
bcu = [bc_noslip]
bcp = [bc_inflow, bc_outflow]
f = Constant(mesh, PETSc.ScalarType((0, 0)))
ns_solver = NSCode1Solver(V, Q, bcu, bcp, f, dt_raw=dt, rho_raw=1, mu_raw=1)
def u_exact(x):
    values = np.zeros((2, x.shape[1]), dtype=PETSc.ScalarType)
    values[0] = 4 * x[1] * (1.0 - x[1])
    return values

u_ex = Function(V)
u_ex.interpolate(u_exact)
L2_error = form(dot(ns_solver.u_ - u_ex, ns_solver.u_ - u_ex) * dx)
for i in range(num_steps):
    # Update current time step
    t += dt
    ns_solver.solve_one_step()
    # Compute error at current time-step
    error_L2 = np.sqrt(mesh.comm.allreduce(assemble_scalar(L2_error), op=MPI.SUM))
    error_max = mesh.comm.allreduce(
        np.max(ns_solver.u_.x.petsc_vec.array - u_ex.x.petsc_vec.array), op=MPI.MAX
    )
    # Print error only every 20th step and at the last step
    if (i % 20 == 0) or (i == num_steps - 1):
        print(f"Time {t:.2f}, L2-error {error_L2:.6e}, Max error {error_max:.6e}")

ns_solver.post_process()

# Time 10.00, L2-error 5.201511e-06, Max error 8.923966e-06