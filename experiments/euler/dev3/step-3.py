
from ChorinSolver import *

rho_raw = 1.0
mu_raw = 1.0
mesh = create_unit_square(MPI.COMM_WORLD, 10, 10)
t = 0
T = 10
num_steps = 500
dt = T / num_steps

v_cg2 = element("Lagrange", mesh.topology.cell_name(),
                2, shape=(mesh.geometry.dim, ))
s_cg1 = element("Lagrange", mesh.topology.cell_name(), 1)
V = functionspace(mesh, v_cg2)
Q = functionspace(mesh, s_cg1)

# 边界条件


def walls(x):
    return np.logical_or(np.isclose(x[1], 0), np.isclose(x[1], 1))


def inflow(x):
    return np.isclose(x[0], 0)


def outflow(x):
    return np.isclose(x[0], 1)


wall_dofs = locate_dofs_geometrical(V, walls)
u_noslip = np.array((0,) * mesh.geometry.dim, dtype=PETSc.ScalarType)
bc_noslip = dirichletbc(u_noslip, wall_dofs, V)
inflow_dofs = locate_dofs_geometrical(Q, inflow)
bc_inflow = dirichletbc(PETSc.ScalarType(8), inflow_dofs, Q)
outflow_dofs = locate_dofs_geometrical(Q, outflow)
bc_outflow = dirichletbc(PETSc.ScalarType(0), outflow_dofs, Q)
bcu = [bc_noslip]
bcp = [bc_inflow, bc_outflow]

solver = ChorinSolver(V, Q, bcu, bcp, dt, rho_raw, mu_raw)


def u_exact(x):
    values = np.zeros((2, x.shape[1]), dtype=PETSc.ScalarType)
    values[0] = 4 * x[1] * (1.0 - x[1])
    return values


u_ex = Function(V)
u_ex.interpolate(u_exact)

L2_error = form(dot(solver.u_ - u_ex, solver.u_ - u_ex) * dx)

t = 0.0
for i in range(100):
    t += dt
    solver.solve_one_step()
    error_L2 = np.sqrt(mesh.comm.allreduce(
        assemble_scalar(L2_error), op=MPI.SUM))
    error_max = mesh.comm.allreduce(
        np.max(solver.u_.x.petsc_vec.array - u_ex.x.petsc_vec.array), op=MPI.MAX)
    if (i % 20 == 0) or (i == num_steps - 1):
        print(
            f"Time {t:.2f}, L2-error {error_L2:.2e}, Max error {error_max:.2e}")

solver.post_process()
