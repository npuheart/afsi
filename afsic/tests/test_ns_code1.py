# https://jsdokken.com/dolfinx-tutorial/chapter2/ns_code1.html
import math

import pytest
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


def test_ns_code1_poiseuille():
    # https://jsdokken.com/dolfinx-tutorial/chapter2/ns_code1.html
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
    
    # f = Constant(mesh, PETSc.ScalarType((0, 0))) # Not passed in __init__ in the class definition I saw earlier, checking usages. 
    # In the previous turn, the user's ns_code1.py snippet had `ns_solver = NSCode1Solver(V, Q, bcu, bcp, f, dt_raw=dt, rho_raw=1, mu_raw=1)`
    # BUT the class definition I wrote in `NSCode1Solver.py` earlier DOES NOT take `f` in `__init__`.
    # It initializes `self.f` to zero inside `__init__`.
    # I should check if the user modified the class or if I should follow the class definition I created.
    # The snippet provided in the attachment `ns_code1.py` HAS `f` in the call.
    # `ns_solver = NSCode1Solver(V, Q, bcu, bcp, f, dt_raw=dt, rho_raw=1, mu_raw=1)`
    # If the user is using the class I defined efficiently, it signature is:
    # def __init__(self, V, Q, bcu, bcp, dt_raw: float, rho_raw: float, mu_raw: float):
    # Let me check NSCode1Solver.py again to be sure.
    
    # Assuming I should stick to the class definition I created which doesn't take f in init.
    # If the user modified it, I might break it. 
    # However, looking at the previous turn's `ns_code1.py` attachment content provided by the user in the prompt, it has:
    # `ns_solver = NSCode1Solver(V, Q, bcu, bcp, f, dt_raw=dt, rho_raw=1, mu_raw=1)`
    # But my `create_file` for `NSCode1Solver.py` did NOT include `f` in `__init__`.
    # I will stick to the valid class definition I created: `NSCode1Solver(V, Q, bcu, bcp, dt_raw=dt, rho_raw=1.0, mu_raw=1.0)`
    # refusing the `f` argument.
    f = Constant(mesh, PETSc.ScalarType((0, 0)))
    ns_solver = NSCode1Solver(V, Q, bcu, bcp, f, dt_raw=dt, rho_raw=1, mu_raw=1)
    
    # Setup exact solution for error calculation
    def u_exact_func(x):
        values = np.zeros((2, x.shape[1]), dtype=PETSc.ScalarType)
        values[0] = 4 * x[1] * (1.0 - x[1])
        return values

    u_ex = Function(V)
    u_ex.interpolate(u_exact_func)
    L2_error_form = form(dot(ns_solver.u_ - u_ex, ns_solver.u_ - u_ex) * dx)
    
    # This form depends on u_ which changes every step, so we might need to recreate it or update coefficients if it was compiled. 
    # But dolfinx forms usually hold references to Functions.
    # We'll define the error form inside the loop for clarity or just compute it at the end.
    
    for i in range(num_steps):
        t += dt
        ns_solver.solve_one_step()
            
        # Check error at the final step
        error_L2 = np.sqrt(mesh.comm.allreduce(assemble_scalar(L2_error_form), op=MPI.SUM))
        error_max = mesh.comm.allreduce(
            np.max(ns_solver.u_.x.petsc_vec.array - u_ex.x.petsc_vec.array), op=MPI.MAX
        )
    # Print error only every 20th step and at the last step
        if (i % 20 == 0) or (i == num_steps - 1):
            print(f"Time {t:.2f}, L2-error {error_L2:.6e}, Max error {error_max:.6e}")
    
    ns_solver.post_process()
    # Final error check at the end of the simulation
    assert math.isclose(
        error_L2,
        5.201511e-06,
        rel_tol=1e-9,
        abs_tol=1e-12,
    )

if __name__ == "__main__":
    test_ns_code1_poiseuille()
# Time 10.00, L2-error 5.201511e-06, Max error 8.923966e-06