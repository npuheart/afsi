"""
ChorinSolver 开发实验 step-2: Poiseuille 流

注意: 这是实验代码，生产环境请使用 afsic.euler.ChorinSolver。
"""

from mpi4py import MPI
from petsc4py import PETSc
import numpy as np

from dolfinx.fem import Constant, Function, functionspace, assemble_scalar, dirichletbc, form, locate_dofs_geometrical
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, create_vector, set_bc
from dolfinx.io import VTXWriter
from dolfinx.mesh import create_unit_square
from dolfinx.plot import vtk_mesh
from basix.ufl import element
from ufl import (FacetNormal, Identity, TestFunction, TrialFunction,
                 div, dot, ds, dx, inner, lhs, nabla_grad, rhs, sym)


class ChorinSolverDev:
    """开发版 ChorinSolver — 仅用于实验"""

    def __init__(self, V, Q, bcu, bcp, dt_raw, rho_raw, mu_raw):
        self.bcu = bcu
        self.bcp = bcp
        self.V = V
        self.Q = Q

        mesh = V.mesh
        u = TrialFunction(V)
        v = TestFunction(V)
        p = TrialFunction(Q)
        q = TestFunction(Q)

        k = Constant(mesh, PETSc.ScalarType(dt_raw))
        mu = Constant(mesh, PETSc.ScalarType(mu_raw))
        rho = Constant(mesh, PETSc.ScalarType(rho_raw))

        self.dt = dt_raw
        self.u_n = Function(V, name="u_n")
        self.u_ = Function(V)
        self.p_ = Function(Q)
        self.p_n = Function(Q, name="p_n")
        self.f = Function(V)

        U = 0.5 * (self.u_n + u)
        n = FacetNormal(mesh)

        def epsilon(u_):
            return sym(nabla_grad(u_))

        def sigma(u_, p_):
            return 2 * mu * epsilon(u_) - p_ * Identity(len(u_))

        # Step 1: tentative velocity
        F1 = rho * dot((u - self.u_n) / k, v) * dx
        F1 += rho * dot(dot(self.u_n, nabla_grad(self.u_n)), v) * dx
        F1 += inner(sigma(U, self.p_n), epsilon(v)) * dx
        F1 += dot(self.p_n * n, v) * ds - dot(mu * nabla_grad(U) * n, v) * ds
        F1 -= dot(self.f, v) * dx
        self.a1 = form(lhs(F1))
        self.L1 = form(rhs(F1))
        A1 = assemble_matrix(self.a1, bcs=bcu)
        A1.assemble()
        self.b1 = create_vector(V)

        # Step 2: pressure correction
        self.a2 = form(dot(nabla_grad(p), nabla_grad(q)) * dx)
        self.L2 = form(dot(nabla_grad(self.p_n), nabla_grad(q)) * dx
                       - (rho / k) * div(self.u_) * q * dx)
        A2 = assemble_matrix(self.a2, bcs=bcp)
        A2.assemble()
        self.b2 = create_vector(Q)

        # Step 3: velocity correction
        self.a3 = form(rho * dot(u, v) * dx)
        self.L3 = form(rho * dot(self.u_, v) * dx
                       - k * dot(nabla_grad(self.p_ - self.p_n), v) * dx)
        A3 = assemble_matrix(self.a3)
        A3.assemble()
        self.b3 = create_vector(V)

        # Solver 1
        self.solver1 = PETSc.KSP().create(mesh.comm)
        self.solver1.setOperators(A1)
        self.solver1.setType(PETSc.KSP.Type.BCGS)
        pc1 = self.solver1.getPC()
        pc1.setType(PETSc.PC.Type.HYPRE)
        pc1.setHYPREType("boomeramg")

        # Solver 2
        self.solver2 = PETSc.KSP().create(mesh.comm)
        self.solver2.setOperators(A2)
        self.solver2.setType(PETSc.KSP.Type.BCGS)
        pc2 = self.solver2.getPC()
        pc2.setType(PETSc.PC.Type.HYPRE)
        pc2.setHYPREType("boomeramg")

        # Solver 3
        self.solver3 = PETSc.KSP().create(mesh.comm)
        self.solver3.setOperators(A3)
        self.solver3.setType(PETSc.KSP.Type.CG)
        pc3 = self.solver3.getPC()
        pc3.setType(PETSc.PC.Type.SOR)

    def solve_one_step(self):
        """执行一次时间步（三步投影法）"""
        # Step 1: Tentative velocity
        with self.b1.localForm() as loc:
            loc.set(0)
        assemble_vector(self.b1, self.L1)
        apply_lifting(self.b1, [self.a1], [self.bcu])
        self.b1.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                            mode=PETSc.ScatterMode.REVERSE)
        set_bc(self.b1, self.bcu)
        self.solver1.solve(self.b1, self.u_.x.petsc_vec)
        self.u_.x.scatter_forward()

        # Step 2: Pressure correction
        with self.b2.localForm() as loc:
            loc.set(0)
        assemble_vector(self.b2, self.L2)
        apply_lifting(self.b2, [self.a2], [self.bcp])
        self.b2.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                            mode=PETSc.ScatterMode.REVERSE)
        set_bc(self.b2, self.bcp)
        self.solver2.solve(self.b2, self.p_.x.petsc_vec)
        self.p_.x.scatter_forward()

        # Step 3: Velocity correction
        with self.b3.localForm() as loc:
            loc.set(0)
        assemble_vector(self.b3, self.L3)
        self.b3.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                            mode=PETSc.ScatterMode.REVERSE)
        self.solver3.solve(self.b3, self.u_.x.petsc_vec)
        self.u_.x.scatter_forward()

        # Update previous time step
        self.u_n.x.array[:] = self.u_.x.array[:]
        self.p_n.x.array[:] = self.p_.x.array[:]

    def cleanup(self):
        self.b1.destroy()
        self.b2.destroy()
        self.b3.destroy()
        self.solver1.destroy()
        self.solver2.destroy()
        self.solver3.destroy()


# =============================================================================
# Poiseuille 流实验 (仅当直接运行时执行)
# =============================================================================
if __name__ == "__main__":
    rho_raw = 1.0
    mu_raw = 1.0
    mesh = create_unit_square(MPI.COMM_WORLD, 10, 10)
    T = 10
    num_steps = 500
    dt = T / num_steps

    v_cg2 = element("Lagrange", mesh.topology.cell_name(), 2, shape=(mesh.geometry.dim,))
    s_cg1 = element("Lagrange", mesh.topology.cell_name(), 1)
    V = functionspace(mesh, v_cg2)
    Q = functionspace(mesh, s_cg1)

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

    solver = ChorinSolverDev(V, Q, bcu, bcp, dt, rho_raw, mu_raw)

    # 解析解: u(y) = 4*y*(1-y), v = 0
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
        error_L2 = np.sqrt(mesh.comm.allreduce(assemble_scalar(L2_error), op=MPI.SUM))
        error_max = mesh.comm.allreduce(
            np.max(solver.u_.x.petsc_vec.array - u_ex.x.petsc_vec.array), op=MPI.MAX)
        if (i % 20 == 0) or (i == num_steps - 1):
            print(f"Time {t:.2f}, L2-error {error_L2:.2e}, Max error {error_max:.2e}")

    solver.cleanup()
    print("Done.")


