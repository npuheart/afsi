
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


# Solver
class ChorinSolver:

    def __init__(self, V, Q, bcu, bcp, dt_raw, rho_raw, mu_raw):
        self.bcu = bcu
        self.bcp = bcp

        self.V = V
        self.Q = Q

        u = TrialFunction(V)
        v = TestFunction(V)
        p = TrialFunction(Q)
        q = TestFunction(Q)
        
        mesh = V.mesh
        self.mesh = mesh

        k = Constant(mesh, PETSc.ScalarType(dt_raw))
        mu = Constant(mesh, PETSc.ScalarType(mu_raw))
        rho = Constant(mesh, PETSc.ScalarType(rho_raw))

        u_n = Function(V)
        u_n.name = "u_n"
        U = 0.5 * (u_n + u)
        n = FacetNormal(mesh)
        f = Function(V)

        # Define strain-rate tensor
        def epsilon(u):
            return sym(nabla_grad(u))

        # Define stress tensor
        def sigma(u, p):
            return 2 * mu * epsilon(u) - p * Identity(len(u))

        # Define the variational problem for the first step
        p_n = Function(Q)
        p_n.name = "p_n"
        F1 = rho * dot((u - u_n) / k, v) * dx
        F1 += rho * dot(dot(u_n, nabla_grad(u_n)), v) * dx
        F1 += inner(sigma(U, p_n), epsilon(v)) * dx
        F1 += dot(p_n * n, v) * ds - dot(mu * nabla_grad(U) * n, v) * ds
        F1 -= dot(f, v) * dx
        a1 = form(lhs(F1))
        L1 = form(rhs(F1))

        A1 = assemble_matrix(a1, bcs=bcu)
        A1.assemble()
        b1 = create_vector(L1)

        # Define variational problem for step 2
        u_ = Function(V)
        a2 = form(dot(nabla_grad(p), nabla_grad(q)) * dx)
        L2 = form(dot(nabla_grad(p_n), nabla_grad(q))
                  * dx - (rho / k) * div(u_) * q * dx)
        A2 = assemble_matrix(a2, bcs=bcp)
        A2.assemble()
        b2 = create_vector(L2)

        # Define variational problem for step 3
        p_ = Function(Q)
        a3 = form(rho * dot(u, v) * dx)
        L3 = form(rho * dot(u_, v) * dx - k *
                  dot(nabla_grad(p_ - p_n), v) * dx)
        A3 = assemble_matrix(a3)
        A3.assemble()
        b3 = create_vector(L3)

        # Solver for step 1
        solver1 = PETSc.KSP().create(mesh.comm)
        solver1.setOperators(A1)
        solver1.setType(PETSc.KSP.Type.BCGS)
        pc1 = solver1.getPC()
        pc1.setType(PETSc.PC.Type.HYPRE)
        pc1.setHYPREType("boomeramg")

        # Solver for step 2
        solver2 = PETSc.KSP().create(mesh.comm)
        solver2.setOperators(A2)
        solver2.setType(PETSc.KSP.Type.BCGS)
        pc2 = solver2.getPC()
        pc2.setType(PETSc.PC.Type.HYPRE)
        pc2.setHYPREType("boomeramg")

        # Solver for step 3
        solver3 = PETSc.KSP().create(mesh.comm)
        solver3.setOperators(A3)
        solver3.setType(PETSc.KSP.Type.CG)
        pc3 = solver3.getPC()
        pc3.setType(PETSc.PC.Type.SOR)

        self.a1 = a1
        self.a2 = a2
        self.a3 = a3

        self.b1 = b1
        self.b2 = b2
        self.b3 = b3
        self.L1 = L1
        self.L2 = L2
        self.L3 = L3
        self.solver1 = solver1
        self.solver2 = solver2
        self.solver3 = solver3

        self.u_n = u_n
        self.u_ = u_
        self.p_ = p_
        self.p_n = p_n
        self.f = f

    def solve_one_step(self):
        # Step 1: Tentative velocity step
        with self.b1.localForm() as loc_1:
            loc_1.set(0)
        assemble_vector(self.b1, self.L1)
        apply_lifting(self.b1, [self.a1], [self.bcu])
        self.b1.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                            mode=PETSc.ScatterMode.REVERSE)
        set_bc(self.b1, self.bcu)
        self.solver1.solve(self.b1, self.u_.x.petsc_vec)
        self.u_.x.scatter_forward()

        # Step 2: Pressure corrrection step
        with self.b2.localForm() as loc_2:
            loc_2.set(0)
        assemble_vector(self.b2, self.L2)
        apply_lifting(self.b2, [self.a2], [self.bcp])
        self.b2.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                            mode=PETSc.ScatterMode.REVERSE)
        set_bc(self.b2, self.bcp)
        self.solver2.solve(self.b2, self.p_.x.petsc_vec)
        self.p_.x.scatter_forward()

        # Step 3: Velocity correction step
        with self.b3.localForm() as loc_3:
            loc_3.set(0)
        assemble_vector(self.b3, self.L3)
        self.b3.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                            mode=PETSc.ScatterMode.REVERSE)
        self.solver3.solve(self.b3, self.u_.x.petsc_vec)
        self.u_.x.scatter_forward()
        # Update variable with solution form this time step
        self.u_n.x.array[:] = self.u_.x.array[:]
        self.p_n.x.array[:] = self.p_.x.array[:]

    def post_process(self):
        self.b1.destroy()
        self.b2.destroy()
        self.b3.destroy()
        self.solver1.destroy()
        self.solver2.destroy()
        self.solver3.destroy()