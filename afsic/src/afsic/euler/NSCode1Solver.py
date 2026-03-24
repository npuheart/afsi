# https://jsdokken.com/dolfinx-tutorial/chapter2/ns_code1.html
from mpi4py import MPI
from petsc4py import PETSc
import numpy as np

from dolfinx.fem import (
    Constant,
    Function,
    extract_function_spaces,
    form,
)
from dolfinx.fem.petsc import (
    assemble_matrix,
    assemble_vector,
    apply_lifting,
    create_vector,
    set_bc,
)
from ufl import (
    FacetNormal,
    Identity,
    TestFunction,
    TrialFunction,
    div,
    dot,
    ds,
    dx,
    inner,
    lhs,
    nabla_grad,
    rhs,
    sym,
)

class NSCode1Solver:
    def epsilon(self, u):
        """Strain-rate tensor."""
        return sym(nabla_grad(u))

    def sigma(self, u, p):
        """Stress tensor."""
        return 2 * self.mu * self.epsilon(u) - p * Identity(len(u))

    def __init__(self, V, Q, bcu, bcp, f, dt_raw, rho_raw, mu_raw):
        self.V = V
        self.Q = Q
        self.bcu = bcu
        self.bcp = bcp
        self.f = f
        
        mesh = V.mesh
        self.mesh = mesh
        
        self.dt = Constant(mesh, PETSc.ScalarType(dt_raw))
        self.mu = Constant(mesh, PETSc.ScalarType(mu_raw))
        self.rho = Constant(mesh, PETSc.ScalarType(rho_raw))

        self.u = TrialFunction(self.V)
        self.v = TestFunction(self.V)
        self.p = TrialFunction(self.Q)
        self.q = TestFunction(self.Q)
        self.u_n = Function(self.V)
        self.u_n.name = "u_n"
        self.U = 0.5 * (self.u_n + self.u)

        n = FacetNormal(mesh)

        self.p_n = Function(self.Q)
        self.p_n.name = "p_n"
        self.F1 = self.rho * dot((self.u - self.u_n) / self.dt, self.v) * dx
        self.F1 += self.rho * dot(dot(self.u_n, nabla_grad(self.u_n)), self.v) * dx
        self.F1 += inner(self.sigma(self.U, self.p_n), self.epsilon(self.v)) * dx
        self.F1 += dot(self.p_n * n, self.v) * ds - dot(self.mu * nabla_grad(self.U) * n, self.v) * ds
        self.F1 -= dot(self.f, self.v) * dx
        self.a1 = form(lhs(self.F1))
        self.L1 = form(rhs(self.F1))
        self.A1 = assemble_matrix(self.a1, bcs=self.bcu)
        self.A1.assemble()
        self.b1 = create_vector(extract_function_spaces(self.L1))

        # Define variational problem for step 2
        self.u_ = Function(self.V)
        self.a2 = form(dot(nabla_grad(self.p), nabla_grad(self.q)) * dx)
        self.L2 = form(dot(nabla_grad(self.p_n), nabla_grad(self.q)) * dx - (self.rho / self.dt) * div(self.u_) * self.q * dx)
        self.A2 = assemble_matrix(self.a2, bcs=bcp)
        self.A2.assemble()
        self.b2 = create_vector(extract_function_spaces(self.L2))

        # Define variational problem for step 3
        self.p_ = Function(self.Q)
        self.a3 = form(self.rho * dot(self.u, self.v) * dx)
        self.L3 = form(self.rho * dot(self.u_, self.v) * dx - self.dt * dot(nabla_grad(self.p_ - self.p_n), self.v) * dx)
        self.A3 = assemble_matrix(self.a3)
        self.A3.assemble()
        self.b3 = create_vector(extract_function_spaces(self.L3))

        # Solver for step 1
        self.solver1 = PETSc.KSP().create(mesh.comm)
        self.solver1.setOperators(self.A1)
        self.solver1.setType(PETSc.KSP.Type.BCGS)
        # pc1 = solver1.getPC()
        # pc1.setType(PETSc.PC.Type.HYPRE)
        # pc1.setHYPREType("boomeramg")

        # Solver for step 2
        self.solver2 = PETSc.KSP().create(mesh.comm)
        self.solver2.setOperators(self.A2)
        self.solver2.setType(PETSc.KSP.Type.BCGS)
        # pc2 = solver2.getPC()
        # pc2.setType(PETSc.PC.Type.HYPRE)
        # pc2.setHYPREType("boomeramg")

        # Solver for step 3
        self.solver3 = PETSc.KSP().create(mesh.comm)
        self.solver3.setOperators(self.A3)
        self.solver3.setType(PETSc.KSP.Type.CG)
        self.pc3 = self.solver3.getPC()
        self.pc3.setType(PETSc.PC.Type.SOR)

    def solve_one_step(self):
        # Step 1: Tentative veolcity step
        with self.b1.localForm() as loc_1:
            loc_1.set(0)
        assemble_vector(self.b1, self.L1)
        apply_lifting(self.b1, [self.a1], [self.bcu])
        self.b1.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        set_bc(self.b1, self.bcu)
        self.solver1.solve(self.b1, self.u_.x.petsc_vec)
        self.u_.x.scatter_forward()

        # Step 2: Pressure corrrection step
        with self.b2.localForm() as loc_2:
            loc_2.set(0)
        assemble_vector(self.b2, self.L2)
        apply_lifting(self.b2, [self.a2], [self.bcp])
        self.b2.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        set_bc(self.b2, self.bcp)
        self.solver2.solve(self.b2, self.p_.x.petsc_vec)
        self.p_.x.scatter_forward()

        # Step 3: Velocity correction step
        with self.b3.localForm() as loc_3:
            loc_3.set(0)
        assemble_vector(self.b3, self.L3)
        self.b3.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
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