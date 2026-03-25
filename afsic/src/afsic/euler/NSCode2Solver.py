# https://jsdokken.com/dolfinx-tutorial/chapter2/ns_code2.html
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx.fem import (
    Constant,
    Function,
    extract_function_spaces,
    form,
    set_bc,
)
from dolfinx.fem.petsc import (
    apply_lifting,
    assemble_matrix,
    assemble_vector,
    create_vector,
    create_matrix,
    set_bc,
)
from ufl import (
    TestFunction,
    TrialFunction,
    div,
    dot,
    dx,
    inner,
    lhs,
    grad,
    nabla_grad,
    rhs,
)

class NSCode2Solver:
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
        
        self.u = TrialFunction(V)
        self.v = TestFunction(V)
        self.u_ = Function(V, name="u")
        self.u_s = Function(V, name="u_tentative")
        self.u_n = Function(V)
        self.u_n1 = Function(V)
        self.p = TrialFunction(Q)
        self.q = TestFunction(Q)
        self.p_ = Function(Q, name="p")
        self.phi = Function(Q, name="phi")
        
        self.F1 = self.rho / self.dt * dot(self.u - self.u_n, self.v) * dx
        self.F1 += inner(dot(1.5 * self.u_n - 0.5 * self.u_n1, 0.5 * nabla_grad(self.u + self.u_n)), self.v) * dx
        self.F1 += 0.5 * self.mu * inner(grad(self.u + self.u_n), grad(self.v)) * dx - dot(self.p_, div(self.v)) * dx
        self.F1 += dot(self.f, self.v) * dx
        self.a1 = form(lhs(self.F1))
        self.L1 = form(rhs(self.F1))
        self.A1 = create_matrix(self.a1)
        self.b1 = create_vector(extract_function_spaces(self.L1))
        self.a2 = form(dot(grad(self.p), grad(self.q)) * dx)
        self.L2 = form(-self.rho / self.dt * dot(div(self.u_s), self.q) * dx)
        self.A2 = assemble_matrix(self.a2, bcs=self.bcp)
        self.A2.assemble()
        self.b2 = create_vector(extract_function_spaces(self.L2))
        self.a3 = form(self.rho * dot(self.u, self.v) * dx)
        self.L3 = form(self.rho * dot(self.u_s, self.v) * dx - self.dt * dot(nabla_grad(self.phi), self.v) * dx)
        self.A3 = assemble_matrix(self.a3)
        self.A3.assemble()
        self.b3 = create_vector(extract_function_spaces(self.L3))

        # Solver for step 1
        self.solver1 = PETSc.KSP().create(mesh.comm)
        self.solver1.setOperators(self.A1)
        self.solver1.setType(PETSc.KSP.Type.BCGS)
        pc1 = self.solver1.getPC()
        pc1.setType(PETSc.PC.Type.JACOBI)

        # Solver for step 2
        self.solver2 = PETSc.KSP().create(mesh.comm)
        self.solver2.setOperators(self.A2)
        self.solver2.setType(PETSc.KSP.Type.MINRES)
        pc2 = self.solver2.getPC()
        pc2.setType(PETSc.PC.Type.HYPRE)
        pc2.setHYPREType("boomeramg")

        # Solver for step 3
        self.solver3 = PETSc.KSP().create(mesh.comm)
        self.solver3.setOperators(self.A3)
        self.solver3.setType(PETSc.KSP.Type.CG)
        pc3 = self.solver3.getPC()
        pc3.setType(PETSc.PC.Type.SOR)

    def solve_one_step(self):
        # Step 1: Tentative velocity step
        self.A1.zeroEntries()
        assemble_matrix(self.A1, self.a1, bcs=self.bcu)
        self.A1.assemble()
        with self.b1.localForm() as loc:
            loc.set(0)
        assemble_vector(self.b1, self.L1)
        apply_lifting(self.b1, [self.a1], [self.bcu])
        self.b1.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        set_bc(self.b1, self.bcu)
        self.solver1.solve(self.b1, self.u_s.x.petsc_vec)
        self.u_s.x.scatter_forward()
        # Step 2: Pressure corrrection step
        with self.b2.localForm() as loc:
            loc.set(0)
        assemble_vector(self.b2, self.L2)
        apply_lifting(self.b2, [self.a2], [self.bcp])
        self.b2.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        set_bc(self.b2, self.bcp)
        self.solver2.solve(self.b2, self.phi.x.petsc_vec)
        self.phi.x.scatter_forward()

        self.p_.x.petsc_vec.axpy(1, self.phi.x.petsc_vec)
        self.p_.x.scatter_forward()

        # Step 3: Velocity correction step
        with self.b3.localForm() as loc:
            loc.set(0)
        assemble_vector(self.b3, self.L3)
        self.b3.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        self.solver3.solve(self.b3, self.u_.x.petsc_vec)
        self.u_.x.scatter_forward()
    
    def post_process(self):
        self.A1.destroy()
        self.A2.destroy()
        self.A3.destroy()
        self.b1.destroy()
        self.b2.destroy()
        self.b3.destroy()
        self.solver1.destroy()
        self.solver2.destroy()
        self.solver3.destroy()
