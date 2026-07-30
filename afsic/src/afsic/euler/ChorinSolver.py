
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx.fem import Constant, Function, form
from dolfinx.fem.petsc import (assemble_matrix, assemble_vector, apply_lifting,
                               create_vector, set_bc)
from ufl import (TestFunction, TrialFunction,
                 div, dot, ds, dx, inner, lhs, nabla_grad, grad, rhs)


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
        self._dt = dt_raw  # store raw dt for direct forcing

        k = Constant(mesh, PETSc.ScalarType(dt_raw))
        mu = Constant(mesh, PETSc.ScalarType(mu_raw))
        rho = Constant(mesh, PETSc.ScalarType(rho_raw))

        u_n = Function(V,name = "u_n")
        u_ = Function(V,name = "u_")
        p_n = Function(Q,name = "p_n")
        p_ = Function(Q,name = "p_")
        f = Function(V,name="force")


        # Define the variational problem for the first step
        F1 = rho * dot((u - u_n) / k, v) * dx
        F1 += rho * inner(dot(grad(u_n), u_n), v)*dx
        F1 += inner(mu * grad(u), grad(v)) * dx
        F1 -= inner(f, v) * dx
        a1 = form(lhs(F1))
        L1 = form(rhs(F1))

        A1 = assemble_matrix(a1, bcs=bcu)
        A1.assemble()
        b1 = create_vector(V)

        # Define variational problem for step 2
        a2 = form(dot(grad(p), grad(q)) * dx)
        L2 = form(dot(-(rho / k) * div(u_), q) * dx)
        A2 = assemble_matrix(a2, bcs=bcp)
        A2.assemble()
        b2 = create_vector(Q)

        # Define variational problem for step 3
        a3 = form(dot(u, v) * dx)
        L3 = form(dot(u_, v) * dx - (k / rho) *
                  dot(grad(p_), v) * dx)
        A3 = assemble_matrix(a3, bcs=bcu)
        A3.assemble()
        b3 = create_vector(V)

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
        apply_lifting(self.b3, [self.a3], [self.bcu])
        self.b3.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                            mode=PETSc.ScatterMode.REVERSE)
        set_bc(self.b3, self.bcu)
        self.solver3.solve(self.b3, self.u_.x.petsc_vec)
        self.u_.x.scatter_forward()

        # Update variable with solution form this time step
        self.u_n.x.array[:] = self.u_.x.array[:]
        self.p_n.x.array[:] = self.p_.x.array[:]

    def solve_one_step_df(self, solid_dofs, bs):
        """Direct Forcing: 同一步内求解 + 修正。

        Algorithm:
          1. Step 1 with f=0 → get ũ (tentative velocity without solid force)
          2. f = (U_solid - ũ)/dt at solid DOFs
          3. Re-solve Step 1 with f → get corrected u*
          4. Steps 2-3 → get u^{n+1}, p^{n+1}

        Parameters
        ----------
        solid_dofs : np.ndarray (int32)
            局部 DOF 索引 (block 索引, 不是分量索引)。
        bs : int
            Block size (gdim)。
        """
        # ---- Step 1a: solve WITHOUT body force → get ũ ----
        self.f.x.array[:] = 0.0
        with self.b1.localForm() as loc:
            loc.set(0)
        assemble_vector(self.b1, self.L1)
        apply_lifting(self.b1, [self.a1], [self.bcu])
        self.b1.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                            mode=PETSc.ScatterMode.REVERSE)
        set_bc(self.b1, self.bcu)
        self.solver1.solve(self.b1, self.u_.x.petsc_vec)
        self.u_.x.scatter_forward()

        # ---- Direct Forcing: compute Δf = -ũ/dt, accumulate f += Δf ----
        u_arr = self.u_.x.array
        f_arr = self.f.x.array
        dt_val = self._dt
        force_sum = [0.0, 0.0]
        for dof in solid_dofs:
            for d in range(bs):
                idx = dof * bs + d
                df = -u_arr[idx] / dt_val
                f_arr[idx] += df              # accumulate, not replace!
                force_sum[d] += u_arr[idx]    # return the Δf (incremental)
        self.f.x.scatter_forward()  # sync ghost values before assembly

        # ---- Step 1b: re-solve WITH force → get corrected u* ----

        # ---- Step 1b: re-solve WITH force → get corrected u* ----
        with self.b1.localForm() as loc:
            loc.set(0)
        assemble_vector(self.b1, self.L1)  # L1 includes f now
        apply_lifting(self.b1, [self.a1], [self.bcu])
        self.b1.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                            mode=PETSc.ScatterMode.REVERSE)
        set_bc(self.b1, self.bcu)
        self.solver1.solve(self.b1, self.u_.x.petsc_vec)
        self.u_.x.scatter_forward()

        # ---- Steps 2-3: pressure correction + velocity correction ----
        with self.b2.localForm() as loc:
            loc.set(0)
        assemble_vector(self.b2, self.L2)
        apply_lifting(self.b2, [self.a2], [self.bcp])
        self.b2.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                            mode=PETSc.ScatterMode.REVERSE)
        set_bc(self.b2, self.bcp)
        self.solver2.solve(self.b2, self.p_.x.petsc_vec)
        self.p_.x.scatter_forward()

        with self.b3.localForm() as loc:
            loc.set(0)
        assemble_vector(self.b3, self.L3)
        apply_lifting(self.b3, [self.a3], [self.bcu])
        self.b3.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                            mode=PETSc.ScatterMode.REVERSE)
        set_bc(self.b3, self.bcu)
        self.solver3.solve(self.b3, self.u_.x.petsc_vec)
        self.u_.x.scatter_forward()

        # ---- Direct velocity correction: u[solid] = 0 ----
        # Applied after Step 3 to ensure final velocity respects the solid.
        u_arr = self.u_.x.array
        for dof in solid_dofs:
            for d in range(bs):
                u_arr[dof * bs + d] = 0.0

        # Update previous time step
        self.u_n.x.array[:] = self.u_.x.array[:]
        self.p_n.x.array[:] = self.p_.x.array[:]

        return force_sum  # (drag_raw, lift_raw) = Σ ũ / dt * dV