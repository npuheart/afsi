"""Serial exactly divergence-conforming RT/DG backward-Euler fluid solver.

SIP viscous terms and lagged-velocity upwind convection follow DOLFINx's
v0.10.0 demo_navier-stokes.py (LGPL-3.0-or-later):
https://github.com/FEniCS/dolfinx/blob/v0.10.0/python/demo/demo_navier-stokes.py
Normal Dirichlet data are strong; tangential data are weak (Nitsche).
Open facets carry zero natural traction. This is not the Chorin algorithm.

By default the saddle point system is factorized directly (serial MUMPS LU).
``linear_solver="block"`` instead assembles the nested (RT velocity) x
(DG pressure) blocks and solves with MINRES (no convection) or FGMRES (upwind
convection on) using a SIMPLE-type fieldsplit preconditioner (SCHUR/UPPER with
the self-P Schur estimate, or SCHUR/DIAG when the system is symmetric): the
velocity block is inverted with LU or hypre BoomerAMG, and the pressure Schur
complement is assembled by PETSc and solved directly. Alternative pressure
treatments: ``pressure_scale="auto"`` uses an additive fieldsplit with a
pressure mass scaling auto-calibrated from a diagonal Schur estimate, a
positive float pins that constant scaling directly, and ``"schur-diag"`` /
``"lsc"`` substitute the assembled per-DOF Schur diagonal / the full sparse
least-squares commutator ``B diag(T)^-1 B^T``. Because the nested solve is
iterative, the mass conservation floor equals the requested ``ksp_rtol``
(times the right hand side scale), not machine precision; the solve raises
instead of returning a nonconverged iterate.
"""
import os

import numpy as np
import ufl
from basix.ufl import element, mixed_element
from petsc4py import PETSc
from dolfinx import fem, mesh as dmesh
from dolfinx.fem.petsc import (assemble_matrix, assemble_vector, apply_lifting,
                               create_vector, set_bc)
from dolfinx.la.petsc import create_vector_wrap


class RTFluidSolver:
    def __init__(self, mesh, dt, rho=1., mu=1., degree=2, convection=True,
                 dirichlet_facets=None, penalty=None, linear_solver="direct",
                 velocity_pc="lu", pressure_scale=None, ksp_rtol=1e-10,
                 ksp_max_it=2000):
        if mesh.comm.size != 1:
            raise NotImplementedError("RTFluidSolver currently supports one MPI rank only")
        if mesh.geometry.dim != 2 or np.issubdtype(PETSc.ScalarType, np.complexfloating):
            raise NotImplementedError("This prototype requires 2D geometry and real PETSc scalars")
        if degree < 2:
            raise ValueError("Start at Basix RT degree 2 for immersed motion")
        if min(dt, rho, mu) <= 0:
            raise ValueError("dt, rho and mu must be positive")
        if linear_solver not in ("direct", "block"):
            raise ValueError("linear_solver must be 'direct' or 'block'")
        if velocity_pc not in ("lu", "hypre"):
            raise ValueError("velocity_pc must be 'lu' or 'hypre'")
        if pressure_scale is not None:
            if isinstance(pressure_scale, str):
                if pressure_scale not in ("auto", "schur-diag", "lsc", "simple"):
                    raise ValueError(
                        "pressure_scale must be a positive number, or one of "
                        "'auto', 'schur-diag', 'lsc', 'simple'")
            elif pressure_scale <= 0:
                raise ValueError("pressure_scale must be positive")
        self.mesh, self.dt = mesh, dt
        self.rho, self.mu = float(rho), float(mu)
        self.convection = convection
        self.linear_solver = linear_solver
        self.velocity_pc = velocity_pc
        self.pressure_scale = None if pressure_scale is None else (
            pressure_scale if isinstance(pressure_scale, str) else float(pressure_scale))
        self.pressure_scale_used = None
        self.last_iterations = 0
        self.iterations_total = 0
        self.iterations_max = 0
        cell = mesh.basix_cell()
        ve = element("RT", cell, degree)
        qe = element("Lagrange", cell, degree - 1, discontinuous=True)
        self.W = fem.functionspace(mesh, mixed_element([ve, qe]))
        self.V, self.u_map = self.W.sub(0).collapse()
        self.Q, self.p_map = self.W.sub(1).collapse()
        self.u_ = fem.Function(self.V, name="velocity")
        self.u_n = fem.Function(self.V, name="previous_velocity")
        self.p_ = fem.Function(self.Q, name="pressure")
        self.solution = fem.Function(self.W)
        self.u_D = fem.Function(self.V)
        self.f = fem.Function(fem.functionspace(mesh, ("DG", degree, (mesh.geometry.dim,))))
        self.ib_load = None
        tdim = mesh.topology.dim
        mesh.topology.create_connectivity(tdim - 1, tdim)
        exterior = dmesh.exterior_facet_indices(mesh.topology)
        facets = exterior if dirichlet_facets is None else np.asarray(dirichlet_facets, dtype=np.int32)
        facets = np.unique(facets).astype(np.int32)
        self.closed = len(facets) == len(exterior)
        tags = dmesh.meshtags(mesh, tdim - 1, facets, np.ones(len(facets), dtype=np.int32))
        self.ds = ufl.Measure("ds", domain=mesh, subdomain_data=tags)
        dofs = fem.locate_dofs_topological((self.W.sub(0), self.V), tdim - 1, facets)
        self.bcs = [fem.dirichletbc(self.u_D, dofs, self.W.sub(0))]
        # Blocked assembly uses boundary conditions on the collapsed space.
        # Function-valued data with a plain dof array: the space is taken from
        # the value function itself (matches the dolfinx demo pattern).
        self.bcs_block = [fem.dirichletbc(
            self.u_D, fem.locate_dofs_topological(self.V, tdim - 1, facets))]
        self.n, self.h = ufl.FacetNormal(mesh), ufl.CellDiameter(mesh)
        self.alpha = fem.Constant(mesh, PETSc.ScalarType(penalty or 20. * degree**2))
        u, p = ufl.TrialFunctions(self.W)
        v, q = ufl.TestFunctions(self.W)
        a_v, a01, a10, L_v = self._terms(u, v, p, q)
        self.a, self.L = fem.form(a_v + a01 + a10), fem.form(L_v)
        if linear_solver == "block":
            uv, vv = ufl.TrialFunction(self.V), ufl.TestFunction(self.V)
            pq, qq = ufl.TrialFunction(self.Q), ufl.TestFunction(self.Q)
            av_b, a01_b, a10_b, Lv_b = self._terms(uv, vv, pq, qq)
            self.a_block = fem.form([[av_b, a01_b], [a10_b, None]])
            self.L_block = fem.form([Lv_b, ufl.ZeroBaseForm((qq,))])
            self.mass_block = fem.form(ufl.inner(pq, qq) * ufl.dx)
            # Standalone forms for the pressure-scale calibration (assembled
            # directly, avoiding borrowed nest submatrix accesses).
            self.a00_calib_form = fem.form(av_b)
            self.a10_calib_form = fem.form(a10_b)
        self.ksp = PETSc.KSP().create(mesh.comm)
        self.ksp.setOptionsPrefix("afsi_rt_")
        opts = PETSc.Options()
        if linear_solver == "direct":
            self.ksp.setType("preonly")
            self.ksp.getPC().setType("lu")
            self.ksp.getPC().setFactorSolverType("mumps")
            opts["afsi_rt_mat_mumps_icntl_24"] = 1
            opts["afsi_rt_mat_mumps_icntl_25"] = 0
        else:
            self.ksp.setType("fgmres" if convection else "minres")
            if convection:
                self.ksp.setPCSide(PETSc.PC.Side.RIGHT)
                # Long restarts: convection-dominated systems need many
                # distinct directions; the default 30 stalls badly. The
                # environment override exists for diagnosing restart stalls.
                opts["afsi_rt_ksp_gmres_restart"] = int(
                    os.environ.get("AFSI_RT_GMRES_RESTART", "150"))
            self.ksp.setTolerances(rtol=ksp_rtol, atol=0., max_it=ksp_max_it)
            # Honest tolerance: measure the true residual. With a block-scaled
            # preconditioner the preconditioned norm can look converged while
            # the true residual is garbage.
            self.ksp.setNormType(PETSc.KSP.NormType.UNPRECONDITIONED)
            pc = self.ksp.getPC()
            pc.setType("fieldsplit")
            self._set_fieldsplit_style(pc)
            # Stored for re-application whenever the operators are rebuilt.
            self.ksp_type_name = self.ksp.getType()
            self.ksp_rtol, self.ksp_max_it = ksp_rtol, ksp_max_it
        self.ksp.setFromOptions()
        self.ksp.setErrorIfNotConverged(True)
        self.A, self.P = None, None
        self.div_form = fem.form(ufl.div(self.u_)**2 * ufl.dx)
        self.pressure_form = fem.form(self.p_ * ufl.dx)
        self.volume = fem.assemble_scalar(fem.form(fem.Constant(mesh, PETSc.ScalarType(1.)) * ufl.dx))

    def _terms(self, u, v, p, q):
        """UFL pieces of the mixed problem for arbitrary trial/test functions.

        Returns the velocity-only bilinear form, the two divergence-coupling
        blocks and the velocity right-hand side. Passing collapsed-space
        functions reproduces exactly the blocks used by the nested solver.
        """
        rho, dt, mu = self.rho, self.dt, self.mu
        n, h, ds, alpha = self.n, self.h, self.ds, self.alpha

        def jump(w):
            return ufl.outer(w('+'), n('+')) + ufl.outer(w('-'), n('-'))

        a_v = rho / dt * ufl.inner(u, v) * ufl.dx
        a_v += mu * (
            ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
            - ufl.inner(ufl.avg(ufl.grad(u)), jump(v)) * ufl.dS
            - ufl.inner(jump(u), ufl.avg(ufl.grad(v))) * ufl.dS
            + alpha / ufl.avg(h) * ufl.inner(jump(u), jump(v)) * ufl.dS
            - ufl.inner(ufl.grad(u), ufl.outer(v, n)) * ds(1)
            - ufl.inner(ufl.outer(u, n), ufl.grad(v)) * ds(1)
            + alpha / h * ufl.inner(ufl.outer(u, n), ufl.outer(v, n)) * ds(1))
        L_v = ufl.inner(rho / dt * self.u_n + self.f, v) * ufl.dx
        L_v += mu * (-ufl.inner(ufl.outer(self.u_D, n), ufl.grad(v)) * ds(1)
                     + alpha / h * ufl.inner(ufl.outer(self.u_D, n), ufl.outer(v, n)) * ds(1))
        if self.convection:
            w = self.u_n
            inflow_selector = ufl.conditional(ufl.gt(ufl.dot(w, n), 0), 1., 0.)
            upwind = inflow_selector('+') * u('+') + inflow_selector('-') * u('-')
            a_v += rho * (-ufl.inner(u, ufl.div(ufl.outer(v, w))) * ufl.dx
                          + ufl.inner(ufl.dot(w, n)('+') * upwind, v('+')) * ufl.dS
                          + ufl.inner(ufl.dot(w, n)('-') * upwind, v('-')) * ufl.dS
                          + ufl.inner(ufl.dot(w, n) * inflow_selector * u, v) * ufl.ds)
            L_v -= rho * ufl.inner(ufl.dot(w, n) * (1 - inflow_selector) * self.u_D, v) * ds(1)
        return a_v, -p * ufl.div(v) * ufl.dx, -q * ufl.div(u) * ufl.dx, L_v

    def _assemble_block_operators(self):
        """Assemble the nested 2x2 operator and the fieldsplit preconditioner.

        P = diag(A00, c M_p): the velocity block of P is the assembled (0, 0)
        block of A; the pressure block is the assembled discontinuous pressure
        mass matrix scaled by ``pressure_scale`` (per-cell block diagonal).

        The solver is reset first: PETSc's fieldsplit state caches references
        to the old nest blocks and index sets, so the KSP/PC must be dropped
        before the old matrices are destroyed and new ones assembled (in-place
        destruction otherwise makes PCSetUp touch freed memory when the
        upwind term forces a rebuild each step).
        """
        self.ksp.reset()
        if self.P is not None:
            self.P.destroy()
            self.P = None
        if self.A is not None:
            self.A.destroy()
        self.A = assemble_matrix(self.a_block, bcs=self.bcs_block, kind="nest")
        self.A.assemble()
        a00 = self.A.getNestSubMatrix(0, 0)
        p11 = assemble_matrix(self.mass_block)
        p11.assemble()
        use_schur_diag = self.pressure_scale == "schur-diag"
        use_lsc = self.pressure_scale == "lsc"
        use_simple = self.pressure_scale in (None, "simple")
        use_calibration = self.pressure_scale in ("auto", "schur-diag", "lsc")
        if use_calibration:
            # Calibrate the pressure mass scaling against a diagonal
            # (SIMPLE-type) Schur estimate: c ~ median(diag(B diag(T)^-1 B^T)
            # / diag(Mp)) with B the divergence block and T the velocity block.
            # The correct scale moves with dt/rho and mu and with the unit
            # system (SI vs CGS), so it is measured, not guessed.
            tblock = assemble_matrix(self.a00_calib_form)
            tblock.assemble()
            bblock = assemble_matrix(self.a10_calib_form)
            bblock.assemble()
            dinv = tblock.getDiagonal()
            dinv.reciprocal()
            dmat = PETSc.Mat().createAIJ((dinv.getLocalSize(), dinv.getSize()))
            dmat.setDiagonal(dinv)
            dmat.assemble()
            dinv.destroy()
            tmp = bblock.matMult(dmat)
            dmat.destroy()
            bt = bblock.transpose()
            schur_mat = tmp.matMult(bt)
            tmp.destroy()
            bt.destroy()
            diag_vec = schur_mat.getDiagonal()
            schur_diag = diag_vec.getArray().copy()
            diag_vec.destroy()
            if not use_lsc:
                schur_mat.destroy()
            bblock.destroy()
            tblock.destroy()
            diag_vec = p11.getDiagonal()
            mass_diag = diag_vec.getArray().copy()
            diag_vec.destroy()
            self.pressure_scale_used = float(np.median(schur_diag / mass_diag))
        else:
            self.pressure_scale_used = self.pressure_scale
        if use_simple:
            # PETSc builds the SIMPLE Schur complement itself from the real
            # off-diagonal blocks (PCFieldSplit SCHUR/UPPER with SELF_P); the
            # pressure sub-solve then uses that assembled S directly.
            p11.destroy()
            self.pressure_scale_used = None
            a01 = self.A.getNestSubMatrix(0, 1)
            a10 = self.A.getNestSubMatrix(1, 0)
        elif use_lsc:
            # Least-squares commutator pressure block: use the assembled
            # sparse Schur approximation S ~ B diag(T)^-1 B^T itself, solved
            # by an inner Krylov method, instead of a scalar/diagonal scaling.
            p11.destroy()
            schur_mat.shift(1e-10 * float(np.median(schur_diag)))
            p11 = schur_mat
        elif use_schur_diag:
            # The pressure scaling itself is genuinely nonuniform across the
            # domain (inflow layers, large rho/dt): keep the per-DOF Schur
            # diagonal assembled above instead of collapsing it to one scalar.
            floor = 1e-12 * float(np.median(schur_diag))
            p11.destroy()
            p11 = PETSc.Mat().createAIJ((schur_diag.size, schur_diag.size))
            diag_vec = PETSc.Vec().createWithArray(np.maximum(schur_diag, floor))
            p11.setDiagonal(diag_vec)
            p11.assemble()
            diag_vec.destroy()
        else:
            p11.scale(self.pressure_scale_used)
        a00.setOption(PETSc.Mat.Option.SPD, True)
        if use_simple:
            self.P = PETSc.Mat().createNest([[a00, a01], [a10, None]])
        else:
            p11.setOption(PETSc.Mat.Option.SPD, True)
            self.P = PETSc.Mat().createNest([[a00, None], [None, p11]])
        self.P.assemble()
        if self.closed:
            null_vec = create_vector(fem.extract_function_spaces(self.L_block), "nest")
            null_vec.getNestSubVecs()[0].set(0.)
            null_vec.getNestSubVecs()[1].set(1.)
            null_vec.normalize()
            self.A.setNullSpace(PETSc.NullSpace().create(vectors=[null_vec]))
        # Re-apply the full solver configuration to the (reset) KSP.
        self.ksp.setType(self.ksp_type_name)
        if self.convection:
            self.ksp.setPCSide(PETSc.PC.Side.RIGHT)
        self.ksp.setTolerances(rtol=self.ksp_rtol, atol=0., max_it=self.ksp_max_it)
        self.ksp.setNormType(PETSc.KSP.NormType.UNPRECONDITIONED)
        pc = self.ksp.getPC()
        pc.setType("fieldsplit")
        self._set_fieldsplit_style(pc)
        self.ksp.setOperators(self.A, self.P)
        nested_is = self.P.getNestISs()
        pc.setFieldSplitIS(("u", nested_is[0][0]), ("p", nested_is[0][1]))
        pc.setUp()
        ksp_u, ksp_p = pc.getFieldSplitSubKSP()
        ksp_u.setType("preonly")
        if self.velocity_pc == "hypre":
            ksp_u.getPC().setType("hypre")
            ksp_u.getPC().setHYPREType("boomeramg")
        else:
            ksp_u.getPC().setType("lu")
            ksp_u.getPC().setFactorSolverType("mumps")
        if use_lsc:
            ksp_p.setType("cg")
            ksp_p.getPC().setType("hypre")
            ksp_p.getPC().setHYPREType("boomeramg")
            ksp_p.setTolerances(rtol=1e-2, atol=0., max_it=200)
        elif use_simple:
            ksp_p.setType("preonly")
            ksp_p.getPC().setType("lu")
            ksp_p.getPC().setFactorSolverType("mumps")
        else:
            ksp_p.setType("preonly")
            ksp_p.getPC().setType("jacobi")
        # Let the options database address the sub-solvers too, e.g.
        # ``-afsi_rt_fieldsplit_u_pc_hypre_boomeramg_...``.
        ksp_u.setFromOptions()
        ksp_p.setFromOptions()
        self.ksp.setFromOptions()
        self.ksp.setErrorIfNotConverged(True)
        if os.environ.get("AFSI_RT_MONITOR"):
            step = max(1, int(os.environ.get("AFSI_RT_MONITOR", "50")))
            self.ksp.monitorCancel()
            self.ksp.setMonitor(
                lambda ksp, it, rnorm: print(f"    ksp it={it} r={rnorm:.3e}", flush=True)
                if it % step == 0 else None)

    def _set_fieldsplit_style(self, pc):
        if self.pressure_scale in (None, "simple"):
            pc.setFieldSplitType(PETSc.PC.CompositeType.SCHUR)
            # UPPER is nonsymmetric: keep MINRES on the symmetric block
            # diagonal form when convection (and thus upwinding) is off.
            pc.setFieldSplitSchurFactType(PETSc.PC.SchurFactType.UPPER if self.convection
                                          else PETSc.PC.SchurFactType.DIAG)
            pc.setFieldSplitSchurPreType(PETSc.PC.SchurPreType.SELFP)
        else:
            pc.setFieldSplitType(PETSc.PC.CompositeType.ADDITIVE)

    def _nonconvergence_message(self):
        reason = self.ksp.getConvergedReason()
        enum_cls = PETSc.KSP.ConvergedReason
        name = next((attr for attr in dir(enum_cls)
                     if not attr.startswith('_') and getattr(enum_cls, attr) == reason),
                    str(reason))
        return (f"RT block solve did not converge (reason={name}({reason}), "
                f"iterations={self.ksp.getIterationNumber()}, rtol={self.ksp_rtol:g})")

    def solve_one_step(self, ib_load=None):
        if ib_load is None:
            ib_load = self.ib_load
        values = None
        if ib_load is not None:
            values = ib_load.getArray(readonly=True) if hasattr(ib_load, 'getArray') else np.asarray(ib_load)
            if values.shape != self.u_.x.array.shape:
                raise ValueError("IB load must use the collapsed RT velocity-space layout")
        if self.linear_solver == "direct":
            if self.A is None or self.convection:
                if self.A is not None:
                    self.A.destroy()
                self.A = assemble_matrix(self.a, bcs=self.bcs)
                self.A.assemble()
                self.ksp.setOperators(self.A)
            b = assemble_vector(self.L)
            apply_lifting(b, [self.a], [self.bcs])
            b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
            if values is not None:
                b.getArray()[self.u_map] += values
            set_bc(b, self.bcs)
            self.ksp.solve(b, self.solution.x.petsc_vec)
            b.destroy()
            self.solution.x.scatter_forward()
            self.u_.x.array[:] = self.solution.x.array[self.u_map]
            self.p_.x.array[:] = self.solution.x.array[self.p_map]
        else:
            if self.A is None or self.convection:
                self._assemble_block_operators()
            b = assemble_vector(self.L_block, kind="nest")
            bcs_lift = fem.bcs_by_block(fem.extract_function_spaces(self.a_block, 1), self.bcs_block)
            apply_lifting(b, self.a_block, bcs=bcs_lift)
            for b_sub in b.getNestSubVecs():
                b_sub.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
            if values is not None:
                b_vel = b.getNestSubVecs()[0].getArray()
                if b_vel.size != values.size:
                    raise ValueError("IB load must match the nested velocity block layout")
                b_vel[:] += values
            bcs_values = fem.bcs_by_block(fem.extract_function_spaces(self.L_block), self.bcs_block)
            set_bc(b, bcs_values)
            x = PETSc.Vec().createNest([create_vector_wrap(self.u_.x), create_vector_wrap(self.p_.x)])
            try:
                self.ksp.solve(b, x)
            except PETSc.Error:
                # setErrorIfNotConverged fires before the explicit guard below.
                if self.ksp.getConvergedReason() >= 0:
                    raise
                raise RuntimeError(self._nonconvergence_message()) from None
            self.last_iterations = self.ksp.getIterationNumber()
            reason = self.ksp.getConvergedReason()
            ok = (PETSc.KSP.ConvergedReason.CONVERGED_RTOL,
                  PETSc.KSP.ConvergedReason.CONVERGED_ATOL,
                  PETSc.KSP.ConvergedReason.CONVERGED_HAPPY_BREAKDOWN)
            if reason not in ok:
                # Hits of max_it can masquerade as CONVERGED_ITS; never let a
                # non-converged block solve return silently.
                raise RuntimeError(self._nonconvergence_message())
            self.iterations_total += self.last_iterations
            self.iterations_max = max(self.iterations_max, self.last_iterations)
            self.u_.x.scatter_forward()
            self.p_.x.scatter_forward()
            self.solution.x.array[self.u_map] = self.u_.x.array
            self.solution.x.array[self.p_map] = self.p_.x.array
            self.solution.x.scatter_forward()
            b.destroy()
        if self.closed:
            self.p_.x.array[:] -= fem.assemble_scalar(self.pressure_form) / self.volume
        self.u_n.x.array[:] = self.u_.x.array
        return self.u_

    def divergence_norm(self):
        return float(np.sqrt(max(0., fem.assemble_scalar(self.div_form))))
