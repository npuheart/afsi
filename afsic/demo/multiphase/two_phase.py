"""Two-phase (diffuse-interface) incompressible flow for AFSI demos.

The machinery AFSI ships (``ChorinSolver`` / ``IPCSSolver`` / ``PeskinRK2Solver``)
assumes a **single** density and viscosity, so two-phase flow needs its own
solver.  This module provides one, in the "one-fluid / diffuse-interface"
formulation that is standard for this purpose:

* **Phase field** ``phi`` in ``[0, 1]`` (``phi ~ 0``: fluid 1, ``phi ~ 1``:
  fluid 2) on CG1, transported by the Allen-Cahn equation

      d phi/dt + u.grad(phi) = M [ eps lap(phi) - W'(phi)/eps ] ,

  with the double well ``W(phi) = phi^2 (1 - phi)^2 / 4`` (so
  ``W'(phi) = phi (1 - phi) (1 - 2 phi) / 2``), mobility ``M`` and interface
  parameter ``eps``.  The equilibrium profile of that equation is

      phi(d) = 0.5 (1 - tanh(d / (2 sqrt(2) eps))) ,

  i.e. an interface of thickness ``4 eps`` - that is the profile the demos
  initialise and the reason ``eps`` is chosen as roughly one mesh cell.

  Advection is explicit, diffusion implicit and the double well is treated with
  the stabilised (semi-implicit) splitting ``W'(phi^{n+1}) ~ W'(phi^n) +
  s (phi^{n+1} - phi^n)`` with ``s = 0.75 >= max |W''| = 1/2``, which keeps the
  scheme linear and - for ``M dt`` small - keeps ``phi`` inside ``[0, 1]``.  The
  result is clipped as a safety net, and that is stated in the demo readmes.

* **Variable properties** by linear interpolation of the phase field,
  ``rho(phi) = rho_1 + (rho_2 - rho_1) phi`` (same for ``mu``), evaluated nodally
  and used in every term of the projection.  Optional gravity/buoyancy enters as
  ``(rho(phi) - rho_1) g``.

* **Surface tension** with the continuum-surface-force (CSF) model,
  ``f_st = sigma kappa grad(phi)`` with ``kappa = -div(grad(phi)/|grad(phi)|)``.
  Both ``kappa`` and the force are computed **on the structured grid** with
  central differences from the nodal phase field and then smoothed.  This
  matters: differentiating the element-wise (discontinuous) CG1 gradient gives a
  curvature dominated by mesh-scale noise - the well known *parasitic currents*
  of CSF surface tension - which destroys the droplet.  For the same reason
  ``kappa`` is evaluated with the non-factorised identity

      kappa = -(lap phi)/|grad phi|
              + (grad phi . Hess(phi) . grad phi) / |grad phi|^3

  so that the exponentially small far-field gradient is never differentiated.
  Grid-based differences of the nodal tanh profile are second-order accurate -
  for a 64x64 droplet of radius 0.25 the measured interface curvature is
  ``3.95 +- 0.5`` against the exact ``4`` (the raw element-wise formula gives
  spikes of +-35 and the run dies within a few steps; see the tables in
  ``demo_801_laplace/readme.md``).

* **Projection** (classical Chorin, i.e. the same scheme as AFSI's
  ``ChorinSolver``, but with the variable coefficient ``1/rho``):

      1.  rho (u* - u^n)/dt + rho (u^n . grad) u^n
              - div( mu (grad u* + grad u*^T) ) = f_st + f_g
      2.  div( (1/rho) grad p^{n+1} ) = div(u*) / dt
      3.  u^{n+1} = u* - dt (1/rho) grad p^{n+1}

  The pressure carries *homogeneous* natural data ``(1/rho) dp/dn = 0``, exactly
  as ``ChorinSolver`` does: ``u*`` already satisfies ``u.n = u_bc.n`` at the
  boundary dofs, so a zero normal pressure derivative preserves it without any
  facet-flux bookkeeping.  (Replacing it by the textbook flux
  ``(1/rho) dp/dn = (u* - u_bc).n/dt`` makes the discrete compatibility depend
  on the quadrature of the boundary flux and the pressure solve then amplifies
  grid-scale modes.)  Likewise the pressure must **not** also be put in the
  predictor: doing both counts it twice and the scheme blows up in a handful of
  steps, which is the first trap to avoid here.

  Because ``rho(phi)`` and ``mu(phi)`` change every step, the two matrices that
  carry them are re-assembled on their (unchanged) sparsity pattern each step;
  when ``rho_1 == rho_2`` and ``mu_1 == mu_2`` (the pure surface-tension case)
  the re-assembly is skipped and the runs are much cheaper.

Everything is serial and 2-D, on a structured quadrilateral mesh.  The class
exposes ``u_n``/``p_``/``phi`` plus measurement helpers (``interior_pressure``,
``droplet_axes``) so the demos can check the Laplace pressure jump or the
deformation of a droplet in shear flow.
"""
import numpy as np

from mpi4py import MPI

from dolfinx import fem, default_scalar_type
from dolfinx.fem import (Constant, Function, dirichletbc, form, functionspace,
                         locate_dofs_topological)
from dolfinx.fem.petsc import (apply_lifting, assemble_matrix, assemble_vector,
                               create_vector, set_bc)
from dolfinx.mesh import CellType, GhostMode, create_rectangle, locate_entities_boundary
from petsc4py import PETSc
from ufl import (TestFunction, TrialFunction, div, dot, dx, grad, inner,
                 lhs, rhs, sym)

comm = MPI.COMM_WORLD
assert comm.size == 1, "these two-phase demos are serial"


class TwoPhaseSolver:
    """Diffuse-interface two-phase solver (variable rho/mu + CSF surface tension).

    Parameters
    ----------
    box : (x0, x1, y0, y1)
    nx, ny : int
        structured grid; velocity is CG2, pressure and phase field are CG1
    dt : float
    rho1, rho2, mu1, mu2 : float
        density/viscosity of fluid 1 (``phi = 0``) and fluid 2 (``phi = 1``)
    sigma : float
        surface-tension coefficient (``0`` disables the CSF force)
    eps : float
        Allen-Cahn interface parameter; the equilibrium profile has thickness
        ``4 eps``, so use ``eps ~ h`` (one mesh cell)
    mobility : float
        Allen-Cahn mobility ``M`` (interface relaxation speed)
    bc_spec : callable or None
        ``bc_spec(x) -> (u_x, u_y)`` applied to the whole boundary
        (``None`` = no Dirichlet velocity, natural conditions)
    gravity : (float, float)
        gravitational acceleration; the body force is ``(rho(phi) - rho_1) g``,
        so a heavier fluid 2 sinks for ``g = (0, -|g|)``
    grad_div : float
        optional grad-div stabilisation coefficient
    smooth_passes : int
        under-relaxed smoothing sweeps applied to the phase field before the
        curvature is differenced (kills parasitic currents; ``0`` = raw)
    fst_smooth : int
        additional sweeps applied to the surface-tension force components
    """

    def __init__(self, box, nx, ny, dt, rho1, rho2, mu1, mu2, sigma, eps,
                 mobility=1.0, bc_spec=None, grad_div=0.0, gravity=(0.0, 0.0),
                 smooth_passes=2, fst_smooth=1):
        self.box = box
        self.dt = float(dt)
        self.rho1, self.rho2 = float(rho1), float(rho2)
        self.mu1, self.mu2 = float(mu1), float(mu2)
        self.sigma, self.eps, self.M = float(sigma), float(eps), float(mobility)
        self.gravity = (float(gravity[0]), float(gravity[1]))
        self.ny, self.nx = int(ny), int(nx)
        self.smooth_passes = int(smooth_passes)
        self.fst_smooth = int(fst_smooth)
        self.hx = (box[1] - box[0]) / self.nx
        self.hy = (box[3] - box[2]) / self.ny

        self.mesh = create_rectangle(comm, ((box[0], box[2]), (box[1], box[3])),
                                     (nx, ny), cell_type=CellType.quadrilateral,
                                     ghost_mode=GhostMode.shared_facet)
        mesh = self.mesh
        self.V = functionspace(mesh, ("Lagrange", 2, (2,)))   # velocity
        self.Q = functionspace(mesh, ("Lagrange", 1))         # pressure
        self.S = functionspace(mesh, ("Lagrange", 1))         # phase field
        self.W1 = functionspace(mesh, ("Lagrange", 1, (2,)))  # surface tension

        self.bcu = []
        fdim = mesh.topology.dim - 1
        facets = locate_entities_boundary(mesh, fdim, lambda x: np.ones(x.shape[1], bool))
        self.bc_fn = Function(self.V)
        if bc_spec is not None:
            dofs = locate_dofs_topological(self.V, fdim, facets)
            self.bc_fn.interpolate(bc_spec)
            self.bcu = [dirichletbc(self.bc_fn, dofs)]

        # fields
        self.u_n = Function(self.V, name="u_n")    # previous velocity
        self.u_ = Function(self.V, name="u_")      # velocity unknown
        self.p_ = Function(self.Q, name="p")       # pressure
        self.phi = Function(self.S, name="phi")    # phase field
        self.phi_n = Function(self.S, name="phi_n")
        self.rho = Function(self.S, name="rho")
        self.mu = Function(self.S, name="mu")
        self.one_over_rho = Function(self.S, name="1/rho")
        self.kappa = Function(self.S, name="kappa")
        self.fst = Function(self.W1, name="f_st")  # surface-tension force
        self.fg = Function(self.W1, name="f_g")    # buoyancy force

        self.k = Constant(mesh, default_scalar_type(self.dt))
        k = self.k

        # --- (1) momentum predictor -------------------------------------------
        u, v = TrialFunction(self.V), TestFunction(self.V)
        F1 = (self.rho * dot((u - self.u_n) / k, v) * dx
              + self.rho * inner(dot(grad(self.u_n), self.u_n), v) * dx
              + inner(2.0 * self.mu * sym(grad(u)), sym(grad(v))) * dx
              - inner(self.fst, v) * dx
              - inner(self.fg, v) * dx)
        if grad_div:
            F1 += Constant(mesh, default_scalar_type(grad_div)) * div(u) * div(v) * dx
        self.a1, self.L1 = form(lhs(F1)), form(rhs(F1))

        # --- (2) pressure Poisson, div((1/rho) grad p) = div(u*)/dt -----------
        # Homogeneous natural data (1/rho) dp/dn = 0, exactly as AFSI's
        # ChorinSolver does: because u* already satisfies u.n = u_bc.n at the
        # boundary dofs, a zero normal pressure derivative keeps u.n = u_bc.n
        # after the correction without any facet-flux bookkeeping.  Adding the
        # flux term (1/rho) dp/dn = (u* - u_bc).n/dt instead makes the discrete
        # compatibility depend on the quadrature of the boundary flux and the
        # pressure solve then amplifies grid-scale modes.
        p, q = TrialFunction(self.Q), TestFunction(self.Q)
        F2 = (dot(self.one_over_rho * grad(p), grad(q)) * dx
              + (1.0 / k) * div(self.u_) * q * dx)
        self.a2, self.L2 = form(lhs(F2)), form(rhs(F2))

        # --- (3) velocity correction ------------------------------------------
        F3 = (dot(u, v) * dx - dot(self.u_, v) * dx
              + k * dot(self.one_over_rho * grad(self.p_), v) * dx)
        self.a3, self.L3 = form(lhs(F3)), form(rhs(F3))

        # --- (4) phase field: Allen-Cahn, IMEX --------------------------------
        ph, ps = TrialFunction(self.S), TestFunction(self.S)
        eps_ = Constant(mesh, default_scalar_type(self.eps))
        M = Constant(mesh, default_scalar_type(self.M))
        stab = Constant(mesh, default_scalar_type(self._stab()))
        F4 = (ph * ps * dx
              + k * M * eps_ * dot(grad(ph), grad(ps)) * dx
              + k * M * (1.0 / eps_) * stab * ph * ps * dx)
        L4 = (self.phi_n * ps * dx
              - k * dot(self.u_n, grad(self.phi_n)) * ps * dx
              + k * M * (1.0 / eps_)
              * (stab * self.phi_n - self._wprime(self.phi_n)) * ps * dx)
        self.a4, self.L4 = form(F4), form(L4)

        # --- matrices / solvers ------------------------------------------------
        self._update_properties()
        self._body_force()
        self.A1 = assemble_matrix(self.a1, bcs=self.bcu); self.A1.assemble()
        self.A2 = assemble_matrix(self.a2); self.A2.assemble()
        self.A3 = assemble_matrix(self.a3, bcs=self.bcu); self.A3.assemble()
        self.A4 = assemble_matrix(self.a4); self.A4.assemble()
        self.b1, self.b2, self.b3, self.b4 = (create_vector(self.V), create_vector(self.Q),
                                              create_vector(self.V), create_vector(self.S))
        self.solver1 = self._ksp(self.A1, "bcgs", "hypre")
        self.solver2 = self._ksp(self.A2, "cg", "hypre")
        self.A2.setNullSpace(PETSc.NullSpace().create(constant=True, comm=comm))
        self.solver3 = self._ksp(self.A3, "cg", "sor")
        self.solver4 = self._ksp(self.A4, "cg", "sor")
        # rho(phi), mu(phi) change every step -> re-assemble their systems (on
        # the unchanged pattern); pure surface tension needs neither
        self._reassemble1 = (self.rho1 != self.rho2) or (self.mu1 != self.mu2)
        self._reassemble2 = (self.rho1 != self.rho2)

        self._t = 0.0
        self.steps = 0

    # ------------------------------------------------------------------ helpers
    def _stab(self):
        """Stabilisation of the double well: ``s >= max |W''| = 1/2``."""
        return 0.75

    @staticmethod
    def _wprime(phi):
        """``W'(phi) = phi (1 - phi) (1 - 2 phi) / 2`` as a UFL expression."""
        return phi * (1 - phi) * (1 - 2 * phi) * 0.5

    @staticmethod
    def _ksp(A, ksp_type, pc_type):
        s = PETSc.KSP().create(comm)
        s.setOperators(A)
        s.setType(getattr(PETSc.KSP.Type, ksp_type.upper()))
        s.setTolerances(rtol=1e-10, atol=1e-13, max_it=10000)
        pc = s.getPC()
        if pc_type == "hypre":
            pc.setType(PETSc.PC.Type.HYPRE)
            pc.setHYPREType("boomeramg")
        else:
            pc.setType(PETSc.PC.Type.SOR)
        return s

    def _grid_index(self):
        """Dof -> (i, j) structured-grid index for the CG1 spaces."""
        if getattr(self, "_gi", None) is None:
            X = self.S.tabulate_dof_coordinates()[:, :2]
            i = np.rint((X[:, 0] - self.box[0]) / self.hx).astype(int)
            j = np.rint((X[:, 1] - self.box[2]) / self.hy).astype(int)
            assert i.min() == 0 and i.max() == self.nx and \
                j.min() == 0 and j.max() == self.ny, "phase dofs are not a full grid"
            assert len(set(zip(i.tolist(), j.tolist()))) == len(i), "non-unique dofs"
            self._gi = (i, j, X.shape[0])
        return self._gi

    def nodal(self, fn):
        """Nodal values of a CG1 field as an ``(ny+1, nx+1)`` array."""
        i, j, n = self._grid_index()
        out = np.full((self.ny + 1, self.nx + 1), np.nan)
        out[j, i] = fn.x.array[: n]
        return out

    def from_nodes(self, A):
        """Inverse of :meth:`nodal` (dof values from an ``(ny+1, nx+1)`` array)."""
        i, j, n = self._grid_index()
        return np.asarray(A)[j, i][: n]

    def nodal_vec(self, fn):
        """Nodal components of a CG1 *vector* field as two grid arrays."""
        i, j, n = self._grid_index()
        A = fn.x.array[: 2 * n].reshape(-1, 2)
        fx = np.full((self.ny + 1, self.nx + 1), np.nan)
        fy = np.full((self.ny + 1, self.nx + 1), np.nan)
        fx[j, i] = A[:, 0]
        fy[j, i] = A[:, 1]
        return fx, fy

    def from_nodes_vec(self, fx, fy):
        """Inverse of :meth:`nodal_vec` (flat dof values)."""
        i, j, n = self._grid_index()
        A = np.empty((n, 2))
        A[:, 0] = np.asarray(fx)[j, i][: n]
        A[:, 1] = np.asarray(fy)[j, i][: n]
        return A.ravel()

    @staticmethod
    def _smooth_grid(A, passes, weight=0.35):
        """Under-relaxed nine-point smoothing of a grid array (edges replicated)."""
        A = np.asarray(A, dtype=float).copy()
        for _ in range(int(passes)):
            pad = np.pad(A, 1, mode="edge")
            nb = (pad[0:-2, 1:-1] + pad[2:, 1:-1] + pad[1:-1, 0:-2] + pad[1:-1, 2:])
            A += weight * (0.25 * nb - A)
        return A

    def _grid_grad(self, A):
        """Central-difference gradient of a grid array (2nd order)."""
        gx = np.gradient(A, self.hx, axis=1)
        gy = np.gradient(A, self.hy, axis=0)
        return gx, gy

    # ------------------------------------------------------------- field set-up
    def set_phase(self, fn):
        """Initialise the phase field from ``fn(x) -> array``."""
        self.phi.interpolate(fn)
        self.phi_n.x.array[:] = self.phi.x.array
        self._update_properties()
        self._body_force()

    def set_velocity(self, ux=None, uy=None):
        """Initialise the velocity (e.g. a linear shear profile)."""
        f = Function(self.V)
        f.interpolate(lambda x: np.array(
            [ux(x) if ux is not None else np.zeros(x.shape[1]),
             uy(x) if uy is not None else np.zeros(x.shape[1])]))
        self.u_n.x.array[:] = f.x.array
        self.u_.x.array[:] = f.x.array

    @staticmethod
    def droplet_phase(centre, radius, eps):
        """The equilibrium Allen-Cahn profile of a circular droplet."""
        def fn(x):
            d = np.hypot(x[0] - centre[0], x[1] - centre[1]) - radius
            return 0.5 * (1.0 - np.tanh(d / (2.0 * np.sqrt(2.0) * eps)))
        return fn

    def _update_properties(self):
        n = self.rho.x.array.size
        phi = self.phi.x.array[: n]
        self.rho.x.array[:] = self.rho1 + (self.rho2 - self.rho1) * phi
        self.mu.x.array[:] = self.mu1 + (self.mu2 - self.mu1) * phi
        self.one_over_rho.x.array[:] = 1.0 / self.rho.x.array

    def _body_force(self):
        """Buoyancy ``(rho(phi) - rho_1) g`` (nodal, straight into the RHS)."""
        gx, gy = self.gravity
        if gx == 0.0 and gy == 0.0:
            self.fg.x.array[:] = 0.0
            return
        A = self.nodal(self.rho) - self.rho1
        self.fg.x.array[:] = self.from_nodes_vec(gx * A, gy * A)

    def _surface_tension(self):
        """CSF force ``sigma kappa grad(phi)`` from a grid-based curvature.

        The curvature uses the *non-factorised* identity

            kappa = -div(grad phi / |grad phi|)
                  = -(lap phi)/|grad phi|
                    + (grad phi . Hess(phi) . grad phi)/|grad phi|^3 ,

        which never differentiates a quantity divided by ``|grad phi|``.  That
        matters a lot: dividing by the exponentially small far-field gradient
        *before* differencing amplifies mesh noise by 1/h and produces the
        notorious parasitic currents (a naive implementation gives curvature
        spikes of +-35 for this droplet instead of the exact 4).
        """
        phi = self._smooth_grid(self.nodal(self.phi), self.smooth_passes)
        gx, gy = self._grid_grad(phi)
        gxx, _ = self._grid_grad(gx)
        _, gyy = self._grid_grad(gy)
        gxy = np.gradient(gx, self.hy, axis=0)
        gmag = np.hypot(gx, gy)
        gmax = gmag.max() if gmag.size else 0.0
        kappa = np.zeros_like(phi)
        if gmax > 0.0:
            keep = gmag > 1e-6 * gmax
            den = np.where(keep, gmag, 1.0)
            kappa = -(gxx + gyy) / den \
                + (gx * gx * gxx + 2.0 * gx * gy * gxy + gy * gy * gyy) / den ** 3
            kappa = np.where(keep, kappa, 0.0)
            kappa = self._smooth_grid(kappa, self.smooth_passes)
        self.kappa.x.array[:] = self.from_nodes(kappa)
        self.fst.x.array[:] = self.from_nodes_vec(
            self._smooth_grid(self.sigma * kappa * gx, self.fst_smooth),
            self._smooth_grid(self.sigma * kappa * gy, self.fst_smooth))

    # ------------------------------------------------------------------ stepping
    def solve_one_step(self):
        # (1) phase transport (IMEX: explicit advection, implicit diffusion)
        self.phi_n.x.array[:] = self.phi.x.array
        with self.b4.localForm() as loc:
            loc.set(0)
        assemble_vector(self.b4, self.L4)
        apply_lifting(self.b4, [self.a4], [[]])
        self.b4.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        set_bc(self.b4, [])
        self.solver4.solve(self.b4, self.phi.x.petsc_vec)
        self.phi.x.scatter_forward()
        np.clip(self.phi.x.array, 0.0, 1.0, out=self.phi.x.array)

        # (2) properties, surface tension and buoyancy for this step
        self._update_properties()
        self._surface_tension()
        self._body_force()
        if self._reassemble1:
            self.A1.zeroEntries()
            assemble_matrix(self.A1, self.a1, bcs=self.bcu); self.A1.assemble()
            self.solver1.setOperators(self.A1)
            self.solver1.getPC().setReusePreconditioner(True)
        if self._reassemble2:
            self.A2.zeroEntries()
            assemble_matrix(self.A2, self.a2); self.A2.assemble()
            self.solver2.setOperators(self.A2)
            self.solver2.getPC().setReusePreconditioner(True)

        # (3) momentum predictor
        with self.b1.localForm() as loc:
            loc.set(0)
        assemble_vector(self.b1, self.L1)
        apply_lifting(self.b1, [self.a1], [self.bcu])
        self.b1.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        set_bc(self.b1, self.bcu)
        self.solver1.solve(self.b1, self.u_.x.petsc_vec)
        self.u_.x.scatter_forward()

        # (4) pressure Poisson
        with self.b2.localForm() as loc:
            loc.set(0)
        assemble_vector(self.b2, self.L2)
        apply_lifting(self.b2, [self.a2], [[]])
        self.b2.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        set_bc(self.b2, [])
        self.solver2.solve(self.b2, self.p_.x.petsc_vec)
        self.p_.x.scatter_forward()

        # (5) velocity correction
        with self.b3.localForm() as loc:
            loc.set(0)
        assemble_vector(self.b3, self.L3)
        apply_lifting(self.b3, [self.a3], [self.bcu])
        self.b3.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        set_bc(self.b3, self.bcu)
        self.solver3.solve(self.b3, self.u_n.x.petsc_vec)
        self.u_n.x.scatter_forward()

        self._t += self.dt
        self.steps += 1

    def run(self, t_end, callback=None, callback_every=1):
        """Advance to ``t_end``; call ``callback(self)`` every ``callback_every``."""
        while self._t < t_end - 1e-12 * self.dt:
            self.solve_one_step()
            if callback is not None and self.steps % callback_every == 0:
                callback(self)
        return self

    # ------------------------------------------------------------- measurements
    def interior_pressure(self, radius, centre, inner=0.7, outer=(1.4, 2.1), level=None):
        """Mean CG1 pressure inside/outside a disc of the given radius."""
        X = self.Q.tabulate_dof_coordinates()[:, :2]
        r = np.hypot(X[:, 0] - centre[0], X[:, 1] - centre[1])
        p = self.p_.x.array[: X.shape[0]]
        inside = r < inner * radius
        outside = (r > outer[0] * radius) & (r < outer[1] * radius)
        if level is not None:
            ph = self.phi.x.array[: X.shape[0]]
            inside &= ph > level
            outside &= ph < level
        return float(p[inside].mean()), float(p[outside].mean())

    def droplet_moments(self, level=0.5, centre=None, window=None):
        """Second moments of the ``phi >= level`` region.

        Returns ``(area, (cx, cy), eigvals, eigvecs)``.  For an ellipse the
        semi-axes are ``2 sqrt(eigval)``, which gives the Taylor deformation
        ``D = (L - B) / (L + B)``.
        """
        A = self.nodal(self.phi)
        X, Y = np.meshgrid(self.box[0] + self.hx * np.arange(self.nx + 1),
                           self.box[2] + self.hy * np.arange(self.ny + 1))
        mask = A >= level
        if centre is not None:
            mask &= np.hypot(X - centre[0], Y - centre[1]) < window
        w = mask.astype(float)
        wsum = w.sum()
        cx = float((w * X).sum() / wsum)
        cy = float((w * Y).sum() / wsum)
        dxx = (w * (X - cx) ** 2).sum()
        dyy = (w * (Y - cy) ** 2).sum()
        dxy = (w * (X - cx) * (Y - cy)).sum()
        cov = np.array([[dxx, dxy], [dxy, dyy]]) / wsum
        evals, evecs = np.linalg.eigh(cov)
        return wsum * self.hx * self.hy, (cx, cy), evals, evecs

    def droplet_axes(self, level=0.5, centre=None, window=None):
        """Equivalent radius, major/minor axis lengths and tilt angle."""
        area, c, evals, evecs = self.droplet_moments(level, centre, window)
        # for an ellipse the semi-axes are 2 sqrt(eigval), so the *lengths* are
        # 4 sqrt(eigval); validated against exact ellipses in demo_802
        axes_len = 4.0 * np.sqrt(np.maximum(evals, 0.0))
        major = evecs[:, 1]
        theta = float(np.arctan2(major[1], major[0]))
        return (np.sqrt(area / np.pi), float(axes_len[1]), float(axes_len[0]), theta)

    def spurious_velocity(self):
        """Max |u| - the parasitic-current level of the current state."""
        return float(np.abs(self.u_n.x.array).max())

    def divergence_norm(self):
        """L2 norm of div(u) (should be round-off small after the projection)."""
        w, z = TrialFunction(self.Q), TestFunction(self.Q)
        A = assemble_matrix(form(w * z * dx)); A.assemble()
        b = create_vector(self.Q)
        assemble_vector(b, form(div(self.u_n) * z * dx))
        out = Function(self.Q)
        self._ksp(A, "cg", "sor").solve(b, out.x.petsc_vec)
        out.x.scatter_forward()
        val = float(np.sqrt(comm.allreduce(fem.assemble_scalar(form(out * out * dx)),
                                           op=MPI.SUM)))
        b.destroy(); A.destroy()
        return val
