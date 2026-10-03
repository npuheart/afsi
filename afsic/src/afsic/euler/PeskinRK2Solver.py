"""Peskin (2002) two-stage IB fluid solver on a doubly periodic box.

This is the time integrator of IB2d (``please_Update_Fluid_Velocity.m``),
re-written with AFSI / DOLFINx building blocks:

* spatial discretisation: Taylor–Hood Q2/Q1 on a structured quadrilateral mesh
  of the box ``[x0, x1] x [y0, y1]``;
* periodicity (IB2d is FFT based): the matrices/vectors assembled by DOLFINx on
  the non-periodic mesh are reduced with a periodic prolongation ``P``
  (``A_per = P^T A P``, ``b_per = P^T b``), i.e. every right/top node is tied
  to its left/bottom image;
* each stage is an *exact* (monolithic) unsteady Stokes solve — no projection
  splitting error, as with the FFT solver of IB2d.

Time stepping (Peskin, Acta Numerica 2002; IB2d)::

    stage 1  (backward Euler over dt/2, explicit skew-symmetric convection)
        rho (u_h - u_n)/(dt/2) + rho C(u_n) = -grad p_h + mu Lap u_h + f
        div u_h = 0
    stage 2  (Crank–Nicolson viscosity, midpoint convection)
        rho (u_{n+1} - u_n)/dt + rho C(u_h) = -grad p + mu/2 Lap(u_{n+1} + u_n) + f
        div u_{n+1} = 0

with ``C(w) = 1/2 [ (w.grad) w + div(w (x) w) ]`` and the same IB force density
``f`` (spread from the half-step Lagrangian configuration) in both stages.
Multiplying stage 2 by 2 gives the *same* left-hand side as stage 1, so the
saddle-point matrix is assembled and LU-factorised only once.

Usage: write the Eulerian force density into ``self.f`` (nodal values, e.g.
from ``IBInterpolation.solid_to_fluid``, followed by ``periodic_fold``) and
call ``solve_one_step``.  Afterwards ``self.u_h`` is the half-step velocity
(used to move the Lagrangian points) and ``self.u_`` / ``self.p_`` hold
``u^{n+1}`` / ``p^{n+1/2}``.

Notes
-----
* With ``(Nx/2, Ny/2)`` Q2 cells the velocity nodes coincide with the
  ``Nx x Ny`` collocated grid of IB2d (``IBMesh(..., order=2)``); with
  ``(Nx, Ny)`` cells the Q1 pressure/vertex nodes do (``IBMesh(..., order=1)``).
* ``grad_div > 0`` adds the consistent grad-div term ``gamma (div u, div v)``;
  it strongly reduces the IB volume leakage of Taylor–Hood.
* The pressure is fixed at one node (constant null space) and then shifted to
  zero mean, the gauge of the FFT solver.  Serial only.
"""

import numpy as np
import scipy.sparse as sp
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import fem
from dolfinx.fem import Constant, Function, form
from dolfinx.fem.petsc import assemble_matrix, assemble_vector
from ufl import (TestFunction, TrialFunction, div, dot, dx, grad, inner, outer)


def periodic_master_map(V, box):
    """For every (block) dof of ``V`` the index of its periodic master dof.

    Right (x = x1) and top (y = y1) nodes are mapped onto their left / bottom
    images; corners map onto (x0, y0).  Masters map onto themselves.
    """
    x0, x1, y0, y1 = box
    Lx, Ly = x1 - x0, y1 - y0
    X = V.tabulate_dof_coordinates()[:, :2]
    h = 1e-8 * min(Lx, Ly)
    nLx, nLy = int(round(Lx / h)), int(round(Ly / h))
    kx = np.mod(np.round((X[:, 0] - x0) / h).astype(np.int64), nLx)
    ky = np.mod(np.round((X[:, 1] - y0) / h).astype(np.int64), nLy)
    slave = np.isclose(X[:, 0], x1) | np.isclose(X[:, 1], y1)
    key = kx * (nLy + 1) + ky
    lookup = dict(zip(key[~slave].tolist(), np.flatnonzero(~slave).tolist()))
    master = np.arange(len(X))
    master[slave] = [lookup[k] for k in key[slave].tolist()]
    return master


def periodic_prolongation(V, box):
    """Sparse ``P`` (n_full x n_reduced) with ``u_full = P u_reduced`` (blocked dofs)."""
    bs = V.dofmap.index_map_bs
    master = periodic_master_map(V, box)
    masters = np.unique(master)
    red = -np.ones(len(master), dtype=np.int64)
    red[masters] = np.arange(len(masters))
    rows = (np.arange(len(master))[:, None] * bs + np.arange(bs)).ravel()
    cols = (red[master][:, None] * bs + np.arange(bs)).ravel()
    P = sp.csr_matrix((np.ones(len(rows)), (rows, cols)),
                      shape=(len(master) * bs, len(masters) * bs))
    return P, master


def _petsc_to_scipy(A):
    ai, aj, av = A.getValuesCSR()
    return sp.csr_matrix((av, aj, ai), shape=A.getSize())


class PeskinRK2Solver:
    """IB2d-equivalent periodic Navier–Stokes solver (see module docstring).

    Parameters
    ----------
    mesh : dolfinx.mesh.Mesh
        Structured quadrilateral mesh of the periodic box.
    box : tuple
        ``(x0, x1, y0, y1)``.
    dt, rho, mu : float
        Time step, density, dynamic viscosity.
    velocity_order, pressure_order : int
        Taylor–Hood orders (default 2 / 1).
    convection : bool
        Include the skew-symmetric convective term (default True).
    grad_div : float
        Grad-div stabilisation parameter gamma (default 0 = plain Taylor–Hood).
    petsc_options : dict, optional
        Options of the saddle-point KSP (default: direct LU with MUMPS).
    """

    def __init__(self, mesh, box, dt, rho, mu, velocity_order=2, pressure_order=1,
                 convection=True, grad_div=0.0, petsc_options=None):
        from basix.ufl import element

        if mesh.comm.size != 1:
            raise NotImplementedError("PeskinRK2Solver is serial (as the afsic IB coupling)")
        self.mesh, self.box, self.dt = mesh, box, dt
        gdim = mesh.geometry.dim
        cell = mesh.topology.cell_name()
        self.V = fem.functionspace(mesh, element("Lagrange", cell, velocity_order, shape=(gdim,)))
        self.Q = fem.functionspace(mesh, element("Lagrange", cell, pressure_order))

        self.u_n = Function(self.V, name="u_n")      # u^n
        self.u_h = Function(self.V, name="u_half")   # u^{n+1/2}
        self.u_ = Function(self.V, name="u")         # u^{n+1}
        self.p_ = Function(self.Q, name="p")         # p^{n+1/2}, zero mean
        self.p_h = Function(self.Q, name="p_half")
        self.f = Function(self.V, name="force")      # Eulerian IB force density

        k = Constant(mesh, PETSc.ScalarType(dt))
        rho_c = Constant(mesh, PETSc.ScalarType(rho))
        mu_c = Constant(mesh, PETSc.ScalarType(mu))
        u, v = TrialFunction(self.V), TestFunction(self.V)
        p, q = TrialFunction(self.Q), TestFunction(self.Q)

        def conv(w):
            # skew-symmetric form, as IB2d: 1/2[(w.grad)w + div(w w)]
            return 0.5 * (dot(grad(w), w) + div(outer(w, w)))

        # common LHS:  (2 rho/dt) M + mu K (+ gamma grad-div),  -p div v,  -q div u
        a_uu = (2.0 * rho_c / k) * inner(u, v) * dx + mu_c * inner(grad(u), grad(v)) * dx
        if grad_div > 0.0:
            a_uu += Constant(mesh, PETSc.ScalarType(grad_div)) * div(u) * div(v) * dx
        a_up = -p * div(v) * dx
        # stage 1 RHS
        L1 = (2.0 * rho_c / k) * inner(self.u_n, v) * dx + inner(self.f, v) * dx
        # stage 2 RHS (equation x 2 -> same LHS; the computed pressure is 2 p)
        L2 = ((2.0 * rho_c / k) * inner(self.u_n, v) * dx
              - mu_c * inner(grad(self.u_n), grad(v)) * dx
              + 2.0 * inner(self.f, v) * dx)
        if convection:
            L1 += -rho_c * inner(conv(self.u_n), v) * dx
            L2 += -2.0 * rho_c * inner(conv(self.u_h), v) * dx
        self.L1, self.L2 = form(L1), form(L2)

        # periodic reduction
        self.Pu, self.master_u = periodic_prolongation(self.V, box)
        self.Pp, self.master_p = periodic_prolongation(self.Q, box)
        Auu = assemble_matrix(form(a_uu))
        Auu.assemble()
        Aup = assemble_matrix(form(a_up))
        Aup.assemble()
        Auu, Aup = _petsc_to_scipy(Auu), _petsc_to_scipy(Aup)
        Ar = self.Pu.T @ Auu @ self.Pu
        Br = self.Pu.T @ Aup @ self.Pp                 # (u, p) block; (p, u) = Br^T
        self.nu, self.np_ = Ar.shape[0], Br.shape[1]
        # pressure gauge: pin the reduced pressure dof nearest the box centre
        xq = self.Q.tabulate_dof_coordinates()[:, :2]
        xc = np.array([0.5 * (box[0] + box[1]), 0.5 * (box[2] + box[3])])
        full_pin = int(np.argmin(np.linalg.norm(xq - xc, axis=1)))
        self.p_pin = int(self.Pp[full_pin].indices[0])
        Br = Br.tolil()
        Br[:, self.p_pin] = 0.0
        Br = Br.tocsr()
        Cpp = sp.csr_matrix(([1.0], ([self.p_pin], [self.p_pin])), shape=(self.np_, self.np_))
        K = sp.bmat([[Ar, Br], [Br.T, Cpp]], format="csr")
        K.eliminate_zeros()
        self.K = PETSc.Mat().createAIJ(size=K.shape, csr=(K.indptr, K.indices, K.data),
                                       comm=MPI.COMM_SELF)
        self.K.assemble()
        self.rhs = self.K.createVecLeft()
        self.sol = self.K.createVecRight()

        self.ksp = PETSc.KSP().create(MPI.COMM_SELF)
        self.ksp.setOperators(self.K)
        self.ksp.setOptionsPrefix("peskin_rk2_")
        opts = PETSc.Options()
        defaults = {"ksp_type": "preonly", "pc_type": "lu",
                    "pc_factor_mat_solver_type": "mumps",
                    "ksp_error_if_not_converged": True}
        defaults.update(petsc_options or {})
        for key, val in defaults.items():
            opts[f"peskin_rk2_{key}"] = val
        self.ksp.setFromOptions()
        for key in defaults:
            del opts[f"peskin_rk2_{key}"]

        self._b = fem.petsc.create_vector(self.V)
        self._area = fem.assemble_scalar(form(Constant(mesh, PETSc.ScalarType(1.0)) * dx))
        self._pmean = {id(self.p_): form(self.p_ * dx), id(self.p_h): form(self.p_h * dx)}

    # ------------------------------------------------------------------
    def _solve(self, L, u_out, p_out, p_scale):
        with self._b.localForm() as loc:
            loc.set(0.0)
        assemble_vector(self._b, L)
        self._b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        r = self.rhs.getArray()
        r[: self.nu] = self.Pu.T @ self._b.array_r
        r[self.nu:] = 0.0
        self.ksp.solve(self.rhs, self.sol)
        x = self.sol.array_r
        u_out.x.array[:] = self.Pu @ x[: self.nu]
        p_out.x.array[:] = p_scale * (self.Pp @ x[self.nu:])
        p_out.x.array[:] -= fem.assemble_scalar(self._pmean[id(p_out)]) / self._area
        u_out.x.scatter_forward()
        p_out.x.scatter_forward()

    def solve_one_step(self):
        """Advance (u_n, f) -> (u_h, u_, p_) and set u_n <- u_."""
        self._solve(self.L1, self.u_h, self.p_h, 1.0)
        self._solve(self.L2, self.u_, self.p_, 0.5)
        self.u_n.x.array[:] = self.u_.x.array

    # ------------------------------------------------------------------
    def periodic_fold(self, fn):
        """Make a nodal (spread) field periodic: add image-node values to the master and copy back.

        ``IBInterpolation.solid_to_fluid`` spreads onto the non-periodic nodal
        grid; contributions landing on right/top nodes are folded onto their
        left/bottom images so that the load is consistent with the periodic space.
        """
        key = id(fn.function_space)
        if not hasattr(self, "_fold"):
            self._fold = {}
        if key not in self._fold:
            m = periodic_master_map(fn.function_space, self.box)
            s = np.flatnonzero(m != np.arange(len(m)))
            self._fold[key] = (s, m[s])
        s, m = self._fold[key]
        bs = fn.function_space.dofmap.index_map_bs
        arr = fn.x.array.reshape(-1, bs)
        np.add.at(arr, m, arr[s])
        arr[s] = arr[m]
        fn.x.scatter_forward()

    def cleanup(self):
        for obj in (self.ksp, self.K, self.rhs, self.sol, self._b):
            obj.destroy()
