"""demo_422 linear algebra helpers: PETSc<->scipy, MUMPS factor, affine maps."""
import os
import numpy as np

from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import default_scalar_type
from scipy.sparse import csr_matrix

from config import comm, rank

def petsc_to_scipy(M):
    """Convert an assembled dolfinx PETSc Mat to a scipy csr_matrix."""
    rowptr, cols, vals = M.getValuesCSR()
    return csr_matrix((vals, cols, rowptr), shape=M.getSize())


def scipy_to_petsc(A, comm=comm):
    """Convert a scipy csr matrix to an assembled PETSc AIJ Mat (local rows)."""
    A = A.tocsr()
    n, m = A.shape
    pM = PETSc.Mat().createAIJ(size=(n, m), comm=comm)
    pM.setValuesCSR(A.indptr, A.indices, A.data)
    pM.assemble()
    return pM


class MumpsFactor:
    """Sparse direct LU factor solved with PETSc KSP (preonly + LU), preferring
    the MUMPS package (fallback: SuperLU_DIST).

    Provides the same ``solve(b)`` interface as
    ``scipy.sparse.linalg.splu`` (1-D or 2-D right-hand sides), so the rest of
    the monolithic code is unchanged.  A fresh factorisation is performed once
    per ``MumpsFactor`` instance (i.e. per Newton linear solve, or once for the
    constant solid mass matrix in the reduced scheme)."""

    def __init__(self, A, comm=comm):
        self._comm = comm
        self._A_p = scipy_to_petsc(A, comm)
        self._ksp = PETSc.KSP().create(comm)
        self._ksp.setOperators(self._A_p)
        self._ksp.setType("preonly")
        pc = self._ksp.getPC()
        pc.setType("lu")
        if PETSc.Sys().hasExternalPackage("mumps"):
            pc.setFactorSolverType("mumps")
        else:
            pc.setFactorSolverType("superlu_dist")
        self._ksp.setFromOptions()
        self._ksp.setUp()

    def solve(self, b):
        b = np.ascontiguousarray(b, dtype=default_scalar_type)
        if b.ndim == 1:
            n = b.size
            pb = PETSc.Vec().createMPI(size=n, comm=self._comm)
            pb.array[:] = b
            x = PETSc.Vec().createMPI(size=n, comm=self._comm)
            self._ksp.solve(pb, x)
            return x.array.copy()
        # multi-column right-hand side: solve column by column (reuses factor)
        return np.column_stack([self.solve(b[:, k]) for k in range(b.shape[1])])


class GAMGSolver:
    """PETSc GMRES + GAMG (algebraic multigrid) solve of a scipy csr matrix.

    Used as an APPROXIMATE inverse inside a flexible preconditioner: only a
    loose tolerance (rtol=1e-2 by default) is enforced, because the outer
    Krylov (FGMRES) absorbs the approximation error.  No direct factorisation
    anywhere -- this is the scalable (large/3D) replacement for MUMPS on the
    fluid blocks.  KSP and PETSc vectors are reused between solves."""

    def __init__(self, A, rtol=1e-2, max_it=100, comm=comm):
        self._n = A.shape[0]
        self._ksp = PETSc.KSP().create(comm)
        self._Ap = scipy_to_petsc(A, comm)
        self._ksp.setOperators(self._Ap)
        self._ksp.setType("gmres")
        pc = self._ksp.getPC()
        pc.setType("gamg")
        self._ksp.setFromOptions()
        self._ksp.setTolerances(rtol=rtol, atol=1e-30, max_it=max_it)
        self._ksp.setUp()
        self._pb = PETSc.Vec().createMPI(size=self._n, comm=comm)
        self._x = PETSc.Vec().createMPI(size=self._n, comm=comm)

    def solve(self, b):
        b = np.ascontiguousarray(b, dtype=float)
        self._pb.array[:] = b
        self._ksp.solve(self._pb, self._x)
        return self._x.array.copy()


class IterativeFluidSaddle:
    """Fluid Stokes saddle  F = [[K, B^T],[B, s11]]  solved WITHOUT any direct
    factorisation: outer FGMRES preconditioned block-diagonally by
        P_f^{-1} = diag( K_amg^{-1},  S_p_amg^{-1} )
    where
      * K_amg^{-1}  : GAMG solve of the velocity block K (BCs applied),
      * S_p_amg^{-1}: GAMG solve of the pressure Schur complement
        S_p = B diag(K)^{-1} B^T (+ the s11 stabilisation), the
        Elman/Silvester/Wathen-style approximation of B K^{-1} B^T.

    The inner K / S_p solves are APPROXIMATE (loose rtol), so the preconditioner
    changes between iterations -> the outer solver must be FGMRES (flexible),
    which is exactly the 3D route: no MUMPS anywhere, only AMG + Krylov.
    Provides the same ``solve(b)`` interface as ``MumpsFactor``."""

    def __init__(self, K, B, Bt, s11=None, comm=comm, Mp=None, Lp=None,
                 rho=1.0, eta=0.01, dt=0.01, **kw):
        from scipy.sparse import bmat, diags, csr_matrix
        self.n_u = K.shape[0]
        self.n_p = B.shape[0]
        if s11 is None:
            s11 = csr_matrix((self.n_p, self.n_p))
        self.F = bmat([[K, Bt], [B, s11]], format="csr")
        # velocity block: GAMG solve (K already has the velocity BCs)
        self.ksp_K = GAMGSolver(K, comm=comm)
        # pressure Schur complement:
        #   default : S_p ~ B diag(K)^{-1} B^T + s11 (Elman/Silvester/Wathen)
        #   if Lp given : Cahouet-Chabard  S_p^{-1} ~ (1/eta) M_p^{-1}
        #                 + (rho/dt) L_p^{-1}, robust in both dt and viscosity
        #                 (captures the mu L half that B diag(K)^{-1} B^T misses)
        self._cc = Lp is not None
        if self._cc:
            self._dMp = 1.0 / Mp.diagonal()          # lumped M_p^{-1}
            self._rho, self._eta, self._dt = rho, eta, dt
            Lp_pin = Lp.tolil()
            Lp_pin[0, :] = 0.0
            Lp_pin[:, 0] = 0.0
            Lp_pin[0, 0] = 1.0
            self.ksp_Sp = GAMGSolver(Lp_pin.tocsr(), comm=comm)
        else:
            dKinv = 1.0 / K.diagonal()
            Sp = (B @ diags(dKinv) @ Bt).tocsr() + s11.tocsr()
            self.ksp_Sp = GAMGSolver(Sp, comm=comm)
        self._comm = comm

    def solve(self, b):
        n_u, n_p = self.n_u, self.n_p
        b = np.ascontiguousarray(b, dtype=float)

        if self._cc:
            dMp, rho, eta, dt = self._dMp, self._rho, self._eta, self._dt
            kspS = self.ksp_Sp

            def p_inv(r):
                r = np.asarray(r, dtype=float)
                u = self.ksp_K.solve(r[:n_u])
                p = (1.0 / eta) * (dMp * r[n_u:]) \
                    + (rho / dt) * kspS.solve(r[n_u:])
                return np.concatenate([u, p])
        else:
            def p_inv(r):
                r = np.asarray(r, dtype=float)
                u = self.ksp_K.solve(r[:n_u])
                p = self.ksp_Sp.solve(r[n_u:])
                return np.concatenate([u, p])

        x, info = fgmres(self.F, b, M=p_inv, rtol=1e-8, atol=0.0,
                         restart=50, maxiter=500)
        if info != 0:
            raise RuntimeError(f"fluid saddle FGMRES did not converge (info={info})")
        return x


def affine_jacobian(verts):
    """Jacobian of an affine triangle  J[i][j] = dx_i/dxi_j  (columns = edges).
    verts: (3, 2) vertex coordinates (reference (0,0),(1,0),(0,1))."""
    v0, v1, v2 = verts[0], verts[1], verts[2]
    return np.array([[v1[0] - v0[0], v2[0] - v0[0]],
                     [v1[1] - v0[1], v2[1] - v0[1]]])


def affine_reference_coords(verts, x):
    """Reference coordinates xi of a physical point x in an affine triangle."""
    J = affine_jacobian(verts)
    v0 = verts[0]
    return np.linalg.solve(J, x - v0)


def fgmres(A, b, M=None, x0=None, rtol=1e-8, atol=0.0, restart=50,
           maxiter=None, callback=None):
    """Flexible GMRES (FGMRES) with RIGHT preconditioning.

    Unlike (left-)GMRES, FGMRES stores the PRECONDITIONED vectors z_j = M v_j
    and solves in span{z_1,...,z_m}, so the preconditioner M may VARY from
    iteration to iteration (flexible preconditioning).  This is the right
    choice when the preconditioner itself uses an iterative inner solve (e.g.
    an AMG / Krylov solve of the fluid block), where each application is only
    approximately the same operator.

    A : matvec callable / scipy csr / LinearOperator (square)
    M : matvec callable (the preconditioner, applied to vectors) or None
    Uses Givens rotations to track the residual incrementally (so convergence
    is detected mid-restart, like standard GMRES) and terminates early.
    Returns (x, info); info = 0 converged, >0 iterations exhausted.
    """
    n = b.shape[0]
    b = np.asarray(b, dtype=float)
    if M is None:
        M = lambda v: v
    if x0 is None:
        x0 = np.zeros(n)
    if maxiter is None:
        maxiter = 100 * (n // max(restart, 1) + 1)
    x = np.asarray(x0, dtype=float).copy()
    r = b - A @ x
    beta = float(np.linalg.norm(r))
    normb = float(np.linalg.norm(b))
    tol = max(rtol * (normb if normb > 0 else 1.0), atol)
    if beta <= tol:
        return x, 0
    it = 0
    while beta > tol and it < maxiter:
        m = min(restart, maxiter - it)
        V = np.zeros((n, m + 1))
        Z = np.zeros((n, m))
        H = np.zeros((m + 1, m))
        cs = np.zeros(m)
        sn = np.zeros(m)
        g = np.zeros(m + 1)
        V[:, 0] = r / beta
        g[0] = beta
        happy = 0
        for j in range(m):
            it += 1
            zj = np.asarray(M(V[:, j]), dtype=float)
            Z[:, j] = zj
            w = A @ zj
            # modified Gram-Schmidt + one reorthogonalisation pass
            for i in range(j + 1):
                H[i, j] = V[:, i] @ w
                w = w - H[i, j] * V[:, i]
            for i in range(j + 1):
                h2 = V[:, i] @ w
                H[i, j] += h2
                w = w - h2 * V[:, i]
            hn = float(np.linalg.norm(w))
            H[j + 1, j] = hn
            # apply previous Givens rotations to the new column
            for i in range(j):
                temp = cs[i] * H[i, j] + sn[i] * H[i + 1, j]
                H[i + 1, j] = -sn[i] * H[i, j] + cs[i] * H[i + 1, j]
                H[i, j] = temp
            # new rotation (H[j+1,j] -> 0)
            if hn > 0.0:
                rot = np.hypot(H[j, j], hn)
                cs[j] = H[j, j] / rot
                sn[j] = hn / rot
                H[j, j] = rot
                H[j + 1, j] = 0.0
            else:
                cs[j] = 1.0
                sn[j] = 0.0
            # update the residual vector g
            g[j + 1] = -sn[j] * g[j]
            g[j] = cs[j] * g[j]
            rnorm = abs(g[j + 1])
            happy = j + 1
            if rnorm <= tol:
                break
            if hn == 0.0:            # happy breakdown: exact in this subspace
                break
            V[:, j + 1] = w / hn
        # solve the (rotated, upper-triangular) least squares: H y = g
        y = np.zeros(happy)
        for i in range(happy - 1, -1, -1):
            y[i] = (g[i] - H[i, i + 1:happy] @ y[i + 1:happy]) / H[i, i]
        x = x + Z[:, :happy] @ y
        r = b - A @ x
        beta = float(np.linalg.norm(r))
        if callback is not None:
            callback(it, beta)
    info = 0 if beta <= tol else it
    return x, info
