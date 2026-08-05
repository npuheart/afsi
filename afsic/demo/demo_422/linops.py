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
