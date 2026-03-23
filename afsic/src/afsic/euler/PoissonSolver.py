from mpi4py import MPI
from petsc4py.PETSc import ScalarType  # type: ignore

import numpy as np

import ufl
from dolfinx import fem, mesh
from dolfinx.fem.petsc import LinearProblem


def define_poisson_problem(u, v, f, g, dx, ds, bc):
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * dx
    L = ufl.inner(f, v) * dx + ufl.inner(g, v) * ds

    return LinearProblem(
        a,
        L,
        bcs=[bc],
        petsc_options_prefix="demo_poisson_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu", "ksp_error_if_not_converged": True},
    )
