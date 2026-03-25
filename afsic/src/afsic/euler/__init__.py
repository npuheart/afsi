"""Euler (fluid) solvers for incompressible Navier-Stokes equations."""
# from .NSBase import NSBase
from .ChorinSolver import ChorinSolver
from .NSCode1Solver import NSCode1Solver
from .NSCode2Solver import NSCode2Solver
from .PoissonSolver import define_poisson_problem
# from .IPCSSolver import IPCSSolver


__all__ = [
    # "NSBase",
    "ChorinSolver",
    "NSCode1Solver",
    "NSCode2Solver",
    "define_poisson_problem",
]
