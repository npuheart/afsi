"""demo_422 configuration (AFSI convention) + MPI helpers."""
import os
import numpy as np

from mpi4py import MPI
from petsc4py import PETSc

comm = MPI.COMM_WORLD
rank = comm.rank


def _env(key, default):
    return os.environ.get(key, default)


def make_config():
    cfg = {
        "project_name": "demo-422",
        "tag": "monolithic-ibfe",
        # discretisation
        "degree": 2,             # P2 velocity / P2 solid displacement
        "pressure_order": 1,     # P1 pressure (Taylor-Hood)
        "Nx": int(_env("NX", 64)),
        "Ny": int(_env("NY", 64)),
        "solid_h": float(_env("SOLID_H", 0.0125)),   # disk mesh size
        # geometry (cavity is the unit square; paper: R=0.2, C=(0.6,0.5))
        "R": 0.2, "cx": 0.6, "cy": 0.5,
        # fluid (paper: rho=1, eta=0.01, lid U=1)
        "eta_f": 0.01, "rho_f": 1.0, "lid": 1.0,
        # solid: incompressible neo-Hookean  P = mu (F - F^{-T})
        "mu_s": float(_env("MU_S", 0.1)), "rho_s": float(_env("RHO_S", 1.0)),
        # time stepping (paper: dt=1e-2, T=8.1, out every 10 steps)
        "dt": float(_env("DT", 1e-2)),
        "T": float(_env("T", 8.1)),
        "out_every": int(_env("OUT", 10)),
        # Newton
        "n_newton_max": 8,
        "newton_rtol": 1e-9,
        "refactor_iter": 4,            # frozen mode: refactorise if Newton needs
                                       # more than this many corrections (keeps
                                       # the factor fresh at large dt)
        "pin_center": int(_env("PIN", 0)) == 1,   # pin disk centre (quasi-static)
        # solver
        "scheme": int(_env("SCHEME", 0)),         # 0 monolithic | 3 reduced 2x2
        "p_stab": 1e-8,                           # eps*M_p on the (1,1) block
        # quasi-Newton: factorise the monolithic Jacobian ONCE and reuse it for
        # all Newton corrections (only the elastic FORCE is re-evaluated per
        # iteration).  The converged solution is unchanged (frozen Jacobian =
        # modified Newton); refactorise adaptively on stall.  0 = full Newton
        # (factorise every iteration), 1 = per-step frozen, 2 = cross-step
        # frozen (reuse the factor for many steps; refactorise on demand).
        "frozen": int(_env("FROZEN", 2)),
        # linear solver for the monolithic Newton system:
        #   "direct" : sparse direct LU (MUMPS) of the full 3x3 Jacobian
        #              (default; frozen cross-step amortises the factorisation)
        #   "gmres"  : FGMRES(50) with a block-LDU preconditioner (RIGHT
        #              preconditioned): the fluid Stokes block [K Bt; B s11]
        #              and the solid mass M_s are CONSTANT and factorised
        #              once; the coupling blocks (-A_uW, -Mfs^T) of the current
        #              Jacobian are kept in L and U (only the 2nd-order
        #              correction A_uW (dt/M_s) Mfs^T is dropped).  Per
        #              iteration: 1 fluid solve + 2 M_s backsolves + 2 coupling
        #              matvecs.  FGMRES (flexible) stays stable when the fluid
        #              solve itself is iterative (AMG/Krylov, the 3D route);
        #              no monolithic factorisation at all.
        "linear_solver": _env("LINEAR_SOLVER", "direct"),
        # how the fluid Stokes block [K Bt; B s11] is solved inside the
        # block-LDU preconditioner (only used with LINEAR_SOLVER=gmres):
        #   "mumps" : sparse direct LU (default; fast in 2D)
        #   "amg"   : NO direct factorisation -- FGMRES + block-diagonal
        #             preconditioner, velocity block K and the pressure Schur
        #             complement S_p = B diag(K)^{-1} B^T both solved with
        #             PETSc GAMG.  Slower in 2D, but the scalable 3D route:
        #             MUMPS is infeasible in 3D, AMG + Krylov is not.
        "fluid_solver": _env("FLUID_SOLVER", "mumps"),
    }
    cfg["num_steps"] = int(_env("STEPS", int(cfg["T"] / cfg["dt"])))
    out = _env("OUTPUT", "output")
    cfg["output_path"] = out if rank == 0 else None
    cfg["output_path"] = comm.bcast(cfg["output_path"], root=0)
    return cfg
