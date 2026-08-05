"""Clean per-phase breakdown of demo_422's step cost at 64x64 (3 steps).
Phases timed separately: interaction, mixed mass, elastic, build, residual,
MUMPS factor (scipy->PETSc + LU), solve."""
import os
import time
import numpy as np
os.environ.setdefault("NX", "64")
os.environ.setdefault("NY", "64")
os.environ.setdefault("SOLID_H", "0.0125")
os.environ.setdefault("STEPS", "3")

from main import make_config, ImmersedFEM
import main as M

cfg = make_config()
cfg["num_steps"] = 3
s = ImmersedFEM(cfg)
s.X[:s.n_u] = s._initial_velocity()

_orig_init = M.MumpsFactor.__init__
_orig_solve = M.MumpsFactor.solve
def _init(self, A, comm=None):
    t0 = time.perf_counter(); _orig_init(self, A, comm)
    _T["factor"] += time.perf_counter() - t0
M.MumpsFactor.__init__ = _init
def _solve(self, b):
    t0 = time.perf_counter(); r = _orig_solve(self, b)
    _T["solve"] += time.perf_counter() - t0
    return r
M.MumpsFactor.solve = _solve

_T = {"inter": 0.0, "mfs": 0.0, "elastic": 0.0, "build": 0.0,
      "resid": 0.0, "factor": 0.0, "solve": 0.0}
nsteps = 0
nnewton = 0
W_old = s.X[s.n_u + s.n_p:].copy()
for step in range(3):
    t = time.perf_counter()
    s.compute_interaction(W_old); _T["inter"] += time.perf_counter() - t
    t = time.perf_counter()
    s.assemble_mixed_mass(); _T["mfs"] += time.perf_counter() - t
    s.X[s.n_u + s.n_p:] = 2.0 * W_old - s.W_prev
    for it in range(cfg["n_newton_max"]):
        W = s.X[s.n_u + s.n_p:]
        t = time.perf_counter()
        f_el, A_uW = s.assemble_elastic(W); _T["elastic"] += time.perf_counter() - t
        t = time.perf_counter()
        A = s.apply_bc(s.build_monolithic(A_uW)); _T["build"] += time.perf_counter() - t
        t = time.perf_counter()
        R_u, R_p, R_W = s._residual(W_old, f_el)
        rhs = np.zeros(s.N); rhs[:s.n_u] = -R_u
        rhs[s.n_u:s.n_u + s.n_p] = -R_p; rhs[s.n_u + s.n_p:] = -R_W
        rhs[s.bc_all] = 0.0; _T["resid"] += time.perf_counter() - t
        lu = M.MumpsFactor(A)
        dX = lu.solve(rhs); dX[s.bc_all] = 0.0; s.X += dX
        nnewton += 1
        if np.linalg.norm(dX) < cfg["newton_rtol"] * (1.0 + np.linalg.norm(s.X)):
            break
    s.W_prev = s.X[s.n_u + s.n_p:].copy()
    nsteps += 1

tot = sum(_T.values())
print(f"{nsteps} steps, {nnewton} Newton linear solves total "
      f"({nnewton / nsteps:.1f}/step), per step:")
for k in ["inter", "mfs", "elastic", "build", "resid", "factor", "solve"]:
    v = _T[k] / nsteps
    print(f"  {k:9s}: {v:8.3f} s/step  ({v / (tot / nsteps) * 100:5.1f}%)")
print(f"  {'TOTAL':9s}: {tot / nsteps:8.3f} s/step")
print(f"\n  per Newton linear solve: factor+LU {_T['factor'] / nnewton:.3f} s, "
      f"solve {_T['solve'] / nnewton:.4f} s")
