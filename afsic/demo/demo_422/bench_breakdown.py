"""Clean per-phase breakdown of demo_422's step cost (current module layout).
Phases: interaction, mfs, elastic, build, residual, MUMPS factor, MUMPS solve.
Runs at 64x64, mu_s=0.1, dt=0.1 (the regime where implicit wins), FROZEN=2."""
import os
import time
import numpy as np
os.environ.setdefault("NX", "64")
os.environ.setdefault("NY", "64")
os.environ.setdefault("SOLID_H", "0.0125")
os.environ.setdefault("STEPS", "10")

from config import make_config
from immersed import ImmersedFEM
from linops import MumpsFactor

cfg = make_config()
cfg["num_steps"] = 10
cfg["dt"] = 0.1
cfg["mu_s"] = 0.1
cfg["frozen"] = 2
s = ImmersedFEM(cfg)
s.X[:s.n_u] = s._initial_velocity()

_orig_init = MumpsFactor.__init__
_orig_solve = MumpsFactor.solve
def _init(self, A, comm=None):
    t0 = time.perf_counter(); _orig_init(self, A, comm)
    _T["factor"] += time.perf_counter() - t0
MumpsFactor.__init__ = _init
def _solve(self, b):
    t0 = time.perf_counter(); r = _orig_solve(self, b)
    _T["solve"] += time.perf_counter() - t0
    return r
MumpsFactor.solve = _solve

_T = {"inter": 0.0, "mfs": 0.0, "elastic": 0.0, "build": 0.0,
      "resid": 0.0, "factor": 0.0, "solve": 0.0}
nnewton = 0
nfactor = 0
W_old = s.X[s.n_u + s.n_p:].copy()
lu_cache = None
for step in range(cfg["num_steps"]):
    t = time.perf_counter(); s.compute_interaction(W_old); _T["inter"] += time.perf_counter() - t
    t = time.perf_counter(); s.assemble_mixed_mass(); _T["mfs"] += time.perf_counter() - t
    s.X[s.n_u + s.n_p:] = 2.0 * W_old - s.W_prev
    # FROZEN=2: reuse the cross-step factor if we have one
    lu = lu_cache
    dX_prev = None
    for it in range(cfg["n_newton_max"]):
        W = s.X[s.n_u + s.n_p:]
        t = time.perf_counter(); f_el, A_uW = s.assemble_elastic(W, tangent=(lu is None)); _T["elastic"] += time.perf_counter() - t
        if lu is None:
            t = time.perf_counter(); A = s.apply_bc(s.build_monolithic(A_uW)); lu = MumpsFactor(A); nfactor += 1; _T["factor"] += time.perf_counter() - t
        t = time.perf_counter(); R_u, R_p, R_W = s._residual(W_old, f_el)
        rhs = np.zeros(s.N); rhs[:s.n_u] = -R_u
        rhs[s.n_u:s.n_u + s.n_p] = -R_p; rhs[s.n_u + s.n_p:] = -R_W
        rhs[s.bc_all] = 0.0; _T["resid"] += time.perf_counter() - t
        dX = lu.solve(rhs); dX[s.bc_all] = 0.0; s.X += dX
        nnewton += 1
        dX_norm = np.linalg.norm(dX)
        if dX_norm < cfg["newton_rtol"] * (1.0 + np.linalg.norm(s.X)):
            break
        # adaptive refactorisation on stagnation (frozen mode only)
        if (it > 0 and dX_prev is not None and dX_norm > 0.5 * dX_prev):
            _, A_uW = s.assemble_elastic(s.X[s.n_u + s.n_p:], tangent=True)
            t = time.perf_counter(); A = s.apply_bc(s.build_monolithic(A_uW)); lu = MumpsFactor(A); nfactor += 1; _T["factor"] += time.perf_counter() - t
        dX_prev = dX_norm
    lu_cache = lu
    s.W_prev = s.X[s.n_u + s.n_p:].copy()

nsteps = cfg["num_steps"]
tot = sum(_T.values())
print(f"{nsteps} steps, {nnewton} Newton linear solves ({nnewton/nsteps:.1f}/step), "
      f"{nfactor} factors ({nfactor/nsteps:.1f}/step), per step:")
for k in ["inter", "mfs", "elastic", "build", "resid", "factor", "solve"]:
    v = _T[k] / nsteps
    print(f"  {k:9s}: {v:8.3f} s/step  ({v/(tot/nsteps)*100:5.1f}%)")
print(f"  {'TOTAL':9s}: {tot/nsteps:8.3f} s/step")
print(f"  per linear solve: factor {_T['factor']/nfactor:.3f} s, "
      f"solve {_T['solve']/nnewton:.4f} s")
