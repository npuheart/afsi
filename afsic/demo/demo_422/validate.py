"""Standalone validation for demo_422 (monolithic IBFE).

Checks
  1. Coupling self-check (built into main.run): projection of (1,0) and (x,y)
     onto the solid is exact to machine precision.
  2. Elastic force and its tangent A_uW by finite differences
       f_el(W=0) == 0 ;  A_uW dW ~= (f_el(W+eps dW) - f_el(W-eps dW)) / 2eps ;
       f_el(rigid translation) ~= 0.
  3. Scheme consistency: SCHEME=0 (full monolithic 3x3) and SCHEME=3 (exact
     Schur-eliminated 2x2) must produce identical solutions after one step.

Run (coarse, ~1 min):
  cd afsic/demo/demo_422
  python validate.py
"""
import os
import numpy as np
from mpi4py import MPI

os.environ.setdefault("NX", "16")
os.environ.setdefault("NY", "16")
os.environ.setdefault("SOLID_H", "0.05")
os.environ.setdefault("STEPS", "0")

from main import ImmersedFEM, make_config

cfg = make_config()
cfg["num_steps"] = 0
cfg["mu_s"] = float(os.environ.get("MU_S", "0.1"))
problem = ImmersedFEM(cfg)


def section(title):
    print(f"\n=== {title} ===")


section("1) coupling self-check (built into ImmersedFEM.verify_coupling)")
problem.verify_coupling()

section("2) elastic force and tangent (finite differences)")
W0 = np.zeros(problem.n_s)
problem.compute_interaction(W0)
problem.assemble_mixed_mass()
f0, A0 = problem.assemble_elastic(W0)
print(f"  f_el(W=0) max: {np.abs(f0).max():.3e}   (should be 0)")

rng = np.random.default_rng(0)
dW = rng.standard_normal(problem.n_s) * 1e-3
eps = 1e-6
fp, _ = problem.assemble_elastic(W0 + eps * dW)
fm, _ = problem.assemble_elastic(W0 - eps * dW)
fd = (fp - fm) / (2 * eps)
Jd = A0 @ dW
err = np.max(np.abs(fd - Jd))
scale = np.max(np.abs(fd))
print(f"  tangent FD: max|A_uW dW - df| = {err:.3e}  "
      f"(rel {err / max(scale, 1e-30):.3e})")

tr = np.zeros(problem.n_s)
for b in range(problem.n_s // 2):
    tr[2 * b] = 0.01
ftr, _ = problem.assemble_elastic(tr)
print(f"  f_el(rigid translation) max: {np.abs(ftr).max():.3e}   (should be ~0)")

section("3) scheme consistency (SCHEME=0 vs SCHEME=3, one step)")
cfg2 = make_config()
cfg2["num_steps"] = 1
cfg2["mu_s"] = cfg["mu_s"]

p0 = ImmersedFEM(cfg2)
p0.X[: p0.n_u] = p0._initial_velocity()
p0.solve_monolithic()
X0 = p0.X.copy()

cfg2["scheme"] = 3
p3 = ImmersedFEM(cfg2)
p3.X[: p3.n_u] = p3._initial_velocity()
p3.solve_monolithic_reduced()
X3 = p3.X.copy()

diff = np.max(np.abs(X0 - X3))
rel = diff / (1.0 + np.max(np.abs(X0)))
print(f"  max|X_scheme0 - X_scheme3| = {diff:.3e}  (rel {rel:.3e})")

section("done")
