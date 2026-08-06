"""Assembly complexity of C = dt A_uW M_s^-1 Mfs^T.

Breaks down the current (dense) assembly time and demonstrates the two
reductions:
  (a) only the band block (nu_band x nu_band) is nonzero -> assemble just that;
  (b) keep M_s^-1 implicit (sparse factor + RHS solves) instead of dense inv;
  (c) the ultimate: never assemble C at all (matvec), cost = one M_s factor
      + sparse matvecs per use.
"""
import os
os.environ["NX"] = "16"; os.environ["NY"] = "16"
os.environ["SOLID_H"] = "0.05"
import time
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import splu
from config import make_config
from immersed import ImmersedFEM

cfg = make_config()
ib = ImmersedFEM(cfg)
ib.compute_interaction(np.zeros(ib.n_s))
ib.assemble_mixed_mass()
nu, np_, ns = ib.n_u, ib.n_p, ib.n_s
dt = cfg["dt"]

Ms = ib.M_s.toarray()
MfsT = ib.MfsT_csr.toarray()
W = 0.01 * np.sin(2.0 * np.pi * np.arange(ns) / ns)
_, A_uW = ib.assemble_elastic(W, tangent=True)
A_uW = A_uW.toarray()
nnzA = np.count_nonzero(A_uW)
nnzM = np.count_nonzero(MfsT)

def t(fn):
    t0 = time.perf_counter(); r = fn(); return r, time.perf_counter() - t0

# (1) naive dense: M_s^-1 (dense inv) + Y + A_uW@Y
Minv, t_inv = t(lambda: np.linalg.inv(Ms))
Y, t_Y = t(lambda: Minv @ MfsT)
C, t_C = t(lambda: dt * (A_uW @ Y))
print(f"ns={ns} nu={nu} nnz(A_uW)={nnzA} nnz(Mfs^T)={nnzM}")
print(f"[1] dense naive:  M_s^-1 {t_inv*1e3:7.1f}ms  "
      f"Y {t_Y*1e3:6.1f}ms  A_uW@Y {t_C*1e3:6.1f}ms  total {(t_inv+t_Y+t_C)*1e3:.1f}ms")

# (2) band block only: C is nonzero only on (band rows) x (band cols)
nr = np.nonzero(np.abs(C).sum(axis=1) > 0)[0]
nc = np.nonzero(np.abs(C).sum(axis=0) > 0)[0]
Yb, t_Yb = t(lambda: Minv @ MfsT[:, nc])          # only band cols
Cb, t_Cb = t(lambda: dt * (A_uW[nr, :] @ Yb))     # only band rows x cols
print(f"[2] band block ({len(nr)}x{len(nc)}):  Yb {t_Yb*1e3:6.1f}ms + "
      f"A_uWb@Yb {t_Cb*1e3:6.1f}ms = {(t_Yb+t_Cb)*1e3:.1f}ms  "
      f"(memory {len(nr)*len(nc)} vs {nu*nu} = {100*len(nr)*len(nc)/(nu*nu):.2f}%)")

# (3) implicit M_s^-1 via sparse factor (no dense inverse)
Ms_lu, t_lu = t(lambda: splu(csr_matrix(Ms)))     # factor once
Y3, t_Y3 = t(lambda: Ms_lu.solve(MfsT[:, nc]))    # multi-RHS (ns small)
C3, t_C3 = t(lambda: dt * (A_uW[nr, :] @ Y3))
print(f"[3] sparse factor M_s^-1:  factor {t_lu*1e3:6.1f}ms + solve {t_Y3*1e3:6.1f}ms "
      f"+ A_uWb@Y {t_C3*1e3:6.1f}ms = {(t_lu+t_Y3+t_C3)*1e3:.1f}ms")
print(f"    (factor is ONE-TIME, reused across steps; band-block err vs dense: "
      f"{np.abs(Cb - C[np.ix_(nr, nc)]).max():.2e})")

# (4) implicit operator (never assemble): per-matvec cost
def C_matvec(v):
    return dt * (A_uW @ Ms_lu.solve(MfsT @ v))
v = np.random.default_rng(0).standard_normal(nu)
_, t_mv = t(lambda: C_matvec(v))
print(f"[4] implicit matvec:  {t_mv*1e6:7.1f}us/apply  (one-time cost = the "
      f"{t_lu*1e3:.1f}ms M_s factor only)")
print("DONE")
