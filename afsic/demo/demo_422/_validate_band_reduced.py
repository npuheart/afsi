"""Validate the band-block sparse reduced scheme (V3):
  1. scheme 0 (full 3x3) vs scheme 3 (band-block reduced): identical solution,
     same Newton iteration count (quadratic preserved).
  2. band-block K~ == dense K~ exactly (the scatter is exact).
  3. assembly time: band vs old dense.
"""
import os
os.environ["NX"] = "16"; os.environ["NY"] = "16"
os.environ["SOLID_H"] = "0.05"
import time
import numpy as np
from scipy.sparse import coo_matrix, bmat, csr_matrix
from config import make_config
from immersed import ImmersedFEM
from linops import MumpsFactor


def run_scheme(scheme, steps=3):
    cfg = make_config()
    cfg["scheme"] = scheme
    cfg["num_steps"] = steps
    cfg["mu_s"] = 0.1
    cfg["frozen"] = 0
    ib = ImmersedFEM(cfg)
    ib.X[:ib.n_u] = ib._initial_velocity()
    ib.W_prev = np.zeros(ib.n_s)
    for _ in range(steps):
        if scheme == 0:
            ib.solve_monolithic()
        else:
            ib.solve_monolithic_reduced()
    return ib


print("== 1. scheme 0 vs scheme 3 (band-block): correctness + Newton ==")
ib0 = run_scheme(0)
ib3 = run_scheme(3)
print(f"  ||X_scheme0 - X_scheme3||/||X_scheme3|| = "
      f"{np.linalg.norm(ib0.X - ib3.X) / np.linalg.norm(ib3.X):.2e}")

print("\n== 2. band-block K~ == dense K~ (exact scatter) ==")
ib = ImmersedFEM(make_config())
ib.cfg["mu_s"] = 0.1
ib.compute_interaction(np.zeros(ib.n_s))
ib.assemble_mixed_mass()
nu, np_, ns = ib.n_u, ib.n_p, ib.n_s
dt = ib.cfg["dt"]
W = 0.01 * np.sin(2.0 * np.pi * np.arange(ns) / ns)
f_el, A_uW = ib.assemble_elastic(W, tangent=True)
Ms_lu = MumpsFactor(ib.M_s)
MfsT_d = ib.MfsT_csr.toarray()
band_cols = np.nonzero(np.abs(MfsT_d).sum(axis=0) > 0)[0]
Y_band = Ms_lu.solve(np.ascontiguousarray(MfsT_d[:, band_cols]))
band_rows = np.nonzero(np.abs(A_uW).sum(axis=1) > 0)[0]
C_band = dt * (A_uW[band_rows, :] @ Y_band)
rows = np.repeat(band_rows, len(band_cols))
cols = np.tile(band_cols, len(band_rows))
C_sp = coo_matrix((C_band.ravel(), (rows, cols)), shape=(nu, nu)).tocsr()
Kt_band = ib.K - C_sp                            # K~ = K - C  (exact)
Kt_dense = csr_matrix(ib.K - dt * (A_uW @ Ms_lu.solve(MfsT_d)))
diff = (Kt_band - Kt_dense).toarray()
print(f"  max|K~_band - K~_dense| = {np.abs(diff).max():.2e}  "
      f"(K~_band nnz={Kt_band.nnz}, dense nu^2={nu*nu})")

print("\n== 3. assembly time: band vs old dense (per Newton) ==")
A_uW_d = A_uW.toarray()
t0 = time.perf_counter(); Y_full = Ms_lu.solve(MfsT_d)
t1 = time.perf_counter(); Kt_full = csr_matrix(ib.K - dt * (A_uW @ Y_full))
t2 = time.perf_counter()
Yb = Ms_lu.solve(np.ascontiguousarray(MfsT_d[:, band_cols]))
t3 = time.perf_counter(); Cb = dt * (A_uW[band_rows, :] @ Yb)
t4 = time.perf_counter()
print(f"  dense:  Y {1000*(t1-t0):6.1f}ms + K~ {1000*(t2-t1):6.1f}ms = "
      f"{1000*(t2-t0):.1f}ms")
print(f"  band:   Yb {1000*(t3-t1):6.1f}ms + C_band+K~ {1000*(t4-t3):6.1f}ms = "
      f"{1000*(t4-t1):.1f}ms   (nnz {C_sp.nnz})")

print("\n== 4. Newton convergence rate (scheme 3, one step) ==")
cfg = make_config()
cfg["scheme"] = 3; cfg["num_steps"] = 1; cfg["mu_s"] = 0.1; cfg["frozen"] = 0
ib4 = ImmersedFEM(cfg)
ib4.X[:ib4.n_u] = ib4._initial_velocity()
ib4.W_prev = np.zeros(ib4.n_s)
ib4.solve_monolithic_reduced()
print("  (see [red3] newton |dX| lines above: should square each iteration)")
print("DONE")
