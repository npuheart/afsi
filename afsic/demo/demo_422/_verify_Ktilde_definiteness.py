"""DECISIVE check (user's proposal): K~ from the SOLVER's OWN assembly path.

K~ = K - dt A_uW M_s^-1 Mfs^T  exactly as solve_monolithic_reduced builds it,
at mu_s=100 and mu_s=100000, dt=0.01, same BCs (velocity + pressure pin via
apply_bc2).  Report min-eig of the constrained K~ and |C|/|K|.
Also compare this C with the diagnostic script's C (consistency).
"""
import os
os.environ["NX"] = "16"; os.environ["NY"] = "16"
os.environ["SOLID_H"] = "0.05"
import numpy as np
from scipy.sparse import csr_matrix, bmat
from config import make_config
from immersed import ImmersedFEM
from linops import MumpsFactor

cfg = make_config()
ib = ImmersedFEM(cfg)
ib.compute_interaction(np.zeros(ib.n_s))
ib.assemble_mixed_mass()
nu, np_, ns = ib.n_u, ib.n_p, ib.n_s
dt = cfg["dt"]
K = ib.K.toarray()
MfsT = ib.MfsT_csr.toarray()
Ms = ib.M_s.toarray()
Minv = np.linalg.inv(Ms)                    # exact M_s^{-1}
Ms_lu = MumpsFactor(ib.M_s)                 # solver's factor
W = 0.01 * np.sin(2.0 * np.pi * np.arange(ns) / ns)
bc_vel = ib.bc_vel
pin = ib.bc_pin[0] - nu


def constrain(A, bc):
    A = A.copy(); A[bc, :] = 0; A[:, bc] = 0; A[bc, bc] = 1.0
    return A


print(f"nu={nu} ns={ns} dt={dt}")
for mu in (100.0, 100000.0):
    ib.cfg["mu_s"] = mu
    f_el, A_uW = ib.assemble_elastic(W, tangent=True)
    A_uW = A_uW.toarray()
    # solver's C: -dt A_uW M_s^{-1} Mfs^T  (K~ = K + C_solver)
    Y = Ms_lu.solve(np.ascontiguousarray(MfsT))         # solver's Y (ns,nu)
    C_solver = -dt * (A_uW @ Y)
    # diagnostic's C: +dt A_uW Minv Mfs^T (same matrices, + sign)
    C_diag = dt * (A_uW @ Minv @ MfsT)
    Kt = constrain(K + C_solver, bc_vel)               # solver's K~ + velocity BCs
    me = np.linalg.eigvalsh(0.5 * (Kt + Kt.T)).min()
    nK = np.abs(K).max(); nC = np.abs(C_solver).max()
    # consistency: solver C vs diagnostic C (sign aside)
    d = np.abs(C_solver + C_diag).max()                # differ by overall sign only
    print(f"mu_s={mu:8.0f}: min-eig(K~_solver)={me:+.3e}   "
          f"|C|/|K|={nC/nK:.1e}   "
          f"max|C_solver + C_diag|={d:.2e} (should be ~0 => same matrices)")
print("DONE")
