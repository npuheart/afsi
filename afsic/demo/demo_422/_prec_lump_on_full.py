"""User's idea: use the LUMPED-structure V3 preconditioner (sparse K~_lump,
diagonal solid solve) as a PRECONDITIONER for the FULL system A.

  A        : full-M_s monolithic (correct, matches deal.II),  A_33 = (1/dt) M_s
  P_lump   : sparse preconditioner built from the LUMPED structure
               saddle [K~_lump B^T; B s11],  K~_lump = K - dt A_uW diag(M_s)^-1 M_fs^T
               solid solve = dt diag(M_s)^-1 (diagonal); L/U coupling diagonal

Outer FGMRES on the FULL A => solution is correct (= A^-1 b).  Since
A - A_lump differs only in the (3,3) block (mu_s-INDEPENDENT), the spectrum
of P_lump^-1 A should be mu_s-independent too (constant iterations) AND the
solution correct.  This would give: correct + sparse + mu_s-robust.

Checks: 1) iterations vs mu_s;  2) solution correctness vs dense LU of A.
"""
import os
os.environ["NX"] = "16"; os.environ["NY"] = "16"
os.environ["SOLID_H"] = "0.05"
import numpy as np
import scipy.linalg
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import splu
from config import make_config
from immersed import ImmersedFEM
from linops import fgmres

cfg = make_config()
ib = ImmersedFEM(cfg)
ib.compute_interaction(np.zeros(ib.n_s))
ib.assemble_mixed_mass()
nu, np_, ns = ib.n_u, ib.n_p, ib.n_s
N = ib.N
dt = cfg["dt"]

K = ib.K.toarray(); Bt = ib.Bt.toarray(); B = ib.B.toarray()
s11 = (cfg["p_stab"] * ib.Mp).toarray()
Ms = ib.M_s.toarray()
MfsT = ib.MfsT_csr.toarray()
dM = np.maximum(Ms.diagonal(), 1e-30)
Minv_lump = 1.0 / dM
bc_vel = ib.bc_vel
pin = ib.bc_pin[0] - nu
bc_all = ib.bc_all


def constrain(A, bc):
    A = A.copy(); A[bc, :] = 0; A[:, bc] = 0; A[bc, bc] = 1.0
    return A


W = 0.01 * np.sin(2.0 * np.pi * np.arange(ns) / ns)
rng = np.random.default_rng(0)

print(f"{'mu_s':>7} | {'iters':>6} {'converged':>10} {'max|err|':>10} "
      f"{'|err|/|x|':>10}")
for mu in (0.1, 1.0, 10.0, 100.0, 1000.0):
    ib.cfg["mu_s"] = mu
    _, A_uW = ib.assemble_elastic(W, tangent=True)
    A_uW = A_uW.toarray()
    A_s = -A_uW; M_wu = -MfsT

    # ---- FULL system A (A_33 = (1/dt) M_s) ----
    A_full = np.block([[K, Bt, A_s], [B, s11, np.zeros((np_, ns))],
                       [M_wu, np.zeros((ns, np_)), (1.0 / dt) * Ms]])
    Af = A_full.copy()
    Af[bc_all, :] = 0; Af[:, bc_all] = 0; Af[bc_all, bc_all] = 1.0
    b = rng.standard_normal(N); b[bc_all] = 0.0

    # ---- sparse lumped preconditioner P_lump^{-1} ----
    Ktl = constrain(K - dt * (A_uW @ np.diag(Minv_lump) @ MfsT), bc_vel)
    S = np.block([[Ktl, Bt], [B, s11]])
    S[nu + pin, :] = 0; S[:, nu + pin] = 0; S[nu + pin, nu + pin] = 1.0
    S_sp = csr_matrix(S)
    lu = splu(S_sp)                     # SPARSE LU of the (sparse) saddle
    Minv = np.diag(dt * Minv_lump)      # diagonal solid inverse
    cnt = [0]
    def p_inv(r):
        cnt[0] += 1
        r = np.asarray(r, float)
        zw0 = Minv @ r[nu + np_:]
        yu = r[:nu] - A_s @ zw0
        zu_zp = lu.solve(np.concatenate([yu, r[nu:nu + np_]]))
        zw = zw0 + Minv @ (M_wu @ zu_zp[:nu])
        return np.concatenate([zu_zp, zw])

    x, info = fgmres(Af, b, M=p_inv, rtol=1e-8, atol=1e-14, restart=50,
                     maxiter=1000)
    # correctness: x must equal A_full^-1 b (dense LU reference)
    x_ref = scipy.linalg.solve(Af, b)
    err = np.linalg.norm(x - x_ref)
    denom = np.linalg.norm(x_ref)
    print(f"{mu:7.1f} | {cnt[0]:>6} {'YES' if info == 0 else 'NO':>10} "
          f"{err:>10.3e} {err / denom:>10.3e}")
print("DONE")
