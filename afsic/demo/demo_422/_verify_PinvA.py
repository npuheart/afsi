"""Verify EXACTLY what V3 changes vs V0: compute P^{-1}A (apply the block-LDU
preconditioner inverse to each column of A) and its spectrum at mu_s=100.

P^{-1} structure (same L/U coupling for both, differs only in the fluid-saddle
(1,1) block):
  zw0 = M_ww^{-1} r_w
  yu  = r_u - A_s zw0                (L^{-1}; A_s = -A_uW)
  [zu;zp] = S^{-1} [yu; r_p]         S = [K~ B^T; B s11],  K~ = K + C
  zw  = zw0 + M_ww^{-1} M_wu zu      (U^{-1}; M_wu = -M_fs^T)
V0: C = 0 (saddle K).   V3: C = -dt A_uW M_s^{-1} M_fs^T (saddle K~).
"""
import os
os.environ["NX"] = "16"; os.environ["NY"] = "16"
os.environ["SOLID_H"] = "0.05"
import numpy as np
from config import make_config
from immersed import ImmersedFEM
import scipy.linalg

cfg = make_config()
ib = ImmersedFEM(cfg)
ib.compute_interaction(np.zeros(ib.n_s))
ib.assemble_mixed_mass()
nu, np_, ns = ib.n_u, ib.n_p, ib.n_s
N = ib.N
dt = cfg["dt"]
mu = 100.0
ib.cfg["mu_s"] = mu

K = ib.K.toarray(); Bt = ib.Bt.toarray(); B = ib.B.toarray()
s11 = (cfg["p_stab"] * ib.Mp).toarray()
Ms = ib.M_s.toarray()
MfsT = ib.MfsT_csr.toarray()
Mfs = MfsT.T
Minv = np.linalg.inv(Ms)
bc_vel = ib.bc_vel
pin = ib.bc_pin[0] - nu
bc_all = ib.bc_all

W = 0.01 * np.sin(2.0 * np.pi * np.arange(ns) / ns)
_, A_uW = ib.assemble_elastic(W, tangent=True)
A_uW = A_uW.toarray()
A_s = -A_uW
M_wu = -MfsT

A_full = np.block([[K, Bt, A_s], [B, s11, np.zeros((np_, ns))],
                   [M_wu, np.zeros((ns, np_)), (1.0 / dt) * Ms]])
A = A_full.copy()
A[bc_all, :] = 0; A[:, bc_all] = 0; A[bc_all, bc_all] = 1.0

C_exact = -dt * (A_uW @ Minv @ MfsT)
Mww = (1.0 / dt) * Ms
Mwwi = dt * Minv


def make_pinv(C):
    Kt = constrain(K + C, bc_vel)
    S = np.block([[Kt, Bt], [B, s11]])
    S[nu + pin, :] = 0; S[:, nu + pin] = 0; S[nu + pin, nu + pin] = 1.0
    lu = scipy.linalg.lu_factor(S)
    def p_inv(r):
        r = np.asarray(r, float)
        zw0 = Mwwi @ r[nu + np_:]
        yu = r[:nu] - A_s @ zw0
        zu_zp = scipy.linalg.lu_solve(lu, np.concatenate([yu, r[nu:nu + np_]]))
        zw = zw0 + Mwwi @ (M_wu @ zu_zp[:nu])
        return np.concatenate([zu_zp, zw])
    return p_inv


def constrain(A, bc):
    A = A.copy(); A[bc, :] = 0; A[:, bc] = 0; A[bc, bc] = 1.0
    return A


print(f"mu_s={mu}:  computing P^{-1}A spectra ...")
for name, C in [("V0 (C=0)", np.zeros_like(K)),
                ("V3 (C=-dt A_uW Ms^-1 Mfs^T)", C_exact)]:
    pinv = make_pinv(C)
    PA = np.column_stack([pinv(A[:, j]) for j in range(N)])
    ev = np.linalg.eigvals(PA)
    re, im = ev.real, ev.imag
    far = np.abs(ev - 1.0)
    # block-level residuals of P^{-1}A
    X11 = PA[:nu, :nu]; X31 = PA[nu + np_:, :nu]
    err11 = np.abs(X11 - np.eye(nu)).max()
    print(f"  {name}: Re in [{re.min():+.3f},{re.max():+.3f}]  "
          f"max|Im|={np.abs(im).max():.3f}  "
          f"max|lambda-1|={far.max():.3f}  "
          f"mean|lambda-1|={far.mean():.3f}  "
          f"max|X11-I|={err11:.3e}")
    if "V3" in name:
        # theory: X_31 should be 2*M_ww^{-1}M_wu (mu_s-independent residual)
        X31_pred = 2.0 * (Mwwi @ M_wu)
        print(f"    max|X31 - 2 Mww^-1 M_wu| = "
              f"{np.abs(X31 - X31_pred).max():.3e}   "
              f"(|2 Mww^-1 M_wu|max={np.abs(X31_pred).max():.3e})")
print("DONE")
