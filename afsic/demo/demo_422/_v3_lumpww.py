"""V3 with LUMPED M_ww, everything else unchanged (user's question).

Full system A is UNCHANGED (A_33 = (1/dt) M_s, correct solution).
V3 preconditioner modified ONLY in M_ww -> diag(M_s) everywhere it appears in
the p_inv (solid solve zw0 AND the L/U coupling), while K~ = K - dt A_uW M_s^-1
M_fs^T in the saddle stays EXACT (dense).

Theory: X11 = F11(K - A_s M'_ww^{-1} M_wu) + F12 B
           = I + F11 A_s (M_ww^{-1} - M'_ww^{-1}) M_wu,
where M'_ww^{-1} = dt diag(M_s)^{-1} != M_ww^{-1} = dt M_s^{-1}.  The extra term
~ mu_s * (lumping error) grows with mu_s -> should fail at large mu_s.
"""
import os
os.environ["NX"] = "16"; os.environ["NY"] = "16"
os.environ["SOLID_H"] = "0.05"
import numpy as np
import scipy.linalg
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
Minv = np.linalg.inv(Ms)                          # EXACT M_s^{-1} (for K~)
Minv_lump = dt * (1.0 / np.maximum(Ms.diagonal(), 1e-30))   # LUMPED M'_ww^{-1} (vector)
bc_vel = ib.bc_vel
pin = ib.bc_pin[0] - nu
bc_all = ib.bc_all
W = 0.01 * np.sin(2.0 * np.pi * np.arange(ns) / ns)
rng = np.random.default_rng(0)


def constrain(A, bc):
    A = A.copy(); A[bc, :] = 0; A[:, bc] = 0; A[bc, bc] = 1.0
    return A


print(f"nu={nu} np={np_} ns={ns}")
print(f"{'mu_s':>7} | {'iters':>6} {'conv':>5} {'|err|/|x|':>10}")
for mu in (0.1, 1.0, 10.0, 100.0, 1000.0):
    ib.cfg["mu_s"] = mu
    _, A_uW = ib.assemble_elastic(W, tangent=True)
    A_uW = A_uW.toarray()
    A_s = -A_uW; M_wu = -MfsT

    # full A (unchanged, correct)
    A_full = np.block([[K, Bt, A_s], [B, s11, np.zeros((np_, ns))],
                       [M_wu, np.zeros((ns, np_)), (1.0 / dt) * Ms]])
    Af = A_full.copy()
    Af[bc_all, :] = 0; Af[:, bc_all] = 0; Af[bc_all, bc_all] = 1.0
    b = rng.standard_normal(N); b[bc_all] = 0.0

    # saddle with EXACT K~ (dense), LU
    Kt = constrain(K - dt * (A_uW @ Minv @ MfsT), bc_vel)
    S = np.block([[Kt, Bt], [B, s11]])
    S[nu + pin, :] = 0; S[:, nu + pin] = 0; S[nu + pin, nu + pin] = 1.0
    lu = scipy.linalg.lu_factor(S)

    # p_inv with LUMPED M_ww (solid solve + L/U coupling), K~ exact
    Mw = np.diag(Minv_lump)          # dt diag(M_s)^{-1}
    cnt = [0]
    def p_inv(r):
        cnt[0] += 1
        r = np.asarray(r, float)
        zw0 = Mw @ r[nu + np_:]                  # LUMPED solid solve
        yu = r[:nu] - A_s @ zw0                  # L^{-1} (lumped coupling)
        zu_zp = scipy.linalg.lu_solve(lu, np.concatenate([yu, r[nu:nu + np_]]))
        zw = zw0 + Mw @ (M_wu @ zu_zp[:nu])      # U^{-1} (lumped coupling)
        return np.concatenate([zu_zp, zw])

    x, info = fgmres(Af, b, M=p_inv, rtol=1e-8, atol=1e-14, restart=50,
                     maxiter=500)
    x_ref = scipy.linalg.solve(Af, b)
    rel = np.linalg.norm(x - x_ref) / np.linalg.norm(x_ref)
    print(f"{mu:7.1f} | {cnt[0]:>6} {'YES' if info == 0 else 'NO':>5} "
          f"{rel:>10.2e}")
print("DONE")
