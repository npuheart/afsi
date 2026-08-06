"""Is the mu_s=50000 stagnation a CONDITION-NUMBER (machine precision) effect?

Error definitions:
  fwd  = ||x_fgmres - x_lu|| / ||x_lu||     (vs dense-LU reference of the SAME A)
  res  = ||A x_fgmres - b|| / ||b||         (FGMRES final relative residual)
  bwd  = ||A x - b|| / (||A|| ||x||)        (backward error; ~eps is "best possible")
  cond = cond_2(A)                          (condition number of the full monolithic A)

If fwd ~ eps * cond, the stagnation is machine-precision limited, NOT a
preconditioner failure (the LU reference itself is only accurate to ~eps*cond).
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
Minv = np.linalg.inv(Ms)
bc_vel = ib.bc_vel
pin = ib.bc_pin[0] - nu
bc_all = ib.bc_all
W = 0.01 * np.sin(2.0 * np.pi * np.arange(ns) / ns)
rng = np.random.default_rng(0)


def constrain(A, bc):
    A = A.copy(); A[bc, :] = 0; A[:, bc] = 0; A[bc, bc] = 1.0
    return A


eps = np.finfo(float).eps
print(f"nu={nu} np={np_} ns={ns}   eps={eps:.1e}")
print(f"{'mu_s':>9} | {'iters':>6} {'res':>9} {'fwd':>9} {'bwd':>9} "
      f"{'cond':>8} {'eps*cond':>9} {'fwd~eps*cond':>13}")
for mu in (1000.0, 5000.0, 10000.0, 50000.0, 100000.0):
    ib.cfg["mu_s"] = mu
    _, A_uW = ib.assemble_elastic(W, tangent=True)
    A_uW = A_uW.toarray()
    A_s = -A_uW; M_wu = -MfsT
    A_full = np.block([[K, Bt, A_s], [B, s11, np.zeros((np_, ns))],
                       [M_wu, np.zeros((ns, np_)), (1.0 / dt) * Ms]])
    Af = A_full.copy()
    Af[bc_all, :] = 0; Af[:, bc_all] = 0; Af[bc_all, bc_all] = 1.0
    b = rng.standard_normal(N); b[bc_all] = 0.0

    C = -dt * (A_uW @ Minv @ MfsT)
    Kt = constrain(K + C, bc_vel)
    S = np.block([[Kt, Bt], [B, s11]])
    S[nu + pin, :] = 0; S[:, nu + pin] = 0; S[nu + pin, nu + pin] = 1.0
    lu = scipy.linalg.lu_factor(S)
    cnt = [0]
    def p_inv(r):
        cnt[0] += 1
        r = np.asarray(r, float)
        zw0 = dt * (Minv @ r[nu + np_:])
        yu = r[:nu] - A_s @ zw0
        zu_zp = scipy.linalg.lu_solve(lu, np.concatenate([yu, r[nu:nu + np_]]))
        zw = zw0 + dt * (Minv @ (M_wu @ zu_zp[:nu]))
        return np.concatenate([zu_zp, zw])

    x, info = fgmres(Af, b, M=p_inv, rtol=1e-8, atol=1e-14, restart=50,
                     maxiter=500)
    x_ref = scipy.linalg.solve(Af, b)
    fwd = np.linalg.norm(x - x_ref) / np.linalg.norm(x_ref)
    res = np.linalg.norm(Af @ x - b) / np.linalg.norm(b)
    nA = scipy.linalg.norm(Af, 2)
    bwd = np.linalg.norm(Af @ x - b) / (nA * np.linalg.norm(x))
    cond = np.linalg.cond(Af)
    ok = "yes" if fwd < 10 * eps * cond else "NO "
    print(f"{mu:9.0f} | {cnt[0]:>6} {res:>9.1e} {fwd:>9.1e} {bwd:>9.1e} "
          f"{cond:>8.1e} {eps * cond:>9.1e} {ok:>13}")
print("DONE")
