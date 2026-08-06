"""DIAGNOSTIC (user's proposal): is $(K+C_L)^{-1}(K+C)$ spectrally equivalent?

C  = dt A_uW M_s^-1 Mfs^T   (exact)
C_L = dt A_uW diag(M_s)^-1 Mfs^T   (lumped)
If M_L ~ M_s spectrally equivalent with constants [0.75,1.3] AND the sandwich
structure transfers it, then (K+C_L)^{-1}(K+C) has spectrum in [0.75,1.3]
independent of mu_s -> GMRES ~5 iters, NOT divergence.  Test BOTH signs:
  '+'  : K + C        (user's physical-PSD framing)
  '-'  : K - C        (actual reduced operator used in the solver)
Report: min-eig of K+s*C (SPD check = adjointness), generalized eigenvalues of
(K+s*C_L)^{-1}(K+s*C), and GMRES iteration count (lumped as preconditioner).
"""
import os
os.environ["NX"] = "8"; os.environ["NY"] = "8"
os.environ["SOLID_H"] = "0.1"
import numpy as np
import scipy.linalg
import scipy.sparse.linalg as spla
from config import make_config
from immersed import ImmersedFEM

cfg = make_config()
ib = ImmersedFEM(cfg)
ib.compute_interaction(np.zeros(ib.n_s))
ib.assemble_mixed_mass()
nu, np_, ns = ib.n_u, ib.n_p, ib.n_s
dt = cfg["dt"]
K = ib.K.toarray()
bc = ib.bc_vel


def constrain(A):
    A = A.copy()
    A[bc, :] = 0.0; A[:, bc] = 0.0; A[bc, bc] = 1.0
    return A


Kc = constrain(K)                       # SPD velocity block (BCs applied)
MfsT = ib.MfsT_csr.toarray()
Ms = ib.M_s.toarray()
Minv = np.linalg.inv(Ms)
dM = np.maximum(Ms.diagonal(), 1e-30)
W = 0.01 * np.sin(2.0 * np.pi * np.arange(ns) / ns)

print(f"nu={nu} ns={ns}  (generalized eig of (K+sC_L)^{-1}(K+sC))")
for mu in (0.1, 10.0, 100.0, 1000.0):
    ib.cfg["mu_s"] = mu
    _, A_uW = ib.assemble_elastic(W, tangent=True)
    A_uW = A_uW.toarray()
    C = dt * (A_uW @ Minv @ MfsT)
    C_L = dt * (A_uW @ np.diag(1.0 / dM) @ MfsT)
    row = []
    for s, nm in [(+1.0, "+"), (-1.0, "-")]:
        Kp = constrain(Kc + s * C)
        KpL = constrain(Kc + s * C_L)
        me = np.linalg.eigvalsh(0.5 * (Kp + Kp.T)).min()   # SPD check
        w = scipy.linalg.eig(Kp, KpL, right=False)           # generalized eig
        re = w.real
        # GMRES: solve Kp x = b, preconditioned by KpL^{-1}
        rng = np.random.default_rng(0)
        b = rng.standard_normal(nu)
        Pinv = spla.LinearOperator((nu, nu), matvec=lambda x: scipy.linalg.solve(KpL, x))
        x, info = spla.gmres(Kp, b, M=Pinv, rtol=1e-8, maxiter=200, restart=50)
        row.append(f"({nm}) min-eig(K)={me:+.1e}  eig(KpL^-1 Kp) Re in "
                   f"[{re.min():+.2e},{re.max():+.2e}]  gmres={info}")
    print(f"mu_s={mu:7.1f}:  " + "   ".join(row))
print("DONE")
