"""WHY does the inner solve diverge at large mu_s?  NOT because M_s is hard.

Inner GMRES solves  S x = b,  S = [K~ B^T; B s11],  K~ = K - dt A_uW M_s^-1 Mfs^T,
preconditioned by the PLAIN saddle S0 = [K B^T; B s11].  Convergence depends on
the spectrum of S0^{-1} S = I + S0^{-1}(S - S0), where
    S - S0 = diag( -dt A_uW M_s^-1 Mfs^T , 0 ) ~ mu_s * (stiffness).
M_s is applied EXACTLY (sparse factor) -- it is NOT the issue; the added
stiffness in K~ that the plain-K preconditioner does not capture is.
"""
import os
os.environ["NX"] = "8"; os.environ["NY"] = "8"
os.environ["SOLID_H"] = "0.1"
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

K = ib.K.toarray(); Bt = ib.Bt.toarray(); B = ib.B.toarray()
s11 = (cfg["p_stab"] * ib.Mp).toarray()
Ms = ib.M_s.toarray()
MfsT = ib.MfsT_csr.toarray()
Minv_full = np.linalg.inv(Ms)
bc_vel = ib.bc_vel
pin = ib.bc_pin[0] - nu
W = 0.01 * np.sin(2.0 * np.pi * np.arange(ns) / ns)

# plain constrained saddle S0 (inner preconditioner)
S0 = np.block([[K.copy(), Bt], [B, s11]])
S0[bc_vel, :] = 0; S0[:, bc_vel] = 0; S0[bc_vel, bc_vel] = 1.0
S0[nu + pin, :] = 0; S0[:, nu + pin] = 0; S0[nu + pin, nu + pin] = 1.0
S0_lu = splu(csr_matrix(S0))
S0inv = lambda v: S0_lu.solve(v)

for mu in (0.1, 100.0):
    ib.cfg["mu_s"] = mu
    _, A_uW = ib.assemble_elastic(W, tangent=True)
    A_uW = A_uW.toarray()
    Kt_dense = K - dt * (A_uW @ Minv_full @ MfsT)
    # constrained K~ saddle S
    S = np.block([[Kt_dense.copy(), Bt], [B, s11]])
    S[bc_vel, :] = 0; S[:, bc_vel] = 0; S[bc_vel, bc_vel] = 1.0
    S[nu + pin, :] = 0; S[:, nu + pin] = 0; S[nu + pin, nu + pin] = 1.0
    m = S.shape[0]
    S0invS = np.column_stack([S0inv(S[:, j]) for j in range(m)])
    ev = np.linalg.eigvals(S0invS)
    far = np.abs(ev - 1.0)
    print(f"mu_s={mu:6.1f}: S0^-1 S eig Re in [{ev.real.min():+.3f},{ev.real.max():+.3f}]"
          f"  max|Im|={np.abs(ev.imag).max():.2f}  max|lambda-1|={far.max():.2f} "
          f"mean|lambda-1|={far.mean():.3f}")
print("(M_s handled EXACTLY via sparse factor in K~; the spread is the added stiffness, "
      "which the plain-K preconditioner S0 does not contain)")
print("DONE")
