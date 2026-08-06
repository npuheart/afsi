"""Matvec-V3 (no dense matrix) quick demo on a SMALL mesh.

Shows:
  (1) the implicit K~ operator matvec == dense K~ (operator correctness);
  (2) outer FGMRES on the FULL A + inner GMRES for [K~ B^T; B s11] converges
      to the CORRECT solution at mu_s=0.1 AND 100 (mu_s-robust, sparse ops);
  (3) inner iteration counts (cost of the nested Krylov).

K~ v = K v - dt A_uW ( M_s^{-1} ( Mfs^T v ) ),  M_s^{-1} via sparse factor.
"""
import os
os.environ["NX"] = "8"; os.environ["NY"] = "8"
os.environ["SOLID_H"] = "0.1"
import numpy as np
import scipy.linalg
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import splu, LinearOperator, gmres
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
Minv_full = np.linalg.inv(Ms)
bc_vel = ib.bc_vel
pin = ib.bc_pin[0] - nu
bc_all = ib.bc_all
W = 0.01 * np.sin(2.0 * np.pi * np.arange(ns) / ns)
rng = np.random.default_rng(0)
print(f"nu={nu} np={np_} ns={ns}  N={N}")

# plain sparse saddle S0 = [K Bt; B s11] (inner preconditioner, factored once)
S0 = np.block([[K.copy(), Bt], [B, s11]])
S0[bc_vel, :] = 0; S0[:, bc_vel] = 0; S0[bc_vel, bc_vel] = 1.0
S0[nu + pin, :] = 0; S0[:, nu + pin] = 0; S0[nu + pin, nu + pin] = 1.0
S0_lu = splu(csr_matrix(S0))

for mu in (0.1, 100.0):
    ib.cfg["mu_s"] = mu
    _, A_uW = ib.assemble_elastic(W, tangent=True)
    A_uW = A_uW.toarray()
    A_s = -A_uW; M_wu = -MfsT

    # (1) operator correctness: K~_op v == (K - dt A_uW M_s^-1 Mfs^T) v
    Ms_lu = splu(csr_matrix(Ms))
    def Kt(v):
        return K @ v - dt * (A_uW @ Ms_lu.solve(MfsT @ v))
    Kt_dense = K - dt * (A_uW @ Minv_full @ MfsT)
    v = rng.standard_normal(nu)
    print(f"  mu_s={mu}: max|K~_op v - K~_dense v| = "
          f"{np.abs(Kt(v) - Kt_dense @ v).max():.2e}")

    # full A + outer FGMRES
    A_full = np.block([[K, Bt, A_s], [B, s11, np.zeros((np_, ns))],
                       [M_wu, np.zeros((ns, np_)), (1.0 / dt) * Ms]])
    Af = A_full.copy()
    Af[bc_all, :] = 0; Af[:, bc_all] = 0; Af[bc_all, bc_all] = 1.0
    b = rng.standard_normal(N); b[bc_all] = 0.0

    def S(up):
        u, p = up[:nu], up[nu:]
        out = np.concatenate([Kt(u) + Bt @ p, B @ u + s11 @ p])
        out[bc_vel] = up[bc_vel]          # homogeneous velocity BC rows: identity
        out[nu + pin] = up[nu + pin]      # pinned pressure dof: identity
        return out
    Sop = LinearOperator((nu + np_, nu + np_), matvec=S, dtype=float)
    S0_prec = LinearOperator((nu + np_, nu + np_),
                             matvec=lambda v: S0_lu.solve(v), dtype=float)

    outer_cnt = [0]; inner_cnt = [0]
    def inner_solve(rhs):
        x, info = gmres(Sop, rhs, M=S0_prec, rtol=1e-4, atol=1e-12,
                        restart=50, maxiter=200,
                        callback=lambda k: inner_cnt.__setitem__(0, inner_cnt[0] + 1))
        return x
    def p_inv(r):
        outer_cnt[0] += 1
        r = np.asarray(r, float)
        zw0 = dt * (Minv_full @ r[nu + np_:])
        yu = r[:nu] - A_s @ zw0
        zu_zp = inner_solve(np.concatenate([yu, r[nu:nu + np_]]))
        zw = zw0 + dt * (Minv_full @ (M_wu @ zu_zp[:nu]))
        return np.concatenate([zu_zp, zw])

    x, info = fgmres(Af, b, M=p_inv, rtol=1e-8, atol=1e-14, restart=50,
                     maxiter=100)
    x_ref = scipy.linalg.solve(Af, b)
    rel = np.linalg.norm(x - x_ref) / np.linalg.norm(x_ref)
    print(f"  mu_s={mu}: outer={outer_cnt[0]} iters (conv={info==0}), "
          f"last inner={inner_cnt[0]}, |err|/|x|={rel:.2e}")
print("DONE")
