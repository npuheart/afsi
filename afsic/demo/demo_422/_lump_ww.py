"""Is lumping M_ww feasible?  Two questions:

A  : full-M_s monolithic (current, matches deal.II):  A_33 = (1/dt) M_s
A' : lumped-M_s monolithic:  A'_33 = (1/dt) diag(M_s), SAME coupling blocks
     (-A_uW, -M_fs^T).  For A' the exact reduced operator is
        K~_lump = K - dt A_uW diag(M_s)^-1 M_fs^T   (SPARSE)
     and the solid block is diagonal => the whole V3-style preconditioner
     is sparse and the solid solve is trivial.

Tests:
  1. mu_s-robustness: FGMRES iterations for (A', V3-sparse) vs mu_s,
     compared to (A, V3-dense) and (A, V0).
  2. Solution fidelity: does A' x' = b ~= A x = b at the benchmark mu_s=0.1?
  3. Sparsity of the lumped added stiffness.
"""
import os
os.environ["NX"] = "16"; os.environ["NY"] = "16"
os.environ["SOLID_H"] = "0.05"
import numpy as np
from config import make_config
from immersed import ImmersedFEM
from linops import fgmres
import scipy.linalg
from scipy import sparse

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
Mfs = MfsT.T
Minv_full = np.linalg.inv(Ms)
dM = np.maximum(Ms.diagonal(), 1e-30)
Minv_lump = 1.0 / dM                      # diag(M_s)^{-1} as vector
bc_vel = ib.bc_vel
pin = ib.bc_pin[0] - nu
bc_all = ib.bc_all


def constrain(A, bc):
    A = A.copy(); A[bc, :] = 0; A[:, bc] = 0; A[bc, bc] = 1.0
    return A


W = 0.01 * np.sin(2.0 * np.pi * np.arange(ns) / ns)


def build(mu, lump_ww):
    ib.cfg["mu_s"] = mu
    _, A_uW = ib.assemble_elastic(W, tangent=True)
    A_uW = A_uW.toarray()
    A_s = -A_uW; M_wu = -MfsT
    if lump_ww:
        Mww33 = (1.0 / dt) * np.diag(dM)
    else:
        Mww33 = (1.0 / dt) * Ms
    Af = np.block([[K, Bt, A_s], [B, s11, np.zeros((np_, ns))],
                   [M_wu, np.zeros((ns, np_)), Mww33]])
    Af[bc_all, :] = 0; Af[:, bc_all] = 0; Af[bc_all, bc_all] = 1.0
    return Af, A_s, M_wu


def make_pinv(Kt, A_s, M_wu, Minv):
    """Minv: full (ns,ns) = dt * M_ww^{-1} (full inverse or diag matrix)."""
    Ktc = constrain(Kt, bc_vel)
    S = np.block([[Ktc, Bt], [B, s11]])
    S[nu + pin, :] = 0; S[:, nu + pin] = 0; S[nu + pin, nu + pin] = 1.0
    lu = scipy.linalg.lu_factor(S)
    def p_inv(r):
        r = np.asarray(r, float)
        zw0 = Minv @ r[nu + np_:]
        yu = r[:nu] - A_s @ zw0
        zu_zp = scipy.linalg.lu_solve(lu, np.concatenate([yu, r[nu:nu + np_]]))
        zw = zw0 + Minv @ (M_wu @ zu_zp[:nu])
        return np.concatenate([zu_zp, zw])
    return p_inv


rng = np.random.default_rng(0)

def count_iters(pinv):
    cnt = [0]
    def wrapped(r):
        cnt[0] += 1
        return pinv(r)
    return wrapped, cnt

print("== 1. mu_s-robustness (FGMRES(50), rtol=1e-8, maxiter=1000; #=iterations) ==")
print(f"{'mu_s':>7} | {'A full + V0':>13} {'A full + V3(dense)':>18} "
      f"{'A_lump + V3(sparse)':>20}")
for mu in (0.1, 1.0, 10.0, 100.0, 1000.0):
    ib.cfg["mu_s"] = mu
    _, A_uW = ib.assemble_elastic(W, tangent=True)
    A_uW = A_uW.toarray()
    A_s = -A_uW; M_wu = -MfsT
    C_full = -dt * (A_uW @ Minv_full @ MfsT)                    # dense
    C_lump = -dt * (A_uW @ np.diag(Minv_lump) @ MfsT)           # sparse-able
    row = []
    # (A full, V0)
    Af, _, _ = build(mu, lump_ww=False)
    b = rng.standard_normal(N); b[bc_all] = 0
    p0, c0 = count_iters(make_pinv(K.copy(), A_s, M_wu, dt * Minv_full))
    x, info = fgmres(Af, b, M=p0, rtol=1e-8, atol=1e-14,
                     restart=50, maxiter=1000)
    row.append(c0[0] if info == 0 else -info)
    # (A full, V3 dense)
    p3, c3 = count_iters(make_pinv(K + C_full, A_s, M_wu, dt * Minv_full))
    x, info = fgmres(Af, b, M=p3, rtol=1e-8, atol=1e-14,
                     restart=50, maxiter=1000)
    row.append(c3[0] if info == 0 else -info)
    # (A_lump, V3 sparse): lumped solid solve, lumped added stiffness
    Al, A_sl, M_wul = build(mu, lump_ww=True)
    b2 = rng.standard_normal(N); b2[bc_all] = 0
    pl, cl = count_iters(make_pinv(K + C_lump, A_sl, M_wul,
                                   dt * np.diag(Minv_lump)))
    x, info = fgmres(Al, b2, M=pl, rtol=1e-8, atol=1e-14,
                     restart=50, maxiter=1000)
    row.append(cl[0] if info == 0 else -info)
    print(f"{mu:7.1f} | {row[0]:>13} {row[1]:>18} {row[2]:>20}")

print("\n== 2. solution fidelity at mu_s=0.1 (same rhs b, dense LU) ==")
Af, _, _ = build(0.1, lump_ww=False)
Al, _, _ = build(0.1, lump_ww=True)
b = rng.standard_normal(N); b[bc_all] = 0
x_full = scipy.linalg.solve(Af, b)
x_lump = scipy.linalg.solve(Al, b)
denom = np.linalg.norm(x_full)
print(f"  ||x_full - x_lump||/||x_full|| = "
      f"{np.linalg.norm(x_full - x_lump) / denom:.3e}   "
      f"(0 => lumped == full; small => good approximation)")

print("\n== 3. sparsity of lumped added stiffness ==")
Cs = sparse.csr_matrix(C_lump)
print(f"  C_lump nnz = {Cs.nnz} / nu^2 = {nu * nu}  "
      f"({100.0 * Cs.nnz / (nu * nu):.2f}% dense)")
print("DONE")
