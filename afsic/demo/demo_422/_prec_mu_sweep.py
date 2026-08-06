"""mu_s-robustness sweep: which preconditioner keeps FGMRES iterations
constant as mu_s grows?

All variants are block-LDU with EXACT L/U coupling (kept in the preconditioner)
and differ ONLY in the fluid-saddle velocity block K~ (the (1,1) block):

  V0 baseline      : K~ = K                                  (current block-LDU)
  V3 exact-reduced : K~ = K - dt A_uW M_s^-1 M_fs^T          (dense, EXACT Schur)
  V2 adjoint-dense : K~ = K + mu_s dt M_fs M_s^-1 K_s0 M_s^-1 M_fs^T  (dense)
  V1 adjoint-lump  : K~ = K + mu_s dt M_fs diag(M_s)^-1 K_s0 diag(M_s)^-1 M_fs^T (sparse)
  V4 lump-exact    : K~ = K - dt A_uW diag(M_s)^-1 M_fs^T    (sparse, EXACT sign)

Theory: V3 == A exactly => P^{-1}A = I => 1 iteration for all mu_s (proves
mu_s-robustness is achievable).  V1 is the scalable sparse-PSD candidate.
The true reduced operator is indefinite at large mu_s (non-adjoint coupling),
so V2/V1 (adjoint PSD) are approximations -- whether they keep iterations
bounded is the empirical question.
"""
import os
os.environ["NX"] = "16"; os.environ["NY"] = "16"
os.environ["SOLID_H"] = "0.05"
import numpy as np
import ufl
from dolfinx import fem
from config import make_config
from immersed import ImmersedFEM
from linops import fgmres, petsc_to_scipy
import scipy.linalg

cfg = make_config()
ib = ImmersedFEM(cfg)
ib.compute_interaction(np.zeros(ib.n_s))
ib.assemble_mixed_mass()
nu, np_, ns = ib.n_u, ib.n_p, ib.n_s
N = ib.N
dt = cfg["dt"]

# ---- dense blocks ----
K = ib.K.toarray(); Bt = ib.Bt.toarray(); B = ib.B.toarray()
s11 = (cfg["p_stab"] * ib.Mp).toarray()
Ms = ib.M_s.toarray()
MfsT = ib.MfsT_csr.toarray()            # (ns, nu) interpolation
Mfs = MfsT.T                            # (nu, ns)
Minv = np.linalg.inv(Ms)                # exact (ns, ns)
diagMinv = 1.0 / np.maximum(Ms.diagonal(), 1e-30)

# solid-side linear elastic stiffness K_s0 (constant, PSD, no mu_s factor)
Vs = ib.Vs
v = ufl.TrialFunction(Vs); uu = ufl.TestFunction(Vs)
a = ufl.inner(ufl.grad(v), ufl.grad(uu)) * ufl.dx
Ks0m = fem.petsc.assemble_matrix(fem.form(a)); Ks0m.assemble()
Ks0 = petsc_to_scipy(Ks0m).toarray()    # (ns, ns)

# ---- BCs ----
bc_vel = ib.bc_vel
pin = ib.bc_pin[0] - nu                 # first pressure dof in the saddle
bc_all = ib.bc_all
bc_v = bc_vel


def constrain_K(A, bc):
    A = A.copy(); A[bc, :] = 0.0; A[:, bc] = 0.0; A[bc, bc] = 1.0
    return A


W = 0.01 * np.sin(2.0 * np.pi * np.arange(ns) / ns)
rng = np.random.default_rng(0)

print(f"nu={nu} np={np_} ns={ns}")
print(f"{'mu_s':>7} | {'V0_base':>9} {'V3_exactR':>10} {'V2_adj_dense':>13} "
      f"{'V1_adj_lump':>12} {'V4_lump_exact':>14}")
for mu in (0.1, 1.0, 10.0, 100.0, 1000.0):
    ib.cfg["mu_s"] = mu
    _, A_uW = ib.assemble_elastic(W, tangent=True)
    A_uW = A_uW.toarray()               # (nu, ns)
    A_s = -A_uW                         # (1,3) block of the monolithic A
    M_wu = -MfsT                        # (3,1) block

    # ---- monolithic A (3x3), BCs applied (velocity + pressure pin) ----
    A_full = np.block([
        [K, Bt, A_s],
        [B, s11, np.zeros((np_, ns))],
        [M_wu, np.zeros((ns, np_)), (1.0 / dt) * Ms]])
    Af = A_full.copy()
    Af[bc_all, :] = 0.0; Af[:, bc_all] = 0.0; Af[bc_all, bc_all] = 1.0
    b = rng.standard_normal(N); b[bc_all] = 0.0

    # ---- added-stiffness corrections ----
    C_exact = -dt * (A_uW @ Minv @ MfsT)                       # V3 (true Schur)
    C_adj = mu * dt * (Mfs @ Minv @ Ks0 @ Minv @ MfsT)         # V2 (dense PSD)
    C_lump = mu * dt * (Mfs @ np.diag(diagMinv) @ Ks0
                        @ np.diag(diagMinv) @ MfsT)            # V1 (lumped)
    C_lump_exact = -dt * (A_uW @ np.diag(diagMinv) @ MfsT)     # V4 (sparse, exact sign)

    variants = {
        "V0_base": K,
        "V3_exactR": K + C_exact,
        "V2_adj_dense": K + C_adj,
        "V1_adj_lump": K + C_lump,
        "V4_lump_exact": K + C_lump_exact,
    }

    def make_pinv(Kt):
        Ktc = constrain_K(Kt, bc_v)
        S = np.block([[Ktc, Bt], [B, s11]])
        S[nu + pin, :] = 0.0; S[:, nu + pin] = 0.0
        S[nu + pin, nu + pin] = 1.0
        lu = scipy.linalg.lu_factor(S)
        def p_inv(r):
            r = np.asarray(r, dtype=float)
            zw0 = dt * (Minv @ r[nu + np_:])             # M_ww^{-1} r_w
            yu = r[:nu] - A_s @ zw0                      # L^{-1}
            zu_zp = scipy.linalg.lu_solve(lu, np.concatenate(
                [yu, r[nu:nu + np_]]))                   # fluid saddle
            zw = zw0 + dt * (Minv @ (M_wu @ zu_zp[:nu])) # U^{-1}
            return np.concatenate([zu_zp, zw])
        return p_inv

    row = []
    for name, Kt in variants.items():
        x, info = fgmres(Af, b, M=make_pinv(Kt), rtol=1e-8, atol=1e-14,
                         restart=50, maxiter=1000)
        row.append(f"{info:>4}" if info != 0 else "  ok")
    print(f"{mu:7.1f} | {row[0]:>9} {row[1]:>10} {row[2]:>13} {row[3]:>12} "
          f"{row[4]:>14}")

# count of iterations for the converged ones, re-run to report iter counts
print("\niteration counts (converged only):")
print(f"{'mu_s':>7} | {'V0_base':>9} {'V3_exactR':>10} {'V2_adj_dense':>13} "
      f"{'V1_adj_lump':>12} {'V4_lump_exact':>14}")
for mu in (0.1, 1.0, 10.0, 100.0, 1000.0):
    ib.cfg["mu_s"] = mu
    _, A_uW = ib.assemble_elastic(W, tangent=True)
    A_uW = A_uW.toarray()
    A_s = -A_uW; M_wu = -MfsT
    A_full = np.block([[K, Bt, A_s], [B, s11, np.zeros((np_, ns))],
                       [M_wu, np.zeros((ns, np_)), (1.0 / dt) * Ms]])
    Af = A_full.copy(); Af[bc_all, :] = 0; Af[:, bc_all] = 0; Af[bc_all, bc_all] = 1
    b = rng.standard_normal(N); b[bc_all] = 0
    C_exact = -dt * (A_uW @ Minv @ MfsT)
    C_adj = mu * dt * (Mfs @ Minv @ Ks0 @ Minv @ MfsT)
    C_lump = mu * dt * (Mfs @ np.diag(diagMinv) @ Ks0 @ np.diag(diagMinv) @ MfsT)
    C_lump_exact = -dt * (A_uW @ np.diag(diagMinv) @ MfsT)
    variants = {"V0_base": K, "V3_exactR": K + C_exact,
                "V2_adj_dense": K + C_adj, "V1_adj_lump": K + C_lump,
                "V4_lump_exact": K + C_lump_exact}
    iters = []
    for name, Kt in variants.items():
        # count iterations by instrumenting M
        cnt = [0]
        base = make_pinv(Kt)
        def pinv(r):
            cnt[0] += 1
            return base(r)
        x, info = fgmres(Af, b, M=pinv, rtol=1e-8, atol=1e-14, restart=50,
                         maxiter=1000)
        iters.append(cnt[0] if info == 0 else -info)
    print(f"{mu:7.1f} | {iters[0]:>9} {iters[1]:>10} {iters[2]:>13} {iters[3]:>12} "
          f"{iters[4]:>14}")
print("DONE")
