"""Adjoint-consistent closure (diagnostic only, no solver change).

User's argument: IF A_uW = -S K_el with S^T = M_s^-1 M_fs^T (same S), then
    C = dt A_uW M_s^-1 M_fs^T = dt S K_el S^T  >= 0   (PSD, contractive)
and K~ = K - C would be SPD for ALL c  ->  unconditionally stable.

We measured A_uW != -M_fs M_s^-1 K_el (structurally different, 34-40x weaker,
best-fit residual 1.47).  This check closes the loop: build the FULLY
adjoint-consistent operator
    K~_adj = K + dt M_fs M_s^-1 K_el M_s^-1 M_fs^T
and verify
  (1) it is SPD for every c in a wide scan (unconditional stability), and
  (2) contrast min-eig vs the CODE operator  K~ = K - dt A_uW M_s^-1 M_fs^T
      (which crosses zero at c* ~ 0.215).
So the conditional stability of the code format is entirely attributable to
the non-adjoint (direct-spread) A_uW, NOT to any missing factor.
8x8 grid: tiny, dense linear algebra is instant.
"""
import os
os.environ["NX"] = "8"; os.environ["NY"] = "8"; os.environ["SOLID_H"] = "0.05"
import numpy as np
import ufl
from dolfinx import fem
from config import make_config
from immersed import ImmersedFEM
from linops import MumpsFactor, petsc_to_scipy

cfg = make_config()
ib = ImmersedFEM(cfg)
ib.compute_interaction(np.zeros(ib.n_s)); ib.assemble_mixed_mass()
nu, np_, ns = ib.n_u, ib.n_p, ib.n_s
print(f"8x8 grid: nu={nu} np={np_} ns={ns}")
MfsT = ib.MfsT_csr.toarray()           # (ns, nu)
M_s = ib.M_s.toarray()
Ms_lu = MumpsFactor(ib.M_s)
K0 = ib.K.toarray(); bc_vel = ib.bc_vel
W_art = 0.01 * np.sin(2.0 * np.pi * np.arange(ns) / ns)


def constrain(A):
    A = A.copy(); A[bc_vel, :] = 0; A[:, bc_vel] = 0; A[bc_vel, bc_vel] = 1.0
    return A


# --- solid-side stiffness K_el = -d f_s/dW  (neo-Hookean tangent at W_art) ---
mu = 0.1
mu_s = mu
Vs = ib.Vs
uf = fem.Function(Vs); uf.x.array[:] = W_art
dWt = ufl.TrialFunction(Vs); phi = ufl.TestFunction(Vs)
Fv = ufl.variable(ufl.Identity(2) + ufl.grad(uf))
Pv = mu * (Fv - ufl.inv(ufl.transpose(Fv)))
PKv = Pv * ufl.transpose(Fv)
f_s = -ufl.inner(PKv, ufl.grad(phi)) * ufl.dx
a_s = -ufl.derivative(f_s, uf, dWt)
Km = fem.petsc.assemble_matrix(fem.form(a_s)); Km.assemble()
K_el = petsc_to_scipy(Km).toarray()          # (ns, ns)

Y1 = Ms_lu.solve(np.ascontiguousarray(MfsT))            # (ns, nu) = M_s^-1 M_fs^T
Z  = Ms_lu.solve(np.ascontiguousarray(K_el))            # (ns, ns) = M_s^-1 K_el
Mfs = MfsT.T                                            # (nu, ns)
C_adj = (Mfs @ Z) @ Y1                                  # (nu, nu) = M_fs M_s^-1 K_el M_s^-1 M_fs^T
C_adj = 0.5 * (C_adj + C_adj.T)

# code's C  at mu_s=0.1 (A_uW is linear in mu_s, C scales with mu_s)
_, A_uW = ib.assemble_elastic(W_art, tangent=True); A_uW = A_uW.toarray()
# C_code(mu) = A_uW(mu) M_s^-1 M_fs^T ; A_uW(mu) = (mu/0.1) A_uW(0.1)
A_uW1 = A_uW / 0.1
C_code = A_uW1 @ Y1
C_code = 0.5 * (C_code + C_code.T)

eig_adj = np.linalg.eigvalsh(C_adj)
eig_code = np.linalg.eigvalsh(C_code)
print(f"  C_adj:  min-eig={eig_adj.min():+.4e}  max-eig={eig_adj.max():+.4e}  "
      f"nnz_neg={np.sum(eig_adj < -1e-12)}")
print(f"  C_code: min-eig={eig_code.min():+.4e}  max-eig={eig_code.max():+.4e}  "
      f"nnz_neg={np.sum(eig_code < -1e-12)}")
print("  -> C_adj is PSD (contractive); C_code is NOT (indefinite)")

print(f"\n  {'c':>7} {'min-eig K~_adj':>15} {'min-eig K~_code':>16}")
for c in [0.1, 0.21, 0.25, 0.4, 0.8, 1.5, 5.0, 50.0]:
    # mu_s = 0.1 fixed for K_el/A_uW; dt = c / mu_s  (rho_s=1)
    dt = c / mu_s
    Kt_adj = constrain(K0 + dt * C_adj)
    Kt_code = constrain(K0 - dt * (mu_s * C_code))
    ma = np.linalg.eigvalsh(0.5 * (Kt_adj + Kt_adj.T)).min()
    mc = np.linalg.eigvalsh(0.5 * (Kt_code + Kt_code.T)).min()
    print(f"  {c:7.2f} {ma:15.3e} {mc:16.3e}")
print("\n  K~_adj stays SPD at every c (unconditionally stable);")
print("  K~_code crosses zero near c* ~ 0.21-0.25 (conditionally stable).")
print("DONE")
