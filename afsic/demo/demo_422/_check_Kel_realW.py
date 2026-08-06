"""Decisive checks (user's proposal #1 + real-W re-diagnosis).

(A) REAL converged W: run scheme 0 at mu_s=0.1 (stable), take the converged W
    from a step.  Re-do min-eig(K~) at THIS W for several mu_s*dt, comparing
    with the ARTIFICIAL W=0.01 sin used before.  If the artificial W is the
    problem, the real W should give SPD K~ even at mu_s*dt=0.3.
(B) Is the solid-side elastic stiffness K_el PSD?  Assemble the neo-Hookean
    solid-side tangent (ufl.derivative of the solid force) at the artificial
    W AND at the real W; report min-eig of -d f_s/dW (should be >= 0 for a
    physical restoring tangent).
"""
import os
os.environ["NX"] = "16"; os.environ["NY"] = "16"
os.environ["SOLID_H"] = "0.05"
import numpy as np
import ufl
from dolfinx import fem
from config import make_config
from immersed import ImmersedFEM
from linops import MumpsFactor, petsc_to_scipy

cfg = make_config()
ib = ImmersedFEM(cfg)
ib.compute_interaction(np.zeros(ib.n_s))
ib.assemble_mixed_mass()
nu, np_, ns = ib.n_u, ib.n_p, ib.n_s
dt = cfg["dt"]
K = ib.K.toarray(); bc_vel = ib.bc_vel
MfsT = ib.MfsT_csr.toarray()
Ms_lu = MumpsFactor(ib.M_s)
W_art = 0.01 * np.sin(2.0 * np.pi * np.arange(ns) / ns)


def constrain(A):
    A = A.copy(); A[bc_vel, :] = 0; A[:, bc_vel] = 0; A[bc_vel, bc_vel] = 1.0
    return A


def min_eig_Kt(W, mu, dtv):
    ib.cfg["mu_s"] = mu
    _, A_uW = ib.assemble_elastic(W, tangent=True)
    A_uW = A_uW.toarray()
    Kt = constrain(K - dtv * (A_uW @ Ms_lu.solve(np.ascontiguousarray(MfsT))))
    return np.linalg.eigvalsh(0.5 * (Kt + Kt.T)).min()


def solid_Kel_min_eig(W):
    """min-eig of -d f_s/dW (solid-side neo-Hookean stiffness)."""
    Vs = ib.Vs
    u = fem.Function(Vs); u.x.array[:] = W
    dWt = ufl.TrialFunction(Vs); phi = ufl.TestFunction(Vs)
    F = ufl.variable(ufl.Identity(2) + ufl.grad(u))
    mu = ib.cfg["mu_s"]
    P = mu * (F - ufl.inv(ufl.transpose(F)))
    PK = P * ufl.transpose(F)
    f_s = -ufl.inner(PK, ufl.grad(phi)) * ufl.dx
    a_s = -ufl.derivative(f_s, u, dWt)      # -d f_s/dW  (the stiffness K_el)
    Km = fem.petsc.assemble_matrix(fem.form(a_s)); Km.assemble()
    Ke = petsc_to_scipy(Km).toarray()
    return np.linalg.eigvalsh(0.5 * (Ke + Ke.T)).min(), Ke


# ---- (A) real converged W ----
print("== (A) real W from a converged step (mu_s=0.1, dt=0.01, 5 steps) ==")
c = make_config(); c["scheme"] = 0; c["num_steps"] = 5; c["mu_s"] = 0.1
c["frozen"] = 0; c["dt"] = 0.01
ib2 = ImmersedFEM(c)
ib2.compute_interaction(np.zeros(ib2.n_s)); ib2.assemble_mixed_mass()
ib2.X[:ib2.n_u] = ib2._initial_velocity(); ib2.W_prev = np.zeros(ib2.n_s)
for _ in range(5):
    ib2.solve_monolithic()
W_real = ib2.X[ib2.n_u + ib2.n_p:].copy()
print(f"  real |W|={np.linalg.norm(W_real):.3e}  (artificial |W|={np.linalg.norm(W_art):.3e})")

print(f"\n  min-eig(K~):  artificial W vs REAL W  (mu_s*dt scans)")
print(f"  {'mu_s':>6} {'dt':>7} {'mu_s*dt':>8} {'W_art':>12} {'W_real':>12}")
for mu, dtv in [(10, 0.01), (30, 0.01), (100, 0.003), (100, 0.01)]:
    ma = min_eig_Kt(W_art, mu, dtv)
    mr = min_eig_Kt(W_real, mu, dtv)
    print(f"  {mu:6.0f} {dtv:7.4f} {mu*dtv:8.3f} {ma:12.3e} {mr:12.3e}")

# ---- (B) solid-side K_el PSD ----
print("\n== (B) solid-side neo-Hookean stiffness -d f_s/dW : min-eig ==")
for name, Wv in [("artificial", W_art), ("real", W_real)]:
    ib.cfg["mu_s"] = 0.1
    me, Ke = solid_Kel_min_eig(Wv)
    print(f"  {name} W:  min-eig(-d f_s/dW) = {me:+.3e}   "
          f"(max|K_el|={np.abs(Ke).max():.2e}, nnz={np.count_nonzero(Ke)})")
print("DONE")
