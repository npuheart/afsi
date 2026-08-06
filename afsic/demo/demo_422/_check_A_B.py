"""User's checks A & B (2026-08-06) -- decisive, no solver changes.

Check A  (partition of unity of the interpolation S = M_s^{-1} M_fs^T):
    S 1_f  =?  (1/rho_s) 1_s    (must be a CONSTANT, per component)
    MfsT 1_f  =?  (1/rho_s) M_s 1_s      (mass conservation)
  If rho_s=1 the constant must be exactly 1.  If S is not constant (e.g. solid
  points dropped outside the mesh) the interpolation loses the partition of
  unity and the spread/interp pair is not adjoint-consistent.

Check A2  (attribute the 34x between A_uW and -M_fs M_s^{-1} K_el):
    A_adj = -M_fs M_s^{-1} K_el   (the "adjoint-consistent" spread, user's S^T K_el)
    column-norm ratios r_j = ||A_uW[:,j]||/||A_adj[:,j]|| :
      * if r_j ~ constant -> a single missing/scaling factor (bug candidate)
      * if r_j varies strongly with j -> structurally different operators
    also report sigma_max ratio and the best-fit alpha residual.

Check B  (genuine conditional instability vs K~ near-singularity artefact):
    sweep c = mu_s*dt/rho in {0.15,0.19,0.21,0.25,0.4,0.8,1.5} at mu_s=100,
    6 steps each, track total energy E and growth factor g_k=E_k/E_{k-1}:
      * genuine instability  : g(c) monotone, crossing 1 smoothly near c*
      * singularity artefact : g(c) SPIKES near c* then RELAXES for larger c
    also report min-eig(K~) per c (constrained) as a cross-check.
"""
import os
os.environ["NX"] = "16"; os.environ["NY"] = "16"; os.environ["SOLID_H"] = "0.05"
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
rho_s = cfg["rho_s"]
MfsT = ib.MfsT_csr.toarray()           # (ns, nu)
M_s = ib.M_s.toarray()                 # (ns, ns), includes rho_s
Ms_lu = MumpsFactor(ib.M_s)
W_art = 0.01 * np.sin(2.0 * np.pi * np.arange(ns) / ns)


def constrain(A):
    A = A.copy(); bc = ib.bc_vel
    A[bc, :] = 0; A[:, bc] = 0; A[bc, bc] = 1.0
    return A


# =====================================================================
print("=" * 70)
print("CHECK A : partition of unity of the interpolation  S = M_s^-1 M_fs^T")
print("=" * 70)
v_one = np.ones(nu)
w = Ms_lu.solve(np.ascontiguousarray(MfsT @ v_one))      # (ns,)
# component 0 and 1 separately.  Fluid/solid dofs are INTERLEAVED
# (node k -> dofs {2k, 2k+1}), so comp-c source = 1 on dof (2k+c).
v0 = np.zeros(nu); v0[0::2] = 1.0
v1 = np.zeros(nu); v1[1::2] = 1.0
w0 = Ms_lu.solve(np.ascontiguousarray(MfsT @ v0))
w1 = Ms_lu.solve(np.ascontiguousarray(MfsT @ v1))
print(f"  rho_s = {rho_s}")
print(f"  w (both comps): min={w.min():.6e} max={w.max():.6e} "
      f"std/mean={w.std()/max(abs(w.mean()),1e-30):.2e}")
print(f"    expected constant = 1/rho_s = {1.0/rho_s:.6f}")
print(f"    max|w - 1/rho_s| = {np.abs(w - 1.0/rho_s).max():.3e}")
# per-component partition of unity: feeding comp-c constant 1 gives
# w = 1 on solid comp-c dofs and 0 on the other comp's dofs
print(f"    comp0 source: max|w0_c0 - 1|={np.abs(w0[0::2]-1).max():.3e}  "
      f"max|w0_c1 - 0|={np.abs(w0[1::2]).max():.3e}")
print(f"    comp1 source: max|w1_c1 - 1|={np.abs(w1[1::2]-1).max():.3e}  "
      f"max|w1_c0 - 0|={np.abs(w1[0::2]).max():.3e}")
# mass conservation  MfsT 1_f  =?  (1/rho_s) M_s 1_s
lhs = MfsT @ v_one
rhs = (1.0 / rho_s) * (M_s @ np.ones(ns))
print(f"  mass conservation: max|MfsT 1_f - (1/rho_s) M_s 1_s| = "
      f"{np.abs(lhs - rhs).max():.3e}")
print(f"    relative (to M_s rowsum) = "
      f"{np.abs(lhs - rhs).max() / np.abs(rhs).max():.3e}")

# =====================================================================
print()
print("=" * 70)
print("CHECK A2 : attribute the 34x  (A_uW vs -M_fs M_s^-1 K_el)")
print("=" * 70)
mu = 0.1
ib.cfg["mu_s"] = mu
_, A_uW = ib.assemble_elastic(W_art, tangent=True)
A_uW = A_uW.toarray()
# solid-side stiffness K_el = -d f_s/dW  (neo-Hookean tangent)
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
# A_adj = -M_fs M_s^{-1} K_el = -(K_el^T (M_s^{-1} M_fs^T)^T)^T = -(K_el Y)^T
Y = Ms_lu.solve(np.ascontiguousarray(MfsT))  # (ns, nu)
A_adj = -(K_el @ Y).T                        # (nu, ns)
# column norms
nA = np.linalg.norm(A_uW, axis=0); nAdj = np.linalg.norm(A_adj, axis=0)
nz = nAdj > 1e-14
r = nA[nz] / nAdj[nz]
print(f"  max|A_uW|={np.abs(A_uW).max():.3e}  max|A_adj|={np.abs(A_adj).max():.3e}")
print(f"  sigma_max: A_uW={np.linalg.norm(A_uW,2):.3e}  A_adj={np.linalg.norm(A_adj,2):.3e}")
print(f"  column-norm ratio r_j=||A_uW[:,j]||/||A_adj[:,j]||  (nz={nz.sum()} cols):")
print(f"    median={np.median(r):.3e}  mean={r.mean():.3e}  "
      f"[min={r.min():.3e}, max={r.max():.3e}]  std/mean={r.std()/r.mean():.2e}")
print(f"    -> ratio {'~ constant (missing global factor!)' if r.std()/r.mean() < 0.3 else 'VARIABLE (structural, not a single factor)'}")
alpha = np.median(r)
resid = np.linalg.norm(A_uW - alpha * A_adj) / np.linalg.norm(A_uW)
print(f"  best-fit alpha={alpha:.3e} (column-median); rel resid ||A_uW - a*A_adj||/||A_uW|| = {resid:.3e}")
# also try rho_s-scaled and det-scaled candidates to pin the factor
print(f"  candidate factors:  h_f/h_s-ish scale check via trace ratio")
print(f"    tr(A_uW^T A_uW)/tr(A_adj^T A_adj) = {np.trace(A_uW.T@A_uW)/np.trace(A_adj.T@A_adj):.3e}  (sqrt={np.sqrt(np.trace(A_uW.T@A_uW)/np.trace(A_adj.T@A_adj)):.3e})")

# =====================================================================
print()
print("=" * 70)
print("CHECK B : genuine instability vs K~ near-singularity artefact")
print("=" * 70)
cs = [0.15, 0.19, 0.21, 0.25, 0.4, 0.8, 1.5]
MU = 100.0
K0 = ib.K.toarray(); bc_vel = ib.bc_vel

# fluid mass for KE_f
us, vs = ufl.TrialFunction(ib.V), ufl.TestFunction(ib.V)
Mf = fem.petsc.assemble_matrix(fem.form(cfg["rho_f"] * ufl.inner(us, vs) * ufl.dx)); Mf.assemble()
M_f = petsc_to_scipy(Mf).toarray()


def PE_s(ib2, W):
    Vs2 = ib2.Vs
    u2 = fem.Function(Vs2); u2.x.array[:] = W
    F2 = ufl.Identity(2) + ufl.grad(u2)
    detF = ufl.det(F2)
    Ws = mu * (0.5 * (ufl.inner(F2, F2) - 2.0) - ufl.ln(detF))
    v = fem.assemble_scalar(fem.form(Ws * ufl.dx(ib2.smsh)))
    return v if np.isfinite(v) else np.inf     # detF<=0 (flipped cell) -> inf


def total_energy(ib2):
    u = ib2.X[:ib2.n_u]; W = ib2.X[ib2.n_u + ib2.n_p:]
    MfsT2 = ib2.MfsT_csr.toarray()
    us_s = Ms_lu.solve(np.ascontiguousarray(MfsT2 @ u))    # interpolated solid velocity
    KE_s = 0.5 * us_s @ (M_s @ us_s)
    KE_f = 0.5 * u @ (M_f @ u)
    e_pe = PE_s(ib2, W)
    return KE_s + KE_f + e_pe, KE_s, KE_f, e_pe


def min_eig_Kt_W0(muv, dtv, MfsT0):
    """min-eig of K~ at the INITIAL (W=0) configuration: format stability bound."""
    ib.cfg["mu_s"] = muv
    _, A = ib.assemble_elastic(np.zeros(ib.n_s), tangent=True); A = A.toarray()
    Kt = constrain(K0 - dtv * (A @ Ms_lu.solve(np.ascontiguousarray(MfsT0))))
    return np.linalg.eigvalsh(0.5 * (Kt + Kt.T)).min()


print(f"  {'c':>6} {'dt':>9} {'min-eig(K~)':>13} {'E0':>10} {'E_end':>10} "
      f"{'g_max':>9} {'E seq (g_k)':>28}")
for c in cs:
    dt = c / MU
    c2 = make_config(); c2["scheme"] = 0; c2["frozen"] = 0
    c2["num_steps"] = 6; c2["mu_s"] = MU; c2["dt"] = dt
    ib2 = ImmersedFEM(c2)
    ib2.compute_interaction(np.zeros(ib2.n_s)); ib2.assemble_mixed_mass()
    ib2.X[:ib2.n_u] = ib2._initial_velocity(); ib2.W_prev = np.zeros(ib2.n_s)
    try:
        E0, *_ = total_energy(ib2)
        Es = [E0]; gs = []
        for _ in range(6):
            ib2.solve_monolithic()
            E, *_ = total_energy(ib2)
            Es.append(E)
            gs.append(E / Es[-2] if Es[-2] > 0 else np.inf)
        mk = min_eig_Kt_W0(MU, dt, ib.MfsT_csr.toarray())
        gseq = " ".join(f"{g:8.2g}" for g in gs)
        print(f"  {c:6.2f} {dt:9.5f} {mk:13.3e} {Es[0]:10.3g} {Es[-1]:10.3g} "
              f"{max(gs):9.2g}  {gseq}")
    except Exception as ex:
        print(f"  {c:6.2f} {dt:9.5f}   (FAILED: {type(ex).__name__}: {str(ex)[:60]})")
