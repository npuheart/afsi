"""Adjoint-consistency check.

Demo reduced correction (non-adjoint coupling):
    C_demo = -dt A_uW M_s^-1 Mfs^T          (A_uW = direct weak-form spread)
Adjoint-consistent correction (spread = adjoint of interpolation J = M_s^-1 Mfs^T):
    C_adj  = mu_s dt M_fs M_s^-1 K_s0 M_s^-1 M_fs^T     (K_s0 = solid stiffness)

Theory: C_adj is PSD (K_s0, M_s^-1 PSD => sandwich PSD), giving the user's
expected "tilde A ~ rho/dt M + nu K_visc + mu_s dt K_s0" (PLUS sign, elliptic).
If the demo's C_demo is NOT PSD while C_adj IS, the sign problem is caused by
the non-adjoint coupling blocks, not by W being displacement vs velocity.
"""
import os
os.environ["NX"] = "16"; os.environ["NY"] = "16"
os.environ["SOLID_H"] = "0.05"
import numpy as np
import ufl
from dolfinx import fem
from basix.ufl import element
from config import make_config
from immersed import ImmersedFEM
from linops import petsc_to_scipy

cfg = make_config()
ib = ImmersedFEM(cfg)
ib.compute_interaction(np.zeros(ib.n_s))
ib.assemble_mixed_mass()
nu, ns = ib.n_u, ib.n_s

Ms = ib.M_s.toarray()
MfsT = ib.MfsT_csr.toarray()          # (ns, nu) interpolation
Mfs = MfsT.T                          # (nu, ns)
Minv = np.linalg.inv(Ms)

# solid-side elastic stiffness K_s0 (P2 vector on the solid mesh, proxy)
Vs2 = ib.Vs
v = ufl.TrialFunction(Vs2); uu = ufl.TestFunction(Vs2)
a = ufl.inner(ufl.grad(v), ufl.grad(uu)) * ufl.dx
Ks0 = fem.petsc.assemble_matrix(fem.form(a))
Ks0.assemble()
Ks0 = petsc_to_scipy(Ks0).toarray()   # (ns, ns) symmetric PSD
print(f"nu={nu} ns={ns}  min eig K_s0 = {np.linalg.eigvalsh(0.5*(Ks0+Ks0.T)).min():+.3e}")

print(f"{'mu_s':>7} {'demo sym(C) min':>16} {'adjoint sym(C) min':>18} {'demo is PSD':>13}")
for mu in (0.1, 1.0, 10.0, 100.0):
    ib.cfg["mu_s"] = mu
    _, A_uW = ib.assemble_elastic(
        0.01 * np.sin(2.0 * np.pi * np.arange(ns) / ns), tangent=True)
    A_uW = A_uW.toarray()              # (nu, ns)
    C_demo = -cfg["dt"] * (A_uW @ Minv @ MfsT)
    C_adj = mu * cfg["dt"] * (Mfs @ Minv @ Ks0 @ Minv @ MfsT)
    ev_demo = np.linalg.eigvalsh(0.5 * (C_demo + C_demo.T))
    ev_adj = np.linalg.eigvalsh(0.5 * (C_adj + C_adj.T))
    print(f"{mu:7.1f} {ev_demo.min():16.3e} {ev_adj.min():18.3e} "
          f"{'YES' if ev_demo.min() > -1e-12 else 'NO':>13}")
print("DONE")
