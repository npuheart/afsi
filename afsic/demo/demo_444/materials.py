"""Material model for demo_444 (thick-band variant).

Copied from demo_441/materials.py (same repository, GPLv3) so the demo stays
self-contained.  ``model="flory"`` is the stabilised neo-Hookean form used in
Li et al. 2025 (arXiv:2412.15408): modified invariants plus a volumetric
penalty, psi(F) = G/2 (J^(-2/3) tr(F^T F) - 3) + kappa/2 (J - 1)^2.
"""
import ufl

__all__ = ["NeoHookean"]


class NeoHookean:
    def __init__(self, mu_s=83.333, lambda_s=388.889, model="flory"):
        self.mu_s = mu_s
        self.lambda_s = lambda_s
        self.model = model

    def strain_energy(self, F):
        J = ufl.det(F)
        Ic = ufl.tr(F.T * F)
        if self.model == "flory":
            return (0.5 * self.mu_s * (J ** (-2.0 / 3.0) * Ic - 3.0)
                    + 0.5 * self.lambda_s * (J - 1.0) ** 2)
        if self.model == "standard":
            return (0.5 * self.mu_s * (Ic - 3.0)
                    - self.mu_s * ufl.ln(J)
                    + 0.5 * self.lambda_s * ufl.ln(J) ** 2)
        raise ValueError(f"unknown model '{self.model}'")

    def first_piola_kirchhoff_stress_v1(self, domain, coords, p=None):
        F = ufl.variable(ufl.grad(coords))
        return ufl.diff(self.strain_energy(F), F)
