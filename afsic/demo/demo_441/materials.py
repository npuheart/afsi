"""Material models for the AFSI solid benchmarks (demos 441 / 443).

Two neo-Hookean variants are provided; both return the first Piola-Kirchhoff
stress P = d psi / dF computed symbolically with ufl.diff so no hand-derived
formula can drift out of sync with the energy.

``model="flory"`` (default) -- the paper's stabilised form used for the IB and
BS kernels in Li et al. 2025 (arXiv:2412.15408): modified (volume-preserving)
invariants plus a volumetric energy, parameterised by the shear modulus G and
the numerical bulk modulus kappa_stab with nu = 0.4:

    psi(F) = G/2 (J^(-2/3) I_c - 3) + kappa_stab/2 (J - 1)^2

``model="standard"`` -- the classical compressible neo-Hookean:

    psi(F) = G/2 (I_c - 3) - G ln J + lambda/2 (ln J)^2

    (mu_s = G, lambda_s = kappa_stab).
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
