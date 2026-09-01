r"""Material model for the static annular solid benchmark.

The paper prescribes the 2nd Piola-Kirchhoff stress

    S^s = mu_s \hat{e}_theta \otimes \hat{e}_theta,

i.e. stiffness only in the circumferential (fiber) direction.  In a total
Lagrangian formulation the first Piola-Kirchhoff stress is

    P = F S,

where F = grad(solid_coords) and \hat{e}_theta is evaluated in the reference
configuration.
"""
import ufl

__all__ = ["CircumferentialMaterial"]


class CircumferentialMaterial:
    def __init__(self, mu_s=1.0, center=(0.5, 0.5)):
        self.mu_s = mu_s
        self.center = ufl.as_vector(center)

    def second_piola_kirchhoff_stress(self, domain):
        X = ufl.SpatialCoordinate(domain)
        rvec = X - self.center
        r = ufl.sqrt(ufl.dot(rvec, rvec))
        e_theta = ufl.as_vector((-rvec[1] / r, rvec[0] / r))
        return self.mu_s * ufl.outer(e_theta, e_theta)

    def first_piola_kirchhoff_stress_v1(self, domain, coords):
        F = ufl.grad(coords)
        S = self.second_piola_kirchhoff_stress(domain)
        return F * S
