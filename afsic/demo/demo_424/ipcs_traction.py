"""IPCSSolver with the open-boundary pressure traction restored.

Why this exists
---------------
The strong form of the momentum equation is

    rho Du/Dt + grad(p) - mu lap(u) - f = 0

Multiplying by a test function v and integrating over the domain gives

    Int rho Du/Dt.v - Int p div(v) + Int mu grad(u):grad(v) - Int f.v
        + Int_Gamma [ p (v.n) - mu (grad(u).n).v ]  =  0

`IPCSSolver` keeps only the volume terms and drops the whole boundary
integral.  Dropping it is equivalent to imposing the natural condition

    mu grad(u).n = p n          (i.e. zero total traction sigma.n = 0)

on every boundary where no velocity is prescribed.  At the *outlet* the
pressure Dirichlet is p = 0, so zero traction is exactly right.  At the
*inlet* the pressure Dirichlet is p = DP, and zero traction is wrong by DP:
the momentum predictor then tries to build a viscous stress of order DP in a
one-cell layer, produces a large spurious div(u*) there, and because IPCS
*accumulates* the pressure from the projection, that divergence permanently
corrupts the pressure field.

`ChorinSolver` does not have `p` in its momentum predictor at all, so its
natural condition is mu grad(u).n = 0, i.e. exactly the physical condition at
a boundary whose outside only supplies pressure.  That is why Chorin
reproduces the exact pressure and IPCS does not.

The fix
-------
Treat the pressure on the Dirichlet-pressure facets as *known data* and keep
its boundary term explicitly:

    F1 += Int_{Gamma_p} p_D (n.v) ds

Then the only condition still imposed naturally is mu grad(u).n = 0, which is
the correct interface condition for a boundary loaded by an external pressure
p_D.

Usage
-----
    solver = IPCSSolverTraction(V, Q, bcu, bcp, dt, rho, mu, ds_p, p_const)
    solver.p_traction.value = p_inlet(t)      # update each time step
"""
from petsc4py import PETSc

from dolfinx.fem import Constant, form
from dolfinx.fem.petsc import assemble_matrix
from ufl import (FacetNormal, TestFunction, TrialFunction, div, dot, dx,
                 grad, inner, lhs, nabla_grad, rhs)

from afsic import IPCSSolver


class IPCSSolverTraction(IPCSSolver):
    """IPCS + explicit pressure traction on the pressure-Dirichlet facets."""

    def __init__(self, V, Q, bcu, bcp, dt_raw, rho_raw, mu_raw, ds_p,
                 p_traction):
        super().__init__(V, Q, bcu, bcp, dt_raw, rho_raw, mu_raw)

        mesh = V.mesh
        u = TrialFunction(V)
        v = TestFunction(V)
        u_n, u_n1, p_, f = self.u_n, self.u_n1, self.p_, self.f
        k = Constant(mesh, PETSc.ScalarType(dt_raw))
        rho = Constant(mesh, PETSc.ScalarType(rho_raw))
        mu = Constant(mesh, PETSc.ScalarType(mu_raw))
        n = FacetNormal(mesh)

        self.p_traction = p_traction

        F1 = rho / k * dot(u - u_n, v) * dx
        F1 += inner(dot(1.5 * u_n - 0.5 * u_n1, 0.5 * nabla_grad(u + u_n)),
                    v) * dx
        F1 += 0.5 * mu * inner(grad(u + u_n), grad(v)) * dx
        F1 -= dot(p_, div(v)) * dx
        F1 += dot(f, v) * dx
        # the boundary term that the volume form drops, with p = p_D known
        F1 += dot(p_traction * n, v) * ds_p

        self.a1 = form(lhs(F1))
        self.L1 = form(rhs(F1))
        self.A1.zeroEntries()
        assemble_matrix(self.A1, self.a1, bcs=self.bcu)
        self.A1.assemble()
