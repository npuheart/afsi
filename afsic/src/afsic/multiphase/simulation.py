"""Complete single-mesh, first-order gas/liquid/elastic-solid time step."""
import numpy as np

from .coupling import MACCoupling
from .fluid import ConservativeMACSolver
from .levelset import solid_distance, reinitialize, correct_volume, liquid_volume


class ThreePhaseSimulation:
    def __init__(self, grid, solid, phi, properties, gravity=(0., -9.81),
                 interface_cells=1.5, reinitialize_every=5, kernel="bspline"):
        if not np.isfinite(interface_cells) or interface_cells <= 0:
            raise ValueError("interface_cells must be finite and positive")
        if int(reinitialize_every) != reinitialize_every or reinitialize_every < 1:
            raise ValueError("reinitialize_every must be a positive integer")
        if not np.isclose(properties.rho_solid, solid.density):
            raise ValueError("Eulerian and solid material densities must agree")
        self.grid, self.solid, self.properties = grid, solid, properties
        self.epsilon = interface_cells * max(grid.dx, grid.dy)
        self.reinitialize_every = int(reinitialize_every)
        self.phi = grid.check_cell(phi, "phi").copy()
        self.points = np.column_stack([c.ravel() for c in grid.coordinates()])
        self.psi = self._solid_distance()
        rho, mu = properties.fields(self.phi, self.psi, self.epsilon)
        self.fluid = ConservativeMACSolver(grid, rho, mu, gravity)
        self.coupling = MACCoupling(grid, solid.positions, kernel=kernel)
        self.liquid_volume0 = liquid_volume(grid, self.phi, self.psi, self.epsilon)
        self.mass0 = float(np.sum(rho) * grid.dx * grid.dy)
        self.step_number = 0
        self.last_volume_shift = 0.

    def _solid_distance(self):
        return solid_distance(self.points, self.solid.boundary_segments()).reshape(self.grid.shape)

    def stable_dt(self):
        # Gravity acceleration and elastic wave estimates supplement the
        # fluid's current-velocity/viscous bound. They do not prove IB stability.
        acceleration = np.linalg.norm(self.fluid.gravity)
        gravity_dt = .2*np.sqrt(min(self.grid.dx, self.grid.dy)/acceleration) if acceleration else np.inf
        return min(self.fluid.stable_dt(), self.solid.stable_dt(), gravity_dt)

    def step(self, dt):
        if not np.isfinite(dt) or dt <= 0 or dt > self.stable_dt() * (1+1e-12):
            raise ValueError("dt exceeds the current fluid/solid/gravity estimate")
        g, fluid = self.grid, self.fluid
        # Both scalar advection and density/momentum transport use u^n.
        phi = self.phi - dt*g.divergence(*g.upwind_flux(self.phi, fluid.u, fluid.v))
        rho_geometry, mu = self.properties.fields(self.phi, self.psi, self.epsilon)
        fluid.mu[:] = mu
        self.coupling.update(self.solid.positions)
        force = self.coupling.spread(self.solid.load())
        fluid.step(dt, force=force, gravity_density=rho_geometry)
        self.solid.advance(self.coupling.interpolate(fluid.u, fluid.v), dt)
        self.coupling.update(self.solid.positions)
        self.psi = self._solid_distance()
        self.step_number += 1
        if self.step_number % self.reinitialize_every == 0:
            phi = reinitialize(g, phi)
        self.phi, self.last_volume_shift = correct_volume(g, phi, self.psi, self.epsilon, self.liquid_volume0)
        # Viscosity is geometric; transported density remains conservative.
        _, fluid.mu[:] = self.properties.fields(self.phi, self.psi, self.epsilon)
        return self.diagnostics()

    def diagnostics(self):
        g, f = self.grid, self.fluid
        rho_geometry, _ = self.properties.fields(self.phi, self.psi, self.epsilon)
        volume = liquid_volume(g, self.phi, self.psi, self.epsilon)
        mass = float(np.sum(f.rho)*g.dx*g.dy)
        ru, rv = g.face_average(f.rho)
        result = dict(step=self.step_number, time=f.time, mass=mass,
                      mass_relative_error=(mass-self.mass0)/self.mass0,
                      liquid_volume=volume, liquid_volume_error=volume-self.liquid_volume0,
                      volume_shift=self.last_volume_shift,
                      rho_min=float(f.rho.min()), rho_max=float(f.rho.max()),
                      density_geometry_relative_l1=float(np.sum(np.abs(f.rho-rho_geometry))/np.sum(rho_geometry)),
                      divergence_max=float(np.abs(g.divergence(f.u, f.v)).max()),
                      speed_max=float(max(np.abs(f.u).max(), np.abs(f.v).max())),
                      kinetic_energy=float(.5*g.dx*g.dy*(np.sum(ru*f.u**2)+np.sum(rv*f.v**2))))
        result.update(self.solid.diagnostics())
        if not np.isfinite(list(result.values())).all():
            raise RuntimeError("Nonfinite simulation diagnostic")
        return result
