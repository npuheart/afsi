"""Conservative first-order, variable-density MAC solver for closed 2D tanks.

Cell mass fluxes are averaged to dual-face mass fluxes. Exactly those fluxes
advect face momentum. This preserves constant velocity under density transport;
in particular, no independent interpolation of rho*u*u is used.
Pressure projection uses -div((1/rho) grad p), with a single pressure gauge.
"""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve


class ConservativeMACSolver:
    def __init__(self, grid, rho, mu, gravity=(0., -9.81)):
        self.grid = grid
        self.rho = grid.check_cell(rho, "rho", positive=True).copy()
        self.mu = grid.check_cell(mu, "mu").copy()
        if np.any(self.mu < 0):
            raise ValueError("Negative viscosity")
        self.gravity = np.asarray(gravity, float)
        if self.gravity.shape != (2,) or not np.isfinite(self.gravity).all():
            raise ValueError("gravity must be a finite 2-vector")
        self.u, self.v = grid.zeros()
        self.p = np.zeros(grid.shape)
        self.time = 0.
        self.last_projection_residual = 0.

    def stable_dt(self, cfl=.4):
        if not 0 < cfl <= .5:
            raise ValueError("Use 0 < cfl <= .5 for this explicit scheme")
        g = self.grid
        rate = np.max(np.abs(self.u)) / g.dx + np.max(np.abs(self.v)) / g.dy
        # Conservative bound using the largest viscosity / smallest density,
        # including the full symmetric-gradient stress operator.
        rate += 4 * np.max(self.mu) / np.min(self.rho) * (g.dx**-2 + g.dy**-2)
        return cfl / rate if rate > 0 else np.inf

    def _viscous_force(self):
        g, u, v, mu = self.grid, self.u, self.v, self.mu
        xx = 2 * mu * np.diff(u, axis=1) / g.dx
        yy = 2 * mu * np.diff(v, axis=0) / g.dy
        # Odd ghosts enforce zero tangential velocity at all four walls.
        ug = np.concatenate([-u[:1], u, -u[-1:]], axis=0)
        vg = np.concatenate([-v[:, :1], v, -v[:, -1:]], axis=1)
        mp = np.pad(mu, 1, mode="edge")
        corner_mu = .25 * (mp[:-1, :-1] + mp[1:, :-1] + mp[:-1, 1:] + mp[1:, 1:])
        xy = corner_mu * (np.diff(ug, axis=0) / g.dy + np.diff(vg, axis=1) / g.dx)
        fu, fv = g.zeros()
        fu[:, 1:-1] = np.diff(xx, axis=1) / g.dx + np.diff(xy[:, 1:-1], axis=0) / g.dy
        fv[1:-1] = np.diff(xy[1:-1], axis=1) / g.dx + np.diff(yy, axis=0) / g.dy
        return fu, fv

    def transport(self, dt):
        """Return transported cell density and face momenta, before other forces."""
        g, u, v = self.grid, self.u, self.v
        fx, fy = g.upwind_flux(self.rho, u, v)
        rho = self.rho - dt * g.divergence(fx, fy)
        ru, rv = g.face_average(self.rho)
        # u control volumes: x fluxes at cell centres, y fluxes at corners.
        mx = .5 * (fx[:, :-1] + fx[:, 1:])
        my = .5 * (fy[:, :-1] + fy[:, 1:])
        flux_x = mx * np.where(mx >= 0, u[:, :-1], u[:, 1:])
        ug = np.pad(u[:, 1:-1], ((1, 1), (0, 0)), mode="edge")
        flux_y = my * np.where(my >= 0, ug[:-1], ug[1:])
        pu = ru * u
        pu[:, 1:-1] -= dt * (np.diff(flux_x, axis=1) / g.dx + np.diff(flux_y, axis=0) / g.dy)
        # v control volumes: x fluxes at corners, y fluxes at cell centres.
        mx = .5 * (fx[:-1] + fx[1:])
        my = .5 * (fy[:-1] + fy[1:])
        vg = np.pad(v[1:-1], ((0, 0), (1, 1)), mode="edge")
        flux_x = mx * np.where(mx >= 0, vg[:, :-1], vg[:, 1:])
        flux_y = my * np.where(my >= 0, v[:-1], v[1:])
        pv = rv * v
        pv[1:-1] -= dt * (np.diff(flux_x, axis=1) / g.dx + np.diff(flux_y, axis=0) / g.dy)
        return rho, pu, pv

    def project(self, u, v, rho, dt):
        """Project prescribed tentative velocities; homogeneous normal wall data."""
        g = self.grid
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        rho = g.check_cell(rho, "rho", positive=True)
        u, v = np.array(u, float, copy=True), np.array(v, float, copy=True)
        if u.shape != self.u.shape or v.shape != self.v.shape or not np.isfinite(u).all() or not np.isfinite(v).all():
            raise ValueError("Invalid MAC velocity arrays")
        u[:, [0, -1]], v[[0, -1], :] = 0., 0.
        ru, rv = g.face_average(rho)
        ids = np.arange(g.nx * g.ny).reshape(g.shape)
        a = np.concatenate([ids[:, :-1].ravel(), ids[:-1].ravel()])
        b = np.concatenate([ids[:, 1:].ravel(), ids[1:].ravel()])
        w = np.concatenate([(1 / ru[:, 1:-1] / g.dx**2).ravel(), (1 / rv[1:-1] / g.dy**2).ravel()])
        matrix = coo_matrix((np.concatenate([w, w, -w, -w]),
                             (np.concatenate([a, b, a, b]), np.concatenate([a, b, b, a]))),
                            shape=(ids.size, ids.size)).tocsr()
        rhs = -g.divergence(u, v).ravel() / dt
        pressure = np.zeros(ids.size)
        # Symmetric pin p[0]=0: remove its row and column, not just its row.
        pressure[1:] = spsolve(matrix[1:, 1:], rhs[1:])
        residual = np.max(np.abs(matrix @ pressure - rhs))
        scale = max(1., float(np.max(np.abs(rhs))))
        if not np.isfinite(pressure).all() or residual > 1e-7 * scale:
            raise RuntimeError(f"Pressure solve failed: residual={residual:g}")
        self.last_projection_residual = float(residual)
        p = pressure.reshape(g.shape)
        u[:, 1:-1] -= dt * np.diff(p, axis=1) / (g.dx * ru[:, 1:-1])
        v[1:-1] -= dt * np.diff(p, axis=0) / (g.dy * rv[1:-1])
        return u, v, p

    def step(self, dt, force=None, gravity_density=None):
        """Advance one explicit transport/stress step followed by projection.

        force contains physical force densities on MAC faces. gravity_density
        may be the current geometric three-phase density (paper equation 24).
        Transported density persists across steps and is never silently reset.
        """
        if not np.isfinite(dt) or dt <= 0 or dt > self.stable_dt() * (1 + 1e-12):
            raise ValueError("dt must be positive and satisfy the explicit CFL/viscous bound")
        g = self.grid
        if force is None:
            force = g.zeros()
        if (len(force) != 2 or np.shape(force[0]) != self.u.shape
                or np.shape(force[1]) != self.v.shape
                or not all(np.isfinite(f).all() for f in force)):
            raise ValueError("force must contain finite MAC face arrays")
        self.mu = g.check_cell(self.mu, "mu")
        if np.any(self.mu < 0):
            raise ValueError("Negative viscosity")
        gravity_density = self.rho if gravity_density is None else g.check_cell(gravity_density, "gravity_density", positive=True)
        rho, pu, pv = self.transport(dt)
        if np.min(rho) <= 0 or not np.isfinite(rho).all():
            raise RuntimeError("Density lost positivity; reduce the time step")
        ru, rv = g.face_average(rho)
        gu, gv = g.face_average(gravity_density)
        fu, fv = self._viscous_force()
        u = (pu + dt * (fu + force[0] + gu * self.gravity[0])) / ru
        v = (pv + dt * (fv + force[1] + gv * self.gravity[1])) / rv
        u, v, p = self.project(u, v, rho, dt)
        self.rho, self.u, self.v, self.p = rho, u, v, p
        self.time += dt
