"""Two signed-distance fields: negative in liquid and in solid, respectively.

Reference: Wei et al., JCP 552 (2026), 114699, sections 2.2 and 4.1-4.2.
The geometry-based reinitialization here is a serial baseline, not their WENO
Hamilton-Jacobi/subcell-fix implementation.
"""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq


def heaviside(phi, epsilon):
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    z = np.clip(np.asarray(phi, dtype=float) / epsilon, -1., 1.)
    return np.clip(.5 * (1 + z + np.sin(np.pi * z) / np.pi), 0., 1.)


@dataclass(frozen=True)
class PhaseProperties:
    rho_gas: float = 1.2
    rho_liquid: float = 1000.
    rho_solid: float = 1100.
    mu_gas: float = 1.8e-5
    mu_liquid: float = 1.e-3
    mu_solid: float = 1.e-3

    def __post_init__(self):
        values = np.array(list(vars(self).values()))
        if not np.isfinite(values).all() or np.any(values[:3] <= 0) or np.any(values[3:] < 0):
            raise ValueError("Densities must be positive; viscosities nonnegative; all finite")

    def fields(self, phi, psi, epsilon):
        gas, outside = heaviside(phi, epsilon), heaviside(psi, epsilon)
        liquid_weight = outside * (1 - gas)
        gas_weight = outside * gas
        solid_weight = 1 - outside
        rho = self.rho_liquid * liquid_weight + self.rho_gas * gas_weight + self.rho_solid * solid_weight
        mu = self.mu_liquid * liquid_weight + self.mu_gas * gas_weight + self.mu_solid * solid_weight
        return rho, mu


def segment_distance(points, segments):
    """Exact nearest distance to piecewise-linear segments, bounded temporary memory."""
    points, segments = np.asarray(points, float), np.asarray(segments, float)
    if points.ndim != 2 or points.shape[1] != 2 or segments.ndim != 3 or segments.shape[1:] != (2, 2):
        raise ValueError("Expected points (n,2) and segments (m,2,2)")
    if not len(segments) or not np.isfinite(points).all() or not np.isfinite(segments).all():
        raise ValueError("Finite points and nonempty finite segments required")
    edge = segments[:, 1] - segments[:, 0]
    length2 = np.sum(edge * edge, axis=1)
    if np.any(length2 <= 0):
        raise ValueError("Degenerate boundary segment")
    result = np.empty(len(points))
    block = max(1, 200000 // len(segments))
    for start in range(0, len(points), block):
        delta = points[start:start + block, None] - segments[None, :, 0]
        t = np.clip(np.sum(delta * edge, axis=2) / length2, 0., 1.)
        result[start:start + block] = np.sqrt(np.min(np.sum((delta - t[..., None] * edge)**2, axis=2), axis=1))
    return result


def solid_distance(points, segments):
    """SDF of closed, non-self-intersecting polygon boundaries, either orientation.

    Odd/even ray crossings support multiple components and holes. The caller
    must supply a closed boundary (e.g. exterior facets of a valid FE mesh).
    """
    points, segments = np.asarray(points, float), np.asarray(segments, float)
    distance = segment_distance(points, segments)
    inside = np.zeros(len(points), dtype=bool)
    x, y = points.T
    for a, b in segments:
        if a[1] == b[1]:
            continue
        crossing = (a[1] > y) != (b[1] > y)
        intersection = a[0] + (y - a[1]) * (b[0] - a[0]) / (b[1] - a[1])
        inside ^= crossing & (x < intersection)
    return np.where(inside, -distance, distance)


def reinitialize(grid, phi):
    """Extract a triangulated zero contour and recompute distances to its segments.

    Edge extension provides ghost cell centres for interfaces meeting walls.
    Zero contours are preserved to the piecewise-linear reconstruction accuracy.
    """
    phi = grid.check_cell(phi, "phi")
    if np.min(phi) >= 0 or np.max(phi) <= 0:
        return phi.copy()
    a = np.pad(phi, 1, mode="edge")
    x = (np.arange(grid.nx + 2) - .5) * grid.dx
    y = (np.arange(grid.ny + 2) - .5) * grid.dy
    segments = []
    # Only visit squares intersected by the interface.
    lo = np.minimum.reduce([a[:-1, :-1], a[1:, :-1], a[:-1, 1:], a[1:, 1:]])
    hi = np.maximum.reduce([a[:-1, :-1], a[1:, :-1], a[:-1, 1:], a[1:, 1:]])
    for j, i in np.argwhere((lo <= 0) & (hi >= 0) & (lo < hi)):
        p = np.array([[x[i], y[j]], [x[i+1], y[j]], [x[i+1], y[j+1]], [x[i], y[j+1]]])
        f = np.array([a[j, i], a[j, i+1], a[j+1, i+1], a[j+1, i]])
        for tri in ((0, 1, 2), (0, 2, 3)):
            cuts = []
            for k, l in zip(tri, (tri[1], tri[2], tri[0])):
                if f[k] == 0:
                    cuts.append(p[k])
                if f[k] * f[l] < 0:
                    cuts.append(p[k] + f[k] / (f[k] - f[l]) * (p[l] - p[k]))
            cuts = np.unique(np.asarray(cuts).reshape(-1, 2), axis=0)
            if len(cuts) == 2 and np.linalg.norm(cuts[0] - cuts[1]) > 1e-14:
                segments.append(cuts)
    if not segments:
        raise RuntimeError("Mixed-sign level set has no reconstructable interface")
    points = np.column_stack([c.ravel() for c in grid.coordinates()])
    d = segment_distance(points, np.asarray(segments)).reshape(grid.shape)
    return np.where(phi < 0, -d, d)


def liquid_volume(grid, phi, psi, epsilon):
    return float(np.sum((1 - heaviside(phi, epsilon)) * heaviside(psi, epsilon)) * grid.dx * grid.dy)


def correct_volume(grid, phi, psi, epsilon, target):
    """Uniform level-set shift enforcing liquid volume outside the current solid.

    For closed tanks only. This scalar correction is not the paper's variational
    Lagrange-multiplier scheme. It conserves volume, not interface shape.
    """
    phi, psi = grid.check_cell(phi, "phi"), grid.check_cell(psi, "psi")
    available = float(np.sum(heaviside(psi, epsilon)) * grid.dx * grid.dy)
    if not np.isfinite(target) or target < 0 or target > available + 1e-13 * grid.lx * grid.ly:
        raise ValueError("Target liquid volume is outside the available fluid volume")
    target = min(target, available)
    bound = float(np.max(np.abs(phi)) + 2 * epsilon)
    if target == 0:
        shift = bound
    elif target == available:
        shift = -bound
    else:
        shift = brentq(lambda s: liquid_volume(grid, phi + s, psi, epsilon) - target,
                       -bound, bound, xtol=1e-13 * min(grid.dx, grid.dy))
    return phi + shift, float(shift)
