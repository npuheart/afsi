"""Adjoint nodal coupling, with a divergence-preserving MAC interpolant.

Composite B-spline construction: Gruninger & Griffith,
https://arxiv.org/abs/2408.08280 . Here B3/B4 denote support widths (degrees 2/3).
The derivative identity maps interpolant divergence to interpolated MAC
divergence on complete interior supports. FE geometry/time discretization can
still introduce solid volume errors; truncated wall supports lose this identity.
"""
import numpy as np
from scipy.sparse import coo_matrix


def peskin4(r):
    r = np.abs(np.asarray(r, float))
    a = (3 - 2*r + np.sqrt(np.maximum(0., 1 + 4*r - 4*r*r))) / 8
    b = (5 - 2*r - np.sqrt(np.maximum(0., -7 + 12*r - 4*r*r))) / 8
    return np.where(r < 1, a, np.where(r < 2, b, 0.))


def bspline3(r):
    """Centred quadratic cardinal B-spline (support width three cells)."""
    r = np.abs(np.asarray(r, float))
    return np.where(r < .5, .75-r*r, np.where(r < 1.5, .5*(1.5-r)**2, 0.))


def bspline4(r):
    """Centred cubic cardinal B-spline; B4'(r)=B3(r+.5)-B3(r-.5)."""
    r = np.abs(np.asarray(r, float))
    return np.where(r < 1, (4-6*r*r+3*r**3)/6, np.where(r < 2, (2-r)**3/6, 0.))


class MACCoupling:
    """U = I u, f = I.T b / cell_area, where b is an assembled FE load.

    No FE mass matrix is applied a second time to b. The discrete power identity
    holds exactly, including truncated wall supports. Wall weights are NOT
    renormalized; zero extension models fixed exterior velocities. Hence constant
    reproduction is only guaranteed for supports wholly inside the domain.
    """
    def __init__(self, grid, points, kernel="bspline"):
        if kernel not in ("bspline", "peskin4"):
            raise ValueError("kernel must be bspline or peskin4")
        self.grid = grid
        self.kernel = kernel
        self.update(points)

    def update(self, points):
        g = self.grid
        points = np.asarray(points, float)
        if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
            raise ValueError("points must have shape (n,2) and be finite")
        if np.any(points < 0) or np.any(points > [g.lx, g.ly]):
            raise ValueError("Solid nodes left the fluid domain")
        self.points = points.copy()
        self.Iu = self._matrix((g.ny, g.nx + 1), (0., .5))
        self.Iv = self._matrix((g.ny + 1, g.nx), (.5, 0.))

    def _matrix(self, shape, offset):
        g = self.grid
        scaled = self.points / [g.dx, g.dy] - offset
        base = np.floor(scaled).astype(int)
        rows, cols, values = [], [], []
        for dj in (-1, 0, 1, 2):
            for di in (-1, 0, 1, 2):
                ij = base + [di, dj]
                valid = ((ij[:, 0] >= 0) & (ij[:, 0] < shape[1])
                         & (ij[:, 1] >= 0) & (ij[:, 1] < shape[0]))
                row = np.flatnonzero(valid)
                rx, ry = scaled[row, 0] - ij[row, 0], scaled[row, 1] - ij[row, 1]
                if self.kernel == "peskin4":
                    weight = peskin4(rx) * peskin4(ry)
                elif offset[0] == 0:  # u: cubic in x, quadratic in y
                    weight = bspline4(rx) * bspline3(ry)
                else:  # v: quadratic in x, cubic in y
                    weight = bspline3(rx) * bspline4(ry)
                rows.extend(row)
                cols.extend(ij[row, 1] * shape[1] + ij[row, 0])
                values.extend(weight)
        return coo_matrix((values, (rows, cols)), shape=(len(self.points), np.prod(shape))).tocsr()

    def interpolate(self, u, v):
        g = self.grid
        if np.shape(u) != (g.ny, g.nx+1) or np.shape(v) != (g.ny+1, g.nx):
            raise ValueError("Wrong MAC velocity shape")
        return np.column_stack([self.Iu @ np.asarray(u).ravel(), self.Iv @ np.asarray(v).ravel()])

    def spread(self, load):
        load = np.asarray(load, float)
        if load.shape != self.points.shape or not np.isfinite(load).all():
            raise ValueError("Expected finite assembled nodal load (n,2)")
        g = self.grid
        return ((self.Iu.T @ load[:, 0]).reshape(g.ny, g.nx+1) / (g.dx*g.dy),
                (self.Iv.T @ load[:, 1]).reshape(g.ny+1, g.nx) / (g.dx*g.dy))
