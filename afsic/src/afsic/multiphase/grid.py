"""Uniform 2D MAC grid; arrays use (y, x) indexing and SI units."""
from dataclasses import dataclass
from numbers import Integral

import numpy as np


@dataclass(frozen=True)
class MACGrid:
    nx: int
    ny: int
    lx: float = 1.0
    ly: float = 1.0

    def __post_init__(self):
        if (isinstance(self.nx, bool) or isinstance(self.ny, bool)
                or not isinstance(self.nx, Integral) or not isinstance(self.ny, Integral)
                or min(self.nx, self.ny) < 2):
            raise ValueError("nx and ny must be integers >= 2")
        if not np.isfinite([self.lx, self.ly]).all() or min(self.lx, self.ly) <= 0:
            raise ValueError("Domain lengths must be finite and positive")

    @property
    def dx(self):
        return self.lx / self.nx

    @property
    def dy(self):
        return self.ly / self.ny

    @property
    def shape(self):
        return (self.ny, self.nx)

    def coordinates(self, location="cell"):
        if location == "cell":
            x, y = (np.arange(self.nx) + .5) * self.dx, (np.arange(self.ny) + .5) * self.dy
        elif location == "u":
            x, y = np.arange(self.nx + 1) * self.dx, (np.arange(self.ny) + .5) * self.dy
        elif location == "v":
            x, y = (np.arange(self.nx) + .5) * self.dx, np.arange(self.ny + 1) * self.dy
        else:
            raise ValueError("location must be cell, u, or v")
        return np.meshgrid(x, y)

    def zeros(self):
        return np.zeros((self.ny, self.nx + 1)), np.zeros((self.ny + 1, self.nx))

    def divergence(self, u, v):
        return np.diff(u, axis=1) / self.dx + np.diff(v, axis=0) / self.dy

    def face_average(self, a):
        x = np.pad(a, ((0, 0), (1, 1)), mode="edge")
        y = np.pad(a, ((1, 1), (0, 0)), mode="edge")
        return .5 * (x[:, :-1] + x[:, 1:]), .5 * (y[:-1] + y[1:])

    def upwind_flux(self, a, u, v):
        x = np.pad(a, ((0, 0), (1, 1)), mode="edge")
        y = np.pad(a, ((1, 1), (0, 0)), mode="edge")
        return (u * np.where(u >= 0, x[:, :-1], x[:, 1:]),
                v * np.where(v >= 0, y[:-1], y[1:]))

    def check_cell(self, a, name, positive=False):
        a = np.asarray(a, dtype=float)
        if a.shape != self.shape or not np.isfinite(a).all():
            raise ValueError(f"{name} must be finite with shape {self.shape}")
        if positive and np.any(a <= 0):
            raise ValueError(f"{name} must be strictly positive")
        return a
