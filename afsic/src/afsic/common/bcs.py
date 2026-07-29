"""
常用速度边界条件 — 消除演示中 ~20 处重复的 Velocity 类定义。

用法:
    from afsic.common import UpVelocity2D, UpVelocity3D, TurekInlet, SinusoidalInlet

    inlet = TurekInlet(Um=1.5, Ly=1.0)
    inlet.update(t)
    u_inlet.interpolate(inlet)
"""

import numpy as np
from petsc4py import PETSc


class UpVelocity2D:
    """2D 常量均匀速度: u = (1, 0)（方腔驱动上壁）。"""

    def __init__(self):
        self.t = 0.0

    def __call__(self, x):
        gdim = 2
        values = np.zeros((gdim, x.shape[1]), dtype=PETSc.ScalarType)
        values[0] = 1.0
        values[1] = 0.0
        return values


class UpVelocity3D:
    """3D 常量均匀速度: u = (1, 0, 0)。"""

    def __init__(self):
        self.t = 0.0

    def __call__(self, x):
        gdim = 3
        values = np.zeros((gdim, x.shape[1]), dtype=PETSc.ScalarType)
        values[0] = 1.0
        values[1] = 0.0
        values[2] = 0.0
        return values


class TurekInlet:
    """
    Turek 抛物线型入口速度剖面 + 余弦斜坡。

    u(y) = 1.5 * Um * y * (H - y) / (H/2)^2

    Parameters
    ----------
    Um : float
        最大入口速度。
    Ly : float
        入口高度。
    t_ramp : float
        斜坡时间（默认 2.0s）。
    """

    def __init__(self, Um=1.5, Ly=1.0, t_ramp=2.0):
        self.t = 0.0
        self.scale = 0.0
        self.Um = Um
        self.Ly = Ly
        self.t_ramp = t_ramp

    def update(self, t):
        self.t = t
        if self.t < self.t_ramp:
            self.scale = self.Um * (1.0 - np.cos(np.pi * self.t / self.t_ramp)) / 2.0
        else:
            self.scale = self.Um

    def __call__(self, x):
        gdim = 2
        values = np.zeros((gdim, x.shape[1]), dtype=PETSc.ScalarType)
        H = self.Ly
        values[0] = 1.5 * self.scale * x[1] * (H - x[1]) / (H / 2.0) ** 2
        return values


class TurekInlet3D:
    """Turek 抛物线型入口（3D 管流）: u(y,z) = Um * 4*y*(H-y)/H^2。"""

    def __init__(self, Um=1.5, Ly=1.0, t_ramp=2.0):
        self.t = 0.0
        self.scale = 0.0
        self.Um = Um
        self.Ly = Ly
        self.t_ramp = t_ramp

    def update(self, t):
        self.t = t
        if self.t < self.t_ramp:
            self.scale = self.Um * (1.0 - np.cos(np.pi * self.t / self.t_ramp)) / 2.0
        else:
            self.scale = self.Um

    def __call__(self, x):
        gdim = 3
        values = np.zeros((gdim, x.shape[1]), dtype=PETSc.ScalarType)
        H = self.Ly
        values[0] = self.scale * 4.0 * x[1] * (H - x[1]) / (H * H)
        return values


class SinusoidalInlet:
    """
    正弦时间变化 + 抛物线 y 分布的入口。

    u(y, t) = amp * (sin(2πt/T) + offset) * y * (Ly - y) / scale
    """

    def __init__(self, amp=5.0, Ly=1.61, period=1.0, offset=1.1, scale=None):
        self.t = 0.0
        self.amp = amp
        self.Ly = Ly
        self.period = period
        self.offset = offset
        self.scale = scale if scale is not None else (Ly / 2.0) ** 2

    def update(self, t):
        self.t = t

    def __call__(self, x):
        gdim = 2
        values = np.zeros((gdim, x.shape[1]), dtype=PETSc.ScalarType)
        coeff = self.amp * (np.sin(2 * np.pi * self.t / self.period) + self.offset)
        values[0] = coeff * x[1] * (self.Ly - x[1]) / self.scale
        return values


class PipeInlet3D:
    """
    3D 管道径向入口: 仅在 r < R_inner 处有 U_max * sin(2π*freq*t)。

    适用于圆形管道入口的 3D FSI 模拟。
    """

    def __init__(self, center_x=0.0, center_z=0.0, R_inner=1.0,
                 U_max=1.0, freq=1.0):
        self.t = 0.0
        self.cx = center_x
        self.cz = center_z
        self.R = R_inner
        self.U_max = U_max
        self.freq = freq

    def update(self, t):
        self.t = t

    def __call__(self, x):
        gdim = 3
        values = np.zeros((gdim, x.shape[1]), dtype=PETSc.ScalarType)
        r = np.sqrt((x[0] - self.cx) ** 2 + (x[2] - self.cz) ** 2)
        mask = r < self.R
        values[1, mask] = self.U_max * np.sin(2 * np.pi * self.freq * self.t)
        return values
