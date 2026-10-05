"""Port of ``please_Compute_External_Forcing.m``
(``Example_HeartTube/Insect_Heart``).

Pulsatile inflow for the insect heart tube (``6.0 x 1.5`` box), driven by a
``|sin|`` (always-positive, twice-per-period) target inside a vertical slot:

    Fx = -kStiff (uX - uX_tar),   uX_tar = uMax 5 |sin(2 pi t)| (
        (MP + w/2) - y ) ( (MP - w/2) - y ) / scale,   Fy = 0,

with ``kStiff = 1e4``, ``uMax = 5``, ``w = 0.84875 - 0.65125 = 0.1975``,
``MP = (0.84875 + 0.65125)/2 = 0.75``, ``scale = -w^2/4`` and the penalty
window ``x in [0.29, 0.315]``, ``y in [0.65125, 0.84875]``.  ``|sin|`` means
two inflow pulses per 1 s period, the physiologically-looking driving of the
valved insect heart.
"""
import numpy as np

K_STIFF = 1e4
W = 0.84875 - 0.65125
MID_POINT = (0.84875 + 0.65125) / 2
U_MAX = 5.0
X_MIN, X_MAX = 0.29, 0.315
Y_MIN, Y_MAX = 0.65125, 0.84875


def _indices_to_apply_force(x, y):
    """``give_Me_Indices_To_Apply_Force`` -> 0-based (i, j) pairs."""
    ix_min = int(np.searchsorted(x, X_MIN, side="left"))
    ix_max = int(np.searchsorted(x, X_MAX, side="right")) - 1
    iy_min = int(np.searchsorted(y, Y_MIN, side="left"))
    iy_max = int(np.searchsorted(y, Y_MAX, side="right")) - 1
    ii, jj = np.meshgrid(np.arange(ix_min, ix_max + 1),
                         np.arange(iy_min, iy_max + 1))
    return np.column_stack([ii.ravel(), jj.ravel()])


def _target_velocity(t, y_grid):
    """``please_Give_Target_Velocity`` (x-component only)."""
    scale = ((MID_POINT + W / 2) - MID_POINT) * ((MID_POINT - W / 2)
                                                 - MID_POINT)
    return U_MAX * 5 * abs(np.sin(2 * np.pi * t)) \
        * ((MID_POINT + W / 2) - y_grid) * ((MID_POINT - W / 2) - y_grid) \
        / scale


def please_Compute_External_Forcing(dt, current_time, x, y, grid_Info, uX, uY,
                                    first, inds):
    Nx, Ny = int(grid_Info[0]), int(grid_Info[1])
    if first == 1:
        inds = _indices_to_apply_force(x, y)
        first = 0
    fx = np.zeros((Ny, Nx))
    fy = np.zeros((Ny, Nx))
    ii = inds[:, 0].astype(int)
    jj = inds[:, 1].astype(int)
    ux_tar = _target_velocity(current_time, y[jj])
    fx[jj, ii] -= K_STIFF * (uX[jj, ii] - ux_tar)
    fy[jj, ii] -= K_STIFF * uY[jj, ii]
    return fx, fy, first, inds
