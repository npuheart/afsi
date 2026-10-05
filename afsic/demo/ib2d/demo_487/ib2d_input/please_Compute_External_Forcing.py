"""Port of ``please_Compute_External_Forcing.m``
(``Example_Meshmerize_Me/Insect_Wing_Veins``).

Thin vertical-slot jet across the wing-vein network (``0.5 x 0.25`` box):

    Fx = -kStiff (uX - uX_tar),   uX_tar = uMax tanh(2t) (
        (MP + w/2) - y ) ( (MP - w/2) - y ) / scale,   Fy = 0,

with ``kStiff = 1e4``, ``uMax = 0.1``, ``w = 0.1978 - 0.1927 = 0.0051``,
``MP = (0.1978 + 0.1927)/2 = 0.19525``, ``scale = -w^2/4``, and the penalty
window ``x in [0.015, 0.017]``, ``y in [0.1927, 0.1978]``.  The /scale
normalisation makes the target exactly ``uMax tanh(2t)`` at the slot centre.

The .m file allocates ``zeros(Ny, Nx)``; this port uses that (y, x) layout.
"""
import numpy as np

K_STIFF = 1e4
W = 0.1978 - 0.1927
MID_POINT = (0.1978 + 0.1927) / 2
U_MAX = 0.1
X_MIN, X_MAX = 0.015, 0.017
Y_MIN, Y_MAX = 0.1927, 0.1978


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
    return U_MAX * np.tanh(2 * t) * ((MID_POINT + W / 2) - y_grid) \
        * ((MID_POINT - W / 2) - y_grid) / scale


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
