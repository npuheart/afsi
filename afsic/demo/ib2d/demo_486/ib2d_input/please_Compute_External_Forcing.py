"""Port of ``please_Compute_External_Forcing.m``
(``Example_Meshmerize_Me/old``, the leaf-vein network).

Like the top-level ``Example_Meshmerize_Me`` this drives a thin horizontal jet
across an open vein network, but with a different profile and window:

    Fy = -kStiff (uY - uY_tar),   uY_tar = -uMax 5 tanh(t) (
        (MP + w/2) - x ) ( (MP - w/2) - x ),   Fx = 0,

with ``kStiff = 1e4``, ``uMax = 250``, ``w = 0.5515 - 0.4029 = 0.1486``,
``MP = (0.5515 + 0.4029)/2 = 0.4772`` and the penalty window
``x in [0.41, 0.545]``, ``y in [0.115, 0.12]``.  Note this file has **no**
``/ scale`` normalisation (the parabola peak is ``w^2/4 * 5 uMax`` at the
window midpoint), which is the upstream difference w.r.t. the newer
``Example_Meshmerize_Me`` module.
"""
import numpy as np

K_STIFF = 1e4
W = 0.5515 - 0.4029
MID_POINT = (0.5515 + 0.4029) / 2
U_MAX = 250.0
X_MIN, X_MAX = 0.41, 0.545
Y_MIN, Y_MAX = 0.115, 0.12


def _indices_to_apply_force(x, y):
    """``give_Me_Indices_To_Apply_Force`` -> 0-based (i, j) pairs."""
    ix_min = int(np.searchsorted(x, X_MIN, side="left"))
    ix_max = int(np.searchsorted(x, X_MAX, side="right")) - 1
    iy_min = int(np.searchsorted(y, Y_MIN, side="left"))
    iy_max = int(np.searchsorted(y, Y_MAX, side="right")) - 1
    ii, jj = np.meshgrid(np.arange(ix_min, ix_max + 1),
                         np.arange(iy_min, iy_max + 1))
    return np.column_stack([ii.ravel(), jj.ravel()])


def _target_velocity(t, x_grid):
    """``please_Give_Target_Velocity`` (y-component only)."""
    return -U_MAX * (5 * np.tanh(t)) * ((MID_POINT + W / 2) - x_grid) \
        * ((MID_POINT - W / 2) - x_grid)


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
    uy_tar = _target_velocity(current_time, x[ii])
    fx[jj, ii] -= K_STIFF * uX[jj, ii]
    fy[jj, ii] -= K_STIFF * (uY[jj, ii] - uy_tar)
    return fx, fy, first, inds
