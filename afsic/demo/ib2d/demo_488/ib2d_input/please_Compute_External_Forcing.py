"""Port of ``please_Compute_External_Forcing.m``
(``Example_Meshmerize_Me/Starfish``).

A straightforward pulsatile channel drive for the starfish flow tank
(``Ly = 0.25``):

    Fx = -kStiff (uX - uX_tar),   uX_tar = -uMax sin(2 pi f t) (
        (Ly/2 + w/2) - y ) ( (Ly/2 - w/2) - y ),   Fy = 0,

with ``kStiff = 1e4``, ``uMax = 500``, ``w = 0.23``, ``f = 2 Hz``, and the
penalty window ``x in [0.015, 0.065]``, ``y in [0.011, 0.239]``.  The parabola
is built from ``Ly`` (it vanishes at ``Ly/2 +- w/2 = -0.005 / 0.24``) so it is
essentially a full-height parabolic channel profile, oscillating at 2 Hz -- the
starfish sees an oscillatory back-and-forth flow.

The .m file allocates ``zeros(Ny, Nx)``; this port uses that (y, x) layout.
"""
import numpy as np

K_STIFF = 1e4
W = 0.23
U_MAX = 500.0
FREQ = 2.0
X_MIN, X_MAX = 0.015, 0.065
Y_MIN, Y_MAX = 0.011, 0.239


def _indices_to_apply_force(x, y):
    """``give_Me_Indices_To_Apply_Force`` -> 0-based (i, j) pairs."""
    ix_min = int(np.searchsorted(x, X_MIN, side="left"))
    ix_max = int(np.searchsorted(x, X_MAX, side="right")) - 1
    iy_min = int(np.searchsorted(y, Y_MIN, side="left"))
    iy_max = int(np.searchsorted(y, Y_MAX, side="right")) - 1
    ii, jj = np.meshgrid(np.arange(ix_min, ix_max + 1),
                         np.arange(iy_min, iy_max + 1))
    return np.column_stack([ii.ravel(), jj.ravel()])


def _target_velocity(t, y_grid, Ly):
    """``please_Give_Target_Velocity`` (x-component only)."""
    return -U_MAX * np.sin(FREQ * 2 * np.pi * t) \
        * ((Ly / 2 + W / 2) - y_grid) * ((Ly / 2 - W / 2) - y_grid)


def please_Compute_External_Forcing(dt, current_time, x, y, grid_Info, uX, uY,
                                    first, inds):
    Nx, Ny = int(grid_Info[0]), int(grid_Info[1])
    Ly = float(grid_Info[3])
    if first == 1:
        inds = _indices_to_apply_force(x, y)
        first = 0
    fx = np.zeros((Ny, Nx))
    fy = np.zeros((Ny, Nx))
    ii = inds[:, 0].astype(int)
    jj = inds[:, 1].astype(int)
    ux_tar = _target_velocity(current_time, y[jj], Ly)
    fx[jj, ii] -= K_STIFF * (uX[jj, ii] - ux_tar)
    fy[jj, ii] -= K_STIFF * uY[jj, ii]
    return fx, fy, first, inds
