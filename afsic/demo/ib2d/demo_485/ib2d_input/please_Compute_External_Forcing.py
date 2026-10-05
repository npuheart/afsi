"""Port of ``please_Compute_External_Forcing.m``
(``Example_Meshmerize_Me/Flow_Around_Starfish``, ``1.0 x 0.25`` tank).

Thin vertical-slot jet for the starfish flow tank: the penalty window is
``x in [0.04, 0.06]`` (narrow in ``x``) and ``y in [0.02, 0.22]`` (almost the
full height), and the target is

    Fx = -kStiff (uX - uX_tar),   uX_tar = -uMax sin(2 pi f t) (
        (Ly/2 + w/2) - y ) ( (Ly/2 - w/2) - y ),   Fy = 0,

with ``kStiff = 1e4``, ``uMax = 2.5e3``, ``w = 0.25``, and a **pulse train**
``f = freq_P = 5 Hz`` with ``rest = 1 * period`` (so the jet is off as long as
it is on).  ``Ly = 0.25`` makes ``Ly/2 +- w/2`` equal ``0`` and ``0.25``, the
full tank height.

The .m file allocates ``zeros(Ny, Nx)``; this port uses that (y, x) layout.
"""
import numpy as np

K_STIFF = 1e4
W = 0.25
U_MAX = 2.5e3
FREQ_P = 5.0
REST_PERIODS = 1.0
X_MIN, X_MAX = 0.04, 0.06
Y_MIN, Y_MAX = 0.02, 0.22


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
    """``please_Give_Target_Velocity`` (x-component only, pulse train)."""
    period = 1.0 / FREQ_P
    tot_period = period + REST_PERIODS * period
    t_mod = np.mod(t, tot_period)
    pulse = np.where(t_mod <= period,
                     -U_MAX * np.sin(FREQ_P * 2 * np.pi * t), 0.0)
    return pulse * ((Ly / 2 + W / 2) - y_grid) * ((Ly / 2 - W / 2) - y_grid)


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
