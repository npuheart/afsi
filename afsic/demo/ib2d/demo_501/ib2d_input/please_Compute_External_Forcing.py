"""Port of ``please_Compute_External_Forcing.m``
(``Example_VIV_Cylinder/Single_VIV_Cylinder_256_Rectangle``).

Same pulsatile velocity-target scheme as the other VIV demos, but the
**rectangle** boxes (``1.0 x 0.25``) shift the parabolic profile down by
``yShift = 0.375`` so that it stays inside the box:

    uX_tar = -uMax 5 sin(2 pi freq_P t) (
        (Lx/2 - yShift + w/2) - y ) ( (Lx/2 - yShift - w/2) - y ),

with ``freq_P = 10``, ``rest = 10 * period``, ``kStiff = 1e4``, ``w = 0.2``,
``uMax = 75`` and the penalty window ``x in [0.1, 0.108]``,
``y in [0.035, 0.225]``.  The .m file allocates ``zeros(Ny, Nx)`` here (unlike
the square-grid demos, which use the transposed ``zeros(Nx, Ny)``), which is
the correct (y, x) layout and is what this port uses throughout.
"""
import numpy as np

K_STIFF = 1e4
W = 0.2
U_MAX = 75.0
X_MIN, X_MAX = 0.1, 0.108
Y_MIN, Y_MAX = 0.035, 0.225
FREQ_P = 10.0
REST_PERIODS = 10.0
Y_SHIFT = 0.375


def _indices_to_apply_force(x, y):
    """``give_Me_Indices_To_Apply_Force`` -> 0-based (i, j) pairs."""
    ix_min = int(np.searchsorted(x, X_MIN, side="left"))
    ix_max = int(np.searchsorted(x, X_MAX, side="right")) - 1
    iy_min = int(np.searchsorted(y, Y_MIN, side="left"))
    iy_max = int(np.searchsorted(y, Y_MAX, side="right")) - 1
    ii, jj = np.meshgrid(np.arange(ix_min, ix_max + 1),
                         np.arange(iy_min, iy_max + 1))
    return np.column_stack([ii.ravel(), jj.ravel()])


def _target_velocity(t, y_grid, Lx):
    """``please_Give_Target_Velocity`` (x-component only, pulse train)."""
    period = 1.0 / FREQ_P
    tot_period = period + REST_PERIODS * period
    t_mod = np.mod(t, tot_period)
    pulse = np.where(t_mod <= period,
                     -U_MAX * (5 * np.sin(FREQ_P * 2 * np.pi * t)), 0.0)
    yc = Lx / 2 - Y_SHIFT
    return pulse * ((yc + W / 2) - y_grid) * ((yc - W / 2) - y_grid)


def please_Compute_External_Forcing(dt, current_time, x, y, grid_Info, uX, uY,
                                    first, inds):
    Nx, Ny = int(grid_Info[0]), int(grid_Info[1])
    Lx = float(grid_Info[2])
    if first == 1:
        inds = _indices_to_apply_force(x, y)
        first = 0
    fx = np.zeros((Ny, Nx))
    fy = np.zeros((Ny, Nx))
    ii = inds[:, 0].astype(int)
    jj = inds[:, 1].astype(int)
    ux_tar = _target_velocity(current_time, y[jj], Lx)
    fx[jj, ii] -= K_STIFF * (uX[jj, ii] - ux_tar)
    fy[jj, ii] -= K_STIFF * uY[jj, ii]
    return fx, fy, first, inds
