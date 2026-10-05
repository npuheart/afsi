"""Port of ``please_Compute_External_Forcing.m``
(``Example_VIV_Cylinder/Single_VIV_Cylinder_512_Rectangle``).

Rectangle-box variant of the VIV pulsatile forcing (see
``Single_VIV_Cylinder_256_Rectangle``): ``yShift = 0.375`` pushes the parabola
down into the ``1.0 x 0.25`` box, the penalty window is ``x in [0.1, 0.108]``,
``y in [0.035, 0.225]``, and the pulse train uses ``freq_P = 10`` with
``rest = 10 * period``.  This flavour differs from the 256 rectangle one in the
**duty cycle**: the pulse is on only for ``t_mod <= period / 2`` (upstream
copy-paste, kept verbatim here), i.e. half of one period.

The .m file allocates ``zeros(Ny, Nx)``; this port always uses the (Ny, Nx)
layout, which is what the driver's ``fx(j, i)`` indexing means.
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
HALF_PERIOD = True


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
    pulse = np.where(t_mod <= period / 2,
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
