"""Python port of the example's own ``please_Compute_External_Forcing.m``.

The Kelvin-Helmholtz "Cat" setup drives four horizontal bands of velocity
(``uX^tar = +/-sF * uMax * tanh(20 t)``, ``sF = 0.25``, ``uMax = 1``) over the
whole domain with a penalty force (``kStiff = 1e4``), and pushes ``uY`` towards
``2.5 * uMax * tanh(20 t)`` inside 0.025 of the top and bottom walls (there with
``2.5 * kStiff``) and towards 0 everywhere else.

Line-by-line port; the only change is 0-based indexing (the MATLAB original is
1-based).  Both codes import *this* module, so the reference and the AFSI run
evaluate the very same forcing.
"""
import numpy as np

# geometric constants of the example (module level so that the demos can print
# them); the window is the entire domain, xMin1 = 0.0, xMax1 = Lx, yMin = 0,
# yMax = Ly, i.e. every Eulerian node
X_MIN, X_MAX = 0.0, 1.0
Y_MIN, Y_MAX = 0.0, 0.5
K_STIFF = 1e4          # penalty stiffness
W = 0.3                # width of channel (unused by the target profile below)
U_MAX = 1.0            # maximum velocity
SPEED_SCALE = 0.25     # sF, speed scale factor
TIME_SCALE = 20.0      # tF, time-scale factor for ramping up the speed


def _first_ge(x, lo):
    """MATLAB ``while x(j) < lo; j++`` -> first 0-based index with x >= lo."""
    return int(np.argmax(x >= lo))


def _last_le(x, hi):
    """MATLAB ``for j=length(x):-1:1 ... x(j) <= hi`` -> last index."""
    return len(x) - 1 - int(np.argmax(x[::-1] <= hi))


def _indices_to_apply_force(x, y, x_min, x_max, y_min, y_max):
    """MATLAB ``give_Me_Indices_To_Apply_Force`` -> (n, 2) (i, j) 0-based."""
    iX = np.arange(_first_ge(x, x_min), _last_le(x, x_max) + 1)
    iY = np.arange(_first_ge(y, y_min), _last_le(y, y_max) + 1)
    ii, jj = np.meshgrid(iX, iY, indexing="ij")
    return np.column_stack([ii.ravel(), jj.ravel()]).astype(int)


def please_Give_Target_Velocity(t, dx, dy, xGrid, yGrid, Lx, Ly, i, j, w, Umax):
    """MATLAB ``please_Give_Target_Velocity``: the four-band horizontal profile."""
    y = yGrid[j]
    ramp = np.tanh(TIME_SCALE * t)
    if y < 0.125:
        uX_Tar = SPEED_SCALE * Umax * ramp
    elif y < 0.25:
        uX_Tar = -SPEED_SCALE * Umax * ramp
    elif y < 0.375:
        uX_Tar = SPEED_SCALE * Umax * ramp
    else:
        uX_Tar = -SPEED_SCALE * Umax * ramp
    return uX_Tar


def please_Compute_External_Forcing(dt, current_time, x, y, grid_Info, uX, uY,
                                    first, inds):
    """Port of the example's MATLAB entry point (same signature/returns)."""
    Nx, Ny, Lx, Ly = (int(grid_Info[0]), int(grid_Info[1]),
                      grid_Info[2], grid_Info[3])
    dx, dy = grid_Info[4], grid_Info[5]

    if first == 1:
        inds = _indices_to_apply_force(x, y, X_MIN, X_MAX, Y_MIN, Y_MAX)
        first = 0

    fx = np.zeros((Ny, Nx))
    fy = np.zeros((Ny, Nx))
    # the MATLAB original applies the force at every time (its ``if t<0.01``
    # guard is commented out), so there is no time window here
    for n in range(inds.shape[0]):
        i, j = int(inds[n, 0]), int(inds[n, 1])
        uX_Tar = please_Give_Target_Velocity(current_time, dx, dy, x, y, Lx, Ly,
                                             i, j, W, U_MAX)
        fx[j, i] -= K_STIFF * (uX[j, i] - uX_Tar)
        if (y[j] < 0.025) or (y[j] > 0.475):
            fy[j, i] -= 2.5 * K_STIFF * (uY[j, i]
                                         - 2.5 * U_MAX * np.tanh(TIME_SCALE * current_time))
        else:
            fy[j, i] -= K_STIFF * (uY[j, i] - 0.0)
    return fx, fy, first, inds
