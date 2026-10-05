"""Python port of the example's own ``please_Compute_External_Forcing.m``
(``Kelvin_Helmholtz_Qualitative_2``).

Five horizontal bands (shear interfaces at ``y = 0.05, 0.15, 0.25, 0.35,
0.45``) are driven through a penalty force (``kStiff = 1e4``) towards

    uX^tar = sF * uMax * tanh(coeff * (..)) * tanh(tF t)
    uY^tar = 0.075 * uMax * sin(2 pi x / 0.2) * tanh(15 t)

with ``sF = 0.25``, ``coeff = 50``, ``tF = 20``, ``uMax = 2.5``, applied to
five **vertical strips** --
``x in [0, 0.02]``, ``[0.23, 0.27]``, ``[0.48, 0.52]``, ``[0.73, 0.77]``,
``[0.98, 1.0]`` -- and ``y in [0, 0.5]`` (the sibling `Kelvin_Helmholtz_Qualitative`
of demo_579 drives every node instead, which is the only difference between the
two ``.m`` files).  The ``%if t<0.01`` guard of the MATLAB original is commented
out, so the force acts at all times.

Line-by-line port; the only change is 0-based indexing (MATLAB is 1-based).
Both codes import *this* module from the run directory, so the reference and the
AFSI run evaluate the very same forcing.
"""
import numpy as np

STRIPS = ((0.0, 0.02), (0.23, 0.27), (0.48, 0.52), (0.73, 0.77), (0.98, 1.0))
Y_MIN, Y_MAX = 0.0, 0.5     # yMin = 0, yMax = Ly
K_STIFF = 1e4               # penalty stiffness
W = 0.3                     # width of channel (unused by the target profile)
U_MAX = 2.5                 # maximum velocity
SPEED_SCALE = 0.25          # sF
TIME_SCALE = 20.0           # tF
COEFF = 50.0                # size of the interfacial region


def _first_ge(x, lo):
    return int(np.argmax(x >= lo))


def _last_le(x, hi):
    return len(x) - 1 - int(np.argmax(x[::-1] <= hi))


def _indices_to_apply_force(x, y, x_min, x_max, y_min, y_max):
    """MATLAB ``give_Me_Indices_To_Apply_Force`` -> (n, 2) (i, j) 0-based."""
    iX = np.arange(_first_ge(x, x_min), _last_le(x, x_max) + 1)
    iY = np.arange(_first_ge(y, y_min), _last_le(y, y_max) + 1)
    ii, jj = np.meshgrid(iX, iY, indexing="ij")
    return np.column_stack([ii.ravel(), jj.ravel()]).astype(int)


def please_Give_Target_Velocity(t, dx, dy, xGrid, yGrid, Lx, Ly, i, j, w, Umax):
    """MATLAB ``please_Give_Target_Velocity``: 5 shear bands + a sine lift."""
    y = yGrid[j]
    x = xGrid[i]
    ramp = np.tanh(TIME_SCALE * t)
    if y <= 0.1:
        uX_Tar = SPEED_SCALE * Umax * np.tanh(COEFF * (0.05 - y)) * ramp
    elif y <= 0.2:
        uX_Tar = SPEED_SCALE * Umax * np.tanh(COEFF * (y - 0.15)) * ramp
    elif y <= 0.3:
        uX_Tar = SPEED_SCALE * Umax * np.tanh(COEFF * (0.25 - y)) * ramp
    elif y <= 0.4:
        uX_Tar = SPEED_SCALE * Umax * np.tanh(COEFF * (y - 0.35)) * ramp
    else:
        uX_Tar = SPEED_SCALE * Umax * np.tanh(COEFF * (0.45 - y)) * ramp
    uY_Tar = 0.075 * Umax * np.sin(2 * np.pi * x / 0.2) * np.tanh(15 * t)
    return uX_Tar, uY_Tar


def please_Compute_External_Forcing(dt, current_time, x, y, grid_Info, uX, uY,
                                    first, inds):
    """Port of the example's MATLAB entry point (same signature/returns)."""
    Nx, Ny = int(grid_Info[0]), int(grid_Info[1])
    Lx, Ly, dx, dy = grid_Info[2], grid_Info[3], grid_Info[4], grid_Info[5]

    if first == 1:
        inds = np.vstack([_indices_to_apply_force(x, y, xa, xb, Y_MIN, Y_MAX)
                          for xa, xb in STRIPS])
        first = 0

    fx = np.zeros((Ny, Nx))
    fy = np.zeros((Ny, Nx))
    # no time window: the MATLAB ``%if t<0.01`` guard is commented out
    for n in range(inds.shape[0]):
        i, j = int(inds[n, 0]), int(inds[n, 1])
        uX_Tar, uY_Tar = please_Give_Target_Velocity(
            current_time, dx, dy, x, y, Lx, Ly, i, j, W, U_MAX)
        fx[j, i] -= K_STIFF * (uX[j, i] - uX_Tar)
        fy[j, i] -= K_STIFF * (uY[j, i] - uY_Tar)
    return fx, fy, first, inds
