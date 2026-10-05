"""IB2d flow-tank forcing of the Meshmerize family -- line-by-line port of

    matIB2d/Examples/Example_Meshmerize_Me/Dragonfly_Wing/please_Compute_External_Forcing.m

(the ``arb_ext_force = 1`` user module pyIB2d imports from the run directory on
every step; ``main.py`` imports the very same file, so both codes drive the tank
with one implementation).

The module penalises the velocity towards a target profile *inside one small
window* of the Eulerian grid (``xMin..xMax`` x ``yMin..yMax``) with

    F = -kStiff (u - u_target),      kStiff = 1e4,

and the window indices are computed once, on the first call (``first == 1``),
by searching the node-coordinate arrays -- so the module is resolution
independent and the same file works for the reduced grids of this demo.

Target profile of this example (an x-jet through a vertical slot in y):

``uX_tar = Umax tanh(2t) ((MP+w/2)-y)((MP-w/2)-y) / scale`` with
``Umax = 0.1``, ``w = 0.1978 - 0.1927 = 0.0051``, ``MP`` its centre; window
``x in [0.015, 0.017]``, ``y in [0.1927, 0.1978]`` (a very thin slot, the wing
fills most of the tank).

Upstream quirks: ``iX_min``/``iX_max`` are found with ``while`` loops that
``j=j+1`` *after* the assignment (harmless, the index is saved before), the
window indices are 1-based MATLAB indices into the node arrays, and
``uX_tar``/``uY_tar`` are assigned to the grid array only inside the window
(everywhere else the forcing is zero).  The port keeps the loop structure but
uses the 0-based node indices directly, and passes the *coordinate values*
``x[i]``/``y[j]`` into the profile instead of re-indexing the arrays inside it.
"""
import numpy as np

K_STIFF = 1e4
W = 0.005099999999999993
MID_POINT = 0.19525
U_MAX = 0.1
X_MIN, X_MAX, Y_MIN, Y_MAX = (0.015, 0.017, 0.1927, 0.1978)


def give_Me_Indices_To_Apply_Force(x, y, xMin, xMax, yMin, yMax):
    """MATLAB ``give_Me_Indices_To_Apply_Force``: (n, 2) pairs of 0-based node
    indices inside the window, ordered x-major (MATLAB's inner loop is over y)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    iX_min = int(np.argmax(x >= xMin))
    iX_max = len(x) - 1 - int(np.argmax(x[::-1] <= xMax))
    iY_min = int(np.argmax(y >= yMin))
    iY_max = len(y) - 1 - int(np.argmax(y[::-1] <= yMax))
    iX_Vec = np.arange(iX_min, iX_max + 1)
    iY_Vec = np.arange(iY_min, iY_max + 1)
    return np.array([(i, j) for i in iX_Vec for j in iY_Vec], dtype=int)


def please_Compute_External_Forcing(dt, current_time, x, y, grid_Info, uX, uY,
                                    first, inds):
    """MATLAB signature; returns ``(Fx, Fy, first, inds)``."""
    Nx, Ny, Lx, Ly, dx, dy = (int(grid_Info[0]), int(grid_Info[1]),
                              grid_Info[2], grid_Info[3], grid_Info[4],
                              grid_Info[5])
    if first == 1:
        inds = give_Me_Indices_To_Apply_Force(x, y, X_MIN, X_MAX, Y_MIN, Y_MAX)
        first = 0
    fx, fy = give_Me_Velocity_Target_External_Force_Density(
        current_time, dx, dy, x, y, Nx, Ny, Lx, Ly, uX, uY, K_STIFF, W, U_MAX,
        inds, MID_POINT)
    return fx, fy, first, inds


def give_Me_Velocity_Target_External_Force_Density(t, dx, dy, x, y, Nx, Ny, Lx,
                                                   Ly, uX, uY, kStiff, w, Umax,
                                                   inds, MP):
    fx = np.zeros((Ny, Nx))
    fy = np.zeros((Ny, Nx))
    for n in range(inds.shape[0]):
        i = int(inds[n, 0])
        j = int(inds[n, 1])
        uX_Tar, uY_Tar = please_Give_Target_Velocity(
            t, dx, dy, x, y, Lx, Ly, i, j, w, Umax, MP)
        fx[j, i] -= kStiff * (uX[j, i] - uX_Tar)
        fy[j, i] -= kStiff * (uY[j, i] - uY_Tar)
    return fx, fy


def please_Give_Target_Velocity(t, dx, dy, xGrid, yGrid, Lx, Ly, i, j, w, Umax,
                                MP):
    """MATLAB ``please_Give_Target_Velocity`` for this example."""
    yj = yGrid[j]
    scale = ((MP + w / 2) - MP) * ((MP - w / 2) - MP)
    uX_Tar = Umax * np.tanh(2 * t) * ((MP + w / 2) - yj) \
        * ((MP - w / 2) - yj) / scale
    uY_Tar = 0.0
    return uX_Tar, uY_Tar
