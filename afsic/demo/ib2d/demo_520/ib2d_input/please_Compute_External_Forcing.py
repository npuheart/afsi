"""Line-by-line port of
``Example_Channel_Flow/Example_Poroelastic_Channel_Flow/please_Compute_External_Forcing.m``.

pyIB2d's driver imports this module from the run directory (once per step, when
``arb_ext_force`` is on) and calls

    Fx, Fy, first, inds = please_Compute_External_Forcing(
        dt, current_time, x, y, grid_Info, U, V, first, inds)

with ``x``/``y`` the 1-D Eulerian coordinate vectors, ``U``/``V`` the (Ny, Nx)
Eulerian velocity arrays and ``first``/``inds`` the persistent state of the
force mask (``first == 1`` on the first call computes the mask).

The .m file adds a penalty body force on a rectangular *patch* of the fluid
grid (``x in [0.1, 0.16]``, ``y in [0.06, 0.19]``) that drags the fluid toward
the desired channel profile

    uX_Tar = -Umax * 5 * sin(2*10*pi*t) * ((Ly/2 + w/2) - y) * ((Ly/2 - w/2) - y),

with ``uY_Tar = 0`` -- i.e. an oscillating parabolic-like jet of maximum speed
``Umax = 200`` (the 5*sin factor makes the instantaneous target up to 1000 in
module) plus the two signed factors; the two zeros at
``y = Ly/2 -+ w/2 = 0.05, 0.2`` are outside the patch, so the target is
one-signed.  ``kStiff = 1e4`` is the penalty stiffness.

MATLAB's 1-based indexing is translated to 0-based: ``inds`` holds 0-based
``(x-index, y-index)`` pairs and ``fx[j, i]``/``fy[j, i]`` are the numpy
``[y, x]`` entries of the (Ny, Nx) force arrays.
"""
import numpy as np

K_STIFF = 1e4          # penalty stiffness for the velocity-target body force
W = 0.15               # width of the desired channel profile (m)
U_MAX = 200.0          # maximum desired velocity (m/s)
FREQ = 10.0            # oscillation frequency (Hz) in  2*10*pi*t
X_MIN, X_MAX = 0.1, 0.16
Y_MIN, Y_MAX = 0.06, 0.19


def give_Me_Indices_To_Apply_Force(x, y, xMin, xMax, yMin, yMax):
    """0-based grid-index pairs (x-index, y-index) of the force patch."""
    iX_min = int(np.argmax(x >= xMin))
    iX_max = int(x.size - 1 - np.argmax(x[::-1] <= xMax))
    iY_min = int(np.argmax(y >= yMin))
    iY_max = int(y.size - 1 - np.argmax(y[::-1] <= yMax))
    iX_vec = np.arange(iX_min, iX_max + 1)
    iY_vec = np.arange(iY_min, iY_max + 1)
    return np.array([(i, j) for i in iX_vec for j in iY_vec], dtype=np.int64)


def please_Give_Target_Velocity(t, dx, dy, xGrid, yGrid, Lx, Ly, i, j, w, Umax):
    y_val = yGrid[j]
    uX_Tar = -Umax * (5 * np.sin(2 * FREQ * np.pi * t)
                      * ((Ly / 2 + w / 2) - y_val)
                      * ((Ly / 2 - w / 2) - y_val))
    uY_Tar = 0.0
    return uX_Tar, uY_Tar


def give_Me_Velocity_Target_External_Force_Density(t, dx, dy, x, y, Nx, Ny,
                                                  Lx, Ly, uX, uY, kStiff, w,
                                                  Umax, inds):
    fx = np.zeros((Ny, Nx))
    fy = np.zeros((Ny, Nx))
    for i, j in inds:
        uX_Tar, uY_Tar = please_Give_Target_Velocity(t, dx, dy, x, y, Lx, Ly,
                                                     i, j, w, Umax)
        fx[j, i] -= kStiff * (uX[j, i] - uX_Tar)
        fy[j, i] -= kStiff * (uY[j, i] - uY_Tar)
    return fx, fy


def please_Compute_External_Forcing(dt, current_time, x, y, grid_Info, uX, uY,
                                    first, inds):
    Nx = int(grid_Info[0])
    Ny = int(grid_Info[1])
    Lx = grid_Info[2]
    Ly = grid_Info[3]
    dx = grid_Info[4]
    dy = grid_Info[5]

    if first == 1:
        inds = give_Me_Indices_To_Apply_Force(x, y, X_MIN, X_MAX, Y_MIN, Y_MAX)
        first = 0

    fx, fy = give_Me_Velocity_Target_External_Force_Density(
        current_time, dx, dy, x, y, Nx, Ny, Lx, Ly, uX, uY, K_STIFF, W, U_MAX,
        inds)

    Fx = fx
    Fy = fy
    return Fx, Fy, first, inds
