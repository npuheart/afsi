"""Port of ``please_Compute_External_Forcing.m`` (Example_Brinkman_Channel).

pyIB2d's driver imports this module from the run directory once per step when
``arb_ext_force`` is on, calls

    Fx, Fy, first, inds = please_Compute_External_Forcing(
        dt, current_time, x, y, grid_Info, uX, uY, first, inds)

with ``x``/``y`` the 1-D node coordinates (``0:dx:Lx-dx``, ``0:dy:Ly-dy``) and
``uX``/``uY`` the (Ny, Nx) velocity arrays, and adds ``(Fx, Fy)`` to the Eulerian
force before the fluid solve.

The example drives a **normalised parabolic jet ramping up as ``tanh(4t)``**
over ``x in [0.1, 0.16]``, ``y in [0.030, 0.220]`` (the whole strip):

    Fx -= kStiff (uX - uX_tar),   uX_tar = uMax tanh(4t) (
        (MP + w/2) - y ) ( (MP - w/2) - y ) / scale,   Fy = 0,

with ``kStiff = 1e4``, ``W = 0.2``, ``uMax = 2.5``, the mid-plane
``MP = (0.035 + 0.215)/2`` and ``scale = ((MP + w/2) - MP)((MP - w/2) - MP)``, so
the profile peaks at exactly ``uMax tanh(4t)`` on the mid-plane.
"""
import numpy as np

K_STIFF = 1e4                     # stiffness parameter
W = 0.2                           # width of the velocity-target profile
U_MAX = 2.5                       # peak target velocity
MID_POINT = (0.035 + 0.215) / 2   # mid-plane of the strip
X_MIN, X_MAX = 0.1, 0.16
Y_MIN, Y_MAX = 0.030, 0.220
SCALE = ((MID_POINT + W / 2) - MID_POINT) * ((MID_POINT - W / 2) - MID_POINT)


def _indices_to_apply_force(x, y):
    """``give_Me_Indices_To_Apply_Force`` -> 0-based (i, j) pairs."""
    ix_min = int(np.searchsorted(x, X_MIN, side="left"))
    ix_max = int(np.searchsorted(x, X_MAX, side="right")) - 1
    iy_min = int(np.searchsorted(y, Y_MIN, side="left"))
    iy_max = int(np.searchsorted(y, Y_MAX, side="right")) - 1
    ii, jj = np.meshgrid(np.arange(ix_min, ix_max + 1),
                         np.arange(iy_min, iy_max + 1))
    return np.column_stack([ii.ravel(), jj.ravel()])


def _target_velocity(t, y_grid, j, Lx, Ly):
    """``please_Give_Target_Velocity`` (x-component only)."""
    y = y_grid[j]
    ux_tar = U_MAX * np.tanh(4 * t) \
        * ((MID_POINT + W / 2) - y) * ((MID_POINT - W / 2) - y) / SCALE
    return ux_tar, 0.0


def please_Compute_External_Forcing(dt, current_time, x, y, grid_Info, uX, uY,
                                    first, inds):
    Nx, Ny, Lx, Ly = (int(grid_Info[0]), int(grid_Info[1]),
                      float(grid_Info[2]), float(grid_Info[3]))

    if first == 1:
        inds = _indices_to_apply_force(x, y)
        first = 0

    fx = np.zeros((Ny, Nx))
    fy = np.zeros((Ny, Nx))
    for i, j in inds:
        ux_tar, uy_tar = _target_velocity(current_time, y, j, Lx, Ly)
        fx[j, i] -= K_STIFF * (uX[j, i] - ux_tar)
        fy[j, i] -= K_STIFF * (uY[j, i] - uy_tar)

    return fx, fy, first, inds
