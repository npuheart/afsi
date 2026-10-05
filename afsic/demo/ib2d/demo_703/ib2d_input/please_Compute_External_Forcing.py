"""Port of ``please_Compute_External_Forcing.m`` (``Example_Turek_Hron``).

The Turek-Hron benchmark drives the channel inlet with a **parabolic
``100 tanh(2t)`` jet** through a velocity-target penalty force over
``x in [0.1, 0.16]``, ``y in [0.035, 0.215]``:

    Fx -= kStiff (uX - uX_tar),  uX_tar = -uMax (100 tanh(2t))
                                  ((Ly/2 + w/2) - y) ((Ly/2 - w/2) - y),
    Fy = 0,

with ``kStiff = 1e3``, ``w = 0.2``, ``uMax = 2.75`` (peak ``uX^tar`` of the
asymptotic profile ≈ 2.75 m/s at ``y = Ly/2 = 0.125``).

pyIB2d imports this module from the run directory and calls it with the 1-D node
coordinates, the (Ny, Nx) velocity arrays and the persistent ``(first, inds)``
pair; the AFSI ``main.py`` evaluates the very same module at its Q1 node dofs,
so both codes use an identical forcing.  MATLAB's 1-based indices are translated
to 0-based here; the index search of the .m file (first ``x(j) >= xMin``, last
``x(j) <= xMax``) is reproduced with ``np.searchsorted``.
"""
import numpy as np

K_STIFF = 1e3
W = 0.2
U_MAX = 2.75
X_MIN, X_MAX = 0.1, 0.16
Y_MIN, Y_MAX = 0.035, 0.215


def _indices_to_apply_force(x, y):
    """``give_Me_Indices_To_Apply_Force`` -> 0-based (i, j) pairs."""
    ix_min = int(np.searchsorted(x, X_MIN, side="left"))
    ix_max = int(np.searchsorted(x, X_MAX, side="right")) - 1
    iy_min = int(np.searchsorted(y, Y_MIN, side="left"))
    iy_max = int(np.searchsorted(y, Y_MAX, side="right")) - 1
    ii, jj = np.meshgrid(np.arange(ix_min, ix_max + 1),
                         np.arange(iy_min, iy_max + 1))
    return np.column_stack([ii.ravel(), jj.ravel()])


def _target_velocity(t, y_grid, j, Ly):
    """``please_Give_Target_Velocity`` (x-component only)."""
    y = y_grid[j]
    ux_tar = -U_MAX * (100 * np.tanh(2 * t)) \
        * ((Ly / 2 + W / 2) - y) * ((Ly / 2 - W / 2) - y)
    return ux_tar, 0.0


def please_Compute_External_Forcing(dt, current_time, x, y, grid_Info, uX, uY,
                                    first, inds):
    Nx, Ny, Ly = int(grid_Info[0]), int(grid_Info[1]), float(grid_Info[3])

    if first == 1:
        inds = _indices_to_apply_force(x, y)
        first = 0

    fx = np.zeros((Ny, Nx))
    fy = np.zeros((Ny, Nx))
    for i, j in inds:
        ux_tar, uy_tar = _target_velocity(current_time, y, j, Ly)
        fx[j, i] -= K_STIFF * (uX[j, i] - ux_tar)
        fy[j, i] -= K_STIFF * (uY[j, i] - uy_tar)

    return fx, fy, first, inds
