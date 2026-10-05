"""Port of ``please_Compute_External_Forcing.m`` (Example_Mass_Points/Massive_Cell_In_Channel).

pyIB2d imports this module from the run directory once per step when
``arb_ext_force`` is on and calls

    Fx, Fy, first, inds = please_Compute_External_Forcing(
        dt, current_time, x, y, grid_Info, uX, uY, first, inds)

with ``x``/``y`` the 1-D node coordinates (``0:dx:Lx-dx``, ``0:dy:Ly-dy``) and
``uX``/``uY`` the (Ny, Nx) velocity arrays; ``(Fx, Fy)`` is added to the
Eulerian force density before the fluid solve.  The very same module is
imported by this demo's ``main.py``, so both codes use one implementation.

A *single* horizontal channel jet through ``x in [0.1, 0.16]``,
``y in [0.41, 0.59]``, with a velocity-target penalty force

    Fx -= kStiff (uX - uX_tar),
    uX_tar = -uMax 5 tanh(t) ((Lx/2 + w/2) - y) ((Lx/2 - w/2) - y),

``kStiff = 1e4``, ``w = 0.2``, ``uMax = 250``; ``uY_tar = 0``.  This is the
single-cell version of demo_572: the massive cell (54 markers, ``M = 0.02``)
sits in this parabolic jet and, with no gravity, is carried along by it.

Note the upstream quirk that the parabolic profile is written in terms of
``Lx/2`` while its argument is ``y``; for this example ``Lx = Ly = 1``, so the
two are numerically identical and the port keeps the .m file's expression.

MATLAB's 1-based indices are translated to 0-based here; the index search of
the .m file (first ``x(j) >= xMin`` / last ``x(j) <= xMax``) is reproduced with
``np.searchsorted``.
"""
import numpy as np

K_STIFF = 1e4                     # stiffness parameter
W = 0.2                           # width of the target-velocity window
U_MAX = 250.0
X_MIN, X_MAX = 0.1, 0.16
Y_MIN, Y_MAX = 0.41, 0.59


def _indices_to_apply_force(x, y):
    """``give_Me_Indices_To_Apply_Force`` -> 0-based (i, j) pairs (i: x)."""
    ix_min = int(np.searchsorted(x, X_MIN, side="left"))
    ix_max = int(np.searchsorted(x, X_MAX, side="right")) - 1
    iy_min = int(np.searchsorted(y, Y_MIN, side="left"))
    iy_max = int(np.searchsorted(y, Y_MAX, side="right")) - 1
    ii, jj = np.meshgrid(np.arange(ix_min, ix_max + 1),
                         np.arange(iy_min, iy_max + 1))
    return np.column_stack([ii.ravel(), jj.ravel()]).astype(float)


def _target_velocity(t, y_grid, j, Lx):
    """``please_Give_Target_Velocity`` (x-component only)."""
    y = y_grid[j]
    ux_tar = -U_MAX * (5 * np.tanh(t)) \
        * ((Lx / 2 + W / 2) - y) * ((Lx / 2 - W / 2) - y)
    return ux_tar, 0.0


def please_Compute_External_Forcing(dt, current_time, x, y, grid_Info, uX, uY,
                                    first, inds):
    Nx, Ny, Lx, Ly = (int(grid_Info[0]), int(grid_Info[1]),
                      float(grid_Info[2]), float(grid_Info[3]))

    if first == 1:
        inds = _indices_to_apply_force(x, y)
        first = 0

    fx = np.zeros_like(uX)
    fy = np.zeros_like(uX)
    for i, j in inds.astype(int):
        ux_tar, uy_tar = _target_velocity(current_time, y, j, Lx)
        fx[j, i] -= K_STIFF * (uX[j, i] - ux_tar)
        fy[j, i] -= K_STIFF * (uY[j, i] - uy_tar)

    return fx, fy, first, inds
