"""Port of ``please_Compute_External_Forcing.m``
(``Example_Meshmerize_Me``, the leaf-vein network).

The ``Meshmerize`` examples fix an open "leaf vein" network with target points
and drive the flow through a **thin horizontal jet** (the window is narrow in
``y``) whose velocity is ramped on as ``tanh(2t)``:

    Fy = -kStiff (uY - uY_tar),   uY_tar = uMax tanh(2t) (
        (MP + w/2) - x ) ( (MP - w/2) - x ) / scale,
    scale = ( (MP + w/2) - MP ) ( (MP - w/2) - MP ) = -w^2/4,   Fx = 0,

with ``kStiff = 1e4``, ``uMax = 2.5``, ``w = 0.112`` and the midpoint
``MP = (0.35955 + 0.247375)/2 = 0.3034625``; the penalty window is
``x in [0.252, 0.364]``, ``y in [0.3625, 0.3675]``.  Because ``scale`` divides
by ``-w^2/4`` the target is exactly ``uMax tanh(2t)`` at ``x = MP`` and zero at
the window walls, i.e. a parabolic piston pushing in ``-y`` ... with the sign
of ``uY_tar`` fixed by ``tanh``.

The same module is imported by this demo's ``main.py`` and evaluated at the
AFSI node dofs, so both codes use an identical forcing formula.  MATLAB's
1-based indices are translated to 0-based; the index search of the .m file
(first ``x(j) >= xMin`` / last ``x(j) <= xMax``) is reproduced with
``np.searchsorted``.  The .m file allocates ``zeros(Ny, Nx)`` and indexes
``fx(j, i)``, which is the (y, x) layout used here.
"""
import numpy as np

K_STIFF = 1e4
W = 0.359550 - 0.247375
MID_POINT = (0.359550 + 0.247375) / 2
U_MAX = 2.5
X_MIN, X_MAX = 0.252, 0.364
Y_MIN, Y_MAX = 0.3625, 0.3675


def _indices_to_apply_force(x, y):
    """``give_Me_Indices_To_Apply_Force`` -> 0-based (i, j) pairs."""
    ix_min = int(np.searchsorted(x, X_MIN, side="left"))
    ix_max = int(np.searchsorted(x, X_MAX, side="right")) - 1
    iy_min = int(np.searchsorted(y, Y_MIN, side="left"))
    iy_max = int(np.searchsorted(y, Y_MAX, side="right")) - 1
    ii, jj = np.meshgrid(np.arange(ix_min, ix_max + 1),
                         np.arange(iy_min, iy_max + 1))
    return np.column_stack([ii.ravel(), jj.ravel()])


def _target_velocity(t, x_grid):
    """``please_Give_Target_Velocity`` (y-component only)."""
    scale = ((MID_POINT + W / 2) - MID_POINT) * ((MID_POINT - W / 2)
                                                 - MID_POINT)
    return U_MAX * np.tanh(2 * t) * ((MID_POINT + W / 2) - x_grid) \
        * ((MID_POINT - W / 2) - x_grid) / scale


def please_Compute_External_Forcing(dt, current_time, x, y, grid_Info, uX, uY,
                                    first, inds):
    Nx, Ny = int(grid_Info[0]), int(grid_Info[1])
    if first == 1:
        inds = _indices_to_apply_force(x, y)
        first = 0
    fx = np.zeros((Ny, Nx))
    fy = np.zeros((Ny, Nx))
    ii = inds[:, 0].astype(int)
    jj = inds[:, 1].astype(int)
    uy_tar = _target_velocity(current_time, x[ii])
    fx[jj, ii] -= K_STIFF * uX[jj, ii]
    fy[jj, ii] -= K_STIFF * (uY[jj, ii] - uy_tar)
    return fx, fy, first, inds
