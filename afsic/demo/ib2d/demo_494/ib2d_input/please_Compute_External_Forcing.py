"""Port of ``please_Compute_External_Forcing.m``
(``Examples_First_Year_Seminar/Leaf``).

A uniform in-``x`` suction/blowing jet on a thin horizontal window
(``x in [0.01 + offset, 0.13 + offset]`` with ``offset = 0.03``,
``y in [0.005, 0.009]``):

    Fy = -kStiff (uY - uY_tar),
    uY_tar = -uMax (poly_coeff * tanh(100 t)) (
        (Lx/2 + w/2) - x ) ( (Lx/2 - w/2) - x ),
    poly_coeff = 2.5 / | ( (Lx/2 + w/2) - Lx/2 ) ( (Lx/2 - w/2) - Lx/2 ) |,

``kStiff = 1e3``, ``uMax = 5``, ``w = 0.08``, ``Lx = 0.2``; ``tanh(100 t)``
saturates almost immediately, so the leaf is hit by a nearly constant
``-y`` jet of magnitude ``5 * 2.5 = 12.5`` at the window centre (the
``poly_coeff`` normalises the parabola to ``2.5`` at ``x = Lx/2``).  ``Fx = 0``.

MATLAB's 1-based indices are translated to 0-based; the index search of the .m
file (first ``x(j) >= xMin`` / last ``x(j) <= xMax``) is reproduced with
``np.searchsorted``.  The .m file allocates ``zeros(Ny, Nx)``; this port uses
that (y, x) layout.
"""
import numpy as np

K_STIFF = 1e3
W = 0.08
U_MAX = 5.0
OFFSET = 0.03
X_MIN, X_MAX = 0.01 + OFFSET, 0.13 + OFFSET
Y_MIN, Y_MAX = 0.005, 0.009


def _indices_to_apply_force(x, y):
    """``give_Me_Indices_To_Apply_Force`` -> 0-based (i, j) pairs."""
    ix_min = int(np.searchsorted(x, X_MIN, side="left"))
    ix_max = int(np.searchsorted(x, X_MAX, side="right")) - 1
    iy_min = int(np.searchsorted(y, Y_MIN, side="left"))
    iy_max = int(np.searchsorted(y, Y_MAX, side="right")) - 1
    ii, jj = np.meshgrid(np.arange(ix_min, ix_max + 1),
                         np.arange(iy_min, iy_max + 1))
    return np.column_stack([ii.ravel(), jj.ravel()])


def _target_velocity(t, x_grid, Lx):
    """``please_Give_Target_Velocity`` (y-component only)."""
    poly_coeff = 2.5 / abs(((Lx / 2 + W / 2) - Lx / 2)
                           * ((Lx / 2 - W / 2) - Lx / 2))
    return -U_MAX * (poly_coeff * np.tanh(100 * t)) \
        * ((Lx / 2 + W / 2) - x_grid) * ((Lx / 2 - W / 2) - x_grid)


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
    uy_tar = _target_velocity(current_time, x[ii], Lx)
    fx[jj, ii] -= K_STIFF * uX[jj, ii]
    fy[jj, ii] -= K_STIFF * (uY[jj, ii] - uy_tar)
    return fx, fy, first, inds
