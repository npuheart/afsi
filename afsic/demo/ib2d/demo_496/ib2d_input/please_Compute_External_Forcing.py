"""Port of ``please_Compute_External_Forcing.m``
(``Examples_First_Year_Seminar/Flow_Around_Object``).

pyIB2d's driver imports this module from the run directory once per step when
``arb_ext_force`` is on and calls

    Fx, Fy, first, inds = please_Compute_External_Forcing(
        dt, current_time, x, y, grid_Info, uX, uY, first, inds)

with ``x``/``y`` the 1-D node coordinates, ``uX``/``uY`` the (Ny, Nx) Eulerian
velocity arrays, and adds ``(Fx, Fy)`` to the Eulerian force before the fluid
solve.  The example drives a **steady parabolic channel profile that ramps up
as ``tanh(t)``** through a velocity-target penalty force applied on the window
``x in [0.1, 0.16]``, ``y in [0.41, 0.59]``:

    Fx = -kStiff (uX - uX_tar),   uX_tar = -uMax 5 tanh(t) (
        (Lx/2 + w/2) - y ) ( (Lx/2 - w/2) - y ),   Fy = 0,

with ``kStiff = 1e4`` (note the *softer* 1e2 of the cylinder example demo_468),
``w = 0.2``, ``uMax = 250``.  The flow is a **body force** on the fluid, not a
prescribed boundary velocity: IB2d applies it as a momentum source over the
window and the periodic box closes the loop.

The very same module is imported by this demo's ``main.py`` and evaluated at
the AFSI node dofs, so both codes use an identical forcing formula.  MATLAB's
1-based indices are translated to 0-based here; the index search of the .m file
(first ``x(j) >= xMin`` / last ``x(j) <= xMax``) is reproduced with
``np.searchsorted``.  The .m file's ``zeros(Nx, Ny)`` is only correct because
this example is square (``Nx == Ny``); it is stored as (Ny, Nx) here, which is
also what the driver's ``fx(j, i)`` indexing means (``j`` = y-index).
"""
import numpy as np

K_STIFF = 1e4
W = 0.2
U_MAX = 250.0
X_MIN, X_MAX = 0.1, 0.16
Y_MIN, Y_MAX = 0.41, 0.59


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
    """``please_Give_Target_Velocity`` (x-component only)."""
    return -U_MAX * (5 * np.tanh(t)) * ((Lx / 2 + W / 2) - y_grid) \
        * ((Lx / 2 - W / 2) - y_grid)


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
