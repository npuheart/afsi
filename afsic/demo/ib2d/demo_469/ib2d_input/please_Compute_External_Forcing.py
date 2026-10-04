"""Port of ``please_Compute_External_Forcing.m``
(Example_Middle_Channel_Flow).

pyIB2d's driver imports this module from the run directory once per step when
``arb_ext_force`` is on, calls

    Fx, Fy, first, inds = please_Compute_External_Forcing(
        dt, current_time, x, y, grid_Info, uX, uY, first, inds)

with ``x``/``y`` the 1-D node coordinates (``0:dx:Lx-dx``, ``0:dy:Ly-dy``) and
``uX``/``uY`` the (Ny, Nx) velocity arrays, and adds ``(Fx, Fy)`` to the Eulerian
force before the fluid solve.  
The example drives a **velocity-target penalty jet** over the window
``x in [0.47, 0.53]``, ``y in [0.4025, 0.5975]`` (as read back from the module):

    Fx -= kStiff (uX - uX_tar),   uX_tar = -uMax 5|sin(10 pi t)| ((Lx/2 + w/2) - y)((Lx/2 - w/2) - y)   Fy = 0,

with ``kStiff = 1e4``, ``W = 0.2`` and ``u_max = 200``.
The same module is imported by this demo's ``main.py`` and evaluated at the
AFSI node dofs, so both codes use an identical forcing formula.  MATLAB's
1-based indices are translated to 0-based here; the index search of the .m file
(first ``x(j) >= xMin`` / last ``x(j) <= xMax``) is reproduced by
``np.searchsorted``.
"""
import numpy as np

K_STIFF = 1e4                     # stiffness parameter
W = 0.2                           # width of the target-velocity window
U_MAX = 200.0
X_MIN, X_MAX = 0.47, 0.53
Y_MIN, Y_MAX = 0.4025, 0.5975


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
    ux_tar = -U_MAX * (5*np.abs(np.sin(10*np.pi*t))) \
        * ((Lx / 2 + W / 2) - y) * ((Lx / 2 - W / 2) - y)
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
