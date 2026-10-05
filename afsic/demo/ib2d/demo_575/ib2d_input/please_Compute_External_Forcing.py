"""Port of ``please_Compute_External_Forcing.m`` (Example_Baby_Spider_Shower/Flow_Based).

pyIB2d imports this module from the run directory once per step when
``arb_ext_force`` is on and calls

    Fx, Fy, first, inds = please_Compute_External_Forcing(
        dt, current_time, x, y, grid_Info, uX, uY, first, inds)

with ``x``/``y`` the 1-D node coordinates (``0:dx:Lx-dx``, ``0:dy:Ly-dy``) and
``uX``/``uY`` the (Ny, Nx) velocity arrays; ``(Fx, Fy)`` is added to the
Eulerian force density before the fluid solve.  The very same module is
imported by this demo's ``main.py``, so both codes use one implementation.

A **horizontal wind** is imposed in the vertical strip ``x in [0.05, 0.09]``
over the whole height (``y in [0, Ly]``) by a velocity-target penalty force

    Fx -= kStiff (uX - uX_tar),   uX_tar = uMax tanh(5 t),   uY_tar = 0,

``kStiff = 1e4``, ``uMax = 0.2 m/s`` (the "shower" of a still-air spider
thread: the wind is 0.2 m/s and the spider's ghost mass is pulled down by
gravity).  The ``tanh`` switch-on over ~0.2 s avoids an impulsive start.

MATLAB's 1-based indices are translated to 0-based here; the index search of
the .m file (first ``x(j) >= xMin`` / last ``x(j) <= xMax``) is reproduced with
``np.searchsorted``.
"""
import numpy as np

K_STIFF = 1e4                     # stiffness parameter
U_MAX = 0.2                       # maximum wind speed (m/s)
X_MIN, X_MAX = 0.05, 0.09         # the vertical forcing strip
TANH_RATE = 5.0                   # tanh switch-on rate of the .m file


def _indices_to_apply_force(x, y, Ly):
    """``give_Me_Indices_To_Apply_Force`` -> 0-based (i, j) pairs (i: x)."""
    ix_min = int(np.searchsorted(x, X_MIN, side="left"))
    ix_max = int(np.searchsorted(x, X_MAX, side="right")) - 1
    iy_min = int(np.searchsorted(y, 0.0, side="left"))
    iy_max = int(np.searchsorted(y, Ly, side="right")) - 1
    ii, jj = np.meshgrid(np.arange(ix_min, ix_max + 1),
                         np.arange(iy_min, iy_max + 1))
    return np.column_stack([ii.ravel(), jj.ravel()]).astype(float)


def please_Compute_External_Forcing(dt, current_time, x, y, grid_Info, uX, uY,
                                    first, inds):
    Nx, Ny, Lx, Ly = (int(grid_Info[0]), int(grid_Info[1]),
                      float(grid_Info[2]), float(grid_Info[3]))

    if first == 1:
        inds = _indices_to_apply_force(x, y, Ly)
        first = 0

    ux_tar = U_MAX * np.tanh(TANH_RATE * current_time)
    fx = np.zeros_like(uX)
    fy = np.zeros_like(uX)
    for i, j in inds.astype(int):
        fx[j, i] -= K_STIFF * (uX[j, i] - ux_tar)
        fy[j, i] -= K_STIFF * (uY[j, i] - 0.0)

    return fx, fy, first, inds
