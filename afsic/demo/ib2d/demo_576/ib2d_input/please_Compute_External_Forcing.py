"""Port of ``please_Compute_External_Forcing.m`` (Example_Concentration_Dynamics/Leaf_64).

pyIB2d imports this module from the run directory once per step when
``arb_ext_force`` is on and calls

    Fx, Fy, first, inds = please_Compute_External_Forcing(
        dt, current_time, x, y, grid_Info, uX, uY, first, inds)

with ``x``/``y`` the 1-D node coordinates (``0:dx:Lx-dx``, ``0:dy:Ly-dy``) and
``uX``/``uY`` the (Ny, Nx) velocity arrays; ``(Fx, Fy)`` is added to the
Eulerian force density before the fluid solve.  The very same module is
imported by this demo's ``main.py``, so both codes use one implementation.

A **horizontal wind** is imposed in the strip ``x in [0, 0.16]`` over
``y in [0.005, 5]`` (the whole height of the 30 x 5 domain) by a
velocity-target penalty force

    Fx -= kStiff (uX - uX_tar),   uX_tar = uMax tanh(t),   uY_tar = 0,

``kStiff = 0.1`` (very soft), ``uMax = 5 m/s``.  The leaf's petiole is pinned by
target points at ``x in [2.94, 3.2]``, so the free part of the leaf bends in
this slowly ramping wind; the leaf's 26 ghost masses (``M = 5e-4``, ``k = 1``)
make the free part much heavier than the fluid it displaces.

The .m file's ``w = 2.5`` only appears in the *commented out* parabolic profile,
so the active target velocity is ``uMax tanh(t)`` (a plug-like wind); the port
keeps the upstream line verbatim.

MATLAB's 1-based indices are translated to 0-based here; the index search of
the .m file (first ``x(j) >= xMin`` / last ``x(j) <= xMax``) is reproduced with
``np.searchsorted``.
"""
import numpy as np

K_STIFF = 1e-1                    # stiffness parameter (very soft)
W = 2.5                           # only used by the commented-out profile
U_MAX = 5.0                       # maximum wind speed (m/s)
X_MIN, X_MAX = 0.0, 0.16          # the vertical forcing strip at the inlet
Y_MIN, Y_MAX = 0.005, 5.0


def _indices_to_apply_force(x, y):
    """``give_Me_Indices_To_Apply_Force`` -> 0-based (i, j) pairs (i: x)."""
    ix_min = int(np.searchsorted(x, X_MIN, side="left"))
    ix_max = int(np.searchsorted(x, X_MAX, side="right")) - 1
    iy_min = int(np.searchsorted(y, Y_MIN, side="left"))
    iy_max = int(np.searchsorted(y, Y_MAX, side="right")) - 1
    ii, jj = np.meshgrid(np.arange(ix_min, ix_max + 1),
                         np.arange(iy_min, iy_max + 1))
    return np.column_stack([ii.ravel(), jj.ravel()]).astype(float)


def please_Compute_External_Forcing(dt, current_time, x, y, grid_Info, uX, uY,
                                    first, inds):
    Nx, Ny, Lx, Ly = (int(grid_Info[0]), int(grid_Info[1]),
                      float(grid_Info[2]), float(grid_Info[3]))

    if first == 1:
        inds = _indices_to_apply_force(x, y)
        first = 0

    ux_tar = U_MAX * np.tanh(current_time)     # SOLELY HORIZONTAL FLOW PROFILE
    fx = np.zeros_like(uX)
    fy = np.zeros_like(uX)
    for i, j in inds.astype(int):
        fx[j, i] -= K_STIFF * (uX[j, i] - ux_tar)
        fy[j, i] -= K_STIFF * uY[j, i]

    return fx, fy, first, inds
