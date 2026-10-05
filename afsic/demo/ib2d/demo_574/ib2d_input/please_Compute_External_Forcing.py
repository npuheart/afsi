"""Port of ``please_Compute_External_Forcing.m`` (Example_Mass_Points/Gravity_vs_Flow_256x256).

pyIB2d imports this module from the run directory once per step when
``arb_ext_force`` is on and calls

    Fx, Fy, first, inds = please_Compute_External_Forcing(
        dt, current_time, x, y, grid_Info, uX, uY, first, inds)

with ``x``/``y`` the 1-D node coordinates (``0:dx:Lx-dx``, ``0:dy:Ly-dy``) and
``uX``/``uY`` the (Ny, Nx) velocity arrays; ``(Fx, Fy)`` is added to the
Eulerian force density before the fluid solve.  The very same module is
imported by this demo's ``main.py``, so both codes use one implementation.

The fine-grid twin of demo_570's forcing: **three vertical (y-directed) jets**
through the thin band ``y in [0.1, 0.13]`` (the 256 x 256 version narrows the
band from ``[0.1, 0.16]``), centred at ``x = 0.2, 0.5, 0.8`` via the MATLAB
``yShift`` of the parabolic x-profile, with

    Fy -= kStiff (uY - uY_tar),
    uY_tar = -uMax 0.5 |sin(20 pi t)| ((Lx/2 + w/2 + yShift) - x)
                                      ((Lx/2 - w/2 + yShift) - x),

``kStiff = 1e4``, ``w = 0.2``, ``uMax = 250``; ``uX_tar = 0``.  The jets are
switched on/off at 10 Hz.

MATLAB's 1-based indices are translated to 0-based here; the index search of
the .m file (first ``x(j) >= xMin`` / last ``x(j) <= xMax``) is reproduced with
``np.searchsorted``.  The upstream ``inds`` container (a MATLAB array whose
first row stores the three window sizes, followed by the three index blocks) is
reproduced verbatim so that the ported extraction code is line-for-line the
same.
"""
import numpy as np

K_STIFF = 1e4                     # stiffness parameter
W = 0.2                           # width of the target-velocity window
U_MAX = 250.0
Y_MIN, Y_MAX = 0.1, 0.13          # thin band in y (the jets blow in y)
X_MIN, X_MAX = 0.41, 0.59         # x-window of the first jet
X_SHIFTS = (0.0, 0.3, -0.3)       # x-shift of the three jet windows


def _indices_to_apply_force(x, y, x_min, x_max, y_min, y_max):
    """``give_Me_Indices_To_Apply_Force`` -> 0-based (i, j) pairs (i: x)."""
    ix_min = int(np.searchsorted(x, x_min, side="left"))
    ix_max = int(np.searchsorted(x, x_max, side="right")) - 1
    iy_min = int(np.searchsorted(y, y_min, side="left"))
    iy_max = int(np.searchsorted(y, y_max, side="right")) - 1
    ii, jj = np.meshgrid(np.arange(ix_min, ix_max + 1),
                         np.arange(iy_min, iy_max + 1))
    return np.column_stack([ii.ravel(), jj.ravel()]).astype(float)


def _target_velocity(t, x_grid, i, Lx, x_shift):
    """``please_Give_Target_Velocity`` (y-component only)."""
    x = x_grid[i]
    uy_tar = -U_MAX * (0.5 * np.abs(np.sin(20 * np.pi * t))) \
        * ((Lx / 2 + W / 2 + x_shift) - x) * ((Lx / 2 - W / 2 + x_shift) - x)
    return 0.0, uy_tar


def _force_density(t, x, y, uX, uY, inds, Lx, x_shift):
    """``give_Me_Velocity_Target_External_Force_Density`` for one window."""
    fx = np.zeros_like(uX)
    fy = np.zeros_like(uX)
    for i, j in inds.astype(int):
        ux_tar, uy_tar = _target_velocity(t, x, i, Lx, x_shift)
        fx[j, i] -= K_STIFF * (uX[j, i] - ux_tar)
        fy[j, i] -= K_STIFF * (uY[j, i] - uy_tar)
    return fx, fy


def please_Compute_External_Forcing(dt, current_time, x, y, grid_Info, uX, uY,
                                    first, inds):
    Nx, Ny, Lx, Ly = (int(grid_Info[0]), int(grid_Info[1]),
                      float(grid_Info[2]), float(grid_Info[3]))

    if first == 1:
        blocks = [_indices_to_apply_force(x, y, X_MIN + s, X_MAX + s,
                                          Y_MIN, Y_MAX) for s in X_SHIFTS]
        n1, n2 = len(blocks[0]), len(blocks[1])
        inds = np.vstack([np.array([[n1, n2]], float), blocks[0], blocks[1],
                          blocks[2]])
        first = 0

    ids1 = inds[1:1 + int(inds[0, 0])]
    ids2 = inds[1 + int(inds[0, 0]):1 + int(inds[0, 0]) + int(inds[0, 1])]
    ids3 = inds[1 + int(inds[0, 0]) + int(inds[0, 1]):]

    fx1, fy1 = _force_density(current_time, x, y, uX, uY, ids1, Lx, 0.0)
    fx2, fy2 = _force_density(current_time, x, y, uX, uY, ids2, Lx, 0.3)
    fx3, fy3 = _force_density(current_time, x, y, uX, uY, ids3, Lx, -0.3)

    return fx1 + fx2 + fx3, fy1 + fy2 + fy3, first, inds
