"""Port of ``please_Compute_External_Forcing.m`` (Example_Mass_Points/Channel_Flow_Cellular_Race).

pyIB2d imports this module from the run directory once per step when
``arb_ext_force`` is on and calls

    Fx, Fy, first, inds = please_Compute_External_Forcing(
        dt, current_time, x, y, grid_Info, uX, uY, first, inds)

with ``x``/``y`` the 1-D node coordinates (``0:dx:Lx-dx``, ``0:dy:Ly-dy``) and
``uX``/``uY`` the (Ny, Nx) velocity arrays; ``(Fx, Fy)`` is added to the
Eulerian force density before the fluid solve.  The very same module is
imported by this demo's ``main.py``, so both codes use one implementation.

The example drives **three horizontal (x-directed) jets** through the thin band
``x in [0.1, 0.16]``, centred at ``y = 0.11, 0.41, 0.71`` (the MATLAB
``yShift`` shifts the *parabolic y-profile* of each jet), with a
velocity-target penalty force

    Fx -= kStiff (uX - uX_tar),
    uX_tar = -uMax 5 tanh(t) ((Lx/2 + w/2 + yShift) - y)
                             ((Lx/2 - w/2 + yShift) - y),

``kStiff = 1e4``, ``w = 0.2``, ``uMax = 250``; ``uY_tar = 0`` (so ``Fy`` only
penalises any spurious y-velocity inside the windows).  The jets are switched
on smoothly (``tanh``) and then held, i.e. this is a steady "channel flow" the
massive cells have to swim against -- the counterpart of the oscillating
``Gravity_vs_Flow`` forcing (demo_570).

Note the upstream quirk that the parabolic profile is written in terms of
``Lx/2`` while its argument is ``y``; for this example ``Lx = Ly = 1``, so the
two are numerically identical and the port keeps the .m file's expression.

MATLAB's 1-based indices are translated to 0-based here; the index search of
the .m file (first ``x(j) >= xMin`` / last ``x(j) <= xMax``) is reproduced with
``np.searchsorted``.  The upstream ``inds`` container (a MATLAB array whose
first row stores the two window sizes, followed by the two index blocks) is
reproduced verbatim so that the ported extraction code is line-for-line the
same.
"""
import numpy as np

K_STIFF = 1e4                     # stiffness parameter
W = 0.2                           # width of the target-velocity window
U_MAX = 250.0
X_MIN, X_MAX = 0.1, 0.16          # thin band in x (the jets blow in x)
Y_MIN, Y_MAX = 0.41, 0.59         # y-window of the first jet
Y_SHIFTS = (0.0, 0.3, -0.3)       # y-shift of the three jet windows


def _indices_to_apply_force(x, y, x_min, x_max, y_min, y_max):
    """``give_Me_Indices_To_Apply_Force`` -> 0-based (i, j) pairs (i: x)."""
    ix_min = int(np.searchsorted(x, x_min, side="left"))
    ix_max = int(np.searchsorted(x, x_max, side="right")) - 1
    iy_min = int(np.searchsorted(y, y_min, side="left"))
    iy_max = int(np.searchsorted(y, y_max, side="right")) - 1
    ii, jj = np.meshgrid(np.arange(ix_min, ix_max + 1),
                         np.arange(iy_min, iy_max + 1))
    return np.column_stack([ii.ravel(), jj.ravel()]).astype(float)


def _target_velocity(t, y_grid, j, Lx, y_shift):
    """``please_Give_Target_Velocity`` (x-component only)."""
    y = y_grid[j]
    ux_tar = -U_MAX * (5 * np.tanh(t)) \
        * ((Lx / 2 + W / 2 + y_shift) - y) * ((Lx / 2 - W / 2 + y_shift) - y)
    return ux_tar, 0.0


def _force_density(t, x, y, uX, uY, inds, Lx, y_shift):
    """``give_Me_Velocity_Target_External_Force_Density`` for one window."""
    fx = np.zeros_like(uX)
    fy = np.zeros_like(uX)
    for i, j in inds.astype(int):
        ux_tar, uy_tar = _target_velocity(t, y, j, Lx, y_shift)
        fx[j, i] -= K_STIFF * (uX[j, i] - ux_tar)
        fy[j, i] -= K_STIFF * (uY[j, i] - uy_tar)
    return fx, fy


def please_Compute_External_Forcing(dt, current_time, x, y, grid_Info, uX, uY,
                                    first, inds):
    Nx, Ny, Lx, Ly = (int(grid_Info[0]), int(grid_Info[1]),
                      float(grid_Info[2]), float(grid_Info[3]))

    if first == 1:
        blocks = [_indices_to_apply_force(x, y, X_MIN, X_MAX,
                                          Y_MIN + s, Y_MAX + s)
                  for s in Y_SHIFTS]
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
