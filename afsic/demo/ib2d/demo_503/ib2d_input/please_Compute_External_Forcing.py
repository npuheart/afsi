"""Port of ``please_Compute_External_Forcing.m``
(``Example_VIV_Cylinder/Single_VIV_Cylinder_512``).

**Identical** forcing module to ``Single_VIV_Cylinder_128`` (byte-identical
upstream .m): pulsatile velocity target on ``x in [0.1, 0.16]``,
``y in [0.41, 0.59]``, ``uMax = 75``, ``w = 0.2``, ``kStiff = 1e4``, pulse
``freq_P = 10`` with ``rest = 2 * period``; only the upstream grid differs
(512 x 512 instead of 128 x 128).  See that demo's port for the formula.

Same .m-file quirk: ``zeros(Nx, Ny)`` indexed ``fx(j, i)``, correct only
because the upstream grid is square; the (y, x) layout is used here.

The VIV family drives the flow with a **pulsatile parabolic channel profile**
imposed by a velocity-target penalty force on the window ``x in [0.1, 0.16]``,
``y in [0.41, 0.59]``:

    Fx = -kStiff (uX - uX_tar),   uX_tar = -uMax 5 sin(2 pi f t) (
        (Lx/2 + w/2) - y ) ( (Lx/2 - w/2) - y ),   Fy = 0,

with ``kStiff = 1e4``, ``w = 0.2``, ``uMax = 75`` and the pulsation

    f = freq_P = 10 Hz,  period = 1/f,  rest = 2 * period,
    t_mod = t mod (period + rest),

so the profile is **on for one period and then off for two** (a pulse train
with pauses, the ``Single_VIV_Cylinder_128/512`` flavour).  The y-parabola
vanishes at ``Lx/2 +- w/2`` (the .m file uses ``Lx`` even though the profile is
in ``y``) with peak ``-5 uMax sin(...)`` at the box mid-height.

The same module is imported by this demo's ``main.py`` and evaluated at the
AFSI node dofs, so both codes share the forcing formula.  MATLAB's 1-based
indices are translated to 0-based; the index search of the .m file (first
``x(j) >= xMin`` / last ``x(j) <= xMax``) is reproduced by ``np.searchsorted``.
The .m file allocates ``zeros(Nx, Ny)`` and then indexes ``fx(j, i)``: that is
only correct because this example is square (``Nx == Ny == 128`` in the
upstream grid); here the array is (Ny, Nx), which is what ``fx(j, i)`` means.
"""
import numpy as np

K_STIFF = 1e4
W = 0.2
U_MAX = 75.0
X_MIN, X_MAX = 0.1, 0.16
Y_MIN, Y_MAX = 0.41, 0.59
FREQ_P = 10.0
REST_PERIODS = 2.0


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
    """``please_Give_Target_Velocity`` (x-component only, pulse train)."""
    period = 1.0 / FREQ_P
    tot_period = period + REST_PERIODS * period
    t_mod = np.mod(t, tot_period)
    pulse = np.where(t_mod <= period,
                     -U_MAX * (5 * np.sin(FREQ_P * 2 * np.pi * t)), 0.0)
    return pulse * ((Lx / 2 + W / 2) - y_grid) * ((Lx / 2 - W / 2) - y_grid)


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
