"""Pumping springs of IB2d -- line-by-line port of the example's own

    matIB2d/Examples/Examples_First_Year_Seminar/Plates/update_Springs.m

(the ``update_springs = 1`` user module pyIB2d imports from the run directory
once per step; ``main.py`` imports the very same file, so both codes drive the
"heart tube" with one implementation).

Only the last two springs of the file are time dependent (the two that attach
the plates): their resting length is pumped as

    RL(i) = d - |0.9 d sin(2 pi freq t)|,   freq = 10, d = 1.0,

for ``i = (N-2)+1 .. N_springs`` in MATLAB (1-based) -- i.e. the last two
springs of the ``springs_info`` array, whose rows are
``[master, slave, k, RL, alpha]`` (columns 3/4 of the ``.spring`` file).  The
plates are pushed apart at 10 Hz.
"""
import numpy as np


def update_Springs(dt, current_time, xLag, yLag, springs):
    """MATLAB signature; returns the updated ``springs_info`` array."""
    N = len(xLag)
    N_springs = springs.shape[0]
    freq = 10.0
    # the MATLAB module says ``d = 1.0`` ("diameter of heart tube"), but the
    # geometry this example generates (``Make_Your_Plates_and_Input_Files.m``)
    # places the two plates only ``0.2`` apart, so ``RL`` would be pushed up to
    # 1.0 -- five times the actual separation -- and both codes blow up to NaN
    # within a few steps (measured).  The demo therefore uses the *actual*
    # plate separation ``d = 0.2``; see readme.md (upstream bug).
    d = 0.2
    springs = np.array(springs, dtype=float, copy=True)
    RL = d - abs(0.9 * d * np.sin(2 * np.pi * freq * current_time))
    # MATLAB: for i = (N-2)+1 : N_springs  ->  0-based N-2 .. N_springs-1
    springs[N - 2:, 3] = RL
    return springs
