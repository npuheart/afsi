"""Line-by-line port of the example's ``update_Springs.m``
(``Example_Jellyfish_Swimming/Failed_Attempts/Jellyfish_Springs_Only``).

pyIB2d's driver imports this module from the run directory (once per step, when
``update_springs`` is on) and calls

    springs_info = update_Springs(dt, current_time, xLag, yLag, springs_info)

with ``springs_info`` the (N_springs, 5) array ``[leader, follower, k, resting
length, alpha]`` and the *full-step* marker positions ``xLag``/``yLag``.

This is one of the authors' own ``Failed_Attempts/``: only the springs
are active (no beams, no target points), so the bell has no bending stiffness.
The .m file drives the 25 springs of the bell margin (MATLAB rows ``107:131``)
with the hand-tabulated resting lengths ``distVec``,

    RL_i = distVec_i * (1 - 0.75 sin(4 pi t)),

which is *not* an absolute value: the resting lengths alternate between 0.25x
and 1.75x of ``distVec`` twice per second.

MATLAB's 1-based indexing is translated to 0-based here; the identifiers
(1..N) are rebased by the reader (``ib2d_io.load_example`` auto-detects the
1-based numbering of this matIB2d example).
"""
import numpy as np

# hand-tabulated (MATLAB distVec, rows 107:131)
DIST_VEC = np.array([
    8.660254e-02, 8.829346e-02, 8.985511e-02, 9.129139e-02, 9.260716e-02,
    9.380641e-02, 9.489209e-02, 9.586726e-02, 9.673485e-02, 9.749671e-02,
    9.815495e-02, 9.871098e-02, 9.916637e-02, 9.952229e-02, 9.977934e-02,
    9.993819e-02, 9.999924e-02, 9.996260e-02, 9.982821e-02, 9.959568e-02,
    9.926457e-02, 9.883399e-02, 9.830305e-02, 9.767028e-02, 9.693457e-02])
FIRST_ROW = 107                       # MATLAB springs_info(107+(i-1), 4)
DIST = 8.649829e-02                   # MATLAB dist (tabulated reference length)


def update_Springs(dt, current_time, xLag, yLag, springs):
    rl = DIST_VEC * (1.0 - 0.75 * np.sin(4.0 * np.pi * current_time))
    springs[FIRST_ROW - 1:FIRST_ROW - 1 + DIST_VEC.size, 3] = rl
    return springs
