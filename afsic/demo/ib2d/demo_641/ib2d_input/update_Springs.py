"""Line-by-line port of the example's ``update_Springs.m``
(``Example_Jellyfish_Swimming/Tentacle_Jelly``).

pyIB2d's driver imports this module from the run directory (once per step, when
``update_springs`` is on) and calls

    springs_info = update_Springs(dt, current_time, xLag, yLag, springs_info)

with ``springs_info`` the (N_springs, 5) array ``[leader, follower, k, resting
length, alpha]`` and the *full-step* marker positions ``xLag``/``yLag``.

The .m file contracts the 39 tentacle springs (MATLAB rows
``starting_muscle:ending_muscle`` = ``318+1 : 318+39``) every step,

    RL_i = |cos(freq t pi)|,   freq = 0.8  ->  period 1.25 s,

so the tentacles beat 1.6 times per second.  The example additionally solves a
poroelastic displacement field (``poroelastic = 1``).

MATLAB's 1-based indexing is translated to 0-based here; the identifiers
(1..N) are rebased by the reader (``ib2d_io.load_example`` auto-detects the
1-based numbering of this matIB2d example).
"""
import numpy as np

FREQ = 0.8        # contraction frequency: RL = |cos(freq t pi)|
FIRST_ROW = 318 + 1        # MATLAB starting_muscle
LAST_ROW = 318 + 39        # MATLAB ending_muscle


def update_Springs(dt, current_time, xLag, yLag, springs):
    springs[FIRST_ROW - 1:LAST_ROW, 3] = \
        abs(np.cos(FREQ * current_time * np.pi))
    return springs
