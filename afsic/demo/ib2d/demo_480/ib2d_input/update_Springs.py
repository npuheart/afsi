"""Line-by-line port of the example's ``update_Springs.m``
(``Example_Jellyfish_Swimming/Hoover_Jellyfish``).

pyIB2d's driver imports this module from the run directory (once per step, when
``update_springs`` is on) and calls

    springs_info = update_Springs(dt, current_time, xLag, yLag, springs_info)

with ``springs_info`` the (N_springs, 5) array ``[leader, follower, k, resting
length, alpha]`` and the *full-step* marker positions ``xLag``/``yLag``.  The .m
file drives the jellyfish bell: the resting lengths of MATLAB rows 319:357
(the 39 springs of the bell margin) are contracted periodically,

    RL_i = |cos(freq t pi)|,   freq = 2  ->  period 1 s,

so the bell pulses twice per second (|cos| gives two contractions per period).

MATLAB's 1-based indexing is translated to 0-based here; the identifiers
(1..N) are rebased by the reader (``ib2d_io.load_example`` auto-detects the
1-based numbering of this matIB2d example).
"""
import numpy as np

FREQ = 2.0        # contraction frequency (Hz-ish): RL = |cos(freq t pi)|
FIRST_ROW = 319   # MATLAB springs_info(319:357, 4)
LAST_ROW = 357


def update_Springs(dt, current_time, xLag, yLag, springs):
    springs[FIRST_ROW - 1:LAST_ROW, 3] = \
        abs(np.cos(FREQ * current_time * np.pi))
    return springs
