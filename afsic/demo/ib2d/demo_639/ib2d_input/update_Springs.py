"""Line-by-line port of the example's ``update_Springs.m``
(``Examples_Education/Convergence/Jellyfish/Simulation_Skeletons/Re150/Res_256_320x96``).

pyIB2d's driver imports this module from the run directory (once per step, when
``update_springs`` is on) and calls

    springs_info = update_Springs(dt, current_time, xLag, yLag, springs_info)

with ``springs_info`` the (N_springs, 5) array ``[leader, follower, k, resting
length, alpha]`` and the *full-step* marker positions ``xLag``/``yLag``.  The .m
file makes the jellyfish bell pulsate by re-setting the resting lengths of the
springs of the bell margin (MATLAB rows ``159:end`` of the 177-spring
jelly256 structure) every step:

    RL_i = |cos(freq t pi)|,   freq = 1  ->  period 1 s,

``|cos|`` gives two contractions per period, so the bell pulses twice per
second.

MATLAB's 1-based indexing is translated to 0-based here; the identifiers
(1..N) are rebased by the reader (``ib2d_io.load_example`` auto-detects the
1-based numbering of this matIB2d example).
"""
import numpy as np

FREQ = 1.0        # contraction frequency: RL = |cos(freq t pi)|
FIRST_ROW = 159    # MATLAB springs_info(159:end, 4)


def update_Springs(dt, current_time, xLag, yLag, springs):
    springs[FIRST_ROW - 1:, 3] = abs(np.cos(FREQ * current_time * np.pi))
    return springs
