"""Line-by-line port of the example's ``update_Springs.m``
(``Example_HeartTube/Sea_Spiders``).

pyIB2d's driver imports this module from the run directory once per step (when
``update_springs`` is on) and calls

    springs_info = update_Springs(dt, current_time, xLag, yLag, springs_info)

with ``springs_info`` the (N_springs, 5) array ``[leader, follower, k, resting
length, alpha]`` and ``xLag``/``yLag`` the *full-step* marker positions.  The
.m file drives a **peristaltic wave** along the sea-spider gut:

    gutD = 0.05 (gut diameter), L = 0.6 (gut length), Lwave = L/4,
    freq = 2/0.6, v = L*freq, xM = 0.2,
    for i = 50 : 1229-50   (MATLAB rows, 1229 springs total)
        delta = sin( (pi/Lwave) ( (x_i - xM) - v t ) )
        RL_i  = gutD (1 - 0.9 delta)   if delta >= 0
              = gutD                   otherwise

i.e. only the *positive* half of the travelling sine pinches the 1130 innermost
springs (the gut wall), leaving the resting length ``gutD`` elsewhere; the
centre of the pinch travels at ``v = 2 m/s`` to the right.

MATLAB's 1-based indexing is translated to 0-based here: rows ``50 : 1179``
become ``49 : 1178`` and column 4 becomes column 3.  The identifiers are
rebased by the reader (``ib2d_io.load_example``, ``index_base=1``).
"""
import numpy as np

GUT_DIAMETER = 0.05
GUT_LENGTH = 0.6
WAVE_LENGTH = GUT_LENGTH / 4.0
FREQ = 2.0 / 0.6
X_MID = 0.2
FIRST_ROW = 50            # MATLAB row, 1-based
LAST_ROW = 1229 - 50      # MATLAB row, 1-based


def update_Springs(dt, current_time, xLag, yLag, springs):
    v = GUT_LENGTH * FREQ
    t = np.remainder(current_time, 1.0 / FREQ)
    i0, i1 = FIRST_ROW - 1, LAST_ROW      # 0-based inclusive slice
    x = xLag[i0:i1 + 1]
    delta = np.sin((np.pi / WAVE_LENGTH) * ((x - X_MID) - v * t))
    rl = np.where(delta >= 0.0, GUT_DIAMETER * (1.0 - 0.90 * delta),
                  GUT_DIAMETER)
    springs[i0:i1 + 1, 3] = rl
    return springs
