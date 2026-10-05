"""Line-by-line port of the example's ``update_nonInv_Beams.m``
(``Examples_Education/Interpolation/Swimmer``).

pyIB2d's driver imports this module from the run directory (once per step, when
``update_nonInv_beams`` is on) and calls

    nonInv_beams_info = update_nonInv_Beams(dt, current_time, nonInv_beams_info)

with ``nonInv_beams_info`` the (N_beams, 6) array ``[p1, p2, p3, kb, Cx, Cy]``
(``C`` is the reference vector ``p1 + p3 - 2 p2`` at rest).  The .m file swims
the body by rewriting the *y*-component of the reference curvature, column 6:
the y-positions of the swimmer's backbone are interpolated between two stored
"phases" (``swimmer.phases``) with a piecewise cubic over a 0.5 s period
(0.25 s down-stroke + 0.25 s up-stroke), and

    Cy = yPts[:-2] + yPts[2:] - 2 yPts[1:-1]

is the discrete second difference of that backbone.

The phases file is looked up next to this module first (``main.py`` imports the
module from ``ib2d_input/`` and runs in the demo directory) and falls back to
the current working directory (pyIB2d imports it from the scratch run
directory, where ``run_reference.py`` copies it verbatim).
"""
import os

import numpy as np

T1 = 0.25          # down-stroke (s)
T2 = 0.25          # up-stroke (s)
TR = 0.0           # resting time between the two strokes
PERIOD = T1 + T2 + TR
P1 = 0.10
P2 = 0.90
A_ = (0.0, 0.0, 0.0, 11.111111111111088)
B_ = (0.013888888888889, -0.416666666666666, 4.166666666666658,
      -2.777777777777770)
C_ = (-10.111111111111073, 33.333333333333222, -33.333333333333222,
      11.111111111111073)


def _phases():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "swimmer.phases")
    if not os.path.exists(path):
        path = "swimmer.phases"
    rows = np.loadtxt(path)
    return rows[:, 0], rows[:, 1], rows[:, 2]


_PHASES = None


def _get_phases():
    global _PHASES
    if _PHASES is None:
        _PHASES = _phases()
    return _PHASES


def _gfunc(t_tilde):
    coef = A_ if t_tilde <= P1 else (B_ if t_tilde <= P2 else C_)
    return (coef[0] + coef[1] * t_tilde + coef[2] * t_tilde ** 2
            + coef[3] * t_tilde ** 3)


def update_nonInv_Beams(dt, current_time, beams_info):
    t = np.mod(current_time, PERIOD)
    xP1, yP1, yP2 = _get_phases()
    del xP1                              # the .m notes xP1 == xP2
    yPts = None
    if t <= T1:                          # STATE A -> STATE B
        yPts = yP1 + _gfunc(t / T1) * (yP2 - yP1)
    elif t >= T1 + TR:                   # STATE B -> STATE A
        yPts = yP2 + _gfunc((t - T1 - TR) / T2) * (yP1 - yP2)
    beams_info[:, 5] = yPts[:-2] + yPts[2:] - 2 * yPts[1:-1]
    return beams_info
