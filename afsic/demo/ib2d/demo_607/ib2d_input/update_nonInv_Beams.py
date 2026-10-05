"""Line-by-line port of the example's ``update_nonInv_Beams.m``.

pyIB2d's driver imports this module from the run directory (once per step, when
``update_nonInv_beams`` is on)::

    from update_nonInv_Beams import update_nonInv_Beams
    nonInv_beams_info = update_nonInv_Beams(dt, current_time, beams_info)

``beams_info`` is the ``(N, 6)`` array ``[p1, p2, p3, kb, Cx, Cy]``.  The
education-interpolation swimmer body is interpolated between the two phases of
``swimmer.phases`` (``x1, y1, y2``) with a *piecewise cubic* in the normalised
phase time (knots ``p1 = 0.10``, ``p2 = 0.90``),

    g(s) = a3 s^3                          (s <= p1)
    g(s) = b0 + b1 s + b2 s^2 + b3 s^3     (p1 < s <= p2)
    g(s) = c0 + c1 s + c2 s^2 + c3 s^3     (otherwise)

with a down-stroke and an up-stroke of 0.25 s each and no rest (``tR = 0``).
Only the **y** geometry moves (the x lines are commented out in the .m, and the
``Cx`` line is commented out too, so column 5 keeps its file value); the
reference y-curvature is the second difference of the interpolated polyline,

    Cy = yPts[0:-2] + yPts[2:] - 2 yPts[1:-1].

Upstream quirks:

* the .m's ``if tTilde<=p1 / elseif (tTilde>p1)&&(tTilde<=p2) / else`` covers
  ``[0, 1]`` but its two branches overlap at the knot ``p1`` (the first wins);
* the assignment again ignores the beam connectivity and relies on the
  ``.nonInv_beam`` ordering (checked: the 182 beams span consecutive points).

Note: ``swimmer.target`` pins 256 extra markers (k = 1e6) -- the "head" of this
education example.
"""
import os

import numpy as np

T1 = 0.25                         # down-stroke [s]
T2 = 0.25                         # up-stroke [s]
TR = 0.0                          # resting time [s]
PERIOD = T1 + T2 + TR
P1_, P2_ = 0.10, 0.90
A0, A1, A2, A3 = 0.0, 0.0, 0.0, 11.111111111111088
B0, B1, B2, B3 = 0.013888888888889, -0.416666666666666, 4.166666666666658, \
    -2.777777777777770
C0, C1, C2, C3 = -10.111111111111073, 33.333333333333222, \
    -33.333333333333222, 11.111111111111073

_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "swimmer.phases")
_CACHE = {}


def _phases():
    if "P" not in _CACHE:
        _CACHE["P"] = np.loadtxt(_FILE, ndmin=2)[:, :3]
    return _CACHE["P"]


def _g(s):
    if s <= P1_:
        return A0 + A1 * s + A2 * s ** 2 + A3 * s ** 3
    if s <= P2_:
        return B0 + B1 * s + B2 * s ** 2 + B3 * s ** 3
    return C0 + C1 * s + C2 * s ** 2 + C3 * s ** 3


def update_nonInv_Beams(dt, current_time, beams_info):
    P = _phases()
    yP1, yP2 = P[:, 1], P[:, 2]

    t = np.remainder(current_time, PERIOD)
    if t <= T1:
        g = _g(t / T1)
        yPts = yP1 + g * (yP2 - yP1)
    elif t >= T1 + TR:
        g = _g((t - T1 - TR) / T2)
        yPts = yP2 + g * (yP1 - yP2)
    else:
        yPts = yP2

    beams_info[:, 5] = yPts[:-2] + yPts[2:] - 2.0 * yPts[1:-1]
    return beams_info
