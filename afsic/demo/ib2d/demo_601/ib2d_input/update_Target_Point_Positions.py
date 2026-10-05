"""Line-by-line port of the example's ``update_Target_Point_Positions.m``.

pyIB2d's driver imports this module from the run directory (once per step, only
when ``update_target`` and ``target_pts`` are both on)::

    from update_Target_Point_Positions import update_Target_Point_Positions
    target_info = update_Target_Point_Positions(dt, current_time, target_info)

``targets`` is the ``(N, 4)`` array ``[lag-id, x, y, stiffness]`` built by the
driver.  The Beating Heart is 771 markers: 770 linear springs (k = 1e3) plus
**771 target points** (k = 2e10) whose anchors interpolate between the two heart
shapes in ``State_A.pts`` / ``State_B.pts`` (row i of each file, no header):

    t1 = 0.1 s (A -> B), tR = 0.02 s (rest), t2 = 0.1 s (B -> A),
    period = 0.22 s,  t = current_time mod period,

    X = A + g(t/t1) (B - A)                 (phase 1)
    X = B + g((t-t1-tR)/t2) (A - B)         (phase 3)

with the piecewise cubic

    g(s) = a3 s^3                          for s <= 0.25,
    g(s) = b0 + b1 s + b2 s^2 + b3 s^3     for 0.25 < s <= 0.925,
    g(s) = c0 + c1 s + c2 s^2 + c3 s^3     otherwise

(a3 = 4.324324324324318; b = 0.123456790123457, -1.481481481481478,
5.925925925925911, -3.576910243576897; c = -16.777777777777700,
53.333333333333101, -53.333333333333101, 17.777777777777700), the same blend as
the Moving_Circle education example (demo_465).  During the 0.02 s rest window
the anchors hold their last prescribed values (the .m has no branch for
``t1 < t < t1+tR``, so the markers are simply not re-targeted).

The .m re-reads the two state files on every call; here they are cached (they
are constant) and resolved relative to this module's directory.
"""
import os

import numpy as np

T1 = 0.1                          # period A -> B [s]
TR = 0.02                         # resting time in between [s]
T2 = 0.1                          # period B -> A [s]
PERIOD = T1 + TR + T2             # 0.22 s
P1, P2 = 0.25, 0.925              # cubic-piece knots
A0, A1, A2, A3 = 0.0, 0.0, 0.0, 4.324324324324318
B0, B1, B2, B3 = 0.123456790123457, -1.481481481481478, 5.925925925925911, \
    -3.576910243576897
C0, C1, C2, C3 = -16.777777777777700, 53.333333333333101, \
    -53.333333333333101, 17.777777777777700

_HERE = os.path.dirname(os.path.abspath(__file__))
_CACHE = {}


def _states():
    if "S" not in _CACHE:
        A = np.loadtxt(os.path.join(_HERE, "State_A.pts"), ndmin=2)
        B = np.loadtxt(os.path.join(_HERE, "State_B.pts"), ndmin=2)
        _CACHE["S"] = (A, B)
    return _CACHE["S"]


def _g(s):
    """The .m's piecewise cubic blend ``g(s)`` on ``s in [0, 1]``."""
    if s <= P1:
        return A0 + A1 * s + A2 * s ** 2 + A3 * s ** 3
    if s <= P2:
        return B0 + B1 * s + B2 * s ** 2 + B3 * s ** 3
    return C0 + C1 * s + C2 * s ** 2 + C3 * s ** 3


def update_Target_Point_Positions(dt, current_time, targets):
    ids = targets[:, 0].astype(np.int64)
    A, B = _states()
    t = np.remainder(current_time, PERIOD)
    if t <= T1:                                   # STATE A -> B
        g = _g(t / T1)
        targets[ids, 1] = A[:, 0] + g * (B[:, 0] - A[:, 0])
        targets[ids, 2] = A[:, 1] + g * (B[:, 1] - A[:, 1])
    elif t >= T1 + TR:                            # STATE B -> A
        g = _g((t - T1 - TR) / T2)
        targets[ids, 1] = B[:, 0] + g * (A[:, 0] - B[:, 0])
        targets[ids, 2] = B[:, 1] + g * (A[:, 1] - B[:, 1])
    return targets
