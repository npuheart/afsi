"""Line-by-line port of the example's ``update_Target_Point_Positions.m``.

pyIB2d's driver imports this module from the run directory (once per step, only
when ``update_target`` and ``target_pts`` are both on)::

    from update_Target_Point_Positions import update_Target_Point_Positions
    target_info = update_Target_Point_Positions(dt, current_time, target_info)

``targets`` is the ``(N, 4)`` array ``[lag-id, x, y, stiffness]`` built by the
driver.  The Pulsing Heart is 6829 target points (k = 1.5e6) whose anchors are
interpolated between two heart shapes stored in ``All_Positions.txt``
(columns ``x1, y1, x2, y2``):

    tP1 = 0.05 s (contraction A -> B), tP2 = 0.05 s (expansion B -> A),
    period = 0.1 s,  t = current_time mod period,

    g(s) = a s^2                       for s <= 0.1,
    g(s) = c s^3 + d s^2 + g s + h     for 0.1 < s < 0.9,
    g(s) = -b (s-1)^2 + 1              for s >= 0.9,

    X = A + g(t/tP1) (B - A)      (phase 1),   X = B + g((t-tP1)/tP2) (A - B) (phase 2)

(a = 2.739726027397260, b = a, c = -2.029426686960933, d = 3.044140030441400,
g = -0.015220700152207, h = 0.000253678335870).  All markers move; the anchor
path is continuous and periodic.  The .m re-reads ``All_Positions.txt`` every
step; here it is cached and resolved relative to this module's directory, so it
works both from the pyIB2d run directory and from ``ib2d_input/``.
"""
import os

import numpy as np

TP1 = 0.05                        # contraction time [s]
TP2 = 0.05                        # expansion time [s]
PERIOD = TP1 + TP2                # 0.1 s
A2, B2 = 2.739726027397260, 2.739726027397260
C2, D2 = -2.029426686960933, 3.044140030441400
G2, H2 = -0.015220700152207, 0.000253678335870

_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "All_Positions.txt")
_CACHE = {}


def _states():
    if "S" not in _CACHE:
        _CACHE["S"] = np.loadtxt(_FILE, ndmin=2)[:, :4]
    return _CACHE["S"]


def _g(s):
    """The .m's piecewise polynomial blend ``g(s)`` on ``s in [0, 1]``."""
    if s <= 0.1:
        return A2 * s ** 2
    if s < 0.9:
        return C2 * s ** 3 + D2 * s ** 2 + G2 * s + H2
    return -B2 * (s - 1.0) ** 2 + 1.0


def update_Target_Point_Positions(dt, current_time, targets):
    ids = targets[:, 0].astype(np.int64)
    S = _states()
    x1, y1, x2, y2 = S[:, 0], S[:, 1], S[:, 2], S[:, 3]

    t = np.remainder(current_time, PERIOD)
    if t <= TP1:                                  # PHASE 1 -> PHASE 2
        g1 = _g(t / TP1)
        x_targets = x1 + g1 * (x2 - x1)
        y_targets = y1 + g1 * (y2 - y1)
    else:                                         # PHASE 2 -> PHASE 1
        g2 = _g((t - TP1) / TP2)
        x_targets = x2 + g2 * (x1 - x2)
        y_targets = y2 + g2 * (y1 - y2)

    targets[ids, 1] = x_targets
    targets[ids, 2] = y_targets
    return targets
