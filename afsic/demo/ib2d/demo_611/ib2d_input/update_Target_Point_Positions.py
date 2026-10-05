"""Line-by-line port of the example's ``update_Target_Point_Positions.m``.

pyIB2d's driver imports this module from the run directory (once per step, only
when ``update_target`` and ``target_pts`` are both on)::

    from update_Target_Point_Positions import update_Target_Point_Positions
    target_info = update_Target_Point_Positions(dt, current_time, target_info)

``targets`` is the ``(N, 4)`` array ``[lag-id, x, y, stiffness]`` built by the
driver (identifiers 0-based after the reader rebased them).  Only the first
``last_id = 284`` markers move (the remaining pinned ones are decoration); they
are animated through the three letter geometries stored in ``All_Positions.txt``
(columns ``x1,y1,x2,y2,x3,y3,xM,yM``) with the phase times

    tP1 = 0.05  (P1 -> P2),   tP2 = 0.01  (wait in P2),
    tP3 = 0.05  (P2 -> P3),   tP4 = 0.01  (wait in P3),   period = 0.12 s,

and the same piecewise-polynomial blend ``g(s)`` used by several IB2d teaching
examples (a parabola up to ``s = 0.1``, a cubic through the middle, an inverted
parabola from ``s = 0.9``), applied to the *normalised* phase time:

    X = P1 + g(t / tP1) (P2 - P1)                    (P1 -> P2),
    X = P2 + g((t - tP1 - tP2) / tP3) (P3 - P2)      (P2 -> P3),

with X = P2 / X = P3 during the two waiting phases.

Notes on the port:

* MATLAB's 1-based indexing is translated to 0-based: the identifiers in
  ``targets[:, 0]`` are 0-based and the moving rows are ``0 .. last_id - 1``
  (in this example the ids are exactly ``0 .. last_id - 1`` in file order, so
  ``xPts(IDs)`` in the .m is just "all off the moving rows");
* the .m re-reads ``All_Positions.txt`` on every call; here the file is read
  once and cached (it is constant), resolved relative to this module's
  directory so the module works both from the pyIB2d run directory and from
  ``ib2d_input/`` (which is how ``main.py`` imports it);
* the commented-out ``please_Print_Vertices_To_File`` debugging code of the .m
  is not ported (it only ran from an uncommented branch that is dead upstream);
* the two waiting phases compute a ``g2`` that multiplies ``(P2 - P2)`` /
  ``(P3 - P3)`` -- i.e. identically zero; kept for faithfulness.
"""
import os

import numpy as np

LAST_ID = 284                    # last index of the moving target points

A = 2.739726027397260            # y1(t) = a t^2
B = 2.739726027397260            # y3(t) = -b (t-1)^2 + 1
C = -2.029426686960933           # y2(t) = c t^3 + d t^2 + g t + h
D = 3.044140030441400
G = -0.015220700152207
H = 0.000253678335870

TP1, TP2, TP3, TP4 = 0.05, 0.01, 0.05, 0.01     # phase durations [s]
PERIOD = TP1 + TP2 + TP3 + TP4                  # 0.12 s

_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "All_Positions.txt")
_CACHE = {}


def _positions():
    """``All_Positions.txt`` -> the (x1, y1, x2, y2, x3, y3) column arrays."""
    if "P" not in _CACHE:
        mat = np.loadtxt(_FILE, ndmin=2)
        _CACHE["P"] = (mat[:LAST_ID, 0], mat[:LAST_ID, 1], mat[:LAST_ID, 2],
                       mat[:LAST_ID, 3], mat[:LAST_ID, 4], mat[:LAST_ID, 5])
    return _CACHE["P"]


def _g(s):
    """The .m's piecewise polynomial blend ``g(s)`` on ``s in [0, 1]``."""
    t1, t2 = 0.1, 0.9
    if s <= t1:
        return A * s ** 2
    if s < t2:
        return C * s ** 3 + D * s ** 2 + G * s + H
    return -B * (s - 1.0) ** 2 + 1.0


def update_Target_Point_Positions(dt, current_time, targets):
    ids = targets[:LAST_ID, 0].astype(np.int64)   # IDs: Lag-Pt IDs
    xPts = targets[:LAST_ID, 1].copy()            # Original x-Values
    yPts = targets[:LAST_ID, 2].copy()            # Original y-Values

    x1, y1, x2, y2, x3, y3 = _positions()

    t = np.remainder(current_time, PERIOD)        # time in the period

    if t <= TP1:                                  # PHASE 1 -> PHASE 2
        g1 = _g(t / TP1)
        xPts[ids] = x1 + g1 * (x2 - x1)
        yPts[ids] = y1 + g1 * (y2 - y1)
    elif t <= TP1 + TP2:                          # PHASE 2 waiting period
        tprev = TP1
        g2 = _g((t - tprev) / TP2)                # noqa: F841
        xPts[ids] = x2 + g2 * (x2 - x2)           # identically x2
        yPts[ids] = y2 + g2 * (y2 - y2)
    elif t <= TP1 + TP2 + TP3:                    # PHASE 2 -> PHASE 3
        tprev = TP1 + TP2
        g2 = _g((t - tprev) / TP3)
        xPts[ids] = x2 + g2 * (x3 - x2)
        yPts[ids] = y2 + g2 * (y3 - y2)
    elif t <= PERIOD:                             # PHASE 3 waiting period
        tprev = TP1 + TP2 + TP3
        g2 = _g((t - tprev) / TP4)
        xPts[ids] = x3 + g2 * (x3 - x3)           # identically x3
        yPts[ids] = y3 + g2 * (y3 - y3)

    targets[:LAST_ID, 1] = xPts
    targets[:LAST_ID, 2] = yPts
    return targets
