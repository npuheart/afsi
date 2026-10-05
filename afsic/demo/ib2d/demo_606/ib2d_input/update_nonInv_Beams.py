"""Line-by-line port of the example's ``update_nonInv_Beams.m``.

pyIB2d's driver imports this module from the run directory (once per step, when
``update_nonInv_beams`` is on)::

    from update_nonInv_Beams import update_nonInv_Beams
    nonInv_beams_info = update_nonInv_Beams(dt, current_time, beams_info)

``beams_info`` is the ``(N, 6)`` array ``[p1, p2, p3, kb, Cx, Cy]``.  The
swimmer's body is interpolated between the two phases recorded in
``swimmer.phases`` (3 columns ``x1, y1, y2``; 184 rows, no header) with the
family's piecewise polynomial in the normalised phase time
(``g(s) = a s^2`` for ``s < 0.1``, a cubic for ``0.1 <= s < 0.9``, and
``-b (s-1)^2 + 1`` otherwise), down-stroke and up-stroke each 0.25 s::

    yPts = yP1 + g(t/tP1) (yP2 - yP1)        (t <= tP1)
    yPts = yP2 + g((t-tP1)/tP2) (yP1 - yP2)  (t > tP1)

The reference curvatures are the **second differences of that interpolated
polyline**, written column-wise:

    Cx = xPts[0:-2] + xPts[2:] - 2 xPts[1:-1]
    Cy = yPts[0:-2] + yPts[2:] - 2 yPts[1:-1]

Upstream quirks:

* the .m sets ``xP2 = xP1`` (the phases differ only in y), so ``xPts`` is the
  constant ``xP1`` and ``Cx`` never changes -- kept verbatim;
* the .m assigns those vectors to ``beams_info(:,5:6)`` blindly, *without*
  using the beam connectivity: it relies on the ``.nonInv_beam`` file being
  ordered so that beam ``s`` spans points ``s, s+1, s+2``.  Kept (and checked:
  the 182 beams of the shipped file are exactly in that order);
* the second difference has ``npts - 2 = 182`` entries for the 182 beams.
"""
import os

import numpy as np

A_ = 2.739726027397260
B_ = 2.739726027397260
C_ = -2.029426686960933
D_ = 3.044140030441400
G_ = -0.015220700152207
H_ = 0.000253678335870
TP1 = 0.25                        # down-stroke [s]
TP2 = 0.25                        # up-stroke [s]
PERIOD = TP1 + TP2

_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "swimmer.phases")
_CACHE = {}


def _phases():
    if "P" not in _CACHE:
        _CACHE["P"] = np.loadtxt(_FILE, ndmin=2)[:, :3]
    return _CACHE["P"]


def _g(s):
    """Piecewise ``g(s)`` of the .m (knots 0.1 / 0.9 on the normalised time)."""
    if s < 0.1:
        return A_ * s ** 2
    if s < 0.9:
        return C_ * s ** 3 + D_ * s ** 2 + G_ * s + H_
    return -B_ * (s - 1.0) ** 2 + 1.0


def update_nonInv_Beams(dt, current_time, beams_info):
    P = _phases()
    xP1, yP1, yP2 = P[:, 0], P[:, 1], P[:, 2]
    xP2 = xP1                                   # NOTE xP1 = xP2 in the .m

    t = np.remainder(current_time, PERIOD)
    if t <= TP1:
        g = _g(t / TP1)
        xPts = xP1 + g * (xP2 - xP1)
        yPts = yP1 + g * (yP2 - yP1)
    else:
        g = _g((t - TP1) / TP2)
        xPts = xP2 + g * (xP1 - xP2)
        yPts = yP2 + g * (yP1 - yP2)

    beams_info[:, 4] = xPts[:-2] + xPts[2:] - 2.0 * xPts[1:-1]
    beams_info[:, 5] = yPts[:-2] + yPts[2:] - 2.0 * yPts[1:-1]
    return beams_info
