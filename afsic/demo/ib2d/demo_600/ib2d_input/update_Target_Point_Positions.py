"""Line-by-line port of the example's ``update_Target_Point_Positions.m``.

pyIB2d's driver imports this module from the run directory (once per step, only
when ``update_target`` and ``target_pts`` are both on)::

    from update_Target_Point_Positions import update_Target_Point_Positions
    target_info = update_Target_Point_Positions(dt, current_time, target_info)

``targets`` is the ``(N, 4)`` array ``[lag-id, x, y, stiffness]`` built by the
driver.  The Tethered Jellyfish is 693 markers, all target points
(k = 5e6); the tether itself is the first marker (row 1 of ``XY_2Pos.txt``,
which has **no header line**: ``N`` rows of ``x1, x2, y1, y2``).  The two bell
arms are interpolated between the two recorded states with the same piecewise
cubic as the Moving_Circle education example,

    g(s) = a3 s^3 (s <= 0.125), cubic b (s <= 0.875), cubic c (otherwise),

with per-arm periods ``tR1 = tR2 = tL1 = tL2 = 0.25 s``, so each arm contracts
and expands over a 0.5 s cycle:

    RIGHT arm (rows 2 .. NArm+1, NArm = 174):
        tR = t mod 0.5;   tR <= 0.25  ->  X = X1 + g(tR/tR1) (X2 - X1)
                          otherwise   ->  X = X2 + g((tR-tR1)/tR2) (X1 - X2)
    LEFT arm (rows NArm+2 .. end):
        tL = t mod 0.5;   the same two-branch interpolation with g(tL/tL1).

Upstream bugs (reproduced faithfully, flagged here):

* ``off_Left = 0`` disables the left-arm delay, so both arms start together;
* in the LEFT arm's *middle* cubic piece the .m tests ``elseif tRTilde <= p2``
  instead of ``tLTilde <= p2`` -- a copy-paste bug that leans on the RIGHT
  arm's ``tRTilde`` from the same (shared) function scope.  This port keeps it
  (using the right arm's value), so the reference and AFSI agree;
* the LEFT arm adds ``x + xc, y + yc`` about the corner points taken from
  ``XY_2Pos.txt`` rows ``NArm+2`` and ``3*NArm/2+2``; the data file's first row
  (the tether) is never moved.
"""
import os

import numpy as np

T_R1 = T_R2 = 0.25                # RIGHT arm periods [s]
T_L1 = T_L2 = 0.25                # LEFT arm periods [s]
PERIOD_R = T_R1 + T_R2
PERIOD_L = T_L1 + T_L2
NA_RM = 346                  # # of pts. on the RIGHT arm
P1, P2 = 0.125, 0.875
A0, A1, A2, A3 = 0.0, 0.0, 0.0, 9.1429
B0, B1, B2, B3 = 0.0238, -0.5714, 4.5714, -3.0476
C0, C1, C2, C3 = -8.1429, 27.4286, -27.4286, 9.1429

_HERE = os.path.dirname(os.path.abspath(__file__))
_CACHE = {}


def _states():
    if "XY" not in _CACHE:
        _CACHE["XY"] = np.loadtxt(os.path.join(_HERE, "XY_2Pos.txt"), ndmin=2)
    return _CACHE["XY"]


def _g(s):
    if s <= P1:
        return A0 + A1 * s + A2 * s ** 2 + A3 * s ** 3
    if s <= P2:
        return B0 + B1 * s + B2 * s ** 2 + B3 * s ** 3
    return C0 + C1 * s + C2 * s ** 2 + C3 * s ** 3


def update_Target_Point_Positions(dt, current_time, targets):
    XY = _states()
    # 1-based rows 2 .. NArm+1  <->  0-based rows 1 .. NArm
    xR1, yR1 = XY[1:NA_RM + 1, 0], XY[1:NA_RM + 1, 2]
    xR2, yR2 = XY[1:NA_RM + 1, 1], XY[1:NA_RM + 1, 3]
    xL1, yL1 = XY[NA_RM + 1:, 0], XY[NA_RM + 1:, 2]
    xL2, yL2 = XY[NA_RM + 1:, 1], XY[NA_RM + 1:, 3]

    tR = np.remainder(current_time, PERIOD_R)
    tL = np.remainder(current_time, PERIOD_L)

    # RIGHT arm (always runs: this also defines tRTilde for the LEFT arm, which
    # the upstream .m accidentally reuses)
    if tR <= T_R1:
        tRTilde = tR / T_R1
        g = _g(tRTilde)
        xR, yR = xR1 + g * (xR2 - xR1), yR1 + g * (yR2 - yR1)
    else:
        tRTilde = (tR - T_R1) / T_R2
        g = _g(tRTilde)
        xR, yR = xR2 + g * (xR1 - xR2), yR2 + g * (yR1 - yR2)

    # LEFT arm
    if tL <= T_L1:
        tLTilde = tL / T_L1
        if tLTilde <= P1:
            g = A0 + A1 * tLTilde + A2 * tLTilde ** 2 + A3 * tLTilde ** 3
        elif tRTilde <= P2:            # <- upstream bug: tRTilde, not tLTilde
            g = B0 + B1 * tLTilde + B2 * tLTilde ** 2 + B3 * tLTilde ** 3
        else:
            g = C0 + C1 * tLTilde + C2 * tLTilde ** 2 + C3 * tLTilde ** 3
        xL, yL = xL1 + g * (xL2 - xL1), yL1 + g * (yL2 - yL1)
    else:
        tLTilde = (tL - T_L1) / T_L2
        if tLTilde <= P1:
            g = A0 + A1 * tLTilde + A2 * tLTilde ** 2 + A3 * tLTilde ** 3
        elif tLTilde <= P2:
            g = B0 + B1 * tLTilde + B2 * tLTilde ** 2 + B3 * tLTilde ** 3
        else:
            g = C0 + C1 * tLTilde + C2 * tLTilde ** 2 + C3 * tLTilde ** 3
        xL, yL = xL2 + g * (xL1 - xL2), yL2 + g * (yL1 - yL2)

    # 0-based target rows 1 .. NArm and NArm+1 .. end (= MATLAB rows 2:NArm+1)
    targets[1:NA_RM + 1, 1] = xR
    targets[1:NA_RM + 1, 2] = yR
    targets[NA_RM + 1:, 1] = xL
    targets[NA_RM + 1:, 2] = yL
    return targets
