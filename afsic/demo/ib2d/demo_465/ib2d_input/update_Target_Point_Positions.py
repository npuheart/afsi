"""Port of the two ``update_Target_Point_Positions.m`` variants of the
Moving_Circle education examples (``Linear_Interp`` and ``Cubic_Interp``).

pyIB2d's driver imports this module from the run directory (once per step, when
``update_target`` and ``target_pts`` are on)::

    from update_Target_Point_Positions import update_Target_Point_Positions
    target_info = update_Target_Point_Positions(dt, current_time, target_info)

A 64-marker circle travels through the three prescribed positions stored in
``State_A.pts`` / ``State_B.pts`` / ``State_C.pts``.  With ``t1 = 0.01`` s
(A <-> B) and ``t2 = 0.02`` s (B <-> C) the 0.06 s period is split into four
phases, and in each of them the positions are interpolated as

    X = S_1 + g(s) (S_2 - S_1),   s = (t - t_phase) / t_phase_duration,

where the interpolation parameter ``g(s)`` is the only difference between the
two examples:

* ``INTERP=linear`` -- ``g(s) = s`` (the ``Linear_Interp`` .m file);
* ``INTERP=cubic``  -- the piecewise cubic of the ``Cubic_Interp`` .m file,
  ``g = a3 s^3`` for ``s <= p1 = 0.25``, ``g = b0 + b1 s + b2 s^2 + b3 s^3``
  for ``s <= p1 + p2 = 0.925``, and ``g = c0 + c1 s + c2 s^2 + c3 s^3``
  afterwards.  Both ends of the cubic pieces are value-continuous at the knots
  (``g(0) = 0``, ``g(1) = 1``), so the circle still lands exactly on A, B and C.

The scheme is selected with the ``INTERP`` environment variable so that one
port serves both examples.  MATLAB's 1-based identifiers are rebased by the
reader (``ib2d_io.load_example(..., index_base=1)``); the .m files read the
state files on every call, which is cached here (the files are constant).
"""
import os

import numpy as np

SCHEME = os.environ.get("INTERP", "linear")

T1, T2 = 0.01, 0.02               # A<->B and B<->C durations
PERIOD = 2 * (T1 + T2)

# piecewise cubic of Cubic_Interp/update_Target_Point_Positions.m
P1, P2 = 0.25, 0.925
A0, A1, A2, A3 = 0.0, 0.0, 0.0, 4.324324324324318
B0, B1, B2, B3 = 0.123456790123457, -1.481481481481478, 5.925925925925911, \
    -3.576910243576897
C0, C1, C2, C3 = -16.777777777777700, 53.333333333333101, \
    -53.333333333333101, 17.777777777777700

_CACHE = {}


def _state(name):
    if name not in _CACHE:
        _CACHE[name] = np.loadtxt(name, ndmin=2)[:, :2]
    return _CACHE[name]


def _g(s):
    """Interpolation parameter: identity (linear) or the piecewise cubic."""
    if SCHEME != "cubic":
        return s
    if s <= P1:
        return A0 + A1 * s + A2 * s ** 2 + A3 * s ** 3
    if s <= P1 + P2:
        return B0 + B1 * s + B2 * s ** 2 + B3 * s ** 3
    return C0 + C1 * s + C2 * s ** 2 + C3 * s ** 3


def update_Target_Point_Positions(dt, current_time, targets):
    t = np.remainder(current_time, PERIOD)

    if t <= T1:                                   # A -> B
        s1, s2, s = _state("State_A.pts"), _state("State_B.pts"), t / T1
    elif t <= T1 + T2:                            # B -> C
        s1, s2, s = _state("State_B.pts"), _state("State_C.pts"), (t - T1) / T2
    elif t <= T1 + 2 * T2:                        # C -> B
        s1, s2, s = _state("State_C.pts"), _state("State_B.pts"), \
            (t - T1 - T2) / T2
    else:                                         # B -> A
        s1, s2, s = _state("State_B.pts"), _state("State_A.pts"), \
            (t - T1 - 2 * T2) / T1

    g = _g(s)
    targets[:, 1] = s1[:, 0] + g * (s2[:, 0] - s1[:, 0])
    targets[:, 2] = s1[:, 1] + g * (s2[:, 1] - s1[:, 1])
    return targets
