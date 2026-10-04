"""Line-by-line port of the example's ``update_Target_Point_Positions.m``.

pyIB2d's driver imports this module from the run directory (once per step,
only when ``update_target`` and ``target_pts`` are both on)::

    from update_Target_Point_Positions import update_Target_Point_Positions
    target_info = update_Target_Point_Positions(dt, current_time, target_info)

``targets`` is the (N, 4) array ``[lag-id, x, y, stiffness]`` built by the
driver.  Unlike most examples this one is *stateless*: the anchor positions are
interpolated between the two shapes stored in ``All_Positions.txt`` (columns
``xP1, yP1, xP2, yP2`` -- the relaxed and the contracted heart, read with
``read_File_In``), driven by a two-phase polynomial in the time within the
period ``tP1 + tP2 = 0.05`` s:

* phase 1->2 (``t <= tP1``), with ``s = t/tP1``:

      s < 0.1        g1 = a s^2
      0.1 <= s < 0.9 g1 = c s^3 + d s^2 + g s + h
      s >= 0.9       g1 = -b (s-1)^2 + 1
      X = P1 + g1 (P2 - P1)

* phase 2->1 (``tP1 < t <= tP1 + tP2``), with ``s = (t - tP1)/tP2``: the same
  polynomial (the same thresholds in ``s``) maps back from ``P2`` to ``P1``,

so the shape pulses between the two prescribed configurations (the polynomial
pieces match in value, not in slope, at the knots -- reproduced verbatim).
MATLAB's 1-based indexing is translated to 0-based here; the identifiers
(1..N) are rebased by the reader (``ib2d_io.load_example(..., index_base=1)``).
"""
import numpy as np

# Coefficients for Polynomial Phase-Interpolation
A = 2.739726027397260     # y1(t) = a t^2
B = 2.739726027397260     # y3(t) = -b (t-1)^2 + 1
C = -2.029426686960933    # y2(t) = c t^3 + d t^2 + g t + h
D = 3.044140030441400
G = -0.015220700152207
H = 0.000253678335870

TP1 = 0.025               # Contraction time
TP2 = 0.025               # Expansion time
PERIOD = TP1 + TP2


def _read_file_in(file_name):
    """``read_File_In``: four columns -> (x1, y1, x2, y2).

    The .m file re-reads (and re-parses) the data file on every call; the
    result is cached here since the file is constant and the reference run
    would otherwise spend most of its time in ``textscan``.
    """
    if file_name not in _CACHE:
        mat = np.loadtxt(file_name, ndmin=2)
        _CACHE[file_name] = (mat[:, 0], mat[:, 1], mat[:, 2], mat[:, 3])
    return _CACHE[file_name]


_CACHE = {}


def _phase(s):
    """The piecewise polynomial ``g(s)`` of the .m file."""
    if s < 0.1:
        return A * s ** 2
    if s < 0.9:
        return C * s ** 3 + D * s ** 2 + G * s + H
    return -B * (s - 1.0) ** 2 + 1.0


def update_Target_Point_Positions(dt, current_time, targets):
    ids = targets[:, 0].astype("int")      # IDs: Lag-Pt IDs

    t = np.remainder(current_time, PERIOD)   # <= period by construction
    x1, y1, x2, y2 = _read_file_in("All_Positions.txt")

    if t <= TP1:                            # PHASE 1 --> PHASE 2
        g = _phase(t / TP1)
        xn = x1 + g * (x2 - x1)
        yn = y1 + g * (y2 - y1)
    else:                                   # PHASE 2 --> PHASE 1
        g = _phase((t - TP1) / TP2)
        xn = x2 + g * (x1 - x2)
        yn = y2 + g * (y1 - y2)

    targets[ids, 1] = xn
    targets[ids, 2] = yn
    return targets
