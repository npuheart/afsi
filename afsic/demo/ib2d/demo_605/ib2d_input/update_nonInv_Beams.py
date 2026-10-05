"""Line-by-line port of the example's ``update_nonInv_Beams.m``.

pyIB2d's driver imports this module from the run directory (once per step, when
``update_nonInv_beams`` is on)::

    from update_nonInv_Beams import update_nonInv_Beams
    nonInv_beams_info = update_nonInv_Beams(dt, current_time, beams_info)

``beams_info`` is the ``(N, 6)`` array ``[p1, p2, p3, kb, Cx, Cy]`` read from
``jelly.nonInv_beam`` (identifiers 0-based after the reader rebased them); the
law rewrites the *reference curvatures* ``Cx, Cy`` (columns 5 and 6).

This is the Alben-Peng-Miller jellyfish bell of the First-Year-Seminar
examples: the bell's centreline is re-derived analytically every step from
``beta(t)``

    pulse_time = t - floor(t),   beta = betao + (betam-betao) (pulse_time/to)
                                 if pulse_time < to
                                 else betam + (betao-betam) ((pulse_time-to)/(1-to))

with ``betao = 0.5``, ``betam = 0.3``, ``to = 0.4`` s, by integrating
``dtheta/ds = -1.55 (1 - exp(-s/beta))`` first along the right half of the bell
outwards from its top and then along the left half.  The curvatures are the
second differences of that polyline.

Porting notes / upstream quirks:

* the .m uses the *1-based* MATLAB middle-node identifier
  (``s1 = beams_info(s,2)``) both as an index into ``Xb_lam`` and in its branch
  conditions; pyIB2d hands the module the 0-based identifier, so the port adds
  1 back (``s1 = int(beams_info[s, 1]) + 1``) and then indexes numpy with
  ``s1 - 1``.  Without that rebasing the curvatures would be shifted by one
  point;
* ``npts_bell`` in the .m (256) exceeds the shipped geometry's point count
  (255): the analytic bell is sampled a little finer than the structure, and
  the extra entry only feeds the ``s1 + 1`` neighbour of the last point;
* the .m's ``N = npts_bell - 1`` (255) and ``npts`` (256) differ from the shipped 255 markers; the port keeps the .m's values.
"""
import os

import numpy as np

L1 = 8.0                          # length of the computational domain [m]
N1 = 512                       # "number of meshwidths at the finest level"
BELL_LENGTH = 2.0                 # bell length [m]
NPTS_BELL = int(np.ceil(2.0 * (BELL_LENGTH / L1) * N1))
N = NPTS_BELL - 1
DS1 = BELL_LENGTH / (NPTS_BELL - 1)
BETAO, BETAM, TO = 0.5, 0.3, 0.4
ZS = L1 / 4.0
XREF = L1 / 8.0


def _bell(beta):
    """Analytic bell centreline ``Xb_lam, Yb_lam`` (MATLAB 1-based layout)."""
    xb = np.zeros(NPTS_BELL)
    yb = np.zeros(NPTS_BELL)
    xb[0], yb[0] = XREF, ZS
    half = int(np.ceil(NPTS_BELL / 2))
    zl, ro = ZS, XREF
    for s1 in range(2, half + 1):                 # right side
        th = -1.55 * (1.0 - np.exp(-(s1) * DS1 / beta))
        zl += DS1 * np.sin(th)
        ro += DS1 * np.cos(th)
        xb[s1 - 1], yb[s1 - 1] = ro, zl
    zl, ro = ZS, XREF
    for s1 in range(half + 1, NPTS_BELL + 1):     # left side
        s2 = s1 - half
        th = -1.55 * (1.0 - np.exp(-(s2) * DS1 / beta))
        zl += DS1 * np.sin(th)
        ro -= DS1 * np.cos(th)
        xb[s1 - 1], yb[s1 - 1] = ro, zl
    return xb, yb


def update_nonInv_Beams(dt, current_time, beams_info):
    pulse_time = current_time - np.floor(current_time)
    if pulse_time < TO:
        beta = BETAO + (BETAM - BETAO) * (pulse_time / TO)
    else:
        beta = BETAM + (BETAO - BETAM) * ((pulse_time - TO) / (1.0 - TO))
    xb, yb = _bell(beta)

    s1 = beams_info[:, 1].astype(np.int64) + 1     # back to MATLAB 1-based ids
    # index helpers: MATLAB X(i) -> xb[i - 1]
    def X(i):
        return xb[i - 1]

    def Y(i):
        return yb[i - 1]

    cx = beams_info[:, 4].copy()
    cy = beams_info[:, 5].copy()
    for k, i in enumerate(s1):
        if i == 1:
            cx[k] = X((N - 1) // 2 + 2) + X(i + 1) - 2.0 * X(i)
            cy[k] = Y((N - 1) // 2 + 2) + Y(i + 1) - 2.0 * Y(i)
        elif i <= (N - 1) // 2:
            cx[k] = X(i - 1) + X(i + 1) - 2.0 * X(i)
            cy[k] = Y(i - 1) + Y(i + 1) - 2.0 * Y(i)
        elif i == (N - 1) // 2 + 2:
            cx[k] = X(i + 1) + X(1) - 2.0 * X(i)
            cy[k] = Y(i + 1) + Y(1) - 2.0 * Y(i)
        elif i > (N - 1) // 2 + 2:
            cx[k] = X(i + 1) + X(i - 1) - 2.0 * X(i)
            cy[k] = Y(i + 1) + Y(i - 1) - 2.0 * Y(i)
    beams_info[:, 4] = cx
    beams_info[:, 5] = cy
    return beams_info
