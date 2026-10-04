"""Line-by-line port of the example's ``update_Target_Point_Positions.m``.

pyIB2d's driver imports this module from the run directory (once per step,
only when ``update_target`` and ``target_pts`` are both on)::

    from update_Target_Point_Positions import update_Target_Point_Positions
    target_info = update_Target_Point_Positions(dt, current_time, target_info)

``targets`` is the (N, 4) array ``[lag-id, x, y, stiffness]`` built by the
driver.  This is the "coral tentacle" model of IB2d's Corals example: the
coral has two arms of ``N_half = 45`` markers each, and the law only moves
their **x** coordinate (`targets(:,3)` is left untouched, so every marker keeps
the y of its initial position).  The x position is a *bending* kinematic model

    X(y, t) = xoffset + q4(t)/S - q3(t) xi + q2(t) S xi^2 - q1(t) S^2 xi^3,
    xi = y - yoffset,

mirrored about ``xoffset2grid = Lx/2`` for the first arm, where the four
coefficients ``q1..q4(t)`` are quintic-in-time polynomial fits of laboratory
video data (``S = 1733.33`` px/dm converts the fit's curvature units, and the
time variable is ``t = 30 * (current_time mod 2.30)`` -- frame number in a
2.30 s pulse, computed with ``F = 30`` frames per second).  For
``t <= tp = 67.5`` the polynomials are evaluated directly; for
``67.5 < t <= tend = 69`` they are linearly blended back to their t = 0 values
so that the law is continuous across the period -- i.e. the tentacle waves
periodically and returns to the geometry stored in ``coral.vertex``
(exactly: the law at t = 0 reproduces the example's vertices to 1.7e-16 m).

Note the polynomial coefficients in the .m file are written as an *ascending*
power series (``qpoly = qc17 + qc16 t + ... + qc11 t^6``); ``QC`` below stores the
same rows (t^6 coefficient first, constant last), which is the order
:func:`numpy.polyval` expects.  MATLAB's 1-based indexing is translated to 0-based; the identifiers
(1..N) are rebased by the reader (``ib2d_io.load_example(..., index_base=1)``).
"""
import numpy as np

# --- constants of the .m file -------------------------------------------------
F = 30.0                          # frames per second in the data
S = 1733.333333333333333333       # pixels per dm in the data
TP = F * 2.25                     # time in frames (mod one pulse)
TEND = F * 2.30
PERIOD = TEND / F                 # 2.30 s
XOFFSET = 0.019                   # initial offset from the centre of the domain
YOFFSET = 0.25
XOFFSET2GRID = 0.3                # = Lx / 2 (adds the domain centre back)
N_HALF = 45                       # number of pts. on ONE tentacle

# coefficients qc*1..qc*7 of the four polynomials, in the *ascending* order in
# which the .m file writes them (qc*1 multiplies t^6, qc*7 is the constant)
QC = np.array([
    [-8.018203211249667e-13, 1.706310971452060e-10, -1.371526883228119e-8,
     5.116506115621489e-7, -8.613944795521634e-6, 4.951095471423292e-5,
     -1.237254488643602e-4],
    [-1.093988700839764e-10, 2.370737151797367e-8, -1.946684756263760e-6,
     7.447999061932255e-5, -1.285757196003425e-3, 7.184871148722453e-3,
     -2.357284935085283e-2],
    [-3.111491615328405e-9, 6.922752279229525e-7, -5.951659336196739e-5,
     2.461541908755476e-3, -4.830999119980220e-2, 3.293063238297988e-1,
     -1.893567487439901],
    [2.028150369106308e-8, -4.188949936239104e-6, 3.237380275082298e-4,
     -1.158763275070524e-2, 1.908244836752949e-1, -1.142372536532416,
     3.584246435337805],
])


def _poly(t):
    """qpoly1..4 at time ``t``.

    ``QC`` stores the rows as the .m file writes them, i.e. with the t^6
    coefficient (qc*1) first and the constant (qc*7) last -- which is exactly
    the order ``np.polyval`` wants.
    """
    return np.array([np.polyval(row, t) for row in QC])


def update_Target_Point_Positions(dt, current_time, targets):
    ids = targets[:, 0].astype("int")     # IDs: Lag-Pt IDs
    yPts = targets[:, 2]                  # Original y-Values (= anchor y)

    t = F * np.remainder(current_time, PERIOD)

    if t <= TP:
        q = _poly(t)
    else:
        qf = _poly(TP)
        w = (t - TP) / (TEND - TP)
        q = qf * (1.0 - w) + QC[:, 6] * w        # blend back to the t = 0 values

    xi = yPts - YOFFSET
    X_target = (XOFFSET + q[3] / S - q[2] * xi + q[1] * S * xi ** 2
                - q[0] * S * S * xi ** 3)
    left = np.arange(ids.size) < N_HALF          # 1-based i <= N_half
    targets[ids, 1] = np.where(left, -X_target + XOFFSET2GRID,
                               X_target + XOFFSET2GRID)
    return targets
