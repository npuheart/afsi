"""Line-by-line port of the example's ``update_Target_Point_Positions.m``
(``Example_Corals/Two_Corals``).

pyIB2d's driver imports this module from the run directory (once per step, when
``update_target`` is on) and calls

    target_info = update_Target_Point_Positions(dt, current_time, target_info)

with ``target_info`` the (N_targets, 4) array ``[id, x_anchor, y_anchor, k]``
(the driver fills columns 2/3 with the *initial* marker positions).  The .m file
makes the two coral tentacles wave: the y-anchors are kept fixed and each
tentacle's x-anchor follows a piecewise polynomial that is matched to a slow
quadratic/cubic interpolation of the tentacle shape, driven by
``t = 30 mod(current_time, 2.30)`` over a 2.30 s period (67.5 s of polynomial
motion + a short 0.05 s return).

MATLAB's 1-based indexing is translated to 0-based here: row ``i`` becomes
``i0 = i - 1``, so the branch conditions ``i <= N_target/2`` / ``i <= N_half``
become ``i0 < N_target/2`` / ``i0 < N_half``.  The identifiers (1..N) are
rebased by the reader (``ib2d_io.load_example`` auto-detects the 1-based
numbering of this matIB2d example).
"""
import numpy as np

N_HALF = 90                     # points on ONE tentacle
F = 30.0
S = 1733.333333333333333333
TP = F * 2.25                   # polynomial phase end
TEND = F * 2.30                 # period end (return phase)
XOFFSET = 0.019
YOFFSET = 0.25
XOFFSET2GRID1 = 0.2
XOFFSET2GRID2 = 0.4
QC1 = (-8.018203211249667e-13, 1.706310971452060e-10, -1.371526883228119e-8,
       5.116506115621489e-7, -8.613944795521634e-6, 4.951095471423292e-5,
       -1.237254488643602e-4)
QC2 = (-1.093988700839764e-10, 2.370737151797367e-8, -1.946684756263760e-6,
       7.447999061932255e-5, -1.285757196003425e-3, 7.184871148722453e-3,
       -2.357284935085283e-2)
QC3 = (-3.111491615328405e-9, 6.922752279229525e-7, -5.951659336196739e-5,
       2.461541908755476e-3, -4.830999119980220e-2, 3.293063238297988e-1,
       -1.893567487439901)
QC4 = (2.028150369106308e-8, -4.188949936239104e-6, 3.237380275082298e-4,
       -1.158763275070524e-2, 1.908244836752949e-1, -1.142372536532416,
       3.584246435337805)


def _poly(qc, tt):
    """``qc7 + qc6 t + ... + qc1 t^6`` (the .m writes the coefficients in
    descending power order)."""
    return sum(c * tt ** (6 - i) for i, c in enumerate(qc))


def update_Target_Point_Positions(dt, current_time, targets):
    yPts = targets[:, 2]
    N_target = len(targets)
    t = F * np.mod(current_time, 2.30)
    if t <= TP:
        qpoly = [_poly(qc, t) for qc in (QC1, QC2, QC3, QC4)]
    else:
        qpolyf = [_poly(qc, TP) for qc in (QC1, QC2, QC3, QC4)]
        fac = (TEND - TP - (t - TP)) / (TEND - TP)
        qpoly = [qf * fac + qc[6] * (t - TP) / (TEND - TP)
                 for qf, qc in zip(qpolyf, (QC1, QC2, QC3, QC4))]
    for i in range(N_target):
        shifted = yPts[i] - YOFFSET
        x_target = (XOFFSET + qpoly[3] / S - qpoly[2] * shifted
                    + qpoly[1] * S * shifted ** 2
                    - qpoly[0] * S * S * shifted ** 3)
        if i < N_target / 2:
            x_target = (-x_target + XOFFSET2GRID1 if i < N_HALF
                        else x_target + XOFFSET2GRID1)
        else:
            x_target = (-x_target + XOFFSET2GRID2 if i < N_target / 2 + N_HALF
                        else x_target + XOFFSET2GRID2)
        targets[i, 1] = x_target
    return targets
