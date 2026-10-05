"""Line-by-line port of the example's ``update_Target_Point_Positions.m``.

pyIB2d's driver imports this module from the run directory (once per step, only
when ``update_target`` and ``target_pts`` are both on)::

    from update_Target_Point_Positions import update_Target_Point_Positions
    target_info = update_Target_Point_Positions(dt, current_time, target_info)

``targets`` is the ``(N, 4)`` array ``[lag-id, x, y, stiffness]`` built by the
driver (identifiers 0-based after the reader rebased them).  The example is the
*Two Corals* variant of demo_461's coral tentacle model: 360 markers, four arms
of ``N_half = 90`` markers (two coral pairs), all target points with k = 1e8.
The law is the same video-fitted bending model as ``Single_Coral``,

    X(y, t) = xoffset + q4(t)/S - q3(t) xi + q2(t) S xi^2 - q1(t) S^2 xi^3,
    xi = y - yoffset,

with the four quintic-in-time coefficient polynomials ``q1..q4(t)`` of the .m
(ascending power series) evaluated at the video frame ``t = 30 (current_time mod
2.30)`` and linearly blended back to the t = 0 values for ``t > tp = 67.5``;
here only **x** is prescribed (column 3 of ``targets`` is left untouched, just
as the .m leaves its ``targets(:,3)`` line commented out).  The four arms are
mirrored about their own pair centres:

    arm 1 (rows 1 .. N_half,               0-based 0 .. N_half-1)   -> -X + 0.2
    arm 2 (rows N_half+1 .. N_target/2)                             -> +X + 0.2
    arm 3 (rows N_target/2+1 .. N_target/2+N_half)                  -> -X + 0.4
    arm 4 (rows N_target/2+N_half+1 .. N_target)                    -> +X + 0.4

(the .m hardcodes ``xoffset2grid1 = 0.2`` and ``xoffset2grid2 = 0.4`` for the
two corals of the 0.6 x 0.6 box).

Porting notes: MATLAB's 1-based indexing is translated to 0-based (``i <=
N_half`` becomes ``i0 < N_half``); the ``yPts`` used in the bending model are
the *initial* y values read from ``targets[:, 2]`` at the start of the call.
"""
import numpy as np

N_HALF = 90                       # # of pts. on ONE tentacle
F = 30.0                          # frames per second in the data
S = 1733.333333333333333333       # pixels per dm in the data
TP = F * 2.25                     # time in frames (mod one pulse)
TEND = F * 2.30
PERIOD = TEND / F                 # 2.30 s pulse
XOFFSET = 0.019                   # initial offset from the centre of the domain
YOFFSET = 0.25
XOFFSET2GRID1 = 0.2               # centre of the first coral pair
XOFFSET2GRID2 = 0.4               # centre of the second coral pair

# coefficients qc*1..qc*7 of the four polynomials, in the *ascending* order of
# the .m file (qc*1 multiplies t^6, qc*7 is the constant), i.e. the order
# np.polyval expects
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


def update_Target_Point_Positions(dt, current_time, targets):
    ids = targets[:, 0].astype(np.int64)          # IDs: Lag-Pt IDs
    yPts = targets[:, 2].copy()                   # Original y-Values
    N_target = len(ids)

    t = F * np.remainder(current_time, PERIOD)    # time in frames (mod pulse)
    if t <= TP:
        q = np.array([np.polyval(row, t) for row in QC])
    else:
        w = (t - TP) / (TEND - TP)
        q = np.array([np.polyval(row, TP) for row in QC]) * (1.0 - w) \
            + QC[:, 6] * w

    xi = yPts - YOFFSET
    Xt = (XOFFSET + q[3] / S - q[2] * xi + q[1] * S * xi ** 2
          - q[0] * S * S * xi ** 3)

    i0 = np.arange(N_target)
    half = N_target // 2
    out = np.empty(N_target)
    left1 = i0 < N_HALF                                  # rows 1 .. N_half
    left2 = i0 < half + N_HALF
    out[:half] = np.where(left1[:half], -Xt[:half] + XOFFSET2GRID1,
                          Xt[:half] + XOFFSET2GRID1)
    out[half:] = np.where(left2[half:], -Xt[half:] + XOFFSET2GRID2,
                          Xt[half:] + XOFFSET2GRID2)

    targets[ids, 1] = out
    return targets
