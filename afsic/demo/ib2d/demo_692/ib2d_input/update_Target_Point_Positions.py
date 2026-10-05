"""Oscillating four-armed coral of IB2d -- line-by-line port of the example's own

    matIB2d/Examples/Examples_First_Year_Seminar/Corals/
        update_Target_Point_Positions.m

(the ``update_target = 1`` user module pyIB2d imports from the run directory
once per step; ``main.py`` imports the very same file).

The geometry (``corals.vertex``, 180 points) is four "arms" of 45 points each;
each arm is rigidly rotated about its inner corner by an oscillating angle and
the result is written into columns 2/3 (x/y) of the driver's ``target_info``
array (``[id, x, y, k]``).  The angles come from the two "angle generators"

    n        = ceil((t + dt)/dt)                      (step counter)
    dtheta_1 = pi dt,   angVec_1 = dtheta_1 : dtheta_1 : 2 pi
    n1       = mod(n, len(angVec_1)),  angInput_1 = angVec_1(n1)   (MATLAB 1-based)
    ... the same with dtheta_2 = pi dt / 2
    ang      = mod(angInput, 2/3 pi);  if ang > pi/3: ang = 2/3 pi - ang

i.e. a triangle wave of amplitude ``pi/3`` and period ``4/3 s`` (``dt = 1e-5``);
the second pair of arms uses half the angular rate.  ``read_File_In`` skips the
header row of the vertex file, exactly like IB2d's other readers.

Upstream quirks (kept verbatim): ``angVec(n1)`` with ``n1 = 0`` would be an
out-of-range MATLAB index -- it only avoids this while ``n < 2/dt`` steps
(``t < 2 s`` at the shipped ``dt`` and ``t < 20 s`` at this demo's); the
rotation is applied to ``targets(IDs, 2:3)``, i.e. the target rows are indexed
by the *identifier* column (they are 1..180 in file order here).  The vertex
file is re-read on every call in MATLAB; here it is cached.
"""
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_CACHE = {}


def read_File_In(file_name):
    """MATLAB ``read_File_In``: reads a two-column vertex file, skipping the
    leading point-count row."""
    if file_name not in _CACHE:
        for d in (os.getcwd(), _HERE, os.path.join(_HERE, "ib2d_input")):
            p = os.path.join(d, file_name)
            if os.path.exists(p):
                with open(p) as fh:
                    fh.readline()          # header (number of points)
                    _CACHE[file_name] = np.loadtxt(fh)
                break
        else:
            raise FileNotFoundError(f"{file_name} not found in {os.getcwd()} / {_HERE}")
    return _CACHE[file_name][:, 0], _CACHE[file_name][:, 1]


def rotate_Geometry(ang, xC, yC, xRef, yRef):
    """MATLAB ``rotate_Geometry``: rotate ``(xRef, yRef)`` about ``(xC, yC)``."""
    xRef = np.asarray(xRef, dtype=float) - xC
    yRef = np.asarray(yRef, dtype=float) - yC
    x = xRef * np.cos(ang) - yRef * np.sin(ang) + xC
    y = xRef * np.sin(ang) + yRef * np.cos(ang) + yC
    return x, y


def give_Me_Oscillatory_Angle(dt, current_time):
    """MATLAB ``give_Me_Oscillatory_Angle`` (1-based ``angVec(n1)``)."""
    n = int(np.ceil((current_time + dt) / dt))
    dtheta = np.pi * dt
    angVec1 = np.arange(dtheta, 2 * np.pi + dtheta / 2, dtheta)
    n1 = int(np.mod(n, len(angVec1)))
    dtheta = np.pi * dt / 2
    angVec2 = np.arange(dtheta, 2 * np.pi + dtheta / 2, dtheta)
    n2 = int(np.mod(n, len(angVec2)))
    assert n1 >= 1 and n2 >= 1, \
        "MATLAB would index angVec(0) here (upstream quirk); keep n < 2/dt"
    return angVec1[n1 - 1], angVec2[n2 - 1]


def update_Target_Point_Positions(dt, current_time, targets):
    targets = np.array(targets, dtype=float, copy=True)
    IDs = targets[:, 0].astype(int)
    angInput1, angInput2 = give_Me_Oscillatory_Angle(dt, current_time)
    full_ang = np.pi / 3
    dbl_ang = 2 * full_ang

    ang1 = np.mod(angInput1, dbl_ang)
    if full_ang < ang1 <= dbl_ang:
        ang1 = dbl_ang - ang1
    ang2 = np.mod(angInput2, dbl_ang)
    if full_ang < ang2 <= dbl_ang:
        ang2 = dbl_ang - ang2

    xRef, yRef = read_File_In("corals.vertex")
    nel = len(xRef)
    xL_1, yL_1 = xRef[0], yRef[0]
    xR_1, yR_1 = xRef[nel // 4], yRef[nel // 4]
    xL_2, yL_2 = xRef[nel // 2], yRef[nel // 2]
    xR_2, yR_2 = xRef[3 * nel // 4], yRef[3 * nel // 4]

    xR_Ref_1, yR_Ref_1 = rotate_Geometry(-ang1, xR_1, yR_1,
                                         xRef[nel // 4:nel // 2],
                                         yRef[nel // 4:nel // 2])
    xL_Ref_1, yL_Ref_1 = rotate_Geometry(ang1, xL_1, yL_1,
                                         xRef[:nel // 4], yRef[:nel // 4])
    xR_Ref_2, yR_Ref_2 = rotate_Geometry(-ang2, xR_2, yR_2,
                                         xRef[3 * nel // 4:], yRef[3 * nel // 4:])
    xL_Ref_2, yL_Ref_2 = rotate_Geometry(ang2, xL_2, yL_2,
                                         xRef[nel // 2:3 * nel // 4],
                                         yRef[nel // 2:3 * nel // 4])

    targets[IDs, 1] = np.concatenate([xL_Ref_1, xR_Ref_1, xL_Ref_2, xR_Ref_2])
    targets[IDs, 2] = np.concatenate([yL_Ref_1, yR_Ref_1, yL_Ref_2, yR_Ref_2])
    return targets
