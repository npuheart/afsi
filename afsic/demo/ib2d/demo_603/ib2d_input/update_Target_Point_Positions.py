"""Line-by-line port of the example's ``update_Target_Point_Positions.m``.

pyIB2d's driver imports this module from the run directory (once per step, only
when ``update_target`` and ``target_pts`` are both on)::

    from update_Target_Point_Positions import update_Target_Point_Positions
    target_info = update_Target_Point_Positions(dt, current_time, target_info)

``targets`` is the ``(N, 4)`` array ``[lag-id, x, y, stiffness]`` built by the
driver.  The structure is 104 target points (k = 1e7) forming two hive pairs (four petals of 26 markers each) in a 1 x 1 periodic box, plus a passive concentration scalar (kDiff from hive.concentration) seeded inside the hive.

The motion is a *rotation* of each quarter of the structure about its own base
point, driven by two oscillatory angles

    n        = ceil((t + dt)/dt),
    angVec1  = (pi dt) : (pi dt) : 2 pi,     ang1 = angVec1[mod(n, L1)],
    angVec2  = (pi dt/2) : (pi dt/2) : 2 pi, ang2 = angVec2[mod(n, L2)],

folded into ``[0, 2*full_ang]`` with ``full_ang = pi/3`` (a triangle wave), and
each quarter of the geometry is rotated by +/- ang about the corner point that
anchors it:

    rows 0   .. N/4-1      rotate by +ang1 about (xRef(1),     yRef(1))
    rows N/4 .. N/2-1      rotate by -ang1 about (xRef(N/4+1), ...)
    rows N/2 .. 3N/4-1     rotate by +ang2 about (xRef(N/2+1), ...)
    rows 3N/4 .. N-1       rotate by -ang2 about (xRef(3N/4+1), ...)

so the two pairs open and close (the reference geometry comes from
``hive.vertex``, row-wise: the .m drops its header line with ``vertices(2:end,:)``).

Upstream quirks / bugs:

* ``give_Me_Oscillatory_Angle`` indexes ``angVec1(mod(n, L1))`` with a 1-based
  index, so ``mod(n, L1) == 0`` would error in MATLAB (reachable only after
  ``L1 = floor(2/dt)`` steps); the port uses the equivalent
  ``angVec1[mod(n, L1) - 1]`` and clamps that unreachable 0 to the last entry;
* the .m reads the vertex file with ``textscan(fileID,'%f %f')`` +
  ``mat_info(2:end,:)``; because the file starts with a bare point count,
  MATLAB's stream-wise numeric parsing would shift the columns (x <-> y, off by
  one point).  This port reads the file row-wise
  (``np.loadtxt(..., skiprows=1)``), which is what ``(2:end, :)`` intends, and
  both codes here import this same module;
* the fold uses ``<=`` at the upper end, so the folded angle never exceeds
  ``full_ang``.
"""
import os

import numpy as np

FULL_ANG = np.pi / 3.0            # how far open the movement goes [rad]
VERTEX_FILE = "hive.vertex"


def _ref_geometry():
    return np.loadtxt(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   VERTEX_FILE), skiprows=1)[:, :2]


def _angle(value, full_ang):
    """Triangle-wave fold of ``mod(value, 2 full_ang)``."""
    a = np.remainder(value, 2.0 * full_ang)
    if full_ang < a <= 2.0 * full_ang:
        a = 2.0 * full_ang - a
    return a


def _give_me_oscillatory_angle(dt, current_time):
    """The .m's local helper ``give_Me_Oscillatory_Angle``."""
    n = int(np.ceil((current_time + dt) / dt))

    dtheta = np.pi * dt
    ang_vec1 = np.arange(1, int(np.floor(2.0 * np.pi / dtheta)) + 1) * dtheta
    n1 = n % ang_vec1.size
    ang_input1 = ang_vec1[n1 - 1] if n1 > 0 else ang_vec1[-1]

    dtheta = np.pi * dt / 2.0
    ang_vec2 = np.arange(1, int(np.floor(2.0 * np.pi / dtheta)) + 1) * dtheta
    n2 = n % ang_vec2.size
    ang_input2 = ang_vec2[n2 - 1] if n2 > 0 else ang_vec2[-1]
    return ang_input1, ang_input2


def _rotate(ang, x_c, y_c, x_ref, y_ref):
    xr, yr = x_ref - x_c, y_ref - y_c
    return (xr * np.cos(ang) - yr * np.sin(ang) + x_c,
            xr * np.sin(ang) + yr * np.cos(ang) + y_c)


def update_Target_Point_Positions(dt, current_time, targets):
    ids = targets[:, 0].astype(np.int64)
    n_t = len(ids)
    a1, a2 = _give_me_oscillatory_angle(dt, current_time)
    ang1 = _angle(a1, FULL_ANG)
    ang2 = _angle(a2, FULL_ANG)

    ref = _ref_geometry()
    q = n_t // 4
    x_ref, y_ref = ref[:, 0], ref[:, 1]
    xl1, yl1 = ref[:q, 0], ref[:q, 1]
    xr1, yr1 = ref[q:2 * q, 0], ref[q:2 * q, 1]
    xl2, yl2 = ref[2 * q:3 * q, 0], ref[2 * q:3 * q, 1]
    xr2, yr2 = ref[3 * q:, 0], ref[3 * q:, 1]

    xR1, yR1 = _rotate(-ang1, x_ref[q], y_ref[q], xr1, yr1)
    xL1, yL1 = _rotate(ang1, x_ref[0], y_ref[0], xl1, yl1)
    xR2, yR2 = _rotate(-ang2, x_ref[3 * q], y_ref[3 * q], xr2, yr2)
    xL2, yL2 = _rotate(ang2, x_ref[2 * q], y_ref[2 * q], xl2, yl2)

    targets[ids, 1] = np.concatenate([xL1, xR1, xL2, xR2])
    targets[ids, 2] = np.concatenate([yL1, yR1, yL2, yR2])
    return targets
