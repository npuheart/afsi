"""Line-by-line port of the example's ``update_Target_Point_Positions.m``.

pyIB2d's driver imports this module from the run directory (once per step, only
when ``update_target`` and ``target_pts`` are both on)::

    from update_Target_Point_Positions import update_Target_Point_Positions
    target_info = update_Target_Point_Positions(dt, current_time, target_info)

``targets`` is the ``(N, 4)`` array ``[lag-id, x, y, stiffness]`` built by the
driver.  The example (a nematocyst firing at its prey) has 212 markers: 160
springs (k = 5e5), 160 non-invariant beams (kb = 1e7) and **52 target points**
(k = 1.75e7).  The law fires the targets in +y with a ramped speed,

    yPts = yPts + tanh(2000 t) * (L/period) * dt,   L = 0.4 m, period = 0.05 s,

again only for ``current_time <= period`` (no return stroke in the .m -- the
harpoon stays extended).  The update is incremental (it reads the current
``targets[:, 2]``), and only the y column moves.
"""

import numpy as np

TANH_GAIN = 2000.0                # ramp steepness of the tanh start-up
PERIOD = 0.05                     # time it takes to move to the prey [s]
L = 0.4                           # distance between prey and nematocyst [m]
SPEED = L / PERIOD                # = 8 m/s


def update_Target_Point_Positions(dt, current_time, targets):
    ids = targets[:, 0].astype("int")
    y_pts = targets[:, 2].copy()
    if current_time <= PERIOD:
        y_pts[ids] = y_pts[ids] + np.tanh(TANH_GAIN * current_time) * SPEED * dt
        targets[ids, 2] = y_pts[ids]
    return targets
