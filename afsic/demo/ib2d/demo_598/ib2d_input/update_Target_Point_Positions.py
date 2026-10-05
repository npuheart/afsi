"""Line-by-line port of the example's ``update_Target_Point_Positions.m``.

pyIB2d's driver imports this module from the run directory (once per step, only
when ``update_target`` and ``target_pts`` are both on)::

    from update_Target_Point_Positions import update_Target_Point_Positions
    target_info = update_Target_Point_Positions(dt, current_time, target_info)

``targets`` is the ``(N, 4)`` array ``[lag-id, x, y, stiffness]`` built by the
driver.  The Insect Flight wing is 205 target points (k = 5e7); the law strokes
the wing to the right by

    xPts = xPts + tanh(200 t) * (L/period) * dt,   L = 0.6 m, period = 0.025 s,

for ``current_time <= period`` and then *stops*: the .m has no return stroke
(the ``else`` branch does not exist), so the wing translates 0.6 m to the right
and stays there -- an upstream quirk of this "Failed_Attempts"-style example
(the MATLAB code and the MATLAB comment both say "Moves wing to the right!").

The update is incremental: ``xPts`` is read from the current ``targets[:, 1]``,
so the displacement accumulates step by step (and the ``tanh(200 t)`` ramp,
≈ 1 after 0.025 s, makes the first few steps slow).  Only the x column moves;
y is untouched (the .m leaves it commented out).
"""

import numpy as np

TANH_GAIN = 200.0                 # ramp steepness of the tanh start-up
PERIOD = 0.025                    # time it takes to move to the right [s]
L = 0.6                           # distance the wing moves [m]
SPEED = L / PERIOD                # = 24 m/s


def update_Target_Point_Positions(dt, current_time, targets):
    ids = targets[:, 0].astype("int")
    x_pts = targets[:, 1].copy()
    if current_time <= PERIOD:
        x_pts[ids] = x_pts[ids] + np.tanh(TANH_GAIN * current_time) * SPEED * dt
        targets[ids, 1] = x_pts[ids]
    return targets
