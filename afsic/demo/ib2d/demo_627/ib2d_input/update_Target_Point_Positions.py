"""Port of the example's ``update_Target_Point_Positions.m`` (the "stirring").

pyIB2d imports this module from the run directory (``sys.path`` entry) and
assigns its return value to ``target_info``, exactly like the MATLAB driver
does; the AFSI port imports the *same* module, so both codes stir with one
implementation.

The piston's target points (all ``targets`` rows; ``NP_before = 0``) are pushed
along +x while ``t <= period`` and back along -x while ``t > period + rest``,
with the offset treated as a per-step velocity::

    t      = rem(current_time, 2*period + rest)          period = 0.05, rest = 0.01
    offset = max_Speed * sin(pi t / period) * dt         max_Speed = 15
    x     += offset                                      (t <= period)
    x     -= max_Speed * sin(pi (t - period - rest) / period) * dt
                                                         (t > period + rest)

so one full cycle (0.11 s) moves the piston out and back with one sinusoidal
stroke of peak speed 15 m/s (a displacement of about 0.477 m each way).  During
the rest window ``period < t <= period + rest`` no offset is applied (upstream:
the second branch is an ``elseif``, so the piston simply sits still).

``targets`` has the IB2d layout ``[id, x, y, stiffness]``; only column 1 (the
x anchor) is modified.

**Stroke speed.**  The shipped ``max_Speed = 15 m/s`` piston is numerically
unstable in *both* modern implementations of these examples (pyIB2d and AFSI
blow up -- NaN markers *and* fluid -- at ``t ~ 0.02`` whatever ``dt`` is used;
see the demo readme).  The stroke speed is therefore read once from the
environment (``MAX_SPEED``, default 15 as upstream) so that a stirred, stable
run is possible; the demos run with a documented reduced value and otherwise
use this module unchanged.
"""
import os

import numpy as np

PERIOD = 0.05
REST = 0.01
L = 0.5
MAX_SPEED = float(os.environ.get("MAX_SPEED", 15.0))


def update_Target_Point_Positions(dt, current_time, targets):
    """Advance the x-anchors of every target point by one step."""
    targets = np.array(targets, copy=True)
    t = np.remainder(current_time, 2 * PERIOD + REST)
    x = targets[:, 1]
    if t <= PERIOD:
        x += MAX_SPEED * np.sin(np.pi * t / PERIOD) * dt
    elif t > PERIOD + REST:
        x -= MAX_SPEED * np.sin(np.pi * (t - PERIOD - REST) / PERIOD) * dt
    targets[:, 1] = x
    return targets
