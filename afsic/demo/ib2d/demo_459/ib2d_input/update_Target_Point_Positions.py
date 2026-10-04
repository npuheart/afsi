"""Line-by-line port of the example's ``update_Target_Point_Positions.m``.

pyIB2d's driver imports this module from the run directory (once per step,
only when ``update_target`` and ``target_pts`` are both on)::

    from update_Target_Point_Positions import update_Target_Point_Positions
    target_info = update_Target_Point_Positions(dt, current_time, target_info)

``targets`` is the (N, 4) array ``[lag-id, x, y, stiffness]`` built by the
driver, with the initial positions of the target points as anchors.  The .m
file moves every target point

* permanently upward (``y += (s/2) dt`` on every call), and
* right for the first half of a period and left for the second half
  (``x += +- s dt`` with ``period = 0.05`` s, ``L = 0.5`` m,
  ``s = L/(period/2) = 20`` m/s),

i.e. one full left-right sweep per 0.05 s period superposed on a constant
upward drift.  MATLAB's 1-based indexing is translated to 0-based here; the
identifiers stored in the file (1..N) are converted to 0-based by the reader
(``ib2d_io.load_example(..., index_base=1)``).
"""
import numpy as np


def update_Target_Point_Positions(dt, current_time, targets):
    ids = targets[:, 0].astype("int")        # IDs: Lag-Pt IDs

    period = 0.05                     # Time it takes to move to the right
    L = 0.5                           # Distance cylinder moves in each dir.
    s = L / (period / 2)              # Speed cylinder moves
    t = np.remainder(current_time, period)

    # Moves cylinder up!
    targets[ids, 2] = targets[ids, 2] + (s / 2) * dt

    # Moves cylinder to the right and then left
    if t < period / 2:
        targets[ids, 1] = targets[ids, 1] + s * dt
    else:
        targets[ids, 1] = targets[ids, 1] - s * dt

    return targets
