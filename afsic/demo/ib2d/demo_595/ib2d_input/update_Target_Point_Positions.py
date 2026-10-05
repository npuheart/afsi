"""Line-by-line port of the example's ``update_Target_Point_Positions.m``.

pyIB2d's driver imports this module from the run directory (once per step, only
when ``update_target`` and ``target_pts`` are both on)::

    from update_Target_Point_Positions import update_Target_Point_Positions
    target_info = update_Target_Point_Positions(dt, current_time, target_info)

``targets`` is the ``(N, 4)`` array ``[lag-id, x, y, stiffness]`` built by the
driver.  The sawtooth pump has 6552 target points (k = 1e11); the first
``N_top = 3276`` (the *upper* jaw, i.e. the first half of the .vertex file)
oscillate in y about their initial position,

    y_i(t) = y_i(0) - (a/2) sin(2 pi freq t),   a = 0.337 m,  freq = 2 Hz,

with the amplitude, frequency and ``N_top`` hardcoded in the .m file.  The lower
jaw (markers 3277..6552) does not move (the .m has that update commented out).

Upstream quirk (documented, deliberately *not* reproduced): the .m reads the
initial y values with ``textscan(fileID,'%f %f','CollectOutput',1)`` and then
takes ``vertices(2:end,2)``.  Because the first line of ``sawtooth.vertex``
holds the bare point count, MATLAB's ``textscan`` -- which treats line breaks as
whitespace for numeric formats -- pairs the numbers *as a stream*
(``(count, x1), (y1, x2), (y2, x3), ...``), so the shipped .m would write the
``x`` coordinates of points 2..3277 into the ``y`` slopes of markers 1..3276,
destroying the sawtooth geometry.  This port reads the file row-wise
(``np.loadtxt(..., skiprows=1)``), which is what the surrounding code
(``vertices(2:end,:)``) clearly intends; the same module is imported by the
pyIB2d reference, so both codes run the intended law here.
"""
import os

import numpy as np

FREQ = 2.0                        # pumping frequency [Hz]
A = 0.337                         # total vertical distance the pump travels [m]
N_TOP = 3276                      # # of points along the top (of 6552)

_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "sawtooth.vertex")
_CACHE = {}


def _y_positions():
    """Row-wise read of the .vertex file -> initial y values (see docstring)."""
    if "y" not in _CACHE:
        PTS = np.loadtxt(_FILE, skiprows=1)[:, 1]
        _CACHE["y"] = PTS
    return _CACHE["y"]


def update_Target_Point_Positions(dt, current_time, targets):
    yPTS = _y_positions()
    targets[0:N_TOP, 2] = yPTS[0:N_TOP] - (A / 2.0) * np.sin(
        2.0 * np.pi * FREQ * current_time)
    return targets
