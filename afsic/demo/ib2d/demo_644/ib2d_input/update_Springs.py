"""Line-by-line port of the example's ``update_Springs.m``
(``Example_Jellyfish_Swimming/Failed_Attempts/Jellyfish_With_Beams/Forward_Swimmer``).

pyIB2d's driver imports this module from the run directory (once per step, when
``update_springs`` is on) and calls

    springs_info = update_Springs(dt, current_time, xLag, yLag, springs_info)

with ``springs_info`` the (N_springs, 5) array ``[leader, follower, k, resting
length, alpha]`` and the *full-step* marker positions ``xLag``/``yLag``.

This is one of the authors' own ``Failed_Attempts/`` (the folder is
called "Forward_Swimmer").  The bell carries *invariant* beams (``jelly.beam``,
``beams = 1``) and the .m file drives only the very last spring,

    RL(end) = dist * (1 - 0.5 |sin(10 pi t)|),   dist = 0.1250077,

so only a single spring is actuated -- 10 contractions per second (the
sub-grid actuation is why the authors filed this run as a failed attempt).

MATLAB's 1-based indexing is translated to 0-based here; the identifiers
(1..N) are rebased by the reader (``ib2d_io.load_example`` auto-detects the
1-based numbering of this matIB2d example).
"""
import numpy as np

DIST = 1.250077e-01    # MATLAB dist (distance between ends of jellyfish bell)


def update_Springs(dt, current_time, xLag, yLag, springs):
    springs[-1, 3] = DIST * (1.0 - 0.5 * abs(np.sin(10.0 * np.pi * current_time)))
    return springs
