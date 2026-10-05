"""Line-by-line port of the example's ``update_Springs.m``
(``Example_Jellyfish_Swimming/Failed_Attempts/Jellyfish_With_Beams/Backwards_Swimmer_for_LOLz``).

pyIB2d's driver imports this module from the run directory (once per step, when
``update_springs`` is on) and calls

    springs_info = update_Springs(dt, current_time, xLag, yLag, springs_info)

with ``springs_info`` the (N_springs, 5) array ``[leader, follower, k, resting
length, alpha]`` and the *full-step* marker positions ``xLag``/``yLag``.

This is one of the authors' own ``Failed_Attempts/`` (the folder name
says it all: "Backwards_Swimmer_for_LOLz").  The bell carries *invariant* beams
(``jelly.beam``, ``beams = 1``) and the .m file drives only the very last
spring,

    RL(end) = dist * |cos(10 pi t)|,   dist = 0.08649829,

i.e. a single spring beating 20 times per second.

MATLAB's 1-based indexing is translated to 0-based here; the identifiers
(1..N) are rebased by the reader (``ib2d_io.load_example`` auto-detects the
1-based numbering of this matIB2d example).
"""
import numpy as np

DIST = 8.649829e-02    # MATLAB dist (distance between ends of jellyfish bell)


def update_Springs(dt, current_time, xLag, yLag, springs):
    springs[-1, 3] = DIST * abs(np.cos(10.0 * np.pi * current_time))
    return springs
