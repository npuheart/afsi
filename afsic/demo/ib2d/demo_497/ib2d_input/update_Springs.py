"""Line-by-line port of the example's ``update_Springs.m``
(``Examples_First_Year_Seminar/Jellyfish_Oscillator``).

pyIB2d's driver imports this module from the run directory once per step (when
``update_springs`` is on) and calls

    springs_info = update_Springs(dt, current_time, xLag, yLag, springs_info)

with ``springs_info`` the (N_springs, 5) array ``[leader, follower, k, resting
length, alpha]``.  The single line of the .m file

    springs_info(end, 4) = dist * ( 1 - 0.5 |sin(10 pi t)| )

sets the resting length of the **last** spring (the one closing the jellyfish
bell, ``dist = 0.1250077`` from ``Jellyfish.m``) with a ``|sin|`` at 5 Hz, so
the bell is contracted twice per 0.2 s cycle and the jellyfish "swims".

MATLAB's 1-based ``end`` is the last row in both languages, and column 4
becomes column 3 here.
"""
import numpy as np

DIST = 1.250077e-01


def update_Springs(dt, current_time, xLag, yLag, springs):
    springs[-1, 3] = DIST * (1.0 - 0.5 * abs(np.sin(10 * np.pi * current_time)))
    return springs
