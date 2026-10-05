"""Port of the Jellyfish convergence-skeleton ``update_Springs.m``
(``Examples_Education/Convergence/Jellyfish/Simulation_Skeletons/<Re>/<Res>/``),
whose *only* active line is

    springs_info(39:end,4) = abs( cos(freq*current_time*pi) );   % freq = 1.0

pyIB2d's driver imports this module from the run directory (once per step, when
``update_springs`` is on) and calls

    springs_info = update_Springs(dt, current_time, xLag, yLag, springs_info)

with ``springs_info`` the (N_springs, 5) array ``[leader, follower, k, resting
length, alpha]`` and ``xLag``/``yLag`` the *full-step* marker positions.

UPSTREAM BUG (found while porting; see the demo readme)
-------------------------------------------------------
The row range ``39:end`` is a copy-paste of the muscle *count* of the original
``Example_Jellyfish_Swimming/Hoover_Jellyfish`` (318 bell springs, then the
muscle block ``319:357``, i.e. 39 springs).  In these skeletons the spring
files store the bell contour first and the muscles **last**, so ``39:end``

* is empty for Res_32 (20 springs) and Res_48 (31 springs) -> *no* activation;
* hits only the muscles for Res_64 (42 springs, 38 of them bell) -> by accident
  correct;
* from Res_96 on also hits bell springs (Res_192: bell rows 1..118, so
  ``39:132`` rewrites 80 bell contour springs, Res_1024: 677 rows).

The bell contour springs (``k = 1e7/ds``) have their resting length equal to the
marker spacing, so rewriting it to ``|cos| <= 1`` stretches them by ``1/ds``
(48x at Res_192) and the example explodes -- measured on demo_521 (verbatim
mode): AFSI becomes NaN at ``t = 6e-3`` and pyIB2d at ``t = 8e-3``.

The generator (``Make_Jelly_Geometry.m``) stores the muscle rest lengths as 0.0
exactly (they are ``k = 1e5`` tension springs that pull the two bell wings
together), so this port recovers the intended, original Hoover-jellyfish
behaviour by updating **only the ``L0 == 0`` muscle block** -- identical to
``springs_info(319:357,4)`` of the working ``Hoover_Jellyfish`` example and to
``starting_muscle = 318+1`` of ``Tentacle_Jelly``.  Set

    JELLY_MUSCLE_ROWS=verbatim

to reproduce the upstream ``39:end`` behaviour instead (both codes then explode
in the very same way, which is what the readme's "上游 bug" section reports).

MATLAB's 1-based indexing is translated to 0-based (row 39 -> 0-based index 38);
the identifiers are rebased by the reader (``ib2d_io.load_example``).
"""
import os

import numpy as np

FREQ = 1.0               # contraction frequency (Hz): RL = |cos(freq*pi*t)|
VERBATIM_FIRST_ROW = 39  # MATLAB "springs_info(39:end,4)"
MODE = os.environ.get("JELLY_MUSCLE_ROWS", "muscles")

_muscle_mask = None


def _mask(springs):
    """Rows of the muscle block (stored rest length exactly 0), cached."""
    global _muscle_mask
    if _muscle_mask is None:
        _muscle_mask = springs[:, 3] == 0.0
    return _muscle_mask


def update_Springs(dt, current_time, xLag, yLag, springs):
    rl = abs(np.cos(FREQ * current_time * np.pi))
    if MODE == "verbatim":
        springs[VERBATIM_FIRST_ROW - 1:, 3] = rl
    else:
        springs[_mask(springs), 3] = rl
    return springs
