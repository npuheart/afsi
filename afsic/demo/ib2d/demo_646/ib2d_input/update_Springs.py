"""Line-by-line port of the example's ``update_Springs.m``
(``Example_Jellyfish_Swimming/Failed_Attempts/Jellyfish_IBAMR_Convert``).

pyIB2d's driver imports this module from the run directory (once per step, when
``update_springs`` is on) and calls

    springs_info = update_Springs(dt, current_time, xLag, yLag, springs_info)

with ``springs_info`` the (N_springs, 5) array ``[leader, follower, k, resting
length, alpha]`` and the *full-step* marker positions ``xLag``/``yLag``.

This is one of the authors' own ``Failed_Attempts/``: a conversion of an
IBAMR jellyfish setup.  The .m file contracts the springs of the bell margin
(MATLAB rows ``319:end``) every step with a *full* sine (note the ``2*pi``, the
other jellyfish examples use ``pi`` and ``|cos|``),

    RL_i = |sin(freq t 2 pi)|,   freq = 0.875  ->  period 1.143 s,

so the bell contracts once per period.  The shipped ``input2d`` uses a 128x1024
box (16 long, 2 wide) with ``dt = 1e-5`` and ``Tfinal = 12``.

MATLAB's 1-based indexing is translated to 0-based here; the identifiers
(1..N) are rebased by the reader (``ib2d_io.load_example`` auto-detects the
1-based numbering of this matIB2d example).
"""
import numpy as np

FREQ = 0.875      # contraction frequency: RL = |sin(freq t 2 pi)|
FIRST_ROW = 319   # MATLAB springs_info(319:end, 4)


def update_Springs(dt, current_time, xLag, yLag, springs):
    springs[FIRST_ROW - 1:, 3] = \
        abs(np.sin(FREQ * current_time * 2.0 * np.pi))
    return springs
