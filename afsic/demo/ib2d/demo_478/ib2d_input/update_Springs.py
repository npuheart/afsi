"""Line-by-line port of the example's ``update_Springs.m``
(``Example_Concentration_Gradients/Impedance_Pumping_Tube``).

pyIB2d's driver imports this module from the run directory (once per step, when
``update_springs`` is on) and calls

    springs_info = update_Springs(dt, current_time, xLag, yLag, springs_info)

with ``springs_info`` the (N_springs, 5) array ``[leader, follower, k, resting
length, alpha]`` and ``xLag``/``yLag`` the *full-step* marker positions.  The .m
file drives an **impedance pump**: the resting lengths of 16 springs (MATLAB
rows ``(N-2)+60 : (N-2)+75``, with ``N = length(xLag) = 412`` markers) are
pinched sinusoidally,

    RL_i = d - |0.9 d sin(2 pi freq t)|,      d = 1, freq = 10 Hz,

so the tube is squeezed twice per period (``|sin|``) at 10 Hz and the resulting
volume change drives the flow -- "impedance pumping".

MATLAB's 1-based indexing is translated to 0-based here; the identifiers
(1..N) are rebased by the reader (``ib2d_io.load_example(..., index_base=1)``).
"""
import numpy as np

D_REST = 1.0          # tube diameter
FREQ = 10.0           # pumping frequency (Hz)
ROW_OFFSET = 60       # MATLAB (N-2)+60 first updated spring
N_ROWS = 16           # MATLAB (N-2)+60 : (N-2)+75


def update_Springs(dt, current_time, xLag, yLag, springs):
    n_lag = xLag.size
    first = (n_lag - 2) + ROW_OFFSET - 1          # -> 0-based row index
    rl = D_REST - abs(0.9 * D_REST
                      * np.sin(2 * np.pi * FREQ * current_time))
    springs[first:first + N_ROWS, 3] = rl
    return springs
