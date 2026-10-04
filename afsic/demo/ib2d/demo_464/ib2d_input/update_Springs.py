"""Line-by-line port of the example's ``update_Springs.m``.

pyIB2d's driver imports this module from the run directory (once per step, when
``update_springs`` is on) and calls

    springs_info = update_Springs(dt, current_time, xLag, yLag, springs_info)

with ``springs_info`` the (N_springs, 5) array
``[leader, follower, k, resting length, alpha]`` and ``xLag``/``yLag`` the
*full-step* marker positions.  The .m file drives a **peristaltic wave**: the
resting lengths of the 76 cross-diameter "pump" springs (MATLAB rows
``(N-2)+2 : N_springs-1``, i.e. the last rows of the ``.spring`` file, joining
marker ``i`` to marker ``i + 78``) follow

    RL_i = d - 0.95 sin( (2 pi / L) (x_i - x_M - v t) )   if the sine >= 0,
    RL_i = d                                              otherwise,

with ``d = 1``, ``L = 2``, ``freq = 4`` Hz, ``v = L freq = 8``,
``t = rem(current_time, 1/freq)`` and ``x_M = xLag(N/4)`` (MATLAB index, the
39th marker of this 156-marker tube).  MATLAB's 1-based indexing is translated
to 0-based; the identifiers (1..N) are rebased by the reader
(``ib2d_io.load_example(..., index_base=1)``).
"""
import numpy as np

D_REST = 1.0
L_WAVE = 2.0
FREQ = 4.0
V_WAVE = L_WAVE * FREQ
N_PROBE = 39          # MATLAB N/4 for the 156-marker tube -> 0-based 38


def update_Springs(dt, current_time, xLag, yLag, springs):
    s_1 = springs[:, 0].astype("int")     # MASTER node indices (0-based)
    n_springs = springs.shape[0]
    n_lag = xLag.size

    x_M = xLag[N_PROBE - 1]
    t = np.remainder(current_time, 1.0 / FREQ)

    # MATLAB: for i = (N-2)+2 : N_springs-1   (1-based)
    first = (n_lag - 2) + 2 - 1               # -> 0-based row index
    last = n_springs - 1 - 1
    for i in range(first, last + 1):
        x = xLag[s_1[i]]
        delta = np.sin((2 * np.pi / L_WAVE) * ((x - x_M) - V_WAVE * t))
        springs[i, 3] = D_REST - 0.95 * delta if delta >= 0 else D_REST

    return springs
