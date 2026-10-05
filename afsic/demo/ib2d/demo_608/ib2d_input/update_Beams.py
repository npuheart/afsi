"""Line-by-line port of the example's ``update_Beams.m``.

pyIB2d's driver imports this module from the run directory (once per step, when
``update_beams`` is on)::

    from update_Beams import update_Beams
    beams_info = update_Beams(dt, current_time, beams_info)

``beams_info`` is the ``(N, 5)`` array ``[p1, p2, p3, kb, C]`` of the
*invariant* torsional beams (N = 81 here); the law rewrites the
reference curvature ``C`` (column 5).

The example (a "Failed_Attempts" jellyfish whose beam curvatures are driven
directly) hardcodes the two curvature states ``C1`` (contracted) and ``C2``
(expanded) of every beam in a local function ``give_Beam_Curvatures``, and
linearly interpolates between them over the 0.1 s cycle,

    P1 = 0.3 period (contraction), P2 = 0.7 period (expansion),
    t  = current_time mod period,
    C  = C1 + (t/P1) (C2 - C1)          for t <= P1,
    C  = C2 + ((t-P1)/P2) (C1 - C2)     otherwise.

Upstream quirks:

* the .m wraps that assignment in a ``for i = 1:length(beams_info(:,5))`` loop
  whose body overwrites the *whole* column every iteration (a no-op loop); the
  port performs the vectorised assignment once;
* the curvature table is hardcoded in the .m and is copied here verbatim
  (81 rows x 2 columns).
"""
import numpy as np

PERIOD = 0.1                      # period of contraction/expansion [s]
P1 = 0.3 * PERIOD                 # contraction fraction
P2 = PERIOD - P1                  # expansion fraction

CURVES = np.array([
    [-2.864292e-07, -8.979307e-08],
    [-2.872202e-07, -9.008286e-08],
    [-2.885419e-07, -9.047027e-08],
    [-2.903989e-07, -9.105332e-08],
    [-2.927983e-07, -9.193269e-08],
    [-2.960176e-07, -9.29165e-08],
    [-2.998039e-07, -9.410602e-08],
    [-3.041696e-07, -9.560733e-08],
    [-3.091325e-07, -9.727721e-08],
    [-3.147086e-07, -9.917227e-08],
    [-3.212043e-07, -1.014054e-07],
    [-3.286533e-07, -1.038846e-07],
    [-3.370976e-07, -1.066737e-07],
    [-3.462828e-07, -1.099535e-07],
    [-3.56546e-07, -1.136386e-07],
    [-3.682544e-07, -1.177511e-07],
    [-3.811728e-07, -1.223174e-07],
    [-3.953674e-07, -1.274862e-07],
    [-4.112494e-07, -1.334263e-07],
    [-4.289288e-07, -1.401956e-07],
    [-4.485268e-07, -1.479964e-07],
    [-4.705421e-07, -1.569195e-07],
    [-4.947855e-07, -1.670684e-07],
    [-5.218025e-07, -1.788625e-07],
    [-5.522347e-07, -1.928e-07],
    [-5.855227e-07, -2.091336e-07],
    [-6.227644e-07, -2.2869e-07],
    [-6.643237e-07, -2.523007e-07],
    [-7.096185e-07, -2.80969e-07],
    [-7.59943e-07, -3.163939e-07],
    [-8.152455e-07, -3.607256e-07],
    [-8.742746e-07, -4.178003e-07],
    [-9.372455e-07, -4.927781e-07],
    [-1.003177e-06, -5.936574e-07],
    [-1.070229e-06, -7.33637e-07],
    [-1.136246e-06, -9.341778e-07],
    [-1.197397e-06, -1.228301e-06],
    [-1.250119e-06, -1.661816e-06],
    [-1.290668e-06, -2.263332e-06],
    [-1.016078e-06, -2.875063e-06],
    [-7.840238e-07, -3.139832e-06],
    [-1.016078e-06, -2.875063e-06],
    [-1.290668e-06, -2.263332e-06],
    [-1.250119e-06, -1.661816e-06],
    [-1.197397e-06, -1.228301e-06],
    [-1.136246e-06, -9.341778e-07],
    [-1.070229e-06, -7.33637e-07],
    [-1.003177e-06, -5.936574e-07],
    [-9.372455e-07, -4.927781e-07],
    [-8.742746e-07, -4.178003e-07],
    [-8.152455e-07, -3.607256e-07],
    [-7.59943e-07, -3.163939e-07],
    [-7.096185e-07, -2.80969e-07],
    [-6.643237e-07, -2.523007e-07],
    [-6.227644e-07, -2.2869e-07],
    [-5.855227e-07, -2.091336e-07],
    [-5.522347e-07, -1.928e-07],
    [-5.218025e-07, -1.788625e-07],
    [-4.947855e-07, -1.670684e-07],
    [-4.705421e-07, -1.569195e-07],
    [-4.485268e-07, -1.479964e-07],
    [-4.289288e-07, -1.401956e-07],
    [-4.112494e-07, -1.334263e-07],
    [-3.953674e-07, -1.274862e-07],
    [-3.811728e-07, -1.223174e-07],
    [-3.682544e-07, -1.177511e-07],
    [-3.56546e-07, -1.136386e-07],
    [-3.462828e-07, -1.099535e-07],
    [-3.370976e-07, -1.066737e-07],
    [-3.286533e-07, -1.038846e-07],
    [-3.212043e-07, -1.014054e-07],
    [-3.147086e-07, -9.917227e-08],
    [-3.091325e-07, -9.727721e-08],
    [-3.041696e-07, -9.560733e-08],
    [-2.998039e-07, -9.410602e-08],
    [-2.960176e-07, -9.29165e-08],
    [-2.927983e-07, -9.193269e-08],
    [-2.903989e-07, -9.105332e-08],
    [-2.885419e-07, -9.047027e-08],
    [-2.872202e-07, -9.008286e-08],
    [-2.864292e-07, -8.979307e-08]
])


def update_Beams(dt, current_time, beams_info):
    t = np.remainder(current_time, PERIOD)
    n = beams_info.shape[0]
    assert CURVES.shape[0] == n, f"curvature table {CURVES.shape} vs {n} beams"
    if t <= P1:
        beams_info[:, 4] = CURVES[:, 0] + (t / P1) * (CURVES[:, 1] - CURVES[:, 0])
    else:
        tt = t - P1
        beams_info[:, 4] = CURVES[:, 1] + (tt / P2) * (CURVES[:, 0] - CURVES[:, 1])
    return beams_info
