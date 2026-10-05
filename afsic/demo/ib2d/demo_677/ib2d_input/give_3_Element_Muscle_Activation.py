"""Python port of the example's own ``give_3_Element_Muscle_Activation.m``.

pyIB2d calls it (from ``give_3_Element_Muscle_Force_Densities``) **without the
per-muscle index** and with *vectors* over all muscle bands::

    Fm = give_3_Element_Muscle_Activation(v, LF, LFO, SK, a, b, Fmax,
                                          current_time, xPt, xLag)

so the MATLAB ``i`` argument is reproduced here as the vector position
(``i = 1, 2, 3`` for the three scallop muscles, in file order).

The upstream MATLAB function is

    Q  = LF/LFO;  F1 = exp(-((Q-1)/SK)^2);  P0 = Fmax;
    F2 = (1/P0)*(b*P0 - a*v)/(v+b);
    Fm = af_Val * abs(Fmax*F1*F2);                       % note the ``abs``

with the *sequential* activation square-wave ``af_Val`` below (a different
window per muscle, ``freq = 10`` Hz: ``muscle 1`` acts in ``t mod 0.1 < 0.01``,
``muscle 2`` in ``0.005 <= t < 0.015``, ``muscle 3`` in ``0.01 <= t < 0.015``;
the ``on``/``PE_coeff`` outputs are unused by pyIB2d's force routine).
"""
import numpy as np


def give_3_Element_Muscle_Activation(v, LF, LFO, SK, a, b, Fmax, current_time,
                                     xPt, xLag):
    Q = LF / LFO
    F1 = np.exp(-((Q - 1.0) / SK) ** 2)
    P0 = Fmax
    F2 = (1.0 / P0) * (b * P0 - a * v) / (v + b)
    af_Val = give_Traveling_Triggering_Coefficient(current_time, xLag, xPt,
                                                   len(np.atleast_1d(LF)))
    return af_Val * np.abs(Fmax * F1 * F2)


def give_Traveling_Triggering_Coefficient(current_time, xLag, xPt, nmuscles):
    """``af_Val`` per muscle: sequential square waves, ``freq = 10`` Hz."""
    freq = 10.0
    t = np.fmod(current_time, 1.0 / freq)
    af = np.zeros(nmuscles)
    if t < 0.04:
        windows = {1: (0.0, 0.01), 2: (0.005, 0.015), 3: (0.01, 0.015)}
        for i in range(1, nmuscles + 1):
            lo, hi = windows.get(i, (np.inf, np.inf))
            if lo <= t < hi:
                af[i - 1] = np.tanh(50.0 * t)
    return af
