"""Python port of the example's own ``give_Muscle_Activation.m`` (verbatim).

Called by pyIB2d's ``give_Muscle_Force_Densities`` (and by ``main.py``) as

    Fm = give_Muscle_Activation(v, LF, LFO, SK, a, b, Fmax, current_time,
                                xPt, xLag)

with vectorised ``LF``/``LFO``/``SK``/``a``/``b``/``Fmax``/``xPt`` (one entry per
muscle band) and ``xLag`` the full 1-D x-coordinate array of the markers.

The upstream MATLAB function is

    Q  = LF/LFO;  F1 = exp(-((Q-1)/SK)^2);
    P0 = Fmax;    F2 = (1/P0)*(b*P0 - a*v)/(v+b);
    Fm = af_Val * Fmax * F1 * F2;

where the activation ``af_Val`` is the *time-only* square wave
``rem(t, 1/3) < 1/6 -> +1 else -2`` (the scallop's triggering coefficient does
**not** depend on ``xPt``, unlike the HeartTube examples).
"""
import numpy as np


def give_Muscle_Activation(v, LF, LFO, SK, a, b, Fmax, current_time, xPt, xLag):
    Q = LF / LFO
    F1 = np.exp(-((Q - 1.0) / SK) ** 2)
    P0 = Fmax
    F2 = (1.0 / P0) * (b * P0 - a * v) / (v + b)
    af_Val = give_Traveling_Triggering_Coefficient(current_time, xLag, xPt)
    return af_Val * Fmax * F1 * F2


def give_Traveling_Triggering_Coefficient(current_time, xLag, xPt):
    """Square wave in time only (``freq = 3`` Hz, ``+1`` / ``-2``)."""
    freq = 3.0
    t = np.fmod(current_time, 1.0 / freq)
    return 1.0 if t < 1.0 / (2.0 * freq) else -2.0
