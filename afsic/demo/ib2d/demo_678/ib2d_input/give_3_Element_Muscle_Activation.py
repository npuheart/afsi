"""Python port of the example's own ``give_3_Element_Muscle_Activation.m``.

Unlike ``Scallop_3Elem_Muscle`` this file has the **10 parameters** (no
per-muscle index ``i``) that pyIB2d's
``give_3_Element_Muscle_Force_Densities`` actually passes, and it is vectorised
over the (single) muscle band::

    Fm = give_3_Element_Muscle_Activation(v, LF, LFO, SK, a, b, Fmax,
                                          current_time, xPt, xLag)

The upstream MATLAB function is

    Q  = LF/LFO;  F1 = exp(-((Q-1)/SK)^2);  P0 = Fmax;
    F2 = (1/P0)*(b*P0 - a*v)/(v+b);
    Fm = af_Val * Fmax * F1 * F2;            % no ``abs`` here

with the *time-only* activation (``freq = 3`` Hz) ``af_Val = sin(pi freq t)``
while ``t mod 1/freq < 1/(3 freq)`` and ``0`` afterwards (so the shell gives one
sine-shaped power stroke per cycle -- "almost swims").
"""
import numpy as np


def give_3_Element_Muscle_Activation(v, LF, LFO, SK, a, b, Fmax, current_time,
                                     xPt, xLag):
    Q = LF / LFO
    F1 = np.exp(-((Q - 1.0) / SK) ** 2)
    P0 = Fmax
    F2 = (1.0 / P0) * (b * P0 - a * v) / (v + b)
    af_Val = give_Traveling_Triggering_Coefficient(current_time, xLag, xPt)
    return af_Val * Fmax * F1 * F2


def give_Traveling_Triggering_Coefficient(current_time, xLag, xPt):
    """``sin(pi freq t)`` for the first third of each ``1/freq`` cycle, else 0."""
    freq = 3.0
    t = np.fmod(current_time, 1.0 / freq)
    if t < 1.0 / (3.0 * freq):
        return np.sin(np.pi * freq * t)
    return 0.0
