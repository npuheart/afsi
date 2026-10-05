"""Port of the example's ``give_Ca_ElectroPhys_Muscle_Activation.m``
(``Example_HeartTube/EM_Pump_w_Calcium_and_Muscles``).

This is the ``electro_phys = 1`` + ``FV_LT_muscle = 1`` combination: the MATLAB
driver's spring-stiffness modulation is **skipped** (``electro_phys_Yes && !muscles_Yes``
is false) and instead the muscle force is driven by the electrophysiology
signal itself,

    a_f = 1e1 * electro_potential(floor(ct + 1), ith_muscle),   ct = t / dt,

where ``ith_muscle`` is the muscle's **leader node index** (the .m passes
``id_Leader``) and ``electro_potential`` is the calcium-driven FitzHugh-Nagumo
history of this example (``FitzHugh_Nagumo_1d.py`` + ``Calcium_Dynamics.py``).
The force law is the ordinary FV_LT one,

    Fm = a_f * Fmax * exp(-((Q-1)/SK)^2) * (1/P0)(b P0 - a v)/(v + b),
    Q = LF/LFO,  P0 = Fmax,

with ``v`` the band's contraction speed.

pyIB2d has no electro branch at all: its FV_LT path simply imports
``give_Muscle_Activation`` from the run directory and calls it with
``(v, LF, LFO, SK, a, b, Fmax, current_time, xPt, xLag)`` (all vectorised over
the muscles).  This module keeps that signature so the *same* file drives the
reference run and ``main.py``; the leader-node index is recovered from the
example's own ``HeartTube.muscle`` file (read at first call), and the
electrophysiology is built once from ``input2d`` (overridable with ``IB2D_DT`` /
``IB2D_TFINAL``).
"""
import os

import numpy as np

from FitzHugh_Nagumo_1d import FitzHugh_Nagumo_1d, example_params

_HERE = os.path.dirname(os.path.abspath(__file__))
_DT = None          # IBM dt used for the a-priori potential
_ELECTRO = None     # (v_interp, ind_start, ind_end)
_COLUMNS = None     # 0-based potential column of each muscle band (file order)


def _leader_columns():
    """Potential column used by each muscle band: MATLAB indexes
    ``electro_potential(:, id_Leader)``, and the columns are the tube nodes in
    order, so the 0-based column is ``id_Leader`` minus the file's index base
    (1 for the verbatim matIB2d files, 0 for the rebased copy pyIB2d reads)."""
    with open(os.path.join(_HERE, "HeartTube.muscle")) as fh:
        n = int(float(fh.readline().split()[0]))
        rows = np.atleast_2d(np.loadtxt(fh, max_rows=n))
    leader = rows[:, 0].astype(int)
    assert leader.min() in (1, 2), "unexpected leader-node numbering"
    return leader - (1 if leader.min() == 2 else 0)


def electro_potential():
    """The a-priori calcium/FitzHugh-Nagumo history (built on first use)."""
    global _DT, _ELECTRO, _COLUMNS
    if _ELECTRO is None:
        dt, tend = example_params(_HERE)
        _DT = dt
        _COLUMNS = _leader_columns()
        _ELECTRO = FitzHugh_Nagumo_1d(dt, tend)
    return _ELECTRO


def give_Muscle_Activation(v, LF, LFO, SK, a, b, Fmax, current_time, xPt, xLag):
    """FV_LT muscle force with the electrophysiology activation coefficient."""
    Q = LF / LFO
    F1 = np.exp(-((Q - 1) / SK) ** 2)
    P0 = Fmax
    # Hill force-velocity term (the .m's ``F2``)
    F2 = (1.0 / P0) * (b * P0 - a * v) / (v + b)
    v_interp, _s, _e = electro_potential()
    row = int(np.floor(current_time / _DT + 1.0)) - 1     # .m: floor(ct+1)
    af_val = 1e1 * v_interp[row, _COLUMNS]
    return af_val * Fmax * F1 * F2
