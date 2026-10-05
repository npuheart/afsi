"""pyIB2d hook that reproduces ``IBM_Driver.m``'s electrophysiology block for
the ``Electromechanical_Pumping`` example.

pyIB2d parses ``electro_phys`` but never implements it (grep the Python driver:
the flag is read into ``electro_phys_Yes`` and then unused), while the MATLAB
driver, at every step, overwrites the *stiffness* of the springs in
``ePhys_Start:ePhys_End`` with

    k_i = (1.6e1 * v_i(t))^4,       v = FitzHugh-Nagumo membrane potential,

before the force computation (``electro_phys == 1 && FV_LT_muscle == 0``).
Note the MATLAB comment in the example's ``update_Springs.m`` claims column 3
is the stiffness and column 4 the resting length -- and the blackbox indeed
writes column 3, so the **stiffness** is modulated (see the readme).

Python's driver calls ``update_Springs(dt, current_time, xLag, yLag,
springs_info)`` once per step when ``update_springs = 1`` (right after the
half-step move, i.e. exactly where the MATLAB electro block sits).  This module
therefore provides the missing electro coupling through that documented hook:
``run_reference.py`` sets ``update_springs = 1`` for the reference run and this
file is copied into the reference run directory (and imported by the driver from
there), while ``main.py`` uses the exact same ``FitzHugh_Nagumo_1d`` module.

``springs_info`` columns are ``[leader, follower, stiffness, resting length,
alpha]`` (0-based indices); the potential is built once, on first call, from the
example's ``input2d`` (overridable with ``IB2D_DT`` / ``IB2D_TFINAL``).
"""
import os

import numpy as np

from FitzHugh_Nagumo_1d import FitzHugh_Nagumo_1d, example_params

_ELECTRO = None     # (v_interp, ind_start, ind_end)
_CT = 1             # the driver's ePhys_Ct counter (1-based, first row t = 0)


def _build():
    global _ELECTRO
    if _ELECTRO is None:
        dt, tend = example_params(os.path.dirname(os.path.abspath(__file__)))
        _ELECTRO = FitzHugh_Nagumo_1d(dt, tend)
    return _ELECTRO


def update_Springs(dt, current_time, xLag, yLag, springs):
    """IBM_Driver.m: ``springs_info(ePhys_Start:ePhys_End, 3) = (16 v)^4``."""
    global _CT
    v_interp, ind_start, ind_end = _build()
    row = _CT - 1
    springs[ind_start - 1:ind_end, 2] = (16.0 * v_interp[row]) ** 4
    _CT += 1
    return springs
