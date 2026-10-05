"""Moving-cylinder target trajectory of IB2d -- line-by-line port of the example's

    matIB2d/Examples/Example_Concentration_Dynamics/Moving_Cylinders_64/
        update_Target_Point_Positions.m

(the ``update_target = 1`` user module pyIB2d imports from the run directory
once per step; ``main.py`` imports the very same file, so both codes follow one
trajectory).

Both cylinders are represented by the same 80 marker/target points and are
translated back and forth between the shipped states ``State_A.pts`` and
``State_B.pts`` (``A = x offset 0.07 -> B = 0.97`` at ``y = 0.18/0.82``, i.e. a
horizontal stroke of 0.9 m) with a *piecewise cubic* easing function

    t      = rem(t, 2 t1),  t1 = 0.5            (a full back-and-forth = 1 s)
    tTilde = t / t1            (A -> B)   or  (t - t1) / t1   (B -> A)
    g      = a-poly for tTilde <= 0.05, b-poly for <= 1.0, c-poly otherwise
    targets(:,2:3) = A + g (B - A)   [or B + g (A - B) on the way back]

with the 12 hard-coded coefficients ``a0..a3, b0..b3, c0..c3`` of the MATLAB
module (they are the solution of the 12x12 Hermite system solved by the
example's design script ``interp_Function_Coeffs.m``: C2-continuous, zero
velocity at both ends, unit displacement, "cruise" in the middle).

Upstream quirks (see the demo readme): the module ``read()``s the two ``.pts``
files in both branches (twice per step, and even though they never change) --
here they are loaded once and cached; ``State_*.pts`` have **no** header line
(unlike the structure files), which is what ``read_In_State`` does.
"""
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_CACHE = {}

# hard-coded in the MATLAB module (solution of interp_Function_Coeffs.m)
_P1, _P2, _T1 = 0.05, 0.95, 0.5
_A = (0.0, 0.0, 0.0, 21.052631578947341)
_B = (0.002923976608187, -0.175438596491228, 3.508771929824557,
      -2.339181286549700)
_C = (-20.052631578947224, 63.157894736841683, -63.157894736841683,
      21.052631578947228)


def _read_state(fname):
    if fname not in _CACHE:
        for d in (os.getcwd(), _HERE, os.path.join(_HERE, "ib2d_input")):
            p = os.path.join(d, fname)
            if os.path.exists(p):
                _CACHE[fname] = np.atleast_2d(np.loadtxt(p))
                break
        else:
            raise FileNotFoundError(f"{fname} not found in {os.getcwd()} / {_HERE}")
    return _CACHE[fname]


def _gfunc(tTilde):
    if tTilde <= _P1:
        return _A[0] + _A[1] * tTilde + _A[2] * tTilde ** 2 + _A[3] * tTilde ** 3
    if tTilde <= _P1 + _P2:
        return _B[0] + _B[1] * tTilde + _B[2] * tTilde ** 2 + _B[3] * tTilde ** 3
    return _C[0] + _C[1] * tTilde + _C[2] * tTilde ** 2 + _C[3] * tTilde ** 3


def update_Target_Point_Positions(dt, current_time, targets):
    A = _read_state("State_A.pts")
    B = _read_state("State_B.pts")
    t = np.remainder(current_time, 2 * _T1)
    if t <= _T1:                       # STATE A -> STATE B
        g = _gfunc(t / _T1)
        x = A[:, 0] + g * (B[:, 0] - A[:, 0])
        y = A[:, 1] + g * (B[:, 1] - A[:, 1])
    else:                              # STATE B -> STATE A
        g = _gfunc((t - _T1) / _T1)
        x = B[:, 0] + g * (A[:, 0] - B[:, 0])
        y = B[:, 1] + g * (A[:, 1] - B[:, 1])
    assert len(x) == targets.shape[0], \
        f"state ({len(x)}) and target array ({targets.shape[0]}) disagree"
    targets = np.array(targets, dtype=float, copy=True)
    targets[:, 1] = x                 # MATLAB targets(:,2) = x
    targets[:, 2] = y                 # MATLAB targets(:,3) = y
    return targets
