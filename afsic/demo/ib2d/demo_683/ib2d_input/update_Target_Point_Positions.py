"""Pulsing-coral geometry of IB2d -- line-by-line port of the example's own

    matIB2d/Examples/Example_Concentration_Dynamics/Corals_64/
        update_Target_Point_Positions.m

(the ``update_target = 1`` user module pyIB2d imports from the run directory
once per step; the same file is used by ``main.py``, so both codes follow one
trajectory).  It rebuilds the polyp outline

    s    = (0 : ds : total_meanL) / total_meanL,      ds = dx/2, dx = Lx/Nx
    XbL  = (C1(1) s^3 + C1(2) s^2 + C1(3) s + C1(4)) * total_meanL / L_ten
           + total_offset
    YbL  = (C2(1) s^3 + C2(2) s^2 + C2(3) s + C2(4)) * total_meanL / L_ten
    x    = [flip(-XbL) XbStem XbL] + xoffset
    y    = [flip( YbL) YbStem YbL] + yoffset

from the cubic coefficients ``c1vals`` / ``c2vals`` (one row per time step,
``indx = ceil(t/dt) + 1``) of ``cval.mat``, with ``total_meanL`` /
``total_offset`` from ``total_coeffs.mat`` and the 5-point stem ``XbStem`` /
``YbStem`` (1 x 5) also stored in ``cval.mat``, and writes columns 2/3 (x/y) of
the driver's ``target_info`` array (``[id, x, y, k]``).

Notes / upstream quirks (see the demo readme):

* ``ds = dx/2`` is **hard-coded with Nx = 64**, so the module only works at the
  shipped resolution: 27 points per tentacle + 5 stem points = 59 markers, which
  is exactly the shipped ``coral.vertex``/``coral.target`` count.  The demo
  therefore keeps ``Nx = 64`` and only reduces ``Tfinal``.
* the module relies on ``XbStem``/``YbStem`` coming from the ``load`` of
  ``cval.mat`` (they are not recomputed here), i.e. the stem is frozen at the
  shape ``Coral_Geometry.m`` generated the example with.
* MATLAB re-``load``s the three ``.mat`` files on *every* call; here they are
  loaded once and cached (the data are read-only, and re-reading 4.3 MB per step
  would dominate the run).  ``corals_coeff_30.mat`` is loaded exactly as in the
  MATLAB original even though the coefficients it holds (``cs1``/``cs2``) are
  never used.
"""
import os

import numpy as np
from scipy.io import loadmat

_HERE = os.path.dirname(os.path.abspath(__file__))
_CACHE = {}


def _mat(name):
    """``loadmat`` of ``name``, searched like MATLAB's ``load`` (run directory
    first, then this module's directory)."""
    if name in _CACHE:
        return _CACHE[name]
    for d in (os.getcwd(), _HERE, os.path.join(_HERE, "ib2d_input")):
        p = os.path.join(d, name)
        if os.path.exists(p):
            _CACHE[name] = loadmat(p)
            return _CACHE[name]
    raise FileNotFoundError(f"{name} not found in {os.getcwd()} or {_HERE}")


def update_Target_Point_Positions(dt, current_time, targets):
    # the five geometry constants are hard-coded in the MATLAB module
    xoffset, yoffset = 1.0, 0.5
    Lx, Nx = 2.0, 64
    dx = Lx / Nx
    ds = dx / 2

    tc = _mat("total_coeffs.mat")
    _mat("coral_coeff_30.mat")           # loaded by the MATLAB module, unused
    cv = _mat("cval.mat")
    total_meanL = float(np.ravel(tc["total_meanL"])[0])
    total_offset = float(np.ravel(tc["total_offset"])[0])
    XbStem = np.ravel(cv["XbStem"])
    YbStem = np.ravel(cv["YbStem"])

    # s = (0:ds:total_meanL)/total_meanL  (MATLAB colon semantics)
    n = int(np.floor(total_meanL / ds + 1e-9))
    s = (np.arange(n + 1) * ds) / total_meanL

    indx = int(np.ceil(current_time / dt)) + 1
    C1 = np.ravel(cv["c1vals"][indx - 1])
    C2 = np.ravel(cv["c2vals"][indx - 1])

    XbL_1 = C1[0] * s ** 3 + C1[1] * s ** 2 + C1[2] * s + C1[3]
    YbL_1 = C2[0] * s ** 3 + C2[1] * s ** 2 + C2[2] * s + C2[3]
    L_ten = np.sum(np.hypot(np.diff(XbL_1), np.diff(YbL_1)))

    XbL = XbL_1 * total_meanL / L_ten + total_offset
    YbL = YbL_1 * total_meanL / L_ten
    XbR, YbR = -XbL, YbL

    x = np.concatenate([XbR[::-1], XbStem, XbL]) + xoffset
    y = np.concatenate([YbR[::-1], YbStem, YbL]) + yoffset

    assert len(x) == targets.shape[0], \
        f"geometry ({len(x)}) and target array ({targets.shape[0]}) disagree"
    targets = np.array(targets, dtype=float, copy=True)
    targets[:, 1] = x          # MATLAB targets(:,2) = x (duplicated as in .m)
    targets[:, 2] = y          # MATLAB targets(:,3) = y
    return targets
