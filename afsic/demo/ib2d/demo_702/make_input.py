"""Reconstruct the ``boussinesq.mesh`` file this example never ships.

**Upstream bug.**  `Example_Boussinesq/Falling_Sphere` sets ``boussinesq = 1``,
and pyIB2d's Boussinesq branch (``please_Form_Boussinesq_Forcing_Terms``) reads a
file ``boussinesq.mesh`` listing the ``(xInd, yInd)`` Eulerian nodes the buoyancy
term acts on -- but the example directory contains **no** ``boussinesq.mesh``,
and its own generator ``Make_Tub_Geometry_and_Initial_Concentration.m`` never
writes one (it writes ``.vertex``, ``.mass``, ``.spring``, ``.target``,
``.beam`` and ``.concentration`` only).  Running the example as shipped
therefore dies with ``FileNotFoundError: boussinesq.mesh`` in *both* codes.

This script writes the mask the generator *implies*: its
``give_Me_Indices_To_Apply_Force`` is called over the whole domain
(``xMin = 0, xMax = Lx``, ``yMin = 0, yMax = Lx``), i.e. every node is listed
(1-based indices, as the MATLAB file reader expects)::

    python make_input.py [--nx 128] [--out ib2d_input]

Run it before both codes so the reference and the AFSI run read byte-identical
input (the grid must match the ``NX`` the runs use).
"""
import argparse
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nx", type=int, default=None)
    ap.add_argument("--out", default=os.path.join(HERE, "ib2d_input"))
    args = ap.parse_args()

    text = open(os.path.join(args.out, "input2d")).read()
    get = lambda k: float(re.search(rf"^\s*{k}\s*=\s*([^%\n]+)", text,
                                    re.M).group(1))
    nx0 = int(get("Nx"))
    nx = args.nx or nx0
    ny = max(int(round(nx * get("Ly") / get("Lx"))), 4)
    # 0-based: the MATLAB reader is 1-based, but pyIB2d's port indexes
    # ``mat[yInd, xInd]`` unchanged, so a 1-based full-domain mask would hit
    # ``index Nx`` and raise IndexError (yet another upstream quirk).  The
    # shipped demos' masks (e.g. demo_577) are offset the same way, i.e. the
    # mask is the full domain up to a one-node shift.
    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    # the reader is np.loadtxt(..., unpack=True) -> one (xInd, yInd) pair per
    # line
    pairs = np.column_stack([ii.ravel(), jj.ravel()])
    with open(os.path.join(args.out, "boussinesq.mesh"), "w") as fh:
        np.savetxt(fh, pairs, fmt="%d %d")
    print(f"wrote boussinesq.mesh: {pairs.shape[0]} nodes (all of "
          f"{nx}x{ny}; input2d says {nx0})")


if __name__ == "__main__":
    main()
