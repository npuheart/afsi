#!/usr/bin/env python
"""Python port of the example's own MATLAB geometry generator ``Planets.m``.

    python Planets.py [--out DIR] [--struct-name NAME] [--nx N]

``Planets.m`` (matIB2d ``Examples/Example_Celestial_Mechanics/Planets``) is the
only example of this repo that ships **no** Lagrangian data files: it *creates*
``<struct_name>.vertex`` / ``.beam`` / ``.target`` on the MATLAB side before
``main2d.m`` runs.  This module reproduces it line by line so that the demo is
self-contained and both codes (AFSI and the pyIB2d reference) read exactly the
same generated files.

Faithfulness notes / upstream quirks (documented in ``readme.md``):

* the MATLAB script hard-codes ``struct_name = 'BeamCurve'`` while input2d asks
  for ``string_name = plantes`` (an unquoted typo) -- the two disagree, so the
  example cannot run as shipped.  Here ``--struct-name`` defaults to the
  ``string_name`` of the sibling ``input2d`` (``plantes``), which is what the
  driver actually looks for;
* the geometry is a shallow downward parabola ``y = -2 (x^2 - (a/2)^2)`` on
  ``x in [-a/2, a/2]``; the line that would centre it in the box
  (``xLag(1) = Lx/2 - a/2``) is *commented out* in the upstream file, so half
  of the structure lies outside ``[0, Lx]`` (both codes then fold the markers
  back periodically).  This is reproduced verbatim;
* beams: ``s = 2 .. N-1`` -> ``(s-1, s, s+1)``, ``kb = 7.5e10``, ``C = 0``;
* targets: ids ``1`` and ``N``, ``k = 2e8``.
"""
import argparse
import os
import re

import numpy as np

# upstream Planet.m constants
KB_BEAM = 7.5e10
C_BEAM = 0.0
K_TARGET = 2e8


def read_string_name(example_dir):
    """``string_name`` of ``input2d`` (the typo ``plantes`` included)."""
    with open(os.path.join(example_dir, "input2d")) as fh:
        for line in fh:
            line = line.split("%", 1)[0].strip()
            m = re.match(r"^string_name\s*=\s*(.+?)\s*$", line)
            if m:
                return m.group(1).strip().strip('"').strip("'")
    return "plantes"


def read_nx(example_dir):
    with open(os.path.join(example_dir, "input2d")) as fh:
        for line in fh:
            line = line.split("%", 1)[0].strip()
            m = re.match(r"^Nx\s*=\s*(\S+)", line)
            if m:
                return int(float(m.group(1)))
    return 32


def geometry(N, Lx, a):
    """``give_Me_Immsersed_Boundary_Geometry`` (verbatim, incl. y(1) = 0)."""
    ds = a / (N - 1)
    x = np.empty(N)
    y = np.empty(N)
    x[0] = -a / 2.0
    y[0] = 0.0
    for i in range(1, N):
        x[i] = x[i - 1] + ds
        y[i] = -2.0 * (x[i] ** 2 - (a / 2.0) ** 2)
    return x, y


def write_vertex(path, x, y):
    with open(path, "w") as fh:
        fh.write(f"{x.size}\n")
        for xi, yi in zip(x, y):
            fh.write(f"{xi:1.16e} {yi:1.16e}\n")


def write_target(path, N, k):
    with open(path, "w") as fh:
        fh.write("2\n")
        fh.write(f"1 {k:1.16e}\n")
        fh.write(f"{N} {k:1.16e}\n")


def write_beam(path, N, k, C):
    with open(path, "w") as fh:
        fh.write(f"{N - 2}\n")
        for s in range(2, N):
            fh.write(f"{s - 1} {s} {s + 1} {k:1.16e} {C:1.16e}\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--out", default=here, help="output directory")
    ap.add_argument("--struct-name", default=None,
                    help="structure name (default: input2d's string_name)")
    ap.add_argument("--nx", type=int, default=None,
                    help="Nx of input2d (default: read from input2d)")
    args = ap.parse_args()

    name = args.struct_name or read_string_name(here)
    nx = args.nx if args.nx is not None else read_nx(here)
    N = 2 * nx
    Lx = 1.0
    a = 0.5

    x, y = geometry(N, Lx, a)
    write_vertex(os.path.join(args.out, f"{name}.vertex"), x, y)
    write_target(os.path.join(args.out, f"{name}.target"), N, K_TARGET)
    write_beam(os.path.join(args.out, f"{name}.beam"), N, KB_BEAM, C_BEAM)
    print(f"wrote {name}.vertex/.target/.beam  N={N} kb={KB_BEAM:g} "
          f"k_target={K_TARGET:g}  x in [{x.min():.4f}, {x.max():.4f}] "
          f"y in [{y.min():.4f}, {y.max():.4f}]")


if __name__ == "__main__":
    main()
