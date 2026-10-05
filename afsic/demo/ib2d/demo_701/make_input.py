"""Port of the example's ``Make_Tub_Geometry_and_Initial_Concentration.m``.

Upstream ships **no data files** for this example: the README of the example
directory says

    Before running main2d.m, please run
    Make_Tub_Geometry_and_Initial_Concentration.m to create the input files!

so this script reproduces what that MATLAB function writes, for the grid given
on the command line (the writer is run before both codes, so reference and AFSI
read byte-identical files)::

    python make_input.py [--nx 256] [--out ib2d_input]

It writes ``<string_name>.vertex`` and ``<string_name>.concentration``:

* ``.vertex`` -- a horizontal line of markers at ``y = 0.25``, spacing ``2 ds``
  with ``ds = 0.25 dx`` (i.e. every other node).  Note the upstream quirk: the
  script *calls* ``give_Me_Immsersed_Boundary_Geometry_and_Concentration``
  (which builds a rectangle outline) but then **overwrites** ``xLag``/``yLag``
  with that line, so the rectangle is discarded -- reproduced here.
* ``.concentration`` -- first line ``kDiff`` (from ``input2d``), then the
  ``(Ny, Nx)`` field built by ``give_Me_Initial_Concentration``: unit
  concentration everywhere except the horizontal bands
  ``y < 0.125 -> 0``, ``0.125 <= y < 0.25 -> 1`` (unchanged), ``0.25 <= y < 0.375
  -> 0``, ``y >= 0.375 -> 1`` (unchanged), i.e. zeros in the first and third
  quarter, ones elsewhere.

Because ``target_pts = 0``, ``springs = 0`` and ``beams = 0`` in this example's
``input2d`` the markers carry **no** force: they are passive points that only
advect (the ``.target``/``.spring`` files the .m also writes are not used and
not needed here).
"""
import argparse
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nx", type=int, default=None,
                    help="grid in x (default: the input2d value)")
    ap.add_argument("--out", default=os.path.join(HERE, "ib2d_input"))
    args = ap.parse_args()

    text = open(os.path.join(args.out, "input2d")).read()
    get = lambda k: float(re.search(rf"^\s*{k}\s*=\s*([^%\n]+)", text,
                                    re.M).group(1))
    name = re.search(r'^\s*string_name\s*=\s*"([^"]+)"', text, re.M).group(1)
    nx0 = int(get("Nx"))
    lx = get("Lx")
    nx = args.nx or nx0
    ny = max(int(round(nx * get("Ly") / lx)), 4)
    dx = lx / nx
    ds = 0.25 * dx

    x = np.arange(0.0, lx + 0.5 * ds, 2 * ds)
    y = 0.25 * np.ones_like(x)
    with open(os.path.join(args.out, f"{name}.vertex"), "w") as fh:
        fh.write(f"{x.size}\n")
        for xv, yv in zip(x, y):
            fh.write(f"{xv:1.16e} {yv:1.16e}\n")

    yv = (np.arange(ny) + 0.5) * (get("Ly") / ny)     # MATLAB linspace nodes
    C = np.ones((ny, nx))
    for j, yy in enumerate(yv):
        if yy < 0.125 or (0.25 <= yy < 0.375):
            C[j, :] = 0.0
    kdiff = get("kDiff")
    with open(os.path.join(args.out, f"{name}.concentration"), "w") as fh:
        fh.write(f"{kdiff:.16e}\n")
        np.savetxt(fh, C, fmt="%.16e")
    print(f"wrote {name}.vertex ({x.size} markers), {name}.concentration "
          f"({C.shape}, kDiff={kdiff:g}) for grid {nx}x{ny} "
          f"(input2d says {nx0})")


if __name__ == "__main__":
    main()
