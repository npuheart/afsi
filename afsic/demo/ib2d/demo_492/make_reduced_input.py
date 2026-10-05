"""Build a reduced-grid copy of this example's input for memory-safe runs.

The shipped example is ``1024 x 128`` (``dt = 5e-5``, ``T = 1``) with a
``(Ny, Nx)`` concentration field of the same size; running it as shipped needs
several GB.  This script writes ``reduced_input/`` next to ``ib2d_input/`` with

* ``input2d`` unchanged except ``Nx``/``Ny`` (same aspect ratio) and, optionally,
  ``Tfinal``/``print_dump``,
* every ``<name>.*`` data file copied verbatim, except ``<name>.concentration``
  which is subsampled by an integer stride so that its shape matches the
  reduced grid (the initial blob is smooth over tens of nodes, so the stride
  only shifts the initial condition by half a node at most).

Both codes are then run against that directory::

    python make_reduced_input.py --nx 128
    python run_reference.py --src reduced_input --tend 0.01 --print-dump 50
    IB2D_EXAMPLE=reduced_input TFINAL=0.01 PRINT_DUMP=50 python main.py
    python compare.py

``reduced_input/`` is git-ignored: ``ib2d_input/`` keeps the upstream files
verbatim.
"""
import argparse
import os
import re
import shutil

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "ib2d_input")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nx", type=int, required=True)
    ap.add_argument("--tend", type=float, default=None)
    ap.add_argument("--print-dump", type=int, default=None)
    ap.add_argument("--out", default=os.path.join(HERE, "reduced_input"))
    args = ap.parse_args()

    text = open(os.path.join(SRC, "input2d")).read()
    get = lambda k: float(re.search(rf"^\s*{k}\s*=\s*([^%\n]+)", text,
                                    re.M).group(1))
    nx0, ny0 = int(get("Nx")), int(get("Ny"))
    lx, ly = get("Lx"), get("Ly")
    nx = args.nx
    ny = max(int(round(nx * ly / lx)), 8)
    assert nx0 % nx == 0 and ny0 % ny == 0, \
        f"reduced grid {nx}x{ny} is not an integer subdivision of {nx0}x{ny0}"
    sx, sy = nx0 // nx, ny0 // ny

    if os.path.exists(args.out):
        shutil.rmtree(args.out)
    os.makedirs(args.out)
    for fn in sorted(os.listdir(SRC)):
        if fn.endswith((".m", ".py")) or not os.path.isfile(os.path.join(SRC, fn)):
            continue
        shutil.copy(os.path.join(SRC, fn), os.path.join(args.out, fn))

    new = re.sub(r"^(\s*Nx\s*=)\s*[^%\n]+", rf"\g<1> {nx}  ", text, flags=re.M)
    new = re.sub(r"^(\s*Ny\s*=)\s*[^%\n]+", rf"\g<1> {ny}  ", new, flags=re.M)
    if args.tend is not None:
        new = re.sub(r"^(\s*Tfinal\s*=)\s*[^%\n]+", rf"\g<1> {args.tend}  ",
                     new, flags=re.M)
    if args.print_dump is not None:
        new = re.sub(r"^(\s*print_dump\s*=)\s*[^%\n]+",
                     rf"\g<1> {args.print_dump}  ", new, flags=re.M)
    open(os.path.join(args.out, "input2d"), "w").write(new)

    name = re.search(r'^\s*string_name\s*=\s*"([^"]+)"', text, re.M).group(1)
    cfile = os.path.join(args.out, f"{name}.concentration")
    if os.path.exists(cfile):
        with open(cfile) as fh:
            k = fh.readline()
        C = np.loadtxt(cfile, skiprows=1)
        assert C.shape == (ny0, nx0), f"unexpected concentration {C.shape}"
        with open(cfile, "w") as fh:
            fh.write(k)
            np.savetxt(fh, C[::sy, ::sx], fmt="%.16e")
        print(f"concentration subsampled {C.shape} -> {C[::sy, ::sx].shape} "
              f"(stride {sy} x {sx})")
    print(f"wrote {args.out}: grid {nx}x{ny} (from {nx0}x{ny0}), "
          f"Tfinal={args.tend}, print_dump={args.print_dump}")


if __name__ == "__main__":
    main()
