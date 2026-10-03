"""Python port of IB2d's ``Rubberband.m`` + ``input2d`` writer (for refinement studies).

    python make_rubberband.py <out_dir> [Nx] [dt] [Tfinal] [print_dump]

Writes ``input2d``, ``rubberband.vertex``, ``rubberband.spring`` for an IB2d run
on an Nx x Nx grid with N = 2 Nx Lagrangian points (as ``Rubberband.m``).

IB2d multiplies the spring forces by ds = Lx/(2 Nx) and the spring length
scales like 1/Nx, so the effective band tension is ~ k / Nx^2.  To keep the
*same physical problem* under refinement the stiffness is scaled as
k = 2.5e4 (Nx/32)^2 (k = 2.5e4 at the original Nx = 32).
"""
import os
import sys

import numpy as np


def main(out, Nx=32, dt=1e-3, Tfinal=1.5, print_dump=20):
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.abspath(os.path.join(here, *[".."] * 4))
    src = os.path.join(repo, "third_party", "ib2d", "matIB2d", "Examples",
                       "Example_Standard_Rubberband", "Rubberband_with_Springs", "input2d")
    os.makedirs(out, exist_ok=True)
    N = 2 * Nx
    a, b = 0.4, 0.2                       # as in Rubberband.m: x-radius b, y-radius a
    i = np.arange(N)
    x = 0.5 + b * np.cos(2 * np.pi / N * i)
    y = 0.5 + a * np.sin(2 * np.pi / N * i)
    k = 2.5e4 * (Nx / 32) ** 2
    with open(os.path.join(out, "rubberband.vertex"), "w") as fh:
        fh.write(f"{N}\n")
        for xv, yv in zip(x, y):
            fh.write(f"{xv:1.16e} {yv:1.16e}\n")
    with open(os.path.join(out, "rubberband.spring"), "w") as fh:
        fh.write(f"{N}\n")
        for s in range(1, N + 1):
            fh.write(f"{s} {s % N + 1} {k:1.16e} {0.0:1.16e}\n")
    txt = open(src).read()
    rep = {"Nx": Nx, "Ny": Nx, "dt": dt, "Tfinal": Tfinal, "print_dump": print_dump, "plot_Matlab": 0}
    lines = []
    for ln in txt.splitlines():
        key = ln.split("=")[0].strip() if "=" in ln else None
        if key in rep:
            comment = ln[ln.index("%"):] if "%" in ln else ""
            ln = f"{key} = {rep[key]}    {comment}"
        lines.append(ln)
    open(os.path.join(out, "input2d"), "w").write("\n".join(lines) + "\n")
    print(f"wrote {out}: Nx={Nx}, N={N}, k={k:g}, dt={dt}")


if __name__ == "__main__":
    args = sys.argv[1:]
    main(args[0], *(t(v) for t, v in zip((int, float, float, int), args[1:])))
