"""Python port of the example's ``Make_Tub_Geometry_and_Initial_Concentration.m``.

The upstream example ships **no** data files -- only this MATLAB script, with
the note "Before running main2d.m, please run
Make_Tub_Geometry_and_Initial_Concentration.m to create the input files for the
simulation".  This is a line-by-line port that writes the same four files into
``ib2d_input/``:

    boussinesq.vertex       3 horizontal lines of (inert) markers
    boussinesq.target       1-based ids, k = 1e6      (target_pts = 0: unused)
    boussinesq.concentration  the (Ny, Nx) 4-band field, matIB2d layout
                              (no kDiff line -- the coefficient lives in input2d)
    boussinesq.mesh         the Eulerian indices the Boussinesq term acts on
                            (1-based; see the upstream quirk below)

Usage::

    python make_geometry.py [--nx 256] [--ny 128] [--src ib2d_input]

``--nx/--ny`` must match the ``NX/NY`` of the run (the shipped ``input2d`` asks
for 1024 x 512; the reduced default of the demo is 256 x 128).

Upstream quirk reproduced here: ``print_Boussinesq_Mesh`` guards its ``fprintf``
with ``if ( ( yInd < 0.425 ) || ( yInd > 0.025 ) )`` where ``yInd`` is an
*integer index*, so the condition is always true and the whole grid is written
(the intended floating-point band test is not applied).
"""
import argparse
import os

import numpy as np


def give_Me_Indices_To_Apply_Force(x, y, x_min, x_max, y_min, y_max):
    """MATLAB ``give_Me_Indices_To_Apply_Force`` -> (n, 2) 0-based indices."""
    iX = np.arange(int(np.argmax(x >= x_min)),
                   len(x) - int(np.argmax(x[::-1] <= x_max)), 1)
    iY = np.arange(int(np.argmax(y >= y_min)),
                   len(y) - int(np.argmax(y[::-1] <= y_max)), 1)
    ii, jj = np.meshgrid(iX, iY, indexing="ij")
    return np.column_stack([ii.ravel(), jj.ravel()]).astype(int)


def give_Me_Initial_Concentration(Lx, Nx, Ny):
    """MATLAB ``give_Me_Initial_Concentration`` -> (C, inds) (0-based inds)."""
    x_min, x_max, y_min, y_max = 0, Lx, 0, Lx / 2
    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Lx / 2, Ny)
    inds = give_Me_Indices_To_Apply_Force(x, y, x_min, x_max, y_min, y_max)
    C = np.ones((Ny, Nx))
    Amp, OffY, lam = 0.025, 0.025, 0.2
    for i, j in inds:
        xPt, yPt = x[i], y[j]
        s = Amp * np.sin(2 * np.pi * xPt / lam)
        if yPt < 0.125 + s + OffY:
            C[j, i] = 0
        elif yPt < 0.25 + s:
            C[j, i] = 1
        elif yPt < 0.375 + s - OffY:
            C[j, i] = 2
        else:
            C[j, i] = 3
    return C, inds


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--nx", type=int, default=1024)
    ap.add_argument("--ny", type=int, default=512)
    ap.add_argument("--src", default=os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "ib2d_input"))
    args = ap.parse_args()

    Nx, Ny, Lx = args.nx, args.ny, 1.0
    dx = Lx / Nx
    ds = 0.25 * dx                             # Lagrangian spacing
    name = "boussinesq"

    # --- geometry: the final override of the MATLAB script -----------------
    # xLagT = 0:2*ds:Lx ; 3 lines at y = 0.15, 0.25, 0.35 (OffY = 0)
    n = int(round(Lx / (2 * ds)))
    xLagT = np.arange(n + 1) * (2 * ds)
    xLag = np.tile(xLagT, 3)
    yLag = np.concatenate([0.15 * np.ones_like(xLagT),
                           0.25 * np.ones_like(xLagT),
                           0.35 * np.ones_like(xLagT)])

    # --- concentration + the Boussinesq Eulerian mask ----------------------
    C, inds = give_Me_Initial_Concentration(Lx, Nx, Ny)

    # --- write the four files ---------------------------------------------
    with open(os.path.join(args.src, f"{name}.vertex"), "w") as fh:
        fh.write(f"{xLag.size}\n")
        for xx, yy in zip(xLag, yLag):
            fh.write(f"{xx:1.16e} {yy:1.16e}\n")

    with open(os.path.join(args.src, f"{name}.target"), "w") as fh:
        fh.write(f"{xLag.size}\n")
        for s in range(1, xLag.size + 1):
            fh.write(f"{s} {1e6:1.16e}\n")

    with open(os.path.join(args.src, f"{name}.concentration"), "w") as fh:
        for j in range(Ny):
            fh.write("".join(f"{C[j, i]:1.16e} " for i in range(Nx)) + "\n")

    with open(os.path.join(args.src, "boussinesq.mesh"), "w") as fh:
        # upstream quirk: the guard is always true, so every index is written
        for xInd, yInd in inds:
            # the file is in MATLAB's 1-based layout, like every boussinesq.mesh
            fh.write(f"{xInd + 1} {yInd + 1}\n")

    print(f"wrote {name}.vertex ({xLag.size} pts), {name}.target, "
          f"{name}.concentration ({Ny}x{Nx}), boussinesq.mesh "
          f"({inds.shape[0]} pts) to {args.src}")


if __name__ == "__main__":
    main()
