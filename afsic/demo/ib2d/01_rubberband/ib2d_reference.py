"""Convert IB2d (MATLAB/Octave) VTK output of the Rubberband example to a .npz file.

Usage
-----
    python ib2d_reference.py <path/to/viz_IB2d> [out.npz]

Reads ``lagsPts.XXXX.vtk`` (Lagrangian points), ``u.XXXX.vtk`` (velocity on the
collocated Nx x Ny grid) and ``P.XXXX.vtk`` (pressure).  The arrays stored in the
npz file are

    t      (nt,)            dump times
    X      (nt, Nb, 2)      Lagrangian point positions
    u      (nt, Ny, Nx, 2)  Eulerian velocity, u[k, j, i] at (x_i, y_j) = (i dx, j dy)
    p      (nt, Ny, Nx)     Eulerian pressure (IB2d gauge: zero mean)
    dx, dy                  grid spacing
"""
import glob
import os
import re
import sys

import numpy as np


def _read_header(lines):
    info = {}
    for k, ln in enumerate(lines):
        s = ln.split()
        if not s:
            continue
        if s[0] == "TIME":
            info["t"] = float(lines[k + 1].split()[0])
        elif s[0] == "DIMENSIONS":
            info["dims"] = tuple(int(v) for v in s[1:4])
        elif s[0] == "SPACING":
            info["spacing"] = tuple(float(v) for v in s[1:4])
        elif s[0] == "POINTS":
            info["npts"] = int(s[1])
            info["data_start"] = k + 1
        elif s[0] in ("VECTORS", "SCALARS"):
            info["data_start"] = k + 1 if s[0] == "VECTORS" else k + 2  # skip LOOKUP_TABLE
            break
    return info


def read_lag_points(fname):
    with open(fname) as fh:
        lines = fh.readlines()
    info = _read_header(lines)
    n = info["npts"]
    vals = np.array(" ".join(lines[info["data_start"]:]).split()[: 3 * n], dtype=float)
    return info["t"], vals.reshape(n, 3)[:, :2]


def read_structured(fname, ncomp):
    with open(fname) as fh:
        lines = fh.readlines()
    info = _read_header(lines)
    nx, ny, _ = info["dims"]
    vals = np.array(" ".join(lines[info["data_start"]:]).split()[: ncomp * nx * ny], dtype=float)
    arr = vals.reshape(ny, nx, ncomp)  # VTK structured points: x fastest
    return info["t"], arr, info["spacing"]


def main(viz_dir, out):
    def idx(f):
        return int(re.findall(r"\.(\d+)\.vtk$", f)[0])

    lag_files = sorted(glob.glob(os.path.join(viz_dir, "lagsPts.*.vtk")), key=idx)
    t, X, U, P = [], [], [], []
    spacing = None
    for f in lag_files:
        k = idx(f)
        tk, xk = read_lag_points(f)
        t.append(tk)
        X.append(xk)
        uf = os.path.join(viz_dir, f"u.{k:04d}.vtk")
        pf = os.path.join(viz_dir, f"P.{k:04d}.vtk")
        if os.path.exists(uf):
            _, uk, spacing = read_structured(uf, 3)
            U.append(uk[..., :2])
        if os.path.exists(pf):
            _, pk, _ = read_structured(pf, 1)
            P.append(pk[..., 0])
    data = dict(t=np.array(t), X=np.array(X))
    if U:
        data["u"] = np.array(U)
    if P:
        data["p"] = np.array(P)
    if spacing is not None:
        data["dx"], data["dy"] = spacing[0], spacing[1]
    np.savez_compressed(out, **data)
    print(f"wrote {out}: {len(t)} frames, t in [{t[0]:.4f}, {t[-1]:.4f}], Nb={X[0].shape[0]}")


if __name__ == "__main__":
    viz = sys.argv[1] if len(sys.argv) > 1 else "ib2d_run/viz_IB2d"
    out = sys.argv[2] if len(sys.argv) > 2 else "ib2d_reference.npz"
    main(viz, out)
