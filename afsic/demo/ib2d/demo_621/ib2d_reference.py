"""Convert pyIB2d VTK output (``viz_IB2d/``) to ``ib2d_reference.npz``.

    python ib2d_reference.py <viz_dir> <out.npz> [dt] [print_dump] [nx] [ny]

Reads ``lagsPts.XXXX.vtk`` (Lagrangian points), ``u.XXXX.vtk`` (velocity) and
``P.XXXX.vtk`` (pressure) written by pyIB2d's VTK writer and stores

    t      (nt,)            dump times, t_k = k * dt * print_dump
    X      (nt, Nb, 2)      Lagrangian point positions
    u      (nt, Ny, Nx, 2)  Eulerian velocity, u[k, j, i] at (x_i, y_j)
    p      (nt, Ny, Nx)     Eulerian pressure
    dx, dy                  grid spacing

i.e. the same field layout as the other ib2d demos.  pyIB2d's VTK files carry
no time stamp, so the dump times are reconstructed from ``dt`` and
``print_dump`` (the driver dumps at step counters 0, print_dump,
2 * print_dump, ...).

Storage conventions of pyIB2d's writer (checked against demo_445, whose fields
were validated against the MATLAB reference):

* ``u.XXXX.vtk`` and the scalar fields carry no extra metadata beyond
  ``DIMENSIONS``; two layouts occur (verified empirically against the marker
  velocities and the momentum balance):

  - *loop* dumps (index >= 1): ``DIMENSIONS (Nx, Ny)`` and the Fortran-order
    values are the field stored as ``[x, y]`` -> reshape to ``(Nx, Ny)`` and
    transpose to the physical ``(Ny, Nx)`` layout;
  - the *initial* dump (index 0): ``DIMENSIONS (Ny, Nx)``, already physical
    (u = 0 at t = 0, so the distinction is immaterial).

  The rule used below: ``reshape(dims, order='F')``, transpose when the shape
  comes out as ``(Nx, Ny)``.  (For a square grid this always transposes,
  which matches all empirical checks.)
"""
import glob
import os
import re
import sys

import numpy as np


def _index(fname):
    return int(re.findall(r"\.(\d+)\.vtk$", fname)[0])


def _header(lines, first_data_token):
    """Return (dims, spacing, data_start) of a structured VTK file."""
    dims = spacing = start = None
    for k, ln in enumerate(lines):
        s = ln.split()
        if not s:
            continue
        if s[0] == "DIMENSIONS":
            dims = tuple(int(v) for v in s[1:4])
        elif s[0] == "SPACING":
            spacing = tuple(float(v) for v in s[1:4])
        elif s[0] == first_data_token:
            start = k + 1 if s[0] == "VECTORS" else k + 2  # skip LOOKUP_TABLE
            break
    assert dims is not None and start is not None, "unexpected VTK layout"
    return dims, spacing, start


def read_lag_points(fname):
    """``lagsPts.XXXX.vtk`` -> (Nb, 2) positions."""
    with open(fname) as fh:
        lines = fh.readlines()
    n = start = None
    for k, ln in enumerate(lines):
        s = ln.split()
        if s and s[0] == "POINTS":
            n, start = int(s[1]), k + 1
            break
    assert n is not None, f"no POINTS found in {fname}"
    vals = np.array(" ".join(lines[start:]).split()[: 3 * n], dtype=float)
    return vals.reshape(n, 3)[:, :2]


def _read_flat(fname, nvals, first_token):
    with open(fname) as fh:
        lines = fh.readlines()
    dims, spacing, start = _header(lines, first_token)
    vals = np.array(" ".join(lines[start:]).split()[:nvals], dtype=float)
    return dims, spacing, vals


def _to_physical(vals, dims, ny, nx):
    """Flat VTK data -> physical (Ny, Nx) array."""
    arr = vals.reshape(dims[0], dims[1], order="F")
    if dims[0] == nx and dims[1] == ny:
        return arr.T
    assert dims[0] == ny and dims[1] == nx, \
        f"unexpected field dims {dims} (grid {ny}x{nx})"
    return arr


def read_velocity(fname, ny, nx):
    """``u.XXXX.vtk`` -> ((Ny, Nx, 2), spacing)."""
    dims, spacing, vals = _read_flat(fname, 3 * ny * nx, "VECTORS")
    pts = vals.reshape(ny * nx, 3)[:, :2]
    u = np.empty((ny, nx, 2))
    for c in range(2):
        u[..., c] = _to_physical(pts[:, c].copy(), dims, ny, nx)
    return u, spacing


def read_scalar(fname, ny, nx):
    """``P.XXXX.vtk`` -> ((Ny, Nx), spacing)."""
    dims, spacing, vals = _read_flat(fname, ny * nx, "SCALARS")
    return _to_physical(vals.copy(), dims, ny, nx), spacing


def main(viz_dir, out, dt=None, print_dump=None, nx=None, ny=None):
    files = sorted(glob.glob(os.path.join(viz_dir, "lagsPts.*.vtk")), key=_index)
    assert files, f"no lagsPts.*.vtk under {viz_dir}"

    u0 = os.path.join(viz_dir, f"u.{_index(files[0]):04d}.vtk")
    have_u = os.path.exists(u0)
    if have_u and (nx is None or ny is None):
        with open(u0) as fh:
            dims, spacing, _ = _header(fh.readlines(), "VECTORS")
        ny, nx = max(dims[:2]), min(dims[:2])
        assert ny > nx, "square grid: pass nx/ny explicitly"

    t, X, U, P = [], [], [], []
    for f in files:
        k = _index(f)
        X.append(read_lag_points(f))
        if have_u:
            uk, spacing = read_velocity(os.path.join(viz_dir, f"u.{k:04d}.vtk"),
                                        ny, nx)
            pf = os.path.join(viz_dir, f"P.{k:04d}.vtk")
            U.append(uk)
            P.append(read_scalar(pf, ny, nx)[0] if os.path.exists(pf) else None)
        t.append(k * dt * print_dump if (dt is not None and print_dump is not None)
                 else float(k))

    data = dict(t=np.array(t), X=np.array(X))
    if have_u:
        data["u"] = np.array(U)
        if P and P[0] is not None:
            data["p"] = np.array(P)
        data["dx"], data["dy"] = (float(spacing[0]), float(spacing[1]))
    np.savez_compressed(out, **data)
    print(f"wrote {out}: {len(t)} frames, t in [{t[0]:.5f}, {t[-1]:.5f}], "
          f"Nb={X[0].shape[0]}"
          + (f", grid {ny}x{nx}" if have_u else ""))


if __name__ == "__main__":
    viz = sys.argv[1] if len(sys.argv) > 1 else "ib2d_run/viz_IB2d"
    out = sys.argv[2] if len(sys.argv) > 2 else "ib2d_reference.npz"
    dt = float(sys.argv[3]) if len(sys.argv) > 3 else None
    print_dump = float(sys.argv[4]) if len(sys.argv) > 4 else None
    nx = int(sys.argv[5]) if len(sys.argv) > 5 else None
    ny = int(sys.argv[6]) if len(sys.argv) > 6 else None
    main(viz, out, dt, print_dump, nx, ny)
