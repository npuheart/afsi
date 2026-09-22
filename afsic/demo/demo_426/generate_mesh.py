"""Generate the Lagrangian solid mesh for demo_426 — the two channel plates.

The benchmark discretises each plate with Lagrangian marker points.  Following
demo_424 (and AFSI's immersed-boundary distributor, which takes an integration
weight per Lagrangian point), the plates are built as genuine 2-D TRIANGULAR
STRIPS of finite thickness rather than 1-D line segments:

  * a 1-D line has zero area, so the tether assembled over it would be a force
    per unit LENGTH, while the immersed force entering a 2-D fluid is a force
    per unit AREA -- the two differ by a factor of the plate thickness;
  * with 2-D cells the weak-form assembly `Int beta*(X-Xref).v dx_s` carries the
    correct area measure automatically, which is exactly how demo_424 works.

Run:
    python generate_mesh.py
"""
import numpy as np
from ufl import Measure
from mpi4py import MPI
import dolfinx
from basix.ufl import element
from dolfinx.io import XDMFFile
from dolfinx.mesh import CellType
from dolfinx.fem import form, assemble_scalar, Constant

import configuration as cfg


def build_plate(side):
    """Points and triangles for one plate strip.

    Parameterised by (s, n): s runs along the plate, n across its thickness,
    so the strip follows the channel exactly and is `PLATE_THICKNESS` wide.
    """
    (x0, y0), (x1, y1) = cfg.wall_endpoints(side)
    ax, ay = np.array([x1 - x0, y1 - y0]), None
    length = float(np.hypot(x1 - x0, y1 - y0))
    tx, ty = (x1 - x0) / length, (y1 - y0) / length      # along the plate
    # inward normal: points from the plate towards the channel centreline
    nx, ny = -ty * side, tx * side

    n_s = max(int(np.ceil(length / cfg.DS_LAG)), 1)
    n_t = max(int(np.round(cfg.PLATE_THICKNESS / cfg.DS_LAG)), 1)

    pts = []
    for j in range(n_t + 1):
        off = j * cfg.PLATE_THICKNESS / n_t
        for i in range(n_s + 1):
            s = i * length / n_s
            pts.append((x0 + s * tx + off * nx, y0 + s * ty + off * ny))

    tris = []
    stride = n_s + 1
    for j in range(n_t):
        for i in range(n_s):
            a = j * stride + i
            b = a + 1
            c = a + stride
            d = c + 1
            tris.append((a, b, d))
            tris.append((a, d, c))
    return np.asarray(pts, dtype=np.float64), np.asarray(tris, dtype=np.int64)


def main():
    pts_all, tris_all = [], []
    for side in (-1, +1):
        p, t = build_plate(side)
        offset = sum(len(x) for x in pts_all)
        pts_all.append(p)
        tris_all.append(t + offset)
    points = np.vstack(pts_all)
    cells = np.vstack(tris_all)

    structure = dolfinx.mesh.create_mesh(
        MPI.COMM_WORLD, cells,
        element("Lagrange", "triangle", 1, shape=(2,)), points)

    # ---- regression guard: total plate area must match the exact value ----
    dxx = Measure("dx", domain=structure)
    area = assemble_scalar(form(Constant(structure, 1.0) * dxx))
    exact = 2.0 * cfg.PLATE_AREA_TARGET
    rel = (area - exact) / exact
    assert abs(rel) < 1e-9, f"plate area mismatch: {area} vs {exact}"

    with XDMFFile(MPI.COMM_WORLD, cfg.solid_mesh_path(), "w") as xdmf:
        xdmf.write_mesh(structure)

    if MPI.COMM_WORLD.rank == 0:
        print(f"solid mesh: {structure.topology.index_map(2).size_local} "
              f"triangles, {structure.topology.index_map(0).size_local} nodes")
        print(f"  plate length {cfg.PLATE_LENGTH:.6g}, thickness "
              f"{cfg.PLATE_THICKNESS:.6g}, ds_lag {cfg.DS_LAG:.6g}")
        print(f"  area = {area:.10e} m^2 (exact {exact:.10e}, rel {rel:+.2e})")
        print(f"  written to {cfg.solid_mesh_path()}")


if __name__ == "__main__":
    main()
