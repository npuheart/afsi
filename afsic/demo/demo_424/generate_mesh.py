"""Generate the Lagrangian solid mesh for demo_424.

CASE=open    two flat wall strips (the 2-D cut of the aortic wall)
CASE=closed  the same strips plus a membrane that occludes the lumen at
             mid-length; the union reads as the letter "H"

The mesh is a Cartesian-product grid whose cells are selected by whether they
fall inside the solid, so the result is conforming and (for the closed case)
connected -- the membrane cells share edges with the wall cells.

Run:
    python generate_mesh.py            # CASE from configuration.py
    NY=45 CASE=closed python generate_mesh.py
"""
import numpy as np

import configuration as cfg
import dolfinx
from mpi4py import MPI
from ufl import Measure
from basix.ufl import element
from dolfinx.fem import form, assemble_scalar, Constant
from dolfinx.io import XDMFFile
from dolfinx.mesh import CellType


def build_nodes():
    """x and y node coordinates, with all material interfaces as grid lines."""
    hs = cfg.HS

    # ---- y: [outer wall | inner wall | lumen | inner wall | outer wall] ----
    n_wy = max(int(round(cfg.T_WALL / hs)), 1)          # cells across the wall
    n_ly = max(int(round(2.0 * cfg.A_LUMEN / hs)), 1)   # cells across the lumen

    y_lo = np.linspace(cfg.Y_OUT_LO, cfg.Y_IN_LO, n_wy + 1)
    y_mid = np.linspace(cfg.Y_IN_LO, cfg.Y_IN_HI, n_ly + 1)
    y_hi = np.linspace(cfg.Y_IN_HI, cfg.Y_OUT_HI, n_wy + 1)
    Y = np.concatenate([y_lo, y_mid[1:], y_hi[1:]])

    # ---- x: [upstream | membrane | downstream] ----
    x_start = cfg.X_OFF
    x_end = cfg.X_OFF + cfg.L_AORTA
    x_disc_c = cfg.X_OFF + cfg.DISC_X

    n1 = max(int(round((cfg.DISC_X - 0.5 * cfg.DISC_T) / hs)), 1)
    n_d = max(int(round(cfg.DISC_T / hs)), 1)
    n2 = max(int(round((cfg.L_AORTA - cfg.DISC_X - 0.5 * cfg.DISC_T) / hs)), 1)

    x1 = np.linspace(x_start, x_disc_c - 0.5 * cfg.DISC_T, n1 + 1)
    x2 = np.linspace(x_disc_c - 0.5 * cfg.DISC_T, x_disc_c + 0.5 * cfg.DISC_T, n_d + 1)
    x3 = np.linspace(x_disc_c + 0.5 * cfg.DISC_T, x_end, n2 + 1)
    X = np.concatenate([x1, x2[1:], x3[1:]])

    return X, Y, (n_wy, n_ly), (n1, n_d, n2)


def build_cells(X, Y, ny_split, nx_split):
    """Select the (i, j) cells that belong to the solid."""
    n_wy, n_ly = ny_split
    n1, n_d, n2 = nx_split
    n_x, n_y = len(X) - 1, len(Y) - 1

    wall_rows = list(range(0, n_wy)) + list(range(n_wy + n_ly, n_y))
    lumen_rows = list(range(n_wy, n_wy + n_ly))
    disc_cols = list(range(n1, n1 + n_d))

    cells_vtk = []
    for j in wall_rows:
        for i in range(n_x):
            cells_vtk.append((i, j))
    if cfg.CASE == "closed":
        for j in lumen_rows:
            for i in disc_cols:
                cells_vtk.append((i, j))

    cells_vtk = np.asarray(cells_vtk, dtype=np.int64)
    # vertex (i, j) lives at index j*(n_x+1) + i in the raveled (n_y+1, n_x+1) grid
    stride = n_x + 1
    il = cells_vtk[:, 1] * stride + cells_vtk[:, 0]
    ir = il + 1
    ol = il + stride
    oo = ol + 1
    # VTK cyclic order for a physical CCW quad: (i,j) (i+1,j) (i+1,j+1) (i,j+1)
    vtk = np.column_stack([il, ir, oo, ol])

    # DOLFINx quadrilateral ordering (see demo_423 for the same guard)
    perm = np.asarray(
        dolfinx.cpp.io.perm_vtk(CellType.quadrilateral, 4), dtype=np.int64)
    return vtk[:, perm], n_x, n_y


def main():
    X, Y, ny_split, nx_split = build_nodes()
    cells, n_x, n_y = build_cells(X, Y, ny_split, nx_split)

    # vertex numbering on the full Cartesian product grid:
    # index (i, j) -> j*(n_x+1) + i  (meshgrid default 'xy' gives shape (n_y+1, n_x+1))
    xx, yy = np.meshgrid(X, Y)
    points = np.column_stack([xx.ravel(), yy.ravel()])

    coord_element = element("Lagrange", "quadrilateral", 1, shape=(2,))
    structure = dolfinx.mesh.create_mesh(MPI.COMM_WORLD, cells, coord_element,
                                         points)

    # ---- regression guard: the solid area must match the exact area ----
    dxx = Measure("dx", domain=structure)
    area = assemble_scalar(form(Constant(structure, 1.0) * dxx))
    exact = 2.0 * cfg.L_AORTA * cfg.T_WALL
    if cfg.CASE == "closed":
        exact += 2.0 * cfg.A_LUMEN * cfg.DISC_T
    assert abs(area - exact) < 1e-6 * exact, \
        f"solid area mismatch: {area} vs {exact}"

    out = cfg.solid_mesh_path()
    with XDMFFile(MPI.COMM_WORLD, out, "w") as xdmf:
        xdmf.write_mesh(structure)

    if MPI.COMM_WORLD.rank == 0:
        print(f"solid mesh [{cfg.CASE}]: {n_x} x {n_y} grid, "
              f"{structure.topology.index_map(2).size_local} cells, "
              f"h_s={cfg.HS:g} m")
        print(f"  area = {area:.10e} m^2 (exact {exact:.10e})")
        print(f"  x in [{X[0]:.6g}, {X[-1]:.6g}], "
              f"y in [{Y[0]:.6g}, {Y[-1]:.6g}]")
        print(f"  written to {out}")


if __name__ == "__main__":
    main()
