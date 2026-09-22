"""Generate the Lagrangian solid mesh for demo_425 — 3-D round aorta.

CASE=open    the cylindrical wall alone (a tube: r = a .. a + t)
CASE=closed  the tube plus a coaxial occluding disc of radius a and
             thickness DISC_T at mid-length

Cross-section through the axis
------------------------------
The solid is built as a set of axis-aligned-looking **cylindrical ring blocks**
in (r, theta), stacked along the pipe axis x.  Cutting the closed-case solid by
a plane through the axis gives exactly demo_424's "H": two wall strips joined by
a radial disc.

    424 (2-D)                    425 (3-D)
    ------------------------     --------------------------------------
    y: wall | lumen | wall       r: [0, a) inner core | [a, a+t) wall
    flat strips                  cylindrical rings

How the mesh is built
---------------------
A structured node grid ``(i_a, i_r, i_t)`` is laid out, where

    i_a  axial index (x),  i_r  radial layer index,  i_t  angular index

Every solid cell is a hexahedron spanning one step in each index.  The four
cells around the axis have all four of their r = 0 corners at the same point,
so they are degenerate hexahedra; they are dropped and the ones next to them
become wedges (zero-volume tets are filtered out automatically).

DOLFINx has no built-in hexahedron-to-tetrahedron conversion, so each
hexahedron is split into 6 tetrahedra using the standard "main diagonal 0-6"
rule applied to the VTK-ordered vertices.  That rule is *globally conforming on
a structured grid*: both hexahedra sharing a face use the same face diagonal,
so no hanging nodes appear.

Run:
    python generate_mesh.py
    CASE=open NY=45 python generate_mesh.py
"""
import math

import numpy as np

import configuration as cfg
import dolfinx
from mpi4py import MPI
from ufl import Measure
from basix.ufl import element
from dolfinx.fem import form, assemble_scalar, Constant
from dolfinx.io import XDMFFile
from dolfinx.mesh import CellType

# The 6 tetrahedra of a cube, using the main diagonal 0-6.  Vertex indices are
# local to the (VTK-ordered) hexahedron: the 6 vertices other than 0 and 6 form
# a hexagonal ring around that diagonal, and each tet is the diagonal plus one
# edge of the ring.  The ring order below is the one that makes all six tets
# positively oriented and exactly fills the cube; _check_cube_rule() enforces
# that at import time (an earlier hand-written table was silently degenerate).
CUBE_RING = (1, 5, 4, 7, 3, 2)
CUBE_TETS = np.array([[0, CUBE_RING[a], CUBE_RING[(a + 1) % 6], 6]
                      for a in range(6)], dtype=np.int64)


def _check_cube_rule():
    """Fail loudly if CUBE_TETS is not an exact, positively oriented tiling."""
    cube = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 1.0],
                     [0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0],
                     [1.0, 1.0, 1.0], [0.0, 1.0, 1.0]])
    p = cube[np.asarray(CUBE_TETS, dtype=np.intp)]
    vols = np.linalg.det(np.stack([p[:, 1] - p[:, 0], p[:, 2] - p[:, 0],
                                   p[:, 3] - p[:, 0]], axis=-1)) / 6.0
    if not np.all(vols > 0.0) or abs(vols.sum() - 1.0) > 1e-12:
        raise AssertionError(
            f"CUBE_TETS is not a valid tiling: vols={vols}, sum={vols.sum()}")


_check_cube_rule()


def build_radial_layers():
    """Radial node radii, the disc-surface layer index and the wall layers.

    Thin wrapper over ``configuration.radial_nodes`` so that this file and the
    volume guard in configuration.py can never describe different geometries.
    """
    radii, i_disc, n_wall = cfg.radial_nodes()
    if np.any(np.diff(radii) <= 0.0):
        raise ValueError("radial layers are not strictly increasing")
    return radii, (i_disc if cfg.CASE == "closed" else None), n_wall


def build_axial_layers():
    """Axial node coordinates and the node-index range occupied by the disc."""
    return cfg.axial_nodes()


class NodeGrid:
    """Structured (i_a, i_r, i_t) node numbering with axis sharing."""

    def __init__(self, X, radii, n_theta):
        self.X = X
        self.radii = radii
        self.n_theta = n_theta
        self.n_a = len(X)
        self.n_r = len(radii)
        self.axis_node = np.full(self.n_a, -1, dtype=np.int64)
        self.ring_node = np.full((self.n_a, self.n_r), -1, dtype=np.int64)

        coords = []
        for i_a in range(self.n_a):
            # axis nodes: the degenerate point at r = 0
            self.axis_node[i_a] = len(coords)
            coords.append((self.X[i_a], cfg.Y_C, cfg.Z_C))
            self.ring_node[i_a, 0] = self.axis_node[i_a]
            for i_r in range(1, self.n_r):
                r = self.radii[i_r]
                # first node of this ring; the rest follow consecutively
                self.ring_node[i_a, i_r] = len(coords)
                for i_t in range(n_theta):
                    th = 2.0 * math.pi * i_t / n_theta
                    coords.append((self.X[i_a],
                                   cfg.Y_C + r * math.cos(th),
                                   cfg.Z_C + r * math.sin(th)))
        self.points = np.asarray(coords, dtype=np.float64)
        self._ring_index = None

    # -- node lookup ------------------------------------------------------
    def _ring_indices(self):
        """(n_a, n_r, n_theta) -> node id, with the wrap-around entry
        collapsed onto the ring's first node (a full circle needs n_theta, not
        n_theta+1, distinct nodes)."""
        if self._ring_index is None:
            idx = np.full((self.n_a, self.n_r, self.n_theta + 1),
                          -1, dtype=np.int64)
            for i_a in range(self.n_a):
                for i_r in range(1, self.n_r):
                    base = self.ring_node[i_a, i_r]
                    for i_t in range(self.n_theta):
                        idx[i_a, i_r, i_t] = base + i_t
                    idx[i_a, i_r, self.n_theta] = base  # wrap
            self._ring_index = idx
        return self._ring_index

    def node(self, i_a, i_r, i_t):
        """Global node id.

        ``i_t`` may be ``n_theta`` (or more) to express the wrap-around; for
        ``i_r == 0`` every angle shares the single axis node.
        """
        if i_r == 0:
            return self.axis_node[i_a]
        # i_t = n_theta means the wrap-around, handled by the double modulo
        return self._ring_indices()[i_a, i_r,
                                    i_t % (self.n_theta + 1) % self.n_theta]

    # -- cells ------------------------------------------------------------
    def add_ring_block(self, cells, a0, a1, r0, r1):
        """Append hexahedra for the index box [a0,a1) x [r0,r1) x [0, n_theta)."""
        for i_a in range(a0, a1):
            for i_r in range(r0, r1):
                for i_t in range(self.n_theta):
                    c = [self.node(i_a, i_r, i_t),
                         self.node(i_a + 1, i_r, i_t),
                         self.node(i_a + 1, i_r, i_t + 1),
                         self.node(i_a, i_r, i_t + 1),
                         self.node(i_a, i_r + 1, i_t),
                         self.node(i_a + 1, i_r + 1, i_t),
                         self.node(i_a + 1, i_r + 1, i_t + 1),
                         self.node(i_a, i_r + 1, i_t + 1)]
                    cells.append(c)


def hexes_to_tets(hexes):
    """Split each hexahedron (VTK order) into 6 tets, VTK order."""
    hexes = np.asarray(hexes, dtype=np.int64)
    tets = hexes[:, CUBE_TETS].reshape(-1, 4)
    return tets


def drop_degenerate(cells, points, tol=0.0):
    """Remove cells with repeated node ids or (near-)zero volume."""
    keep = np.ones(len(cells), dtype=bool)
    for c, cell in enumerate(cells):
        if len(set(cell.tolist())) < len(cell):
            keep[c] = False
            continue
        if tol > 0.0:
            p = points[np.asarray(cell)]
            if abs(np.linalg.det(np.column_stack([p[1] - p[0],
                                                  p[2] - p[0],
                                                  p[3] - p[0]]))) <= tol:
                keep[c] = False
    return cells[keep]


def main():
    radii, i_disc, n_wall = build_radial_layers()
    X, (i_a0, i_a1) = build_axial_layers()
    grid = NodeGrid(X, radii, cfg.N_THETA_DIV)

    # The blocks come straight from configuration.solid_node_blocks(), which is
    # also what cfg.VOL_SOLID is computed from -- so the guard below verifies
    # the mesh that was actually emitted, not a parallel derivation of it.
    blocks, _, _ = cfg.solid_node_blocks()

    hexes = []
    for a_lo, a_hi, r_lo, r_hi, _label in blocks:
        grid.add_ring_block(hexes, a_lo, a_hi - 1, r_lo, r_hi - 1)

    hexes = np.asarray(hexes, dtype=np.int64)
    tets = drop_degenerate(hexes_to_tets(hexes), grid.points)

    coord_element = element("Lagrange", "tetrahedron", 1, shape=(3,))
    structure = dolfinx.mesh.create_mesh(MPI.COMM_WORLD, tets, coord_element,
                                         grid.points)

    # ---- regression guard: the solid volume must match the analytic value ----
    # cfg.VOL_SOLID is the exact-circle volume (cylinder + disc) and is
    # *independent* of this generator's ring construction, so this is a genuine
    # cross-check.  The mesh inscribes an N_THETA_DIV-gon in every circle, so
    # it comes out slightly below the ideal value (about 0.1 % at N_theta=104);
    # the tolerance is 1 %.  A topology mistake -- wrong node ordering, dropped
    # or duplicated cells, or a spurious core block -- moves the volume by tens
    # of percent, which is how this guard earned its keep during development.
    dxx = Measure("dx", domain=structure)
    volume = assemble_scalar(form(Constant(structure, 1.0) * dxx))
    exact = cfg.VOL_SOLID
    rel = (volume - exact) / exact
    assert abs(rel) < 1e-2, (
        f"solid volume mismatch: {volume:.10e} vs {exact:.10e} "
        f"(rel {rel:+.4e})")

    out = cfg.solid_mesh_path()
    with XDMFFile(MPI.COMM_WORLD, out, "w") as xdmf:
        xdmf.write_mesh(structure)

    n_cells = structure.topology.index_map(3).size_local
    if MPI.COMM_WORLD.rank == 0:
        print(f"solid mesh [{cfg.CASE}]: {n_cells} tets, "
              f"{structure.topology.index_map(0).size_local} nodes")
        print(f"  hexes {len(hexes)} -> tets {len(tets)} "
              f"(dropped {len(hexes) * 6 - len(tets)} degenerate)")
        print(f"  wall {n_wall} radial layers, disc radius a={cfg.A_LUMEN} m, "
              f"{cfg.N_THETA_DIV} angular divisions")
        print(f"  r in [{radii[0]:.6g}, {radii[-1]:.6g}], "
              f"x in [{X[0]:.6g}, {X[-1]:.6g}]")
        print(f"  volume = {volume:.10e} m^3 (exact {exact:.10e}, "
              f"rel {rel:.2e})")
        print(f"  written to {out}")


if __name__ == "__main__":
    main()
