"""Generate the structured quadrilateral mesh of the immersed annular solid.

The annular solid is treated as a rectangle whose short ends are joined:
    n_theta = 28 * M  elements around the circumference
    n_r     = M       elements across the thickness
with M = 2N/16 = N/8, matching the fluid mesh size ratio in the benchmark.

Run:
    conda activate afsi-dolfinx
    python generate_mesh.py            # default N=32
    N=64 python generate_mesh.py
"""
import os
import numpy as np
from ufl import Measure
from mpi4py import MPI
import dolfinx
from basix.ufl import element
from dolfinx.io import XDMFFile
from dolfinx.mesh import CellType
from dolfinx.fem import form, assemble_scalar, Constant

# Physical parameters of the benchmark
R = 0.25          # inner radius [m]
w = 0.0625        # annulus width [m]
cx, cy = 0.5, 0.5 # center

N = int(os.environ.get("N", "32"))
CELL_TYPE = os.environ.get("CELL_TYPE", "quadrilateral").lower()
M = 2 * N // 16
if M < 1:
    raise ValueError("N must be at least 8 so that M=N/8 >= 1")
n_theta = 28 * M
n_r = M

# Structured vertices: n_theta points on each radial layer, no duplicate seam.
# Vertex (i, j): i = 0..n_theta-1 around the circumference, j = 0..n_r radial.
n_vertices = n_theta * (n_r + 1)
x = np.zeros((n_vertices, 3), dtype=np.float64)
for j in range(n_r + 1):
    r = R + w * j / n_r
    for i in range(n_theta):
        theta = 2.0 * np.pi * i / n_theta
        idx = j * n_theta + i
        x[idx, 0] = cx + r * np.cos(theta)
        x[idx, 1] = cy + r * np.sin(theta)

if CELL_TYPE == "quadrilateral":
    # Quadrilateral cells in VTK cyclic order: [inner-left, inner-right,
    # outer-right, outer-left].  Then convert to DOLFINx order with the
    # official permutation instead of handwriting a cell ordering.
    cells_vtk = np.zeros((n_theta * n_r, 4), dtype=np.int64)
    c = 0
    for j in range(n_r):
        for i in range(n_theta):
            i_next = (i + 1) % n_theta
            il = j * n_theta + i
            ir = j * n_theta + i_next
            oo = (j + 1) * n_theta + i_next
            ol = (j + 1) * n_theta + i
            cells_vtk[c] = [il, ir, oo, ol]
            c += 1

    perm = np.asarray(
        dolfinx.cpp.io.perm_vtk(CellType.quadrilateral, 4), dtype=np.int64)
    cells = cells_vtk[:, perm]
    cell_type = CellType.quadrilateral
    coord_cell = "quadrilateral"
elif CELL_TYPE == "triangle":
    # Split each quadrilateral (physical CCW order) into two triangles.
    # DOLFINx triangle ordering is simply counter-clockwise.
    cells = np.zeros((2 * n_theta * n_r, 3), dtype=np.int64)
    c = 0
    for j in range(n_r):
        for i in range(n_theta):
            i_next = (i + 1) % n_theta
            il = j * n_theta + i
            ir = j * n_theta + i_next
            oo = (j + 1) * n_theta + i_next
            ol = (j + 1) * n_theta + i
            # physical CCW quad: il -> ol -> oo -> ir
            cells[c] = [il, ol, oo]      # lower triangle
            cells[c + 1] = [il, oo, ir]  # upper triangle
            c += 2
    cell_type = CellType.triangle
    coord_cell = "triangle"
else:
    raise ValueError("CELL_TYPE must be 'quadrilateral' or 'triangle'")

coord_element = element("Lagrange", coord_cell, 1, shape=(2,))
structure = dolfinx.mesh.create_mesh(
    MPI.COMM_WORLD, cells, coord_element, x[:, :2]
)

# Regression guard: the solid FE area must match the annular area.  A wrong
# DOLFINx quadrilateral vertex ordering silently corrupts all solid integrals.
dxx = Measure("dx", domain=structure)
A = assemble_scalar(form(Constant(structure, 1.0) * dxx))
A_exact = np.pi * ((R + w) ** 2 - R**2)
assert abs(A - A_exact) < 1e-2 * A, f"Solid mesh area mismatch: {A} vs {A_exact}"

outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plot")
os.makedirs(outdir, exist_ok=True)
outfile = os.path.join(outdir, "mesh-423.xdmf")
with XDMFFile(MPI.COMM_WORLD, outfile, "w") as xdmf:
    xdmf.write_mesh(structure)

if MPI.COMM_WORLD.rank == 0:
    print(f"Annular solid mesh: n_theta={n_theta}, n_r={n_r}, "
          f"cells={structure.topology.index_map(2).size_local}")
    print(f"Written to {outfile}")
