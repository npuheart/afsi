"""
Generate the solid plate mesh for demo_403 (beamInCrossFlow).

Creates a 3D elastic thick plate:
  x in [45, 55] cm    (thickness = 10 cm along flow)
  y in [0,  20] cm    (height = 20 cm from bottom)
  z in [0,  20] cm    (half-width = 20 cm, symmetry at z=0)

Output: plate_mesh.xdmf / plate_mesh.h5
"""

import numpy as np
from mpi4py import MPI
import dolfinx
from dolfinx.mesh import CellType, GhostMode

# Plate dimensions (CGS: cm)
plate_x0, plate_x1 = 45.0, 55.0   # along flow
plate_y0, plate_y1 = 0.0, 20.0     # height from bottom
plate_z0, plate_z1 = 0.0, 20.0     # half-width (symmetry at z=0)

# Mesh resolution
nx, ny, nz = 5, 8, 8  # cells in x, y, z

# Create plate mesh
plate = dolfinx.mesh.create_box(
    comm=MPI.COMM_WORLD,
    points=((plate_x0, plate_y0, plate_z0), (plate_x1, plate_y1, plate_z1)),
    n=(nx, ny, nz),
    cell_type=CellType.hexahedron,
    ghost_mode=GhostMode.shared_facet,
)

# Tag boundaries for later use (bottom fixation, etc.)
plate.topology.create_connectivity(plate.topology.dim - 1, plate.topology.dim)

# Save mesh
import os
output_dir = os.path.dirname(os.path.abspath(__file__))
mesh_path = os.path.join(output_dir, "plate_mesh.xdmf")

with dolfinx.io.XDMFFile(MPI.COMM_WORLD, mesh_path, "w") as xdmf:
    xdmf.write_mesh(plate)

print(f"Plate mesh saved to {mesh_path}")
print(f"  Cells: {plate.topology.index_map(plate.topology.dim).size_global}")
print(f"  Vertices: {plate.geometry.x.shape[0]}")
