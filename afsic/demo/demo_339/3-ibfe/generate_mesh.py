"""Generate solid disk mesh for IB-FE cylinder demo."""

from mpi4py import MPI
import gmsh
import dolfinx
from dolfinx.io import XDMFFile, gmsh as gmshio  # dolfinx 0.10.0: gmshio 更名 gmsh

gmsh.initialize()
gmsh.model.add("cylinder_solid")
gmsh.merge("cylinder_solid.geo")
gmsh.model.mesh.generate(dim=2)

model_rank = 0
gdim = 2
mesh_data = gmshio.model_to_mesh(
    gmsh.model, MPI.COMM_WORLD, model_rank, gdim=gdim
)
mesh, cell_tags, facet_tags = mesh_data[0], mesh_data[1], mesh_data[2]
gmsh.finalize()

# Scale to match fluid domain units (geo is in metres, same as fluid)
# No scaling needed — fluid and solid both in SI [m]

with XDMFFile(MPI.COMM_WORLD, "cylinder_solid.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_meshtags(cell_tags, mesh.geometry)
    xdmf.write_meshtags(facet_tags, mesh.geometry)
