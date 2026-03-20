from mpi4py import MPI
import gmsh  # type: ignore
from dolfinx.io import XDMFFile, gmshio

def gmsh_rectangle(model: gmsh.model, name: str, x0=0.0, y0=0.0, lx=1.0, ly=1.0, mesh_size=0.05) -> gmsh.model:
    """
    Create a Gmsh model of a 2D rectangle and tag its four boundaries.

    Args:
        model: Gmsh model to add the mesh to.
        name: Name (identifier) of the mesh to add.

    Returns:
        Gmsh model with a rectangle mesh added.
    """
    model.add(name)
    model.setCurrent(name)

    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)

    # Add rectangle surface
    rect = model.occ.addRectangle(x0, y0, 0, lx, ly, tag=1)
    model.occ.synchronize()

    # Get boundary curves (edges)
    curves = model.getBoundary([(2, rect)], oriented=False)
    # curves: list of (dim, tag), dim=1 for curves

    # Assign physical tags to boundaries: left, right, bottom, top
    # Find curves by their coordinates
    curve_tags = [c[1] for c in curves]
    curve_coords = [gmsh.model.occ.getCenterOfMass(1, tag) for tag in curve_tags]

    left = [curve_tags[i] for i, c in enumerate(curve_coords) if abs(c[0] - x0) < 1e-8]
    right = [curve_tags[i] for i, c in enumerate(curve_coords) if abs(c[0] - (x0 + lx)) < 1e-8]
    bottom = [curve_tags[i] for i, c in enumerate(curve_coords) if abs(c[1] - y0) < 1e-8]
    top = [curve_tags[i] for i, c in enumerate(curve_coords) if abs(c[1] - (y0 + ly)) < 1e-8]

    model.add_physical_group(2, [rect], tag=1)  # domain
    model.add_physical_group(1, left, tag=2)    # left
    model.add_physical_group(1, right, tag=3)   # right
    model.add_physical_group(1, bottom, tag=4)  # bottom
    model.add_physical_group(1, top, tag=5)     # top

    model.mesh.generate(dim=2)
    return model

gmsh.initialize()
model = gmsh.model()
model = gmsh_rectangle(model, "Rectangle")
model.setCurrent("Rectangle")

model_rank = 0
gdim = 2
mesh_data = gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, model_rank, gdim=gdim)
mesh = mesh_data[0]
ct = mesh_data[1]
ft = mesh_data[2]
ft.name = "Facet markers"

import dolfinx
with dolfinx.io.XDMFFile(MPI.COMM_WORLD, 'mesh-340.xdmf', "w", encoding=dolfinx.io.XDMFFile.Encoding.HDF5) as file:
    file.write_mesh(mesh)
    file.write_meshtags(ft, mesh.geometry)