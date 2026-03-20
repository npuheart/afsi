from mpi4py import MPI
import gmsh  # type: ignore
from dolfinx.io import XDMFFile, gmshio

def gmsh_rectangle(model: gmsh.model, name: str, x0=0.0, y0=0.0, lx=1.0, ly=1.0, mesh_size=0.05, tag_offset=0) -> int:
    """
    Create a Gmsh model of a 2D rectangle and tag its four boundaries.
    Returns the rectangle tag.
    """
    rect = model.occ.addRectangle(x0, y0, 0, lx, ly)
    model.occ.synchronize()

    curves = model.getBoundary([(2, rect)], oriented=False)
    curve_tags = [c[1] for c in curves]
    curve_coords = [gmsh.model.occ.getCenterOfMass(1, tag) for tag in curve_tags]

    left = [curve_tags[i] for i, c in enumerate(curve_coords) if abs(c[0] - x0) < 1e-8]
    right = [curve_tags[i] for i, c in enumerate(curve_coords) if abs(c[0] - (x0 + lx)) < 1e-8]
    bottom = [curve_tags[i] for i, c in enumerate(curve_coords) if abs(c[1] - y0) < 1e-8]
    top = [curve_tags[i] for i, c in enumerate(curve_coords) if abs(c[1] - (y0 + ly)) < 1e-8]

    model.add_physical_group(2, [rect], tag=1 + tag_offset)  # domain
    model.add_physical_group(1, left, tag=2 + tag_offset)    # left
    model.add_physical_group(1, right, tag=3 + tag_offset)   # right
    model.add_physical_group(1, bottom, tag=4 + tag_offset)  # bottom
    model.add_physical_group(1, top, tag=5 + tag_offset)     # top

    return rect

def gmsh_circle(model: gmsh.model, name: str, x0=0.0, y0=0.0, r=0.5, mesh_size=0.05, tag_offset=0) -> int:
    """
    Create a Gmsh model of a 2D circle and tag its boundary.
    Returns the circle tag.
    """
    circle = model.occ.addDisk(x0, y0, 0, r, r)
    model.occ.synchronize()

    curves = model.getBoundary([(2, circle)], oriented=False)
    curve_tags = [c[1] for c in curves]

    model.add_physical_group(2, [circle], tag=1 + tag_offset)  # domain
    model.add_physical_group(1, curve_tags, tag=2 + tag_offset)  # boundary

    return circle

gmsh.initialize()
gmsh.model.add("TwoRectangles")
gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.01)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.02)

model = gmsh.model

rect1 = gmsh_rectangle(model, "Rect1",   x0=2.0-0.0212, y0=0.00, lx=0.0212, ly=0.7, tag_offset=0)
rect2 = gmsh_rectangle(model, "Rect2",   x0=2.0-0.0212, y0=0.91, lx=0.0212, ly=0.7, tag_offset=10)
circ1 = gmsh_circle(   model, "Circle1", x0=0.5,        y0=0.5,  r=0.2,             tag_offset=20)
circ2 = gmsh_circle(   model, "Circle2", x0=0.5,        y0=1.1,  r=0.2,             tag_offset=30)

model.occ.synchronize()
model.mesh.generate(dim=2)

model_rank = 0
gdim = 2
mesh_data = gmshio.model_to_mesh(model, MPI.COMM_WORLD, model_rank, gdim=gdim)
mesh = mesh_data[0]
ct = mesh_data[1]
ft = mesh_data[2]
ft.name = "Facet markers"
ct.name = "Cell markers"

import dolfinx
with dolfinx.io.XDMFFile(MPI.COMM_WORLD, 'mesh-343.xdmf', "w", encoding=dolfinx.io.XDMFFile.Encoding.HDF5) as file:
    file.write_mesh(mesh)
    file.write_meshtags(ft, mesh.geometry)
    file.write_meshtags(ct, mesh.geometry)
