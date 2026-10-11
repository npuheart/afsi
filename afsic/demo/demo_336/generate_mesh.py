"""Generate the disk mesh for demo_336 (lid-driven cavity with an elastic disk).

Disk centre (0.6, 0.5), radius 0.2 (Zhao, Freund & Moser 2008), linear triangles
(the solver uses P2 on top of this mesh).

Usage:
    H=0.008 OUT=plot/disk.xdmf python generate_mesh.py
    MESH_PATH=plot/disk.xdmf python fsi_paralell.py
"""
import os
import gmsh
from mpi4py import MPI
import dolfinx

H = float(os.environ.get("H", "0.008"))
OUT = os.environ.get("OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "plot", "disk.xdmf"))
os.makedirs(os.path.dirname(OUT), exist_ok=True)

gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)
gmsh.model.add("disk")
if MPI.COMM_WORLD.rank == 0:
    disk = gmsh.model.occ.addDisk(0.6, 0.5, 0.0, 0.2, 0.2)
    gmsh.model.occ.synchronize()
    gmsh.model.addPhysicalGroup(2, [disk], 1)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", H)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", H)
    gmsh.model.mesh.generate(2)

try:  # dolfinx >= 0.10
    from dolfinx.io import gmsh as _gio
    data = _gio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=2)
    mesh = data.mesh if hasattr(data, "mesh") else data[0]
except ImportError:  # older dolfinx
    from dolfinx.io import gmshio as _gio
    mesh = _gio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=2)[0]
gmsh.finalize()

with dolfinx.io.XDMFFile(MPI.COMM_WORLD, OUT, "w") as f:
    f.write_mesh(mesh)
if MPI.COMM_WORLD.rank == 0:
    print(f"wrote {OUT} (H={H})")
