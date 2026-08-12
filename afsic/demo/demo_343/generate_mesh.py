"""生成 demo_343 固体网格（上下瓣膜 + 两圆盘）→ plot/mesh-343.xdmf。

固体部件：
  rect1  下瓣膜   (cell tag 1,  boundary tag 2/3/4/5)
  rect2  上瓣膜   (cell tag 11, boundary tag 12/13/14/15)
  circ1  圆盘1    (cell tag 21, boundary tag 22) 中心 (0.5, 0.5) r=0.2
  circ2  圆盘2    (cell tag 31, boundary tag 32) 中心 (0.5, 1.1) r=0.2

运行：
    conda activate afsi-dolfinx
    python generate_mesh.py
"""
import os
from mpi4py import MPI

import gmsh  # type: ignore
from dolfinx.io import XDMFFile, gmsh as gmshio  # dolfinx 0.10: gmshio 更名 gmsh


def gmsh_rectangle(model, name, x0=0.0, y0=0.0, lx=1.0, ly=1.0, tag_offset=0):
    """矩形（瓣膜）：domain tag=1+offset, 边界 tag=2..5+offset。"""
    rect = model.occ.addRectangle(x0, y0, 0, lx, ly)
    model.occ.synchronize()
    curves = model.getBoundary([(2, rect)], oriented=False)
    curve_tags = [c[1] for c in curves]
    curve_coords = [gmsh.model.occ.getCenterOfMass(1, tag) for tag in curve_tags]
    left = [curve_tags[i] for i, c in enumerate(curve_coords) if abs(c[0] - x0) < 1e-8]
    right = [curve_tags[i] for i, c in enumerate(curve_coords) if abs(c[0] - (x0 + lx)) < 1e-8]
    bottom = [curve_tags[i] for i, c in enumerate(curve_coords) if abs(c[1] - y0) < 1e-8]
    top = [curve_tags[i] for i, c in enumerate(curve_coords) if abs(c[1] - (y0 + ly)) < 1e-8]
    model.add_physical_group(2, [rect], tag=1 + tag_offset)
    model.add_physical_group(1, left, tag=2 + tag_offset)
    model.add_physical_group(1, right, tag=3 + tag_offset)
    model.add_physical_group(1, bottom, tag=4 + tag_offset)
    model.add_physical_group(1, top, tag=5 + tag_offset)
    return rect


def gmsh_circle(model, name, x0=0.0, y0=0.0, r=0.2, tag_offset=0):
    """圆盘：domain tag=1+offset, 边界 tag=2+offset。"""
    circle = model.occ.addDisk(x0, y0, 0, r, r)
    model.occ.synchronize()
    curves = model.getBoundary([(2, circle)], oriented=False)
    curve_tags = [c[1] for c in curves]
    model.add_physical_group(2, [circle], tag=1 + tag_offset)
    model.add_physical_group(1, curve_tags, tag=2 + tag_offset)
    return circle


gmsh.initialize()
gmsh.model.add("ValveCircle")
gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.01)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.02)

model = gmsh.model
gmsh_rectangle(model, "Rect1", 2.0 - 0.0212, 0.00, 0.0212, 0.7, tag_offset=0)   # 下瓣膜
gmsh_rectangle(model, "Rect2", 2.0 - 0.0212, 0.91, 0.0212, 0.7, tag_offset=10)  # 上瓣膜
gmsh_circle(model, "Circle1", 0.5, 0.5, 0.2, tag_offset=20)                     # 圆盘1
gmsh_circle(model, "Circle2", 0.5, 1.1, 0.2, tag_offset=30)                     # 圆盘2

model.occ.synchronize()
model.mesh.generate(dim=2)

mesh_data = gmshio.model_to_mesh(model, MPI.COMM_WORLD, rank=0, gdim=2)
mesh, ct, ft = mesh_data[0], mesh_data[1], mesh_data[2]
ft.name = "Facet markers"
ct.name = "Cell markers"
gmsh.finalize()

_out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plot", "mesh-343.xdmf")
os.makedirs(os.path.dirname(_out), exist_ok=True)
with XDMFFile(MPI.COMM_WORLD, _out, "w", encoding=XDMFFile.Encoding.HDF5) as file:
    file.write_mesh(mesh)
    file.write_meshtags(ft, mesh.geometry)
    file.write_meshtags(ct, mesh.geometry)
print("written:", _out)
