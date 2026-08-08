"""生成 demo_341 三维方腔驱动圆球的固体网格（圆球）→ plot/mesh-341.xdmf。

流体网格由 navier-stokes.py / fsi_paralell.py 在脚本内用 create_box 生成，
本脚本只生成被驱动的固体（圆球）网格。

运行：
    conda activate afsi-dolfinx
    python generate_mesh.py
"""
import os
from mpi4py import MPI

import gmsh  # type: ignore
from dolfinx.io import XDMFFile, gmsh as gmshio  # dolfinx 0.10: gmshio 更名 gmsh


def gmsh_sphere(model, name, center=(0.6, 0.5, 0.5), radius=0.2, mesh_size=0.01):
    """Gmsh 圆球网格：球体（dim 3）+ 各余维子实体（peaks/ridges/facets）标记。

    Args:
        model: Gmsh model to add the mesh to.
        name: Name (identifier) of the mesh to add.

    Returns:
        Gmsh model with a sphere mesh added.
    """
    model.add(name)
    model.setCurrent(name)

    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)

    sphere = model.occ.addSphere(center[0], center[1], center[2], radius, tag=1)
    model.occ.synchronize()

    # 球体物理标签
    model.add_physical_group(dim=3, tags=[sphere], tag=1)

    # 把各余维子实体嵌入球体并标记（peaks/ridges/facets）
    for dim in [0, 1, 2]:
        entities = model.getEntities(dim)
        entity_ids = [entity[1] for entity in entities]
        model.mesh.embed(dim, entity_ids, 3, sphere)
        model.add_physical_group(dim=dim, tags=entity_ids, tag=dim)

    model.mesh.generate(dim=3)
    return model


gmsh.initialize()
model = gmsh.model()
model = gmsh_sphere(model, "Sphere")
model.setCurrent("Sphere")

mesh_data = gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, rank=0, gdim=3)
mesh, ct, ft = mesh_data[0], mesh_data[1], mesh_data[2]
ft.name = "Facet markers"
gmsh.finalize()

_out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plot", "mesh-341.xdmf")
os.makedirs(os.path.dirname(_out), exist_ok=True)
with XDMFFile(MPI.COMM_WORLD, _out, "w", encoding=XDMFFile.Encoding.HDF5) as file:
    file.write_mesh(mesh)
    file.write_meshtags(ft, mesh.geometry)
print("written:", _out)
