from mpi4py import MPI
import gmsh
import dolfinx
from dolfinx.io import XDMFFile, gmshio

# 1. 初始化 Gmsh，加载 .geo 文件并生成网格
gmsh.initialize()
gmsh.model.add("turek")
gmsh.merge("turek.geo")  # 加载几何
gmsh.model.mesh.generate(dim=2)  # 生成 2D 网格

# 2. 转为 DOLFINx 网格（含 cell/facet 标记）
model_rank = 0
gdim = 2
mesh_data = gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, model_rank, gdim=gdim)
mesh, cell_tags, facet_tags = mesh_data[0], mesh_data[1], mesh_data[2]
gmsh.finalize()

# 3. 写出 XDMF（ParaView 可直接打开 .xdmf 文件）
with XDMFFile(MPI.COMM_WORLD, "turek_mesh.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_meshtags(cell_tags, mesh.geometry)  # Physical Surface 标记
    xdmf.write_meshtags(
        facet_tags, mesh.geometry
    )  # Physical Line 标记 (11=boundary, 20=flag tip)
