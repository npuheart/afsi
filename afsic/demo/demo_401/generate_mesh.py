from mpi4py import MPI
from dolfinx.io import gmshio   # 若这行报错，改用: from dolfinx.io import gmsh as gmshio
from dolfinx.io import XDMFFile
mesh_data  = gmshio.read_from_msh("sperm3d.msh", MPI.COMM_WORLD, gdim=3)
mesh       = mesh_data.mesh
cell_tags  = mesh_data.cell_tags
facet_tags = mesh_data.facet_tags

# 3. 写出 XDMF（ParaView 可直接打开 .xdmf 文件）
with XDMFFile(MPI.COMM_WORLD, "sperm-2.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_meshtags(cell_tags, mesh.geometry)   # Physical Surface 标记
    xdmf.write_meshtags(facet_tags, mesh.geometry)  # Physical Line 标记

# 4. 如果还要输出场数据（如速度、位移），在时间步内追加写
# with XDMFFile(MPI.COMM_WORLD, "result.xdmf", "w") as out:
#     out.write_mesh(mesh)
#     out.write_function(u, t=0.0)