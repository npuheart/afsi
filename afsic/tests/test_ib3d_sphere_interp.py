"""
IB 3D 插值演示：背景网格 [0,1]^3 上的三角函数场 f
→ 插值到固体球体网格得到 g（解析真值）与 h（IB 插值算子），输出可视化。

步骤：
  1. 在背景网格上定义三角函数场 f
  2. 解析计算 f 在固体节点上的值 → g（真值）
  3. 通过 IBInterpolation3D.fluid_to_solid 从背景网格插值到固体 → h
  4. h、g 输出为 xdmf 可视化文件

运行：
    conda activate afsi-dolfinx
    python test_ib3d_sphere_interp.py
"""
import os
import numpy as np
from mpi4py import MPI
import gmsh

import dolfinx
from dolfinx.fem import Function, functionspace, form, assemble_scalar
from basix.ufl import element
from dolfinx.mesh import CellType, GhostMode
from dolfinx.io import XDMFFile, gmsh as gmshio  # dolfinx 0.10: gmshio 更名 gmsh

from ufl import dot, dx
from afsic import IBMesh3D, IBInterpolation3D

gdim = 3

# ============================================================================
# 1. 背景网格 + 三角函数场 f
# ============================================================================
Nx, Ny, Nz = 32, 32, 32
mesh = dolfinx.mesh.create_box(
    comm=MPI.COMM_WORLD,
    points=((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
    n=(Nx, Ny, Nz),
    cell_type=CellType.hexahedron,
    ghost_mode=GhostMode.shared_facet,
)

v_cg2 = element("Lagrange", mesh.topology.cell_name(), 2, shape=(gdim,))
v_cg1 = element("Lagrange", mesh.topology.cell_name(), 1, shape=(gdim,))
V = functionspace(mesh, v_cg2)
V_io = functionspace(mesh, v_cg1)


def trig_field(x):
    """3 分量三角函数场 f（解析定义，用于插值对比）。"""
    return np.array([
        0.5 * np.sin(2 * np.pi * x[0]) * np.cos(2 * np.pi * x[1]) * np.cos(2 * np.pi * x[2]),
        0.5 * np.cos(2 * np.pi * x[0]) * np.sin(2 * np.pi * x[1]) * np.cos(2 * np.pi * x[2]),
        0.5 * np.cos(2 * np.pi * x[0]) * np.cos(2 * np.pi * x[1]) * np.sin(2 * np.pi * x[2]),
    ])


f = Function(V, name="f_trig")
f.interpolate(trig_field)

# ============================================================================
# 2. 固体球体网格（gmsh，中心 (0.5,0.5,0.5)，半径 0.3，位于背景域内）
# ============================================================================
gmsh.initialize()
gmsh.model.add("sphere")
s = gmsh.model.occ.addSphere(0.5, 0.5, 0.5, 0.3)
gmsh.model.occ.synchronize()
gmsh.model.addPhysicalGroup(3, [s], 1)
gmsh.model.mesh.setSize(gmsh.model.getEntities(0), 0.02)  # h=0.06 -> 0.02（固体加密，1/3）
gmsh.model.mesh.generate(3)
mesh_data = gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, rank=0, gdim=3)
structure, ct, ft = mesh_data[0], mesh_data[1], mesh_data[2]
gmsh.finalize()

Vs = functionspace(structure, element(
    "Lagrange", structure.topology.cell_name(), 2, shape=(gdim,)))
Vs_io = functionspace(structure, element(
    "Lagrange", structure.topology.cell_name(), 1, shape=(gdim,)))

solid_coords = Function(Vs, name="solid_coords")
solid_coords.interpolate(lambda x: np.array([x[0], x[1], x[2]]))

# ============================================================================
# 3. g：解析真值（三角函数直接作用在固体节点坐标上）
# ============================================================================
g = Function(Vs, name="g_exact")
g.interpolate(trig_field)

# ============================================================================
# 4. h：IB 插值算子（MAC 背景网格 → 固体网格）
# ============================================================================
ibmesh = IBMesh3D(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, Nx, Ny, Nz, 2)
ib_interp = IBInterpolation3D(ibmesh)

coords_bg = Function(V)
coords_bg.interpolate(lambda x: np.array([x[0], x[1], x[2]]))
ibmesh.build_map(coords_bg._cpp_object)
ib_interp.evaluate_current_points(solid_coords._cpp_object)

h = Function(Vs, name="h_ib")
ib_interp.fluid_to_solid(f._cpp_object, h._cpp_object)
h.x.scatter_forward()

# ============================================================================
# 5. 输出可视化（h: IB 插值 t=0，g: 解析真值 t=1）
# ============================================================================
# 误差场 error = g - h（IB 插值误差）
e = Function(Vs)
e.x.array[:] = g.x.array - h.x.array
e.x.scatter_forward()

h_io = Function(Vs_io, name="h_ib")
h_io.interpolate(h)
g_io = Function(Vs_io, name="g_exact")
g_io.interpolate(g)
e_io = Function(Vs_io, name="error")
e_io.interpolate(e)

out_dir = os.path.dirname(os.path.abspath(__file__))
with XDMFFile(MPI.COMM_WORLD, os.path.join(out_dir, "sphere_interp.xdmf"), "w") as xf:
    xf.write_mesh(structure)
    xf.write_function(h_io, 0.0)   # h: IB 插值
    xf.write_function(g_io, 0.0)   # g: 解析真值（同帧，便于对比）
    xf.write_function(e_io, 0.0)   # error = g - h（插值误差场）

# 背景场 f（可选查看）
f_io = Function(V_io, name="f_trig")
f_io.interpolate(f)
with XDMFFile(MPI.COMM_WORLD, os.path.join(out_dir, "bg_field.xdmf"), "w") as xf:
    xf.write_mesh(mesh)
    xf.write_function(f_io, 0.0)

# ============================================================================
# 误差统计（g vs h）
# ============================================================================
err2 = MPI.COMM_WORLD.allreduce(
    assemble_scalar(form(dot(g - h, g - h) * dx)), op=MPI.SUM)
g2 = MPI.COMM_WORLD.allreduce(
    assemble_scalar(form(dot(g, g) * dx)), op=MPI.SUM)
if MPI.COMM_WORLD.rank == 0:
    print(f"\n固体球体节点数: {structure.geometry.x.shape[0]}, 单元数: "
          f"{structure.topology.index_map(3).size_global}")
    print(f"L2 相对误差 ||g - h|| / ||g|| = {np.sqrt(err2 / g2):.4e}")
    print(f"输出: {os.path.join(out_dir, 'sphere_interp.xdmf')} "
          f"(函数 h_ib=IB 插值, g_exact=解析真值, error=g-h 插值误差)")
