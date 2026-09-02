"""
纯 Python 版 IB 3D 插值演示（不依赖 afsic）：
背景网格 [0,1]^3 上的三角函数场 f → 插值到固体球体网格，核函数为 Peskin 4 点 δ。

- f     ：numpy 数组存背景网格顶点值（三角函数场）
- g     ：解析真值（三角函数直接作用在固体节点）
- h     ：手动实现 Peskin 4 点核插值（背景网格 → 固体）
- error ：g - h 插值误差场

三个场输出为 xdmf 可视化文件。仅依赖 numpy / gmsh / dolfinx（无 afsic）。

运行：
    conda activate afsi-dolfinx
    python test_ib3d_sphere_interp_pure.py
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

gdim = 3

# ============================================================================
# Peskin 4 点核（与 afsic kernel_expression.h 的 evaluate_kernel_peskin<4> 一致）
# ============================================================================
def phi(r):
    """Peskin 4-point delta function（r 为以网格间距归一化的距离）。"""
    r = abs(r)
    if r <= 1.0:
        return (3.0 - 2.0 * r + np.sqrt(1.0 + 4.0 * r - 4.0 * r * r)) / 8.0
    elif r <= 2.0:
        return (5.0 - 2.0 * r - np.sqrt(-7.0 + 12.0 * r - 4.0 * r * r)) / 8.0
    return 0.0


def peskin_interp_3d(f, x, dh, origin=(0.0, 0.0, 0.0)):
    """从背景网格顶点场 f 用 Peskin 4 点核插值到点 x。

    f    : (Nx+1, Ny+1, Nz+1, 3) 背景网格顶点值
    x    : (3,) 目标点坐标
    dh   : (3,) 背景网格间距
    """
    N = (f.shape[0] - 1, f.shape[1] - 1, f.shape[2] - 1)
    x0 = (np.asarray(x) - np.asarray(origin)) / np.asarray(dh)   # 归一化
    base = np.floor(x0 - 1.0).astype(int)   # 4 点核 base node
    val = np.zeros(3)
    for i in range(4):
        xi = base[0] + i
        wi = phi(x0[0] - xi)
        if wi == 0.0 or xi < 0 or xi > N[0]:
            continue
        for j in range(4):
            yj = base[1] + j
            wj = phi(x0[1] - yj)
            if wj == 0.0 or yj < 0 or yj > N[1]:
                continue
            wij = wi * wj
            for k in range(4):
                zk = base[2] + k
                wk = phi(x0[2] - zk)
                if wk == 0.0 or zk < 0 or zk > N[2]:
                    continue
                val += wij * wk * f[xi, yj, zk]
    return val


# ============================================================================
# 1. 背景网格 [0,1]^3 + 三角函数场 f（numpy 顶点值）
# ============================================================================
Nx, Ny, Nz = 32, 32, 32
dh = (1.0 / Nx, 1.0 / Ny, 1.0 / Nz)

xs = np.linspace(0.0, 1.0, Nx + 1)
ys = np.linspace(0.0, 1.0, Ny + 1)
zs = np.linspace(0.0, 1.0, Nz + 1)
X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")

fx = 0.5 * np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y) * np.cos(2 * np.pi * Z)
fy = 0.5 * np.cos(2 * np.pi * X) * np.sin(2 * np.pi * Y) * np.cos(2 * np.pi * Z)
fz = 0.5 * np.cos(2 * np.pi * X) * np.cos(2 * np.pi * Y) * np.sin(2 * np.pi * Z)
f_vals = np.stack([fx, fy, fz], axis=-1)   # (Nx+1, Ny+1, Nz+1, 3)


def trig_field(x):
    """解析三角函数场（用于 g）。"""
    return np.array([
        0.5 * np.sin(2 * np.pi * x[0]) * np.cos(2 * np.pi * x[1]) * np.cos(2 * np.pi * x[2]),
        0.5 * np.cos(2 * np.pi * x[0]) * np.sin(2 * np.pi * x[1]) * np.cos(2 * np.pi * x[2]),
        0.5 * np.cos(2 * np.pi * x[0]) * np.cos(2 * np.pi * x[1]) * np.sin(2 * np.pi * x[2]),
    ])


# ============================================================================
# 2. 固体球体网格（gmsh，中心 (0.5,0.5,0.5)，半径 0.3）
# ============================================================================
gmsh.initialize()
gmsh.model.add("sphere")
s = gmsh.model.occ.addSphere(0.5, 0.5, 0.5, 0.3)
gmsh.model.occ.synchronize()
gmsh.model.addPhysicalGroup(3, [s], 1)
gmsh.model.mesh.setSize(gmsh.model.getEntities(0), 0.02)   # 固体网格尺寸 h
gmsh.model.mesh.generate(3)
mesh_data = gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, rank=0, gdim=3)
structure, ct, ft = mesh_data[0], mesh_data[1], mesh_data[2]
gmsh.finalize()

# CG1 向量空间（dof = 顶点，便于手动填值 + 可视化）
Vs_io = functionspace(structure, element(
    "Lagrange", structure.topology.cell_name(), 1, shape=(gdim,)))

# ============================================================================
# 3. g：解析真值（手动按 dof 坐标计算）
# ============================================================================
# 注意：block 向量空间（shape=(3,)）的 tabulate_dof_coordinates() 已按顶点去重
# （每顶点一个坐标），直接使用，切勿再 [::3]。
points = Vs_io.tabulate_dof_coordinates()          # (Nv, 3) 全部顶点

g = Function(Vs_io, name="g_exact")
for i, p in enumerate(points):
    g.x.array[3 * i:3 * i + 3] = trig_field(p)

# ============================================================================
# 4. h：Peskin 4 点核插值（纯 Python 手写）
# ============================================================================
h = Function(Vs_io, name="h_ib")
for i, p in enumerate(points):
    h.x.array[3 * i:3 * i + 3] = peskin_interp_3d(f_vals, p, dh, (0.0, 0.0, 0.0))

# 误差场
e = Function(Vs_io, name="error")
e.x.array[:] = g.x.array - h.x.array

# ============================================================================
# 5. 输出可视化
# ============================================================================
out_dir = os.path.dirname(os.path.abspath(__file__))
with XDMFFile(MPI.COMM_WORLD, os.path.join(out_dir, "sphere_interp_pure.xdmf"), "w") as xf:
    xf.write_mesh(structure)
    xf.write_function(h, 0.0)   # h: Peskin 插值
    xf.write_function(g, 0.0)   # g: 解析真值
    xf.write_function(e, 0.0)   # error = g - h

# 背景网格（顶点值 f 写到 CG1 场用于查看）
mesh = dolfinx.mesh.create_box(
    comm=MPI.COMM_WORLD,
    points=((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
    n=(Nx, Ny, Nz),
    cell_type=CellType.hexahedron,
    ghost_mode=GhostMode.shared_facet,
)
V1 = functionspace(mesh, element("Lagrange", mesh.topology.cell_name(), 1, shape=(gdim,)))
f_bg = Function(V1, name="f_trig")
dc_bg = V1.tabulate_dof_coordinates()
for i, p in enumerate(dc_bg):
    f_bg.x.array[3 * i:3 * i + 3] = trig_field(p)
with XDMFFile(MPI.COMM_WORLD, os.path.join(out_dir, "bg_field_pure.xdmf"), "w") as xf:
    xf.write_mesh(mesh)
    xf.write_function(f_bg, 0.0)

# ============================================================================
# 6. 误差统计
# ============================================================================
err2 = MPI.COMM_WORLD.allreduce(
    assemble_scalar(form(dot(g - h, g - h) * dx)), op=MPI.SUM)
g2 = MPI.COMM_WORLD.allreduce(
    assemble_scalar(form(dot(g, g) * dx)), op=MPI.SUM)
if MPI.COMM_WORLD.rank == 0:
    print(f"\n纯 Python 版（Peskin 4 点核）")
    print(f"固体球体顶点数: {points.shape[0]}, 单元数: "
          f"{structure.topology.index_map(3).size_global}")
    print(f"L2 相对误差 ||g - h|| / ||g|| = {np.sqrt(err2 / g2):.4e}")
    print(f"输出: {os.path.join(out_dir, 'sphere_interp_pure.xdmf')} "
          f"(h_ib, g_exact, error)")
