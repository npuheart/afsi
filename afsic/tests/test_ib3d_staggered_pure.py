"""
纯 Python 交错网格（MAC 风格）IB 3D 插值演示（无有限元空间，不依赖 afsic/dolfinx）。

背景网格 [0,1]^3 为交错网格：
  - fx 存于 x 方向面中心（x 偏移 dh/2）
  - fy 存于 y 方向面中心（y 偏移 dh/2）
  - fz 存于 z 方向面中心（z 偏移 dh/2）
每个分量在各自的偏移网格上用 Peskin 4 点核插值到固体球体节点。

- g     ：解析真值（三角函数）
- h     ：交错网格 Peskin 4 点核插值
- error ：g - h
输出为 VTU（ParaView 可直接打开）。仅依赖 numpy / gmsh。

运行：
    conda activate afsi-dolfinx
    python test_ib3d_staggered_pure.py
"""
import os
import numpy as np
import gmsh

gdim = 3

# ============================================================================
# Peskin 4 点核
# ============================================================================
def phi(r):
    r = abs(r)
    if r <= 1.0:
        return (3.0 - 2.0 * r + np.sqrt(1.0 + 4.0 * r - 4.0 * r * r)) / 8.0
    elif r <= 2.0:
        return (5.0 - 2.0 * r - np.sqrt(-7.0 + 12.0 * r - 4.0 * r * r)) / 8.0
    return 0.0


def peskin_interp_scalar(f, origin, dh, x):
    """在网格上 Peskin 4 点核标量插值。

    f      : numpy 数组 (n0, n1, n2)，索引 i 对应坐标 origin + i*dh
    origin : (3,) 网格原点（第一个点的坐标）
    dh     : (3,) 网格间距
    x      : (3,) 目标点
    """
    x0 = (np.asarray(x) - np.asarray(origin)) / np.asarray(dh)
    base = np.floor(x0 - 1.0).astype(int)
    s = 0.0
    for i in range(4):
        xi = base[0] + i
        wi = phi(x0[0] - xi)
        if wi == 0.0 or xi < 0 or xi >= f.shape[0]:
            continue
        for j in range(4):
            yj = base[1] + j
            wj = phi(x0[1] - yj)
            if wj == 0.0 or yj < 0 or yj >= f.shape[1]:
                continue
            wij = wi * wj
            for k in range(4):
                zk = base[2] + k
                wk = phi(x0[2] - zk)
                if wk == 0.0 or zk < 0 or zk >= f.shape[2]:
                    continue
                s += wij * wk * f[xi, yj, zk]
    return s


def trig_field(x):
    """解析三角函数场（(3, n) 输入，dolfinx 约定）。"""
    return np.array([
        0.5 * np.sin(2 * np.pi * x[0]) * np.cos(2 * np.pi * x[1]) * np.cos(2 * np.pi * x[2]),
        0.5 * np.cos(2 * np.pi * x[0]) * np.sin(2 * np.pi * x[1]) * np.cos(2 * np.pi * x[2]),
        0.5 * np.cos(2 * np.pi * x[0]) * np.cos(2 * np.pi * x[1]) * np.sin(2 * np.pi * x[2]),
    ])


def trig_on_points(P):
    """解析三角函数场，(n, 3) 输入 -> (n, 3) 输出。"""
    X, Y, Z = P[:, 0], P[:, 1], P[:, 2]
    return np.column_stack([
        0.5 * np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y) * np.cos(2 * np.pi * Z),
        0.5 * np.cos(2 * np.pi * X) * np.sin(2 * np.pi * Y) * np.cos(2 * np.pi * Z),
        0.5 * np.cos(2 * np.pi * X) * np.cos(2 * np.pi * Y) * np.sin(2 * np.pi * Z),
    ])


# ============================================================================
# 1. 交错背景网格 [0,1]^3（MAC 风格，各分量偏移 dh/2）
# ============================================================================
N = 32                      # 每方向单元数
dh = (1.0 / N,) * 3

# 索引网格
ix = np.arange(N)           # 面中心索引（偏移方向）
inode = np.arange(N + 1)    # 节点索引

# fx: x 面中心（x 偏移 +0.5，y/z 在节点）  shape (N, N+1, N+1)
Xx = (ix + 0.5) / N; Yx = inode / N; Zx = inode / N
FX, FY, FZ = np.meshgrid(Xx, Yx, Zx, indexing="ij")
fx = trig_on_points(np.stack([FX.ravel(), FY.ravel(), FZ.ravel()], axis=-1))
fx = fx[:, 0].reshape(N, N + 1, N + 1)

# fy: y 面中心（y 偏移 +0.5，x/z 在节点）  shape (N+1, N, N+1)
Yy = (ix + 0.5) / N; Xy = inode / N; Zy = inode / N
FX, FY, FZ = np.meshgrid(Xy, Yy, Zy, indexing="ij")
fy = trig_on_points(np.stack([FX.ravel(), FY.ravel(), FZ.ravel()], axis=-1))
fy = fy[:, 1].reshape(N + 1, N, N + 1)

# fz: z 面中心（z 偏移 +0.5，x/y 在节点）  shape (N+1, N+1, N)
Zz = (ix + 0.5) / N; Xz = inode / N; Yz = inode / N
FX, FY, FZ = np.meshgrid(Xz, Yz, Zz, indexing="ij")
fz = trig_on_points(np.stack([FX.ravel(), FY.ravel(), FZ.ravel()], axis=-1))
fz = fz[:, 2].reshape(N + 1, N + 1, N)

# 各分量网格的原点（第一个点的坐标）
origin_x = (0.5 * dh[0], 0.0, 0.0)
origin_y = (0.0, 0.5 * dh[1], 0.0)
origin_z = (0.0, 0.0, 0.5 * dh[2])

# ============================================================================
# 2. 固体球体网格（gmsh，中心 (0.5,0.5,0.5)，半径 0.3）
# ============================================================================
gmsh.initialize()
gmsh.model.add("sphere")
s = gmsh.model.occ.addSphere(0.5, 0.5, 0.5, 0.3)
gmsh.model.occ.synchronize()
gmsh.model.addPhysicalGroup(3, [s], 1)
gmsh.model.mesh.setSize(gmsh.model.getEntities(0), 0.02)
gmsh.model.mesh.generate(3)

# 纯 gmsh 提取节点与四面体单元（无有限元空间）
node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
points = node_coords.reshape(-1, 3)            # (Np, 3)
node_idx = (node_tags - 1).astype(int)          # tag -> 行号（gmsh 通常连续）
points = points[node_idx]

cells = None
elem_types, _, node_tags_elem = gmsh.model.mesh.getElements(3)
for t, nte in zip(elem_types, node_tags_elem):
    if t == 4:                                   # 4 节点四面体
        cells = (nte.reshape(-1, 4) - 1).astype(int)
gmsh.finalize()
assert cells is not None, "未找到四面体单元"

# ============================================================================
# 3. g：解析真值
# ============================================================================
g = trig_on_points(points)    # (Np, 3)

# ============================================================================
# 4. h：交错网格 Peskin 插值（每分量用自己的偏移网格）
# ============================================================================
h = np.zeros_like(g)
for i, p in enumerate(points):
    h[i, 0] = peskin_interp_scalar(fx, origin_x, dh, p)
    h[i, 1] = peskin_interp_scalar(fy, origin_y, dh, p)
    h[i, 2] = peskin_interp_scalar(fz, origin_z, dh, p)

error = g - h

# ============================================================================
# 5. 输出 VTU（纯 Python 手写，ParaView 可打开）
# ============================================================================
out_dir = os.path.dirname(os.path.abspath(__file__))
vtu = os.path.join(out_dir, "sphere_interp_staggered.vtu")


def write_vtu(filename, pts, cells, point_data):
    npts, ncell = len(pts), len(cells)
    offsets = np.cumsum([4] * ncell)
    types = np.full(ncell, 10, dtype=np.uint8)   # VTK tetrahedron
    with open(filename, "w") as fh:
        fh.write('<?xml version="1.0"?>\n')
        fh.write('<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">\n')
        fh.write('  <UnstructuredGrid>\n')
        fh.write(f'    <Piece NumberOfPoints="{npts}" NumberOfCells="{ncell}">\n')
        fh.write('      <Points>\n')
        fh.write('        <DataArray type="Float64" NumberOfComponents="3" format="ascii">\n')
        np.savetxt(fh, pts, fmt="%.8e")
        fh.write('        </DataArray>\n      </Points>\n      <Cells>\n')
        fh.write('        <DataArray type="Int64" Name="connectivity" format="ascii">\n')
        np.savetxt(fh, cells, fmt="%d")
        fh.write('        </DataArray>\n')
        fh.write('        <DataArray type="Int64" Name="offsets" format="ascii">\n')
        np.savetxt(fh, offsets.reshape(-1, 1), fmt="%d")
        fh.write('        </DataArray>\n')
        fh.write('        <DataArray type="UInt8" Name="types" format="ascii">\n')
        np.savetxt(fh, types.reshape(-1, 1), fmt="%d")
        fh.write('        </DataArray>\n      </Cells>\n      <PointData>\n')
        for name, arr in point_data.items():
            fh.write(f'        <DataArray type="Float64" Name="{name}" NumberOfComponents="3" format="ascii">\n')
            np.savetxt(fh, arr, fmt="%.8e")
            fh.write('        </DataArray>\n')
        fh.write('      </PointData>\n    </Piece>\n  </UnstructuredGrid>\n</VTKFile>\n')


write_vtu(vtu, points, cells, {"h_ib": h, "g_exact": g, "error": error})

# ============================================================================
# 6. 误差统计（纯 numpy）
# ============================================================================
err_rel = np.linalg.norm(error) / np.linalg.norm(g)
wsum = phi(-1.5) + phi(-0.5) + phi(0.5) + phi(1.5)   # 核权重和（应为 1）
print(f"\n纯 Python 交错网格版（MAC，无有限元空间）")
print(f"背景交错网格: {N}^3，各分量偏移 dh/2")
print(f"固体球体顶点: {points.shape[0]}, 四面体单元: {len(cells)}")
print(f"Peskin 4 点核权重和(理论=1): {wsum:.6f}")
print(f"L2 相对误差 ||g - h|| / ||g|| = {err_rel:.4e}")
print(f"输出: {vtu}")
