"""demo_341 后处理：t=1 时刻三条中心线速度插值对比 + 三幅图。

遍历 plot/<case>_N<grid>/velocity.xdmf（NS 与 FSI、不同背景网格密度），
取时间最接近 t=1.0 的帧，沿三条中心线采样速度分量：
    (x, 0.5, 0.5)  → 采样 u_x   → line_x.csv / line_x.png
    (0.5, y, 0.5)  → 采样 u_y   → line_y.csv / line_y.png
    (0.5, 0.5, z)  → 采样 u_z   → line_z.csv / line_z.png

三幅图上叠加不同背景网格密度（及 NS/FSI）的结果。

运行：
    conda activate afsi-dolfinx
    python plot_lines.py
"""
import os
import glob
import numpy as np
import h5py
from mpi4py import MPI

import dolfinx
from dolfinx.io import XDMFFile
from dolfinx.fem import Function, functionspace
from basix.ufl import element
import dolfinx.geometry as dg

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

comm = MPI.COMM_WORLD
PLOT = os.path.dirname(os.path.abspath(__file__))
T_TARGET = 1.0
NP = 101
AXIS = {"x": 0, "y": 1, "z": 2}


def kt(k):
    return float(k.replace("_", "."))


def load_case(folder):
    """加载某 run 的 velocity 场（最接近 t=1.0 的帧）。返回 (mesh, u, t)。"""
    xdmf = os.path.join(folder, "velocity.xdmf")
    h5f = os.path.join(folder, "velocity.h5")
    if not (os.path.exists(xdmf) and os.path.exists(h5f)):
        return None
    with XDMFFile(comm, xdmf, "r") as f:
        mesh = f.read_mesh()
    V = functionspace(mesh, element("Lagrange", mesh.topology.cell_name(), 1, shape=(3,)))
    u = Function(V)
    with h5py.File(h5f, "r") as h:
        keys = sorted(h["Function/f"].keys(), key=kt)
        key = min(keys, key=lambda k: abs(kt(k) - T_TARGET))
        t = kt(key)
        u.x.array[:] = h[f"Function/f/{key}"][:].ravel()
    return mesh, u, t


def sample_line(mesh, u, axis, comp):
    """沿 (变化轴=axis, 其余=0.5) 的线采样速度分量 comp。"""
    pts = np.full((NP, 3), 0.5)
    coords = np.linspace(0.0, 1.0, NP)
    pts[:, AXIS[axis]] = coords
    tree = dg.bb_tree(mesh, mesh.topology.dim)
    vals = np.full(NP, np.nan)
    for i in range(NP):
        q = pts[i:i + 1]
        cand = dg.compute_collisions_points(tree, q)
        col = dg.compute_colliding_cells(mesh, cand, q)
        links = col.links(0)
        if len(links) > 0:
            uu = u.eval(q, np.array([links[0]], dtype=np.int32))
            vals[i] = np.ravel(uu)[comp]
    return coords, vals


def main():
    # 收集所有 run 目录（ns_N<grid> / fsi_N<grid>）
    runs = []
    for v in glob.glob(os.path.join(PLOT, "*_N*", "velocity.xdmf")):
        folder = os.path.dirname(v)
        name = os.path.basename(folder)
        runs.append((name, folder))
    runs.sort()
    if not runs:
        print("未找到 *_N<grid>/velocity.xdmf，请先运行求解脚本。")
        return
    print("找到 runs:", [r[0] for r in runs])

    # 每条线：axis -> [(run_name, coords, vals)]
    lines = {"x": [], "y": [], "z": []}
    for name, folder in runs:
        res = load_case(folder)
        if res is None:
            print(f"  [skip] {name}: 缺 velocity 文件")
            continue
        mesh, u, t = res
        print(f"  {name}: t={t:.3f}")
        for axis in ["x", "y", "z"]:
            coords, vals = sample_line(mesh, u, axis, AXIS[axis])
            lines[axis].append((name, coords, vals))

    # 写 csv + 画图
    for axis in ["x", "y", "z"]:
        data = lines[axis]
        if not data:
            continue
        coords0 = data[0][1]
        header = [axis] + [f"{name}_u{axis}" for name, _, _ in data]
        rows = np.column_stack([coords0] + [vals for _, _, vals in data])
        csv_path = os.path.join(PLOT, f"line_{axis}.csv")
        np.savetxt(csv_path, rows, delimiter=",", header=",".join(header),
                   comments="", fmt="%.6g")
        print("  写:", csv_path)

        # 画图（叠加不同网格密度；NS 实线 / FSI 虚线）
        plt.figure(figsize=(7, 5))
        colors = plt.cm.tab10(np.linspace(0, 1, len(data)))
        for (name, coords, vals), c in zip(data, colors):
            ls = "-" if name.startswith("ns") else "--"
            plt.plot(coords, vals, color=c, linestyle=ls, marker="",
                     linewidth=1.8, label=name)
        plt.xlabel(f"{axis}")
        plt.ylabel(f"u_{axis}  (t={T_TARGET}s)")
        plt.title(f"demo_341 velocity along center line {axis} @ t={T_TARGET}s")
        plt.grid(True, alpha=0.4)
        plt.legend(loc="best", fontsize=8)
        plt.tight_layout()
        png = os.path.join(PLOT, f"line_{axis}.png")
        plt.savefig(png, dpi=150)
        plt.close()
        print("  画:", png)


if __name__ == "__main__":
    main()
