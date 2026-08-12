"""demo_343 后处理：圆盘中心线速度采样 + 画图。

读取 plot/velocity.xdmf（demo_343 运行输出），沿两个圆盘中心的水平线：
    y = 0.5  （圆盘1 中心） → 采样 u_x → line_disk1.csv / .png
    y = 1.1  （圆盘2 中心） → 采样 u_x → line_disk2.csv / .png

运行：
    conda activate afsi-dolfinx
    python plot_centerline.py
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
NP = 201
DISK_Y = [0.5, 1.1]   # 圆盘中心 y 坐标（generate_mesh.py: Circle1/Circle2）


def kt(k):
    return float(k.replace("_", "."))


def load_last(folder):
    """读取某 run 最后一个时间步的速度场。返回 (mesh, u)。"""
    xdmf = os.path.join(folder, "velocity.xdmf")
    h5f = os.path.join(folder, "velocity.h5")
    with XDMFFile(comm, xdmf, "r") as f:
        mesh = f.read_mesh()
    V = functionspace(mesh, element("Lagrange", mesh.topology.cell_name(), 1, shape=(2,)))
    u = Function(V)
    with h5py.File(h5f, "r") as h:
        keys = sorted(h["Function/f"].keys(), key=kt)
        last = keys[-1]
        u.x.array[:] = h[f"Function/f/{last}"][:][:, :2].ravel()  # h5 存 3 分量，取前 2
    return mesh, u, kt(last)


def sample_horizontal(mesh, u, y0):
    """沿 y=y0 水平线采样 u_x。"""
    pts = np.zeros((NP, 3))
    xs = np.linspace(0.0, 8.0, NP)   # Lx = 8
    pts[:, 0] = xs
    pts[:, 1] = y0
    tree = dg.bb_tree(mesh, mesh.topology.dim)
    vals = np.full(NP, np.nan)
    for i in range(NP):
        q = pts[i:i + 1]
        cand = dg.compute_collisions_points(tree, q)
        col = dg.compute_colliding_cells(mesh, cand, q)
        links = col.links(0)
        if len(links) > 0:
            uu = u.eval(q, np.array([links[0]], dtype=np.int32))
            vals[i] = np.ravel(uu)[0]   # u_x 分量
    return xs, vals


def main():
    # 优先遍历子目录（circle1/有圆盘、circle0/无圆盘对照）；根目录仅作后备
    folders = [os.path.dirname(v) for v in
               glob.glob(os.path.join(PLOT, "*", "velocity.xdmf"))]
    if not folders and os.path.exists(os.path.join(PLOT, "velocity.xdmf")):
        folders = [PLOT]
    folders = [f for f in folders if os.path.exists(os.path.join(f, "velocity.xdmf"))]
    if not folders:
        print("未找到 velocity.xdmf，请先运行 fsi_paralell.py")
        return

    # 汇总各 run 数据
    for y0 in DISK_Y:
        runs = []
        for folder in folders:
            name = os.path.basename(folder) if os.path.dirname(folder) == PLOT else "plot"
            mesh, u, t = load_last(folder)
            xs, vals = sample_horizontal(mesh, u, y0)
            runs.append((name, xs, vals, t))
            print(f"  y={y0}: {name} t={t:.3f}")

        # 写 csv
        header = "x," + ",".join(f"{n}_ux" for n, _, _, _ in runs)
        rows = np.column_stack([runs[0][1]] + [v for _, _, v, _ in runs])
        csv_path = os.path.join(PLOT, f"line_disk{y0}.csv")
        np.savetxt(csv_path, rows, delimiter=",", header=header,
                   comments="", fmt="%.6g")
        print("  写:", csv_path)

        # 画图
        plt.figure(figsize=(8, 5))
        colors = plt.cm.tab10(np.linspace(0, 1, len(runs)))
        for (name, xs, vals, t), c in zip(runs, colors):
            plt.plot(xs, vals, color=c, linewidth=1.8, label=f"{name} (t={t:.2f}s)")
        plt.xlabel("x")
        plt.ylabel("u_x")
        plt.title(f"demo_343 velocity along disk center line y={y0}")
        plt.grid(True, alpha=0.4)
        plt.legend(loc="best", fontsize=9)
        plt.tight_layout()
        png = os.path.join(PLOT, f"line_disk{y0}.png")
        plt.savefig(png, dpi=150)
        plt.close()
        print("  画:", png)


if __name__ == "__main__":
    main()
