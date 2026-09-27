#!/usr/bin/env python
"""为 demo_402 报告生成 PyVista 图件（读取 plot/full_chorin 的 h5 输出）。

用法（afsi-dolfinx 环境）：
    cd afsic/demo/demo_402
    python report/make_figures.py

输出：report/figures/*.png
"""
import os
import csv

import numpy as np
import h5py
import pyvista as pv

pv.OFF_SCREEN = True

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.normpath(os.path.join(HERE, "..", "plot", "full_chorin"))
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)

VEL = os.path.join(RUN, "velocity.h5")
PRE = os.path.join(RUN, "pressure.h5")
SOL = os.path.join(RUN, "solid.h5")


# ------------------------------------------------------------------ 读取工具 --
def _time_of(key: str) -> float:
    return float(key.replace("_", ".", 1))


def _func_keys(path):
    with h5py.File(path, "r") as h:
        name = list(h["Function"].keys())[0]
        keys = list(h[f"Function/{name}"].keys())
    return name, sorted(keys, key=_time_of)


def read_at(path, t_target: float):
    """读取最接近 t_target 的时刻，返回 (数据数组, 实际时刻)。"""
    name, keys = _func_keys(path)
    key = min(keys, key=lambda k: abs(_time_of(k) - t_target))
    with h5py.File(path, "r") as h:
        data = h[f"Function/{name}/{key}"][:]
    return data, _time_of(key)


def make_grid(path):
    """从 dolfinx 写出的 h5 重建 PyVista 网格（四边形/三角形单元）。"""
    with h5py.File(path, "r") as h:
        topo = h["Mesh/mesh/topology"][:].astype(np.int64)
        geom = h["Mesh/mesh/geometry"][:]
    if geom.shape[1] == 2:
        geom = np.c_[geom, np.zeros(len(geom))]
    n, k = topo.shape
    cells = np.hstack([np.full((n, 1), k, dtype=np.int64), topo]).ravel()
    ctype = {3: pv.CellType.TRIANGLE, 4: pv.CellType.QUAD}[k]
    return pv.UnstructuredGrid(cells, np.full(n, ctype, np.uint8), geom)


# ------------------------------------------------------------------ 数据准备 --
fluid = make_grid(VEL)
solid = make_grid(SOL)

u_end, t_end = read_at(VEL, 10.0)
fluid.point_data["|u|"] = np.linalg.norm(u_end[:, :2], axis=1)

p_end, _ = read_at(PRE, 10.0)
fluid.point_data["p"] = p_end[:, 0]

sc_end, _ = read_at(SOL, 10.0)
sc_ref, _ = read_at(SOL, 0.0)
solid.point_data["|d|"] = np.linalg.norm(sc_end[:, :2] - sc_ref[:, :2], axis=1)
solid.points = np.c_[sc_end[:, :2], np.zeros(len(sc_end))]

# 无标量数据的副本：叠加到流场图上（避免色标被 |d| 数组抢走）
solid_ov = solid.copy()
del solid_ov.point_data["|d|"]
solid_ref = solid.copy()
del solid_ref.point_data["|d|"]
solid_ref.points = np.c_[sc_ref[:, :2], np.zeros(len(sc_ref))]

UM = 200.0                     # 入口平均速度 [cm/s]
CLIM_U = (0.0, 2.0 * UM)       # |u| 色标上限 = 2*Um
print(f"t_end = {t_end:.4f} s, |u| max = {fluid.point_data['|u|'].max():.1f} cm/s")

# ------------------------------------------------------------------- 图 1/2 --
for tag, scal, clim, cmap, title, cb in [
    ("velocity", "|u|", CLIM_U, "turbo",
     f"|u| [cm/s]  (t = {t_end:.2f} s)", "|u| [cm/s]"),
    ("pressure", "p", (-4.0e4, 4.0e4), "RdBu_r",
     f"p [dyne/cm$^2$]  (t = {t_end:.2f} s)", "p"),
]:
    pl = pv.Plotter(window_size=(1600, 420))
    pl.add_mesh(fluid, scalars=scal, cmap=cmap, clim=clim, show_edges=False,
                scalar_bar_args=dict(title=cb, vertical=True, fmt="%.0f",
                                     n_labels=5, position_x=0.86,
                                     position_y=0.15, width=0.03, height=0.7,
                                     title_font_size=14, label_font_size=12))
    pl.add_mesh(solid_ov, color="black", style="wireframe", line_width=1.5)
    pl.view_xy()
    pl.camera.zoom(1.03)
    pl.screenshot(os.path.join(OUT, f"fig_{tag}.png"))
    pl.close()
    print("saved", f"fig_{tag}.png")

# --------------------------------------------------------------------- 图 3 --
pl = pv.Plotter(window_size=(1500, 520))
pl.add_mesh(solid, scalars="|d|", cmap="viridis", show_edges=True,
            line_width=0.4, clim=(0.0, float(solid.point_data["|d|"].max())),
            scalar_bar_args=dict(title="|d| [cm]", vertical=True, fmt="%.2f",
                                 n_labels=5, position_x=0.86,
                                 position_y=0.15, width=0.03, height=0.7,
                                 title_font_size=14, label_font_size=12))
pl.add_mesh(solid_ref, color="gray", style="wireframe", line_width=1.0,
            opacity=0.6)
pl.view_xy()
pl.reset_camera()
pl.screenshot(os.path.join(OUT, "fig_solid.png"))
pl.close()
print("saved fig_solid.png")

# --------------------------------------------------------------------- 图 4 --
snap_t = [2.0, 4.0, 7.0, 10.0]
pl = pv.Plotter(shape=(len(snap_t), 1), window_size=(1500, 260 * len(snap_t)))
for i, t in enumerate(snap_t):
    pl.subplot(i, 0)
    snap = make_grid(VEL)
    u_t, t_real = read_at(VEL, t)
    snap.point_data["|u|"] = np.linalg.norm(u_t[:, :2], axis=1)
    if i == len(snap_t) - 1:
        pl.add_mesh(snap, scalars="|u|", cmap="turbo", clim=CLIM_U,
                    show_edges=False,
                    scalar_bar_args=dict(title="|u| [cm/s]", vertical=False,
                                         fmt="%.0f", n_labels=5,
                                         position_x=0.55, position_y=0.02,
                                         width=0.3, height=0.06,
                                         title_font_size=12,
                                         label_font_size=10))
    else:
        pl.add_mesh(snap, scalars="|u|", cmap="turbo", clim=CLIM_U,
                    show_edges=False, show_scalar_bar=False)
    pl.add_mesh(solid_ov, color="black", style="wireframe", line_width=1.0)
    pl.view_xy()
    pl.add_text(f"t = {t_real:.2f} s", position="upper_left", font_size=11,
                color="black")
pl.screenshot(os.path.join(OUT, "fig_snapshots.png"))
pl.close()
print("saved fig_snapshots.png")

# --------------------------------------------------------------------- 图 5 --
# 尾尖位移时程（PyVista Chart2D，失败则退回 matplotlib）
rows = list(csv.DictReader(open(os.path.join(RUN, "history.csv"))))
t_h = np.array([float(r["t"]) for r in rows])
tipx = np.array([float(r["tip_dx"]) for r in rows])
tipy = np.array([float(r["tip_dy"]) for r in rows])

ok = False
try:
    ch = pv.Chart2D(size=(1400, 620))
    ch.line(t_h, tipx, color="tab:blue", label="tip dx")
    ch.line(t_h, tipy, color="tab:orange", label="tip dy")
    ch.y_label = "tip displacement [cm]"
    ch.x_label = "t [s]"
    ch.legend_visible = True
    ch.to_image(os.path.join(OUT, "fig_tip.png")) if hasattr(ch, "to_image") else None
    ok = os.path.exists(os.path.join(OUT, "fig_tip.png"))
except Exception as exc:  # noqa: BLE001
    print("PyVista Chart2D 不可用:", exc)

if not ok:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(t_h, tipx, label="tip dx")
    ax.plot(t_h, tipy, label="tip dy")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("tip displacement [cm]")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_tip.png"), dpi=120)
    plt.close(fig)
print("saved fig_tip.png")
print("所有图件已输出到", OUT)
