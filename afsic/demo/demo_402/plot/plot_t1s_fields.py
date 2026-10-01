#!/usr/bin/env python
"""t=1s 的涡量场快照（论文 Fig.25 风格；h5 直读 + pyvista 离屏渲染）。

用法：conda run -n afsi-dolfinx python plot_t1s_fields.py [run_dir ...]
默认处理 plot/t1s/{chorin,ipcs}。

读取方式与 report/make_figures.py 相同（h5: Mesh/mesh/{topology,geometry} +
Function/<name>/<time>，函数值按顶点存放）。
"""
import os
import sys

import numpy as np
import h5py
import pyvista as pv

pv.OFF_SCREEN = True

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = sys.argv[1:] or [os.path.join(HERE, "t1s", n) for n in ("chorin", "ipcs")]


def _time_of(key):
    return float(key.replace("_", ".", 1))


def read_last(path):
    with h5py.File(path, "r") as h:
        name = list(h["Function"].keys())[0]
        keys = sorted(h[f"Function/{name}"].keys(), key=_time_of)
        key = keys[-1]
        return h[f"Function/{name}/{key}"][:], _time_of(key)


def make_grid(path):
    with h5py.File(path, "r") as h:
        topo = h["Mesh/mesh/topology"][:].astype(np.int64)
        geom = h["Mesh/mesh/geometry"][:]
    if geom.shape[1] == 2:
        geom = np.c_[geom, np.zeros(len(geom))]
    n, k = topo.shape
    cells = np.hstack([np.full((n, 1), k, dtype=np.int64), topo]).ravel()
    ctype = {3: pv.CellType.TRIANGLE, 4: pv.CellType.QUAD}[k]
    return pv.UnstructuredGrid(cells, np.full(n, ctype, np.uint8), geom)


def snapshot(run):
    fluid = make_grid(os.path.join(run, "velocity.h5"))
    u, t = read_last(os.path.join(run, "velocity.h5"))
    fluid.point_data["u"] = u[:, :2]
    deriv = fluid.compute_derivative(scalars="u")
    grad = np.asarray(deriv.point_data["gradient"])
    # pyvista 的梯度按 (du0/dx, du0/dy, du0/dz, du1/dx, du1/dy, du1/dz) 展平
    fluid.point_data["vorticity_z"] = grad[:, 3] - grad[:, 1]

    solid = make_grid(os.path.join(run, "solid.h5"))
    sc, _ = read_last(os.path.join(run, "solid.h5"))
    solid.points = np.c_[sc[:, :2], np.zeros(len(sc))]

    w = fluid.point_data["vorticity_z"]
    clim = float(np.abs(w).max())
    pl = pv.Plotter(off_screen=True, window_size=(1600, 500))
    pl.add_mesh(fluid, scalars="vorticity_z", cmap="RdBu_r", clim=(-clim, clim),
                scalar_bar_args=dict(title="omega_z [1/s]", vertical=True,
                                     position_x=0.88, position_y=0.2,
                                     width=0.03, height=0.6))
    pl.add_mesh(solid, color="black", style="wireframe", line_width=2.0)
    pl.view_xy()
    pl.camera.zoom(1.03)
    pl.add_text(f"{os.path.basename(run)}   t = {t:.2f} s", font_size=12)
    out = os.path.join(run, "vorticity_t_end.png")
    pl.screenshot(out)
    pl.close()
    print(f"-> {out}  (omega_z: {fluid.point_data['vorticity_z'].min():.0f} .. "
          f"{fluid.point_data['vorticity_z'].max():.0f})")

    zoom = fluid.clip_box(bounds=(15, 65, 8, 32, -1, 1), invert=False)
    pl = pv.Plotter(off_screen=True, window_size=(1400, 700))
    pl.add_mesh(zoom, scalars="vorticity_z", cmap="RdBu_r", clim=(-clim, clim),
                scalar_bar_args=dict(title="omega_z [1/s]", vertical=True,
                                     position_x=0.88, position_y=0.2,
                                     width=0.03, height=0.6))
    pl.add_mesh(solid, color="black", style="wireframe", line_width=2.0)
    pl.view_xy()
    pl.reset_camera()
    out = os.path.join(run, "vorticity_t_end_zoom.png")
    pl.screenshot(out)
    pl.close()
    print(f"-> {out}")


def main():
    for run in RUNS:
        if not os.path.exists(os.path.join(run, "velocity.h5")):
            print(f"[skip] {run}: 无 velocity.h5")
            continue
        snapshot(run)


if __name__ == "__main__":
    main()
