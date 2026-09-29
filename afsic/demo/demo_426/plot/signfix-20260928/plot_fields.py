#!/usr/bin/env python
"""demo_426 流场图（PyVista, off-screen）。

用法：
    python plot_fields.py <run_output_dir> [figdir]

读取 <run_output_dir>/velocity.xdmf 的最后一个时间步，渲染：
    field_velocity.png      |u| 伪彩 + 两块板（红线）
    field_streamlines.png   流线（前沿入口播撒）+ 半透明 |u| 背景
依赖：pyvista / meshio（afsi-dolfinx 环境自带）。
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEMO = os.environ.get("DEMO426_DIR",
                      "/Users/pengfei/GitHub/afsi/afsic/demo/demo_426")
sys.path.insert(0, DEMO)
import configuration as cfg  # noqa: E402

import meshio  # noqa: E402
import pyvista as pv  # noqa: E402

CELL_MAP = {
    "quad": pv.CellType.QUAD,
    "triangle": pv.CellType.TRIANGLE,
    "tetra": pv.CellType.TETRA,
    "hexahedron": pv.CellType.HEXAHEDRON,
}


def read_last_step(xdmf_path):
    with meshio.xdmf.TimeSeriesReader(xdmf_path) as reader:
        points, cells = reader.read_points_cells()
        t, pd, cd = None, None, None
        for k in range(reader.num_steps):
            t, pd, cd = reader.read_data(k)
    return t, points, cells, pd, cd


def pick_vector(point_data):
    for k, v in point_data.items():
        v = np.asarray(v)
        if v.ndim == 2 and v.shape[1] in (2, 3):
            return k
    raise KeyError(f"no vector field in {list(point_data)}")


def to_grid(points, cells, point_data):
    pts = np.asarray(points, dtype=float)
    if pts.shape[1] == 2:
        pts = np.c_[pts, np.zeros(len(pts))]
    pv_cells = {CELL_MAP[c.type]: c.data for c in cells}
    grid = pv.UnstructuredGrid(pv_cells, pts)
    for k, v in point_data.items():
        v = np.asarray(v)
        if v.ndim == 1 or (v.ndim == 2 and v.shape[1] == 1):
            grid.point_data[k] = v.reshape(len(pts), -1)[:, 0]
        else:
            vv = (v[:, :3] if v.shape[1] >= 3
                  else np.c_[v, np.zeros((len(v), 3 - v.shape[1]))])
            grid.point_data[k] = vv
    return grid


def add_plates(pl):
    for side in (-1, +1):
        (x0, y0), (x1, y1) = cfg.wall_endpoints(side)
        pl.add_mesh(pv.Line((x0, y0, 0.0), (x1, y1, 0.0)),
                    color="red", line_width=4)


def main():
    run_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    out_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "figures")
    os.makedirs(out_dir, exist_ok=True)

    t, points, cells, pd, _ = read_last_step(
        os.path.join(run_dir, "velocity.xdmf"))
    print("velocity.xdmf fields:", list(pd.keys()), "| t =", t)
    grid = to_grid(points, cells, pd)
    u_name = pick_vector(pd)
    u = np.asarray(pd[u_name], dtype=float)
    print(f"field '{u_name}': shape {u.shape}")
    if u.shape[1] == 2:
        u = np.c_[u, np.zeros(len(u))]
    grid.point_data["umag"] = np.linalg.norm(u, axis=1)
    grid.point_data["u"] = u
    print(f"grid: {grid.n_points} points, |u| max = "
          f"{grid.point_data['umag'].max():.5f} m/s")

    pv.OFF_SCREEN = True

    # --- |u| pseudocolor ----------------------------------------------------
    pl = pv.Plotter(off_screen=True, window_size=(780, 1000))
    pl.add_mesh(grid, scalars="umag", cmap="turbo",
                scalar_bar_args={"title": "|u| [m/s]"})
    add_plates(pl)
    pl.show_grid(color="gray")
    pl.view_xy()
    pl.add_text(f"demo_426  tether + direct load  |u|  (t = {t:.2f} s)",
                font_size=10)
    out1 = os.path.join(out_dir, "field_velocity.png")
    pl.screenshot(out1)
    pl.close()
    print("written:", out1)

    # --- streamlines --------------------------------------------------------
    try:
        yl, yu = cfg.inlet_interval()
        ys = np.linspace(yl + 0.05, yu - 0.05, 21)
        xs = np.full_like(ys, cfg.X_MIN + 0.02)
        src = pv.PolyData(np.c_[xs, ys, np.zeros_like(ys)])
        streams = grid.streamlines_from_source(
            src, vectors="u", integration_direction="forward",
            max_length=30.0, initial_step_length=0.01,
            terminal_speed=1e-9, max_steps=20000)
        print("streamline points:", streams.n_points)
        pl2 = pv.Plotter(off_screen=True, window_size=(780, 1000))
        pl2.add_mesh(grid, scalars="umag", cmap="turbo", opacity=0.45,
                     scalar_bar_args={"title": "|u| [m/s]"})
        if streams.n_points > 0:
            pl2.add_mesh(streams, color="white", line_width=2.5,
                         render_lines_as_tubes=True)
        add_plates(pl2)
        pl2.show_grid(color="gray")
        pl2.view_xy()
        pl2.add_text(f"demo_426  streamlines over |u|  (t = {t:.2f} s)",
                     font_size=10)
        out2 = os.path.join(out_dir, "field_streamlines.png")
        pl2.screenshot(out2)
        pl2.close()
        print("written:", out2)
    except Exception as exc:  # noqa: BLE001
        print("streamlines failed:", repr(exc))


if __name__ == "__main__":
    main()
