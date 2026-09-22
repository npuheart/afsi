#!/usr/bin/env python3
"""Create PyVista + matplotlib figures for demo_426 (slanted channel).

Follows the workflow in demo_424/plot/PLOTTING_GUIDE.md:

  * read the AFSI XDMF with meshio's TimeSeriesReader (VTK's XdmfReader is
    unreliable on these 2-D XY files) and convert to a pyvista grid
  * render off-screen with the OSMesa software GL backend
  * overlay the analytic solution and the channel geometry, which is what this
    benchmark is about

Figures produced in this directory:

  00_overview.png             2x2: |u|, p, velocity vectors, streamlines
  01_velocity_magnitude.png   |u| with the two channel walls drawn
  02_pressure.png             p with the walls drawn
  03_velocity_vectors.png     subsampled vectors
  04_streamlines.png          streamlines seeded across the channel
  05_profile_x.png            |u| across the channel at x=0.5 vs the exact
                              parabola (the benchmark's Fig. 23 comparison)
  06_history_matplotlib.png   max|u| time history (matplotlib, from history.csv)
  index.html                  preview page

Run (headless):
    AFSI_USE_OSMESA=1 \
    LD_LIBRARY_PATH=/home/fenics/spack/opt/spack/linux-ubuntu24.04-icelake/\
gcc-13.3.0/mesa-23.3.6-2kpaaupbkyp44j6jw6urs3y75yvr6ca7/lib \
    python make_examples.py
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import numpy as np
import meshio
from meshio.xdmf import TimeSeriesReader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- make the demo's configuration importable (analytic solution, geometry) --
HERE = Path(__file__).resolve().parent
CASE_DIR = HERE.parent               # plot/N32_ipcs/

# configuration.py lives in the demo root, a variable number of levels up
# (plot/<case>/example_figures/ -> demo_<n>/), so search upward for it.
DEMO_DIR = HERE
for _ in range(6):
    DEMO_DIR = DEMO_DIR.parent
    if (DEMO_DIR / "configuration.py").exists():
        break
else:
    raise RuntimeError("could not locate configuration.py above "
                       f"{HERE}")
sys.path.insert(0, str(DEMO_DIR))
import configuration as cfg          # noqa: E402

pv = None


def _init_pyvista():
    global pv
    import pyvista as _pv
    _pv.OFF_SCREEN = True
    if os.environ.get("AFSI_USE_OSMESA") == "1":
        import vtk
        from pyvista import _vtk
        _vtk.vtkRenderWindow = vtk.vtkOSOpenGLRenderWindow
    pv = _pv


X_PROFILE = 0.5


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def read_series(path: Path):
    with TimeSeriesReader(str(path)) as ts:
        points, cells = ts.read_points_cells()
        steps = [ts.read_data(k) for k in range(ts.num_steps)]
    return np.asarray(points), cells, steps


def load_mesh():
    points, cells, vel_steps = read_series(CASE_DIR / "velocity.xdmf")
    _, _, pres_steps = read_series(CASE_DIR / "pressure.xdmf")

    t = vel_steps[-1][0]
    point_data = dict(vel_steps[-1][1])
    point_data.update(pres_steps[-1][1])

    mesh = pv.from_meshio(meshio.Mesh(points, cells, point_data=point_data))
    mesh["velocity"] = np.asarray(mesh["f"], dtype=float)
    p_key = "p" if "p" in mesh.point_data else "p_"
    mesh["pressure"] = np.asarray(mesh[p_key], dtype=float).ravel()
    mesh["speed"] = np.linalg.norm(mesh["velocity"], axis=1)
    # IPCS accumulates the pressure increment with no Dirichlet datum anywhere
    # in this case (the flow is driven by the body force), so p_ is fixed only
    # up to an additive constant -- it comes out around +3.25e13 while its
    # standard deviation is ~0.46 and its range ~12.9, all of which are
    # physical.  Remove that arbitrary offset for display; the *variation* is
    # what is meaningful.
    raw = np.asarray(mesh["pressure"], dtype=float)
    mesh["pressure"] = raw - raw.mean()
    print(f"pressure: raw mean {raw.mean():.6e} removed; "
          f"std {raw.std():.4e}, range {raw.max() - raw.min():.4e}")
    mesh.set_active_vectors("velocity")
    return t, mesh


def wall_polylines():
    """The two channel walls as (x, y) arrays of pyvista PolyData."""
    out = []
    for side in (-1, +1):
        (x0, y0), (x1, y1) = cfg.wall_endpoints(side)
        x = np.linspace(x0, x1, 200)
        y = np.interp(x, [x0, x1], [y0, y1])
        out.append(pv.PolyData(np.c_[x, y, np.zeros_like(x)]))
    return out


def view_2d(pl):
    pl.view_xy()
    pl.enable_parallel_projection()


def add_walls(pl, color="black", width=3):
    for line in wall_polylines():
        pl.add_mesh(line, color=color, line_width=width)


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------
def make_individual(t, mesh, streamlines, seeds):
    walls = wall_polylines()

    # 01 |u|
    pl = pv.Plotter(off_screen=True, window_size=(1500, 700))
    pl.set_background("white")
    pl.add_mesh(mesh, scalars="speed", cmap="turbo", show_edges=False,
                scalar_bar_args={"title": "|u| (m/s)"})
    add_walls(pl)
    pl.add_text(f"demo_426  N={cfg.N}  theta={cfg.THETA_DEG:g} deg   "
                f"|u|   t={t:g}", position="upper_left")
    view_2d(pl)
    pl.screenshot(str(HERE / "01_velocity_magnitude.png"))
    pl.close()

    # 02 p
    pl = pv.Plotter(off_screen=True, window_size=(1500, 700))
    pl.set_background("white")
    pl.add_mesh(mesh, scalars="pressure", cmap="coolwarm", show_edges=False,
                scalar_bar_args={"title": "p - mean (Pa)"})
    add_walls(pl)
    pl.add_text(f"demo_426   pressure (mean removed)   t={t:g}", position="upper_left")
    view_2d(pl)
    pl.screenshot(str(HERE / "02_pressure.png"))
    pl.close()

    # 03 vectors
    idx = np.arange(0, mesh.n_points, 12)
    vec = mesh["velocity"][idx]
    keep = np.linalg.norm(vec, axis=1) > 1e-3
    pl = pv.Plotter(off_screen=True, window_size=(1500, 700))
    pl.set_background("white")
    pl.add_mesh(mesh, scalars="speed", cmap="Greys", opacity=0.7,
                show_edges=False)
    add_walls(pl)
    if keep.any():
        pl.add_arrows(mesh.points[idx][keep], vec[keep], mag=0.6, color="black")
    pl.add_text(f"demo_426  velocity vectors (every 12th node)",
                position="upper_left")
    view_2d(pl)
    pl.screenshot(str(HERE / "03_velocity_vectors.png"))
    pl.close()

    # 04 streamlines
    pl = pv.Plotter(off_screen=True, window_size=(1500, 700))
    pl.set_background("white")
    pl.add_mesh(mesh, scalars="speed", cmap="Blues", opacity=0.85,
                show_edges=False)
    if streamlines.n_points:
        pl.add_mesh(streamlines, color="black", line_width=2)
    add_walls(pl, color="red", width=4)
    pl.add_text("demo_426  streamlines (red: channel walls)",
                position="upper_left")
    view_2d(pl)
    pl.screenshot(str(HERE / "04_streamlines.png"))
    pl.close()


def make_overview(t, mesh, streamlines, seeds):
    pl = pv.Plotter(shape=(2, 2), off_screen=True, window_size=(1800, 1200),
                    border=False)
    for (i, j), (scal, cmap, label) in zip(
            ((0, 0), (0, 1)),
            (("speed", "turbo", "(a) |u| (m/s)"),
             ("pressure", "coolwarm", "(b) p - mean (Pa)"))):
        pl.subplot(i, j)
        pl.set_background("white")
        pl.add_mesh(mesh, scalars=scal, cmap=cmap, show_edges=False)
        add_walls(pl)
        pl.add_text(label, position="upper_left")
        view_2d(pl)

    pl.subplot(1, 0)
    pl.set_background("white")
    idx = np.arange(0, mesh.n_points, 12)
    vec = mesh["velocity"][idx]
    keep = np.linalg.norm(vec, axis=1) > 1e-3
    pl.add_mesh(mesh, scalars="speed", cmap="Greys", opacity=0.7,
                show_edges=False, show_scalar_bar=False)
    add_walls(pl)
    if keep.any():
        pl.add_arrows(mesh.points[idx][keep], vec[keep], mag=0.6, color="black")
    pl.add_text("(c) velocity vectors", position="upper_left")
    view_2d(pl)

    pl.subplot(1, 1)
    pl.set_background("white")
    pl.add_mesh(mesh, scalars="speed", cmap="Blues", opacity=0.85,
                show_edges=False, show_scalar_bar=False)
    if streamlines.n_points:
        pl.add_mesh(streamlines, color="black", line_width=2)
    add_walls(pl, color="red", width=4)
    pl.add_text("(d) streamlines", position="upper_left")
    view_2d(pl)

    pl.screenshot(str(HERE / "00_overview.png"))
    pl.close()


def sample_at(mesh, x, y):
    """Nearest-node sampling of |u| on the (structured) fluid grid."""
    pts = mesh.points
    d = (pts[:, 0] - x) ** 2 + (pts[:, 1] - y) ** 2
    i = int(np.argmin(d))
    return float(mesh["speed"][i]), mesh["velocity"][i]


def make_profile(mesh):
    """|u| across the channel at x = X_PROFILE vs the exact parabola."""
    t = np.linspace(-cfg.R_HALF, cfg.R_HALF, 161)
    xs = X_PROFILE * cfg.COS_T - t * cfg.SIN_T
    ys = X_PROFILE * cfg.SIN_T + t * cfg.COS_T
    num = np.array([sample_at(mesh, xi, yi)[0] for xi, yi in zip(xs, ys)])
    ux, uy = cfg.analytic(xs, ys)
    exact = np.hypot(ux, uy)
    rel = np.linalg.norm(num - exact) / np.linalg.norm(exact)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 5.2),
                                  gridspec_kw={"width_ratios": [2, 1]})
    ax.plot(exact, t, "k-", lw=2.5, label="exact (plane Poiseuille)")
    ax.plot(num, t, "o", ms=5, color="tab:red",
            label=f"AFSI N={cfg.N} (relative L2 {rel:.2%})")
    ax.axhline(0.0, color="grey", lw=0.8, ls=":")
    ax.set_xlabel("|u| (m/s)")
    ax.set_ylabel(r"$\xi$ across the channel (m)")
    ax.set_title(f"demo_426 profile at x={X_PROFILE:g}  "
                 f"(benchmark Fig. 23 equivalent)")
    ax.grid(alpha=0.3)
    ax.legend()

    e = num - exact
    ax2.plot(e, t, "o-", ms=4, color="tab:blue")
    ax2.axvline(0.0, color="k", lw=0.8)
    ax2.set_xlabel("|u|_num - |u|_exact (m/s)")
    ax2.set_title(f"error   Linf = {np.max(np.abs(e)):.3e}")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    out = HERE / "05_profile_x.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("wrote", out, f"(relative L2 {rel:.4%})")
    return rel


def make_history():
    src = CASE_DIR / "history.csv"
    if not src.exists():
        print("no history.csv, skipping history figure")
        return
    rows = list(csv.DictReader(open(src)))
    t = np.array([float(r["t"]) for r in rows])
    umax = np.array([float(r["max_u"]) for r in rows])
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t, umax, "-", lw=1.6, color="tab:green")
    ax.axhline(cfg.U_MAX, color="k", ls="--", lw=1.2,
               label=f"analytic u_max = {cfg.U_MAX:.4f} m/s")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("max|u| (m/s)")
    ax.set_title(f"demo_426 N={cfg.N}: global maximum velocity")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = HERE / "06_history_matplotlib.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("wrote", out)


def write_index():
    pngs = sorted(p.name for p in HERE.glob("*.png"))
    html = ["<!DOCTYPE html><html><head><meta charset='utf-8'>",
            "<title>demo_426 figures</title>",
            "<style>body{font-family:sans-serif;margin:24px}"
            "img{max-width:100%;border:1px solid #ddd;margin:12px 0}"
            "h2{font-size:15px;margin:4px 0}</style></head><body>",
            "<h1>demo_426 — slanted channel (2-D IB)</h1>"]
    for p in pngs:
        html += [f"<h2>{p}</h2>", f"<img src='{p}'>"]
    html.append("</body></html>")
    (HERE / "index.html").write_text("\n".join(html))
    print(f"wrote index.html ({len(pngs)} images)")


def main():
    _init_pyvista()
    t, mesh = load_mesh()
    print(f"read t={t:g}: {mesh.n_points} points, {mesh.n_cells} cells")
    print("point arrays:", mesh.array_names)

    # seed streamlines just inside the inlet, across the channel only
    y_lo, y_hi = cfg.inlet_interval()
    x_seed = cfg.X_MIN + 1e-3
    ys = np.linspace(y_lo + 0.05, y_hi - 0.05, 21)
    seeds = pv.PolyData(np.c_[np.full_like(ys, x_seed), ys,
                              np.zeros_like(ys)])
    streamlines = mesh.streamlines_from_source(
        seeds, vectors="velocity", integration_direction="forward",
        surface_streamlines=True,
        max_length=(mesh.bounds[3] - mesh.bounds[2]) * 2.0,
    )
    print(f"streamlines: {streamlines.n_points} points")

    make_overview(t, mesh, streamlines, seeds)
    make_individual(t, mesh, streamlines, seeds)
    make_profile(mesh)
    make_history()
    write_index()


if __name__ == "__main__":
    main()
