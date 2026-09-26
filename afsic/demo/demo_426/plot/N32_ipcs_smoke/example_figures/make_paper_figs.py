#!/usr/bin/env python3
"""PyVista figures for the demo_426 tether run (2-D triangular plates).

Uses meshio to read the AFSI XDMF (per demo_424/plot/PLOTTING_GUIDE.md) and
renders off-screen with OSMesa.  Draws the fluid |u| field, the two plates, the
Lagrangian markers and the x=0.5 measurement line.
"""
from __future__ import annotations
import os, sys
from pathlib import Path
import numpy as np, meshio
from meshio.xdmf import TimeSeriesReader

HERE = Path(__file__).resolve().parent
CASE = HERE.parent
DEMO = HERE
for _ in range(6):
    DEMO = DEMO.parent
    if (DEMO / "configuration.py").exists():
        break
sys.path.insert(0, str(DEMO))
import configuration as cfg

import pyvista as pv
pv.OFF_SCREEN = True
if os.environ.get("AFSI_USE_OSMESA") == "1":
    import vtk
    from pyvista import _vtk
    _vtk.vtkRenderWindow = vtk.vtkOSOpenGLRenderWindow

X_LINE = 0.5


def load():
    with TimeSeriesReader(str(CASE / "velocity.xdmf")) as ts:
        pts, cells = ts.read_points_cells()
        _, pd, _ = ts.read_data(ts.num_steps - 1)
    m = pv.from_meshio(meshio.Mesh(pts, cells, point_data=pd))
    m["velocity"] = np.asarray(m["f"], dtype=float)
    m["speed"] = np.linalg.norm(m["velocity"], axis=1)
    m.set_active_vectors("velocity")
    # plateau the colour scale at twice the analytic u_max so the 0..0.25-ish
    # range the benchmark uses is comparable
    m["speed_c"] = np.minimum(m["speed"], 0.4)
    return m


def plate_lines():
    out = []
    for side in (-1, +1):
        (x0, y0), (x1, y1) = cfg.wall_endpoints(side)
        x = np.linspace(x0, x1, 400)
        y = np.interp(x, [x0, x1], [y0, y1])
        out.append(pv.PolyData(np.c_[x, y, np.zeros_like(x)]))
    return out


def markers():
    """Lagrangian marker positions, from the solid mesh, in the reference frame."""
    from dolfinx.io import XDMFFile
    from mpi4py import MPI
    import dolfinx
    with XDMFFile(MPI.COMM_WORLD, cfg.solid_mesh_path(), "r") as f:
        st = f.read_mesh(name="mesh")
    x = st.geometry.x[:, :2]
    return pv.PolyData(np.c_[x, np.zeros(len(x))])


def load_solid():
    """Reference and displaced plate geometry from the solid XDMF/HDF5 output.

    Read straight from HDF5: meshio's TimeSeriesReader refuses this file, and
    the layout is simple (geometry/topology under /Mesh, the nodal vector under
    /Function/<name>/<time>).
    """
    import h5py
    with h5py.File(CASE / "solid_coords.h5", "r") as f:
        ref = np.asarray(f["Mesh/mesh/geometry"], dtype=float)
        topo = np.asarray(f["Mesh/mesh/topology"], dtype=np.int64)
        key = list(f["Function/solid_coords_io"].keys())[-1]
        cur = np.asarray(f["Function/solid_coords_io"][key], dtype=float)[:, :2]
    cells = [("triangle", topo)]
    g = pv.from_meshio(meshio.Mesh(np.c_[ref, np.zeros(len(ref))], cells))
    g["disp"] = np.linalg.norm(cur - ref, axis=1)
    g["disp_vec"] = np.c_[cur - ref, np.zeros(len(ref))]
    return g, ref, cur


def make_solid_figure():
    g, ref, cur = load_solid()
    print(f"solid: {len(ref)} nodes, max|disp| = {g['disp'].max():.4g}, "
          f"mean = {g['disp'].mean():.4g}")
    scale = float(os.environ.get("WARP", "1"))

    pl = pv.Plotter(shape=(1, 2), off_screen=True, window_size=(2200, 1150))

    # (a) displacement magnitude on the plates + reference outlines
    pl.subplot(0, 0)
    pl.set_background("white")
    pl.add_mesh(g, scalars="disp", cmap="turbo", show_edges=False,
                scalar_bar_args={"title": "|displacement| (m)"})
    for side in (-1, +1):
        (x0, y0), (x1, y1) = cfg.wall_endpoints(side)
        xs = np.linspace(x0, x1, 300)
        ys = np.interp(xs, [x0, x1], [y0, y1])
        pl.add_mesh(pv.PolyData(np.c_[xs, ys, np.zeros_like(xs)]),
                    color="black", line_width=4)
    pl.add_text(f"plates: |d|,  black = reference (X_ref)",
                position="upper_left")
    pl.view_xy(); pl.enable_parallel_projection()

    # (b) warped shape: how far the plates have drifted
    pl.subplot(0, 1)
    pl.set_background("white")
    w = g.warp_by_vector("disp_vec", factor=scale) if scale != 1 else g
    pl.add_mesh(w, color="#cc3333", opacity=0.9)
    for side in (-1, +1):
        (x0, y0), (x1, y1) = cfg.wall_endpoints(side)
        xs = np.linspace(x0, x1, 300)
        ys = np.interp(xs, [x0, x1], [y0, y1])
        pl.add_mesh(pv.PolyData(np.c_[xs, ys, np.zeros_like(xs)]),
                    color="black", line_width=4)
    pl.add_text(f"displaced (warp x{scale:g}) vs reference (black)",
                position="upper_left")
    pl.view_xy(); pl.enable_parallel_projection()

    pl.screenshot(str(HERE / "02_solid_displacement_pyvista.png"))
    pl.close()
    print("wrote 02_solid_displacement_pyvista.png")

    # magnified view: the drift is only ~5% of the plate length, so overlay the
    # reference and the DISPLACED shapes with an amplification factor
    amp = float(os.environ.get("AMP", "10"))
    pl = pv.Plotter(off_screen=True, window_size=(2000, 900))
    pl.set_background("white")
    ref3 = np.c_[ref, np.zeros(len(ref))]
    moved3 = np.c_[ref + amp * (cur - ref), np.zeros(len(ref))]
    pl.add_mesh(pv.PolyData(ref3), color="black", point_size=3.0,
                render_points_as_spheres=True)
    pl.add_mesh(pv.PolyData(moved3), color="#dd2222", point_size=2.5,
                render_points_as_spheres=True)
    # sparse arrows: every 40th node, drawn at the true (already x amp) offset
    sel = np.arange(0, len(ref), 40)
    endpoints = np.c_[ref[sel] + amp * (cur[sel] - ref[sel]),
                      np.zeros(len(sel))]
    pl.add_mesh(pv.PolyData(endpoints), color="#1f77b4", point_size=5.0,
                render_points_as_spheres=True)
    pl.add_text(f"black = reference (X_ref)   red = displaced (x{amp:g})   "
                f"blue dots = same nodes on the displaced plate\n"
                f"max |d| = {g['disp'].max():.4g} m   mean = {g['disp'].mean():.4g} m"
                f"   (h/2 = {0.5*cfg.DX:.4g} m)",
                position="upper_left", font_size=11)
    pl.view_xy(); pl.enable_parallel_projection()
    pl.camera.zoom(1.25)
    pl.screenshot(str(HERE / "03_solid_displacement_magnified.png"))
    pl.close()
    print("wrote 03_solid_displacement_magnified.png")


def main():
    m = load()
    print(f"{m.n_points} nodes, {m.n_cells} cells; |u|max={m['speed'].max():.4f}")
    lines, mk = plate_lines(), markers()

    pl = pv.Plotter(off_screen=True, window_size=(1150, 1500))
    pl.set_background("white")
    pl.add_mesh(m, scalars="speed_c", cmap="jet", show_edges=False,
                scalar_bar_args={"title": "|u| (m/s)"})
    for ln in lines:
        pl.add_mesh(ln, color="#222222", line_width=4)
    pl.add_mesh(mk, color="white", point_size=2.0,
                render_points_as_spheres=True)
    pl.add_mesh(pv.PolyData(np.array([[X_LINE, cfg.Y_MIN, 0.],
                                      [X_LINE, cfg.Y_MAX, 0.]])),
                color="white", line_width=3)
    pl.view_xy(); pl.enable_parallel_projection()
    pl.add_text(f"demo_426  theta={cfg.THETA_DEG:g} deg  N={cfg.N}  "
                f"beta={os.environ.get('BETA', 'default')}")
    pl.screenshot(str(HERE / "00_setup_pyvista.png"))
    pl.close()
    print("wrote 00_setup_pyvista.png")

    # overview 2x2
    pl = pv.Plotter(shape=(1, 2), off_screen=True, window_size=(2200, 1100))
    pl.subplot(0, 0)
    pl.set_background("white")
    pl.add_mesh(m, scalars="speed_c", cmap="jet", show_edges=False)
    for ln in lines: pl.add_mesh(ln, color="#222222", line_width=4)
    pl.add_text("|u| with 2-D triangular plates", position="upper_left")
    pl.view_xy(); pl.enable_parallel_projection()
    pl.subplot(0, 1)
    pl.set_background("white")
    idx = np.arange(0, m.n_points, 22)
    vec = m["velocity"][idx]
    keep = np.linalg.norm(vec, axis=1) > 1e-3
    pl.add_mesh(m, scalars="speed_c", cmap="Greys", opacity=0.75,
                show_edges=False, show_scalar_bar=False)
    for ln in lines: pl.add_mesh(ln, color="red", line_width=4)
    if keep.any():
        pl.add_arrows(m.points[idx][keep], vec[keep], mag=0.4, color="black")
    pl.add_text("velocity vectors", position="upper_left")
    pl.view_xy(); pl.enable_parallel_projection()
    pl.screenshot(str(HERE / "01_overview_pyvista.png"))
    pl.close()
    print("wrote 01_overview_pyvista.png")
    make_solid_figure()


if __name__ == "__main__":
    main()
