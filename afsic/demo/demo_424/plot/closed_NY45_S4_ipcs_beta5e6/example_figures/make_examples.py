#!/usr/bin/env python3
"""Create PyVista example figures for the closed_NY45 AFSI final state.

Reads velocity.xdmf and pressure.xdmf with meshio, converts to a PyVista
UnstructuredGrid, and renders PNGs.

If you are on a headless machine without X/OpenGL, set AFSI_USE_OSMESA=1 and
make sure libOSMesa can be found (e.g. LD_LIBRARY_PATH).  On a normal desktop
or Jupyter setup just run it with python.
"""
from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import meshio
from meshio.xdmf import TimeSeriesReader
import pyvista as pv

pv.OFF_SCREEN = True

if os.environ.get("AFSI_USE_OSMESA") == "1":
    import vtk
    from pyvista import _vtk
    _vtk.vtkRenderWindow = vtk.vtkOSOpenGLRenderWindow

HERE = Path(__file__).resolve().parent
CASE_DIR = HERE.parent
OUT = HERE
CASE_NAME = CASE_DIR.name


def read_series(path: Path):
    """Return points, cells, and a list of (time, point_data, cell_data)."""
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

    # Normalise the variable names used by this AFSI output.
    mesh["velocity"] = np.asarray(mesh["f"], dtype=float)
    p_key = "p" if "p" in mesh.point_data else "p_"
    mesh["pressure"] = np.asarray(mesh[p_key], dtype=float).ravel()
    mesh["speed"] = np.linalg.norm(mesh["velocity"], axis=1)
    mesh.set_active_vectors("velocity")
    return t, mesh


def view_2d(pl: pv.Plotter):
    pl.view_xy()
    pl.enable_parallel_projection()


def blank_axes(pl: pv.Plotter):
    pl.show_axes()
    pl.add_text("x", position="lower_right", font_size=10)


def make_individual(t, mesh, streamlines, seeds):
    # 01 velocity magnitude
    pl = pv.Plotter(off_screen=True, window_size=(1500, 650))
    pl.set_background("white")
    pl.add_mesh(
        mesh,
        scalars="speed",
        cmap="turbo",
        show_edges=False,
        scalar_bar_args={"title": "|u| (m/s)"},
    )
    pl.add_text(f"{CASE_NAME}   t = {t:g} s   velocity magnitude", position="upper_left")
    view_2d(pl)
    pl.screenshot(str(OUT / "01_velocity_magnitude.png"))
    pl.close()

    # 02 pressure
    pl = pv.Plotter(off_screen=True, window_size=(1500, 650))
    pl.set_background("white")
    pl.add_mesh(
        mesh,
        scalars="pressure",
        cmap="coolwarm",
        show_edges=False,
        scalar_bar_args={"title": "p (Pa)"},
    )
    pl.add_text(f"{CASE_NAME}   t = {t:g} s   pressure", position="upper_left")
    view_2d(pl)
    pl.screenshot(str(OUT / "02_pressure.png"))
    pl.close()

    # 03 velocity vectors, subsampled
    idx = np.arange(0, mesh.n_points, 40)
    vec = mesh["velocity"][idx]
    keep = np.linalg.norm(vec, axis=1) > 1e-5
    centers = mesh.points[idx][keep]
    vectors = vec[keep]

    pl = pv.Plotter(off_screen=True, window_size=(1500, 650))
    pl.set_background("white")
    pl.add_mesh(mesh, scalars="speed", cmap="Greys", opacity=0.75, show_edges=False)
    pl.add_arrows(centers, vectors, mag=0.15, color="black")
    pl.add_text(f"{CASE_NAME}   t = {t:g} s   velocity vectors (every 40th node)", position="upper_left")
    view_2d(pl)
    pl.screenshot(str(OUT / "03_velocity_vectors.png"))
    pl.close()

    # 04 streamlines
    pl = pv.Plotter(off_screen=True, window_size=(1500, 650))
    pl.set_background("white")
    pl.add_mesh(mesh, scalars="speed", cmap="Blues", opacity=0.85, show_edges=False)
    if streamlines.n_points:
        pl.add_mesh(streamlines, color="black", line_width=2)
    pl.add_mesh(seeds, color="red", point_size=9, render_points_as_spheres=True)
    pl.add_text(f"{CASE_NAME}   streamlines (red: seed line at x = 1 mm)", position="upper_left")
    view_2d(pl)
    pl.screenshot(str(OUT / "04_streamlines.png"))
    pl.close()

    # 05 x = 0.05 m cross-section (a line in this 2-D case)
    section = mesh.slice(normal="x", origin=(0.05, 0.0, 0.0))
    pl = pv.Plotter(off_screen=True, window_size=(1500, 650))
    pl.set_background("white")
    pl.add_mesh(mesh, color="#ebebeb", show_edges=False)
    if section.n_points:
        pl.add_mesh(
            section,
            scalars="speed",
            cmap="turbo",
            line_width=10,
            render_lines_as_tubes=True,
            scalar_bar_args={"title": "|u| (m/s)"},
        )
    pl.add_text(f"{CASE_NAME}   cross-section x = 0.05 m", position="upper_left")
    view_2d(pl)
    pl.screenshot(str(OUT / "05_cross_section_x.png"))
    pl.close()


def make_overview(t, mesh, streamlines, seeds):
    pl = pv.Plotter(shape=(2, 2), off_screen=True, window_size=(1800, 1100), border=False)

    # (a) speed
    pl.subplot(0, 0)
    pl.set_background("white")
    pl.add_mesh(mesh, scalars="speed", cmap="turbo", show_edges=False, show_scalar_bar=False)
    pl.add_text("(a) |u|", position="upper_left")
    view_2d(pl)

    # (b) pressure
    pl.subplot(0, 1)
    pl.set_background("white")
    pl.add_mesh(mesh, scalars="pressure", cmap="coolwarm", show_edges=False, show_scalar_bar=False)
    pl.add_text("(b) p", position="upper_left")
    view_2d(pl)

    # (c) vectors
    pl.subplot(1, 0)
    pl.set_background("white")
    idx = np.arange(0, mesh.n_points, 40)
    vec = mesh["velocity"][idx]
    keep = np.linalg.norm(vec, axis=1) > 1e-5
    pl.add_mesh(mesh, scalars="speed", cmap="Greys", opacity=0.75, show_edges=False, show_scalar_bar=False)
    pl.add_arrows(mesh.points[idx][keep], vec[keep], mag=0.15, color="black")
    pl.add_text("(c) velocity vectors", position="upper_left")
    view_2d(pl)

    # (d) streamlines
    pl.subplot(1, 1)
    pl.set_background("white")
    pl.add_mesh(mesh, scalars="speed", cmap="Blues", opacity=0.85, show_edges=False, show_scalar_bar=False)
    if streamlines.n_points:
        pl.add_mesh(streamlines, color="black", line_width=2)
    pl.add_mesh(seeds, color="red", point_size=8, render_points_as_spheres=True)
    pl.add_text("(d) streamlines", position="upper_left")
    view_2d(pl)

    pl.screenshot(str(OUT / "00_overview.png"))
    pl.close()


def main():
    t, mesh = load_mesh()
    print(f"Read {CASE_NAME}: t={t:g}, {mesh.n_points} points, {mesh.n_cells} cells")
    print("point arrays:", mesh.array_names)

    # Seed points just downstream of the left boundary, spread over the box height.
    x_seed = mesh.bounds[0] + 0.001
    ys = np.linspace(mesh.bounds[2] + 0.001, mesh.bounds[3] - 0.001, 25)
    seeds = pv.PolyData(np.c_[np.full_like(ys, x_seed), ys, np.zeros_like(ys)])

    streamlines = mesh.streamlines_from_source(
        seeds,
        vectors="velocity",
        integration_direction="forward",
        surface_streamlines=True,
        max_length=(mesh.bounds[1] - mesh.bounds[0]) * 1.2,
    )
    if streamlines.n_points and "velocity" in streamlines.array_names:
        streamlines["speed"] = np.linalg.norm(streamlines["velocity"], axis=1)
    print(f"Streamlines: {streamlines.n_points} points, {streamlines.n_cells} lines")

    make_overview(t, mesh, streamlines, seeds)
    make_individual(t, mesh, streamlines, seeds)
    print("Wrote figures to", OUT)


if __name__ == "__main__":
    main()
