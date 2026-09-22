#!/usr/bin/env python3
"""PyVista figures for the CGS + inflow-ramp runs of demo_426.

One figure per kappa: fluid |u| with the plates at their reference position
(black) and displaced (red, magnified), plus the displacement magnitude on the
plates themselves.
"""
from __future__ import annotations
import os, sys
from pathlib import Path
import numpy as np, meshio, h5py
from meshio.xdmf import TimeSeriesReader

HERE = Path(__file__).resolve().parent
DEMO = HERE.parent
sys.path.insert(0, str(DEMO))
import configuration as cfg

import pyvista as pv
pv.OFF_SCREEN = True
if os.environ.get("AFSI_USE_OSMESA") == "1":
    import vtk
    from pyvista import _vtk
    _vtk.vtkRenderWindow = vtk.vtkOSOpenGLRenderWindow

AMP = float(os.environ.get("AMP", "2"))
def _key(p):
    try:
        return (0, float(p.name.replace("kappa", "")))
    except ValueError:
        return (1, p.name)


DIRS = sorted((HERE / "cgs_bcramp").glob("kappa*"), key=_key)


def read_solid(d):
    with h5py.File(d / "solid_coords.h5", "r") as f:
        ref = np.asarray(f["Mesh/mesh/geometry"], float)
        topo = np.asarray(f["Mesh/mesh/topology"], np.int64)
        key = list(f["Function/solid_coords_io"].keys())[-1]
        cur = np.asarray(f["Function/solid_coords_io"][key], float)[:, :2]
    return ref, cur, topo


def read_fluid(d):
    with TimeSeriesReader(str(d / "velocity.xdmf")) as ts:
        pts, cells = ts.read_points_cells()
        _, pd, _ = ts.read_data(ts.num_steps - 1)
    g = pv.from_meshio(meshio.Mesh(pts, cells, point_data=pd))
    g["speed"] = np.linalg.norm(np.asarray(g["f"], float), axis=1)
    return g


def main():
    h2 = 0.5 * cfg.DX
    for d in DIRS:
        kap = d.name
        g = read_fluid(d)
        ref, cur, topo = read_solid(d)
        dep = np.linalg.norm(cur - ref, axis=1)

        pl = pv.Plotter(shape=(1, 2), off_screen=True, window_size=(2300, 1250))
        pl.subplot(0, 0)
        pl.set_background("white")
        pl.add_mesh(g, scalars="speed", cmap="jet", show_edges=False,
                    scalar_bar_args={"title": "|u| (cm/s)"})
        pl.add_mesh(pv.PolyData(np.c_[ref, np.zeros(len(ref))]), color="black",
                    point_size=3, render_points_as_spheres=True)
        pl.add_mesh(pv.PolyData(np.c_[ref + AMP * (cur - ref),
                                      np.zeros(len(ref))]), color="#dd2222",
                    point_size=2.5, render_points_as_spheres=True)
        pl.add_text(f"demo_426  CGS  {kap}  inflow ramp\n"
                    f"black=reference  red=displaced x{AMP:g}",
                    position="upper_left")
        pl.view_xy(); pl.enable_parallel_projection()

        sm = pv.from_meshio(meshio.Mesh(np.c_[ref, np.zeros(len(ref))],
                                        [("triangle", topo)]))
        sm["disp"] = dep
        pl.subplot(0, 1)
        pl.set_background("white")
        pl.add_mesh(sm, scalars="disp", cmap="turbo", show_edges=False,
                    scalar_bar_args={"title": "|displacement| (cm)"})
        pl.add_text(f"max |d| = {dep.max():.4g} cm  =  "
                    f"{dep.max()/h2:.1f} x (h/2)   [criterion 1.0]",
                    position="upper_left")
        pl.view_xy(); pl.enable_parallel_projection()

        out = HERE / "cgs_bcramp" / f"figure_{kap}.png"
        pl.screenshot(str(out)); pl.close()
        print(f"wrote {out}  max|d|={dep.max():.4g} "
              f"({dep.max()/h2:.1f} x h/2)")


if __name__ == "__main__":
    main()
