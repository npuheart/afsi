#!/usr/bin/env python3
"""PyVista figures for the demo_426 tether-beta sweep.

For each beta: run the case, keep its fluid + solid output in its own
directory, then render a figure showing the velocity field, the plates at their
REFERENCE position (black) and at their DISPLACED position (red, magnified),
with the displacement magnitude on a colour bar.

Run:
    python make_beta_figures.py
"""
from __future__ import annotations
import os, shutil, subprocess, sys
from pathlib import Path
import numpy as np
import meshio, h5py
from meshio.xdmf import TimeSeriesReader

HERE = Path(__file__).resolve().parent
DEMO = HERE.parent
RUNNER = "/home/deepseek-harness/afsi/.tools/afsi-run.sh"
STEPS = int(os.environ.get("STEPS", "320"))
BETAS = [float(x) for x in os.environ.get("BETAS", "64,200,600,2000").split(",")]
AMP = float(os.environ.get("AMP", "2"))
SWEEP = HERE / "_beta_sweep"

sys.path.insert(0, str(DEMO))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplcache")


def run_beta(beta):
    from mpi4py import MPI
    out = SWEEP / f"beta{beta:g}"
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update(USE_IMPLICIT_DRAG="0", BETA=repr(beta), DT_FACTOR="0.2",
               SMOKE="1", SMOKE_STEPS=str(STEPS), T_END=repr(STEPS * 0.00625),
               HOME="/home/deepseek-harness/afsi/.home")
    log = out / "run.log"
    with open(log, "w") as fh:
        subprocess.run([RUNNER, "python", "-B", "-u", "main.py"],
                       cwd=str(DEMO), env=env, stdout=fh,
                       stderr=subprocess.STDOUT, check=False)
    src = DEMO / "plot" / "N32_ipcs_smoke"
    for pat in ("velocity.*", "solid_coords.*", "history.csv", "verify.json"):
        for f in src.glob(pat):
            shutil.copy(f, out / f.name)
    return out


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
    g["speed_c"] = np.minimum(g["speed"], 0.4)
    return g


def main():
    import pyvista as _pv
    global pv
    _pv.OFF_SCREEN = True
    if os.environ.get("AFSI_USE_OSMESA") == "1":
        import vtk
        from pyvista import _vtk
        _vtk.vtkRenderWindow = vtk.vtkOSOpenGLRenderWindow
    pv = _pv

    dirs = []
    for b in BETAS:
        d = SWEEP / f"beta{b:g}"
        if not (d / "solid_coords.h5").exists():
            print(f"running beta={b:g} ...")
            run_beta(b)
        dirs.append((b, d))

    figdir = SWEEP / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    for b, d in dirs:
        g = read_fluid(d)
        ref, cur, topo = read_solid(d)
        dep = np.linalg.norm(cur - ref, axis=1)

        pl = pv.Plotter(shape=(1, 2), off_screen=True,
                        window_size=(2300, 1250))
        # left: fluid |u| with displaced plates drawn on top
        pl.subplot(0, 0)
        pl.set_background("white")
        pl.add_mesh(g, scalars="speed_c", cmap="jet", show_edges=False,
                    scalar_bar_args={"title": "|u| (m/s)"})
        pl.add_mesh(pv.PolyData(np.c_[ref, np.zeros(len(ref))]),
                    color="black", point_size=3, render_points_as_spheres=True)
        pl.add_mesh(pv.PolyData(np.c_[ref + AMP * (cur - ref),
                                      np.zeros(len(ref))]),
                    color="#dd2222", point_size=2.5,
                    render_points_as_spheres=True)
        pl.add_text(f"beta={b:g}   black=reference  red=displaced x{AMP:g}",
                    position="upper_left")
        pl.view_xy(); pl.enable_parallel_projection()

        # right: displacement magnitude on the solid
        sm = pv.from_meshio(meshio.Mesh(np.c_[ref, np.zeros(len(ref))],
                                        [("triangle", topo)]))
        sm["disp"] = dep
        pl.subplot(0, 1)
        pl.set_background("white")
        pl.add_mesh(sm, scalars="disp", cmap="turbo", show_edges=False,
                    scalar_bar_args={"title": "|displacement| (m)"})
        pl.add_text(f"max |d| = {dep.max():.4g} m  =  "
                    f"{dep.max()/(0.5/32):.1f} x h/2", position="upper_left")
        pl.view_xy(); pl.enable_parallel_projection()

        out = figdir / f"beta{b:g}.png"
        pl.screenshot(str(out))
        pl.close()
        print(f"wrote {out}  (max|d|={dep.max():.4g}, "
              f"{dep.max()/(0.5/32):.1f} x h/2)")

    # combined figure
    n = len(dirs)
    # fixed frame = fluid box (not autoscaled)
    allx, ally = [], []
    for _b, _d in dirs:
        _r, _c, _ = read_solid(_d)
        allx += [ _r[:,0].min(), _c[:,0].min() ]; ally += [ _r[:,1].min(), _c[:,1].min() ]
        allx += [ _r[:,0].max(), _c[:,0].max() ]; ally += [ _r[:,1].max(), _c[:,1].max() ]
    cfg_cx = 0.5*(min(allx)+max(allx)); cfg_cy = 0.5*(min(ally)+max(ally))
    cfg_half = 0.5*max(max(allx)-min(allx), max(ally)-min(ally)) * 1.05
    print(f"combined figure frame: centre=({cfg_cx:.3g},{cfg_cy:.3g}) half={cfg_half:.3g}")
    pl = pv.Plotter(shape=(1, n), off_screen=True,
                    window_size=(1150 * n, 1300))
    for i, (b, d) in enumerate(dirs):
        g = read_fluid(d)
        ref, cur, topo = read_solid(d)
        dep = np.linalg.norm(cur - ref, axis=1)
        pl.subplot(0, i)
        pl.set_background("white")
        pl.add_mesh(g, scalars="speed_c", cmap="jet", show_edges=False,
                    show_scalar_bar=(i == n - 1),
                    scalar_bar_args={"title": "|u| (m/s)"})
        pl.add_mesh(pv.PolyData(np.c_[ref, np.zeros(len(ref))]), color="black",
                    point_size=3, render_points_as_spheres=True)
        pl.add_mesh(pv.PolyData(np.c_[ref + AMP * (cur - ref),
                                      np.zeros(len(ref))]), color="#dd2222",
                    point_size=2.5, render_points_as_spheres=True)
        pl.add_text(f"beta={b:g}  max|d|={dep.max():.3g}\n"
                    f"({dep.max()/(0.5/32):.0f} x h/2)", position="upper_left")
        pl.view_xy(); pl.enable_parallel_projection()
        # CONSTANT frame across panels: otherwise the camera rescales to fit the
        # drifting plate and the comparison reads backwards
        pl.camera.position = (cfg_cx, cfg_cy, 1.0)
        pl.camera.focal_point = (cfg_cx, cfg_cy, 0.0)
        pl.camera.up = (0.0, 1.0, 0.0)
        pl.camera.parallel_scale = cfg_half
    out = figdir / "beta_sweep_combined.png"
    pl.screenshot(str(out))
    pl.close()
    print("wrote", out)


if __name__ == "__main__":
    main()
