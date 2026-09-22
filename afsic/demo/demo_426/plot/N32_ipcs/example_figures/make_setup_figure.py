#!/usr/bin/env python3
"""Reproduce the benchmark's computational-setup figure for the slanted channel.

Target figure (Grüninger et al., "Computational setup of the slanted channel
flow problem, inclination angle pi/6"):

  * the rectangular box, with equal aspect so the channel runs at a true 30 deg
  * the velocity-magnitude colour map (jet, 0 .. 0.25 in the benchmark)
  * the two channel plates drawn as rows of DOTS -- the Lagrangian markers
  * a white vertical line at x = 0.5 marking where the profile is measured

The velocity field is read from the AFSI XDMF output with meshio (per
demo_424/plot/PLOTTING_GUIDE.md) and drawn with matplotlib, so no OSMesa is
needed.  A version that additionally draws the exact solution for comparison
is written as `paper_setup_with_exact.png`.

Run:
    python make_setup_figure.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from meshio.xdmf import TimeSeriesReader

HERE = Path(__file__).resolve().parent
CASE_DIR = HERE.parent

DEMO_DIR = HERE
for _ in range(6):
    DEMO_DIR = DEMO_DIR.parent
    if (DEMO_DIR / "configuration.py").exists():
        break
sys.path.insert(0, str(DEMO_DIR))
import configuration as cfg          # noqa: E402

X_LINE = 0.5                          # profile-measurement station


def read_last_step(path: Path):
    with TimeSeriesReader(str(path)) as ts:
        points, cells = ts.read_points_cells()
        _, point_data, _ = ts.read_data(ts.num_steps - 1)
    return np.asarray(points), point_data


def load():
    pts, vel = read_last_step(CASE_DIR / "velocity.xdmf")
    u = np.asarray(vel["f"], dtype=float)
    speed = np.linalg.norm(u[:, :2], axis=1)

    # cell connectivity -> triangles for a tripcolor field plot (the fluid mesh
    # is a structured quad grid; split each quad into two triangles)
    import meshio
    with TimeSeriesReader(str(CASE_DIR / "velocity.xdmf")) as ts:
        _, cells = ts.read_points_cells()
    quads = None
    for block in cells:
        if block.type == "quad":
            quads = np.asarray(block.data)
    if quads is None:
        raise RuntimeError("expected quad cells in the fluid mesh")
    tris = np.vstack([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]])
    return pts[:, :2], tris, speed


def marker_points():
    """Lagrangian marker positions along both plates.

    The plates sit at xi = +-D/2; markers are placed at the axial positions the
    IB coupling uses (spacing ds_lag from configuration.py), so the dots show
    the actual marker distribution.
    """
    out = []
    for side in (-1, +1):
        (x0, y0), (x1, y1) = cfg.wall_endpoints(side)
        length = float(np.hypot(x1 - x0, y1 - y0))
        n = max(int(np.ceil(length / cfg.DS_LAG)), 1)
        s = np.linspace(0.0, 1.0, n + 1)
        out.append(np.column_stack([x0 + s * (x1 - x0), y0 + s * (y1 - y0)]))
    return out


def draw(ax, pts, tris, speed, use_exact=False, vmax=None):
    if use_exact:
        ux, uy = cfg.analytic(pts[:, 0], pts[:, 1])
        field = np.hypot(ux, uy)
        label = "|u| exact (m/s)"
    else:
        field = speed
        label = "|u| (m/s)"

    vmax = vmax if vmax is not None else float(cfg.U_MAX_PAPER)
    tpc = ax.tripcolor(pts[:, 0], pts[:, 1], tris, field,
                       shading="gouraud", cmap="jet",
                       norm=Normalize(vmin=0.0, vmax=vmax))
    cb = plt.colorbar(tpc, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label(label, fontsize=11)

    # Lagrangian markers (the benchmark's dots)
    for mk in marker_points():
        ax.plot(mk[:, 0], mk[:, 1], ".", color="white", ms=2.6, zorder=4)

    # profile-measurement station
    ax.axvline(X_LINE, color="white", lw=1.6, zorder=5)

    ax.set_xlim(cfg.X_MIN, cfg.X_MAX)
    ax.set_ylim(cfg.Y_MIN, cfg.Y_MAX)
    ax.set_aspect("equal")
    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("y", fontsize=12)
    ax.grid(False)


def main():
    pts, tris, speed = load()
    print(f"read {len(pts)} points, {len(tris)} triangles; "
          f"|u| max = {speed.max():.4f} m/s")
    print(f"domain x in [{cfg.X_MIN}, {cfg.X_MAX}], "
          f"y in [{cfg.Y_MIN}, {cfg.Y_MAX}], "
          f"markers {sum(len(m) for m in marker_points())}")

    # size the canvas from the domain aspect ratio (equal-aspect axes), so the
    # box fills the figure instead of leaving whitespace
    dx_ = cfg.X_MAX - cfg.X_MIN
    dy_ = cfg.Y_MAX - cfg.Y_MIN
    width = 6.0
    fig, ax = plt.subplots(figsize=(width, width * dy_ / dx_ + 1.1))
    draw(ax, pts, tris, speed)
    ax.set_title(f"slanted channel, theta = {cfg.THETA_DEG:g} deg\n"
                 f"AFSI N={cfg.N}, dt = {cfg.DT_FACTOR:g} dx, "
                 f"t = 20 s", fontsize=12)
    fig.tight_layout()
    out = HERE / "paper_setup.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("wrote", out)

    # side-by-side with the exact solution
    fig, axes = plt.subplots(1, 2, figsize=(2.0 * width,
                                            width * dy_ / dx_ + 1.1))
    draw(axes[0], pts, tris, speed)
    axes[0].set_title("AFSI (numerical)", fontsize=12)
    draw(axes[1], pts, tris, speed, use_exact=True)
    axes[1].set_title("exact solution", fontsize=12)
    fig.tight_layout()
    out2 = HERE / "paper_setup_with_exact.png"
    fig.savefig(out2, dpi=200)
    plt.close(fig)
    print("wrote", out2)


if __name__ == "__main__":
    main()
