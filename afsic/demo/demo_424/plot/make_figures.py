#!/usr/bin/env python3
"""Render demo_424 field figures with matplotlib.

Follows demo_424/plot/PLOTTING_GUIDE.md: read the AFSI XDMF with meshio's
TimeSeriesReader (VTK's XdmfReader is unreliable on these 2-D XY files), then
draw with matplotlib so the figures work even without PyVista/OSMesa.

The 424 fluid mesh is a structured rectangle, and dolfinx numbers its vertices
along diagonals (i + j = const), so the grid indices are recovered from the
coordinates instead of assuming a reshape.

Usage:
    python make_figures.py                          # the CASES below
    python make_figures.py <case_dir_name> ...
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from meshio.xdmf import TimeSeriesReader

HERE = Path(__file__).resolve().parent

CASES = [
    "open_NY45_S4_ipcs_s4_repeat",
    "closed_NY45_S4_ipcs_s4_repeat",
]
NX, NY = 101, 45          # fluid grid at NY=45 (from configuration.py)


def read_xdmf(path: Path):
    """Return (points, cells, values) for the last time step."""
    with TimeSeriesReader(str(path)) as ts:
        points, cells = ts.read_points_cells()
        _, point_data, _ = ts.read_data(ts.num_steps - 1)
    return np.asarray(points), cells, point_data


def to_grid(points, values, nx=NX, ny=NY):
    xs = np.unique(np.round(points[:, 0], 12))
    ys = np.unique(np.round(points[:, 1], 12))
    if xs.size != nx + 1 or ys.size != ny + 1:
        raise ValueError(f"grid mismatch: {xs.size}x{ys.size}, "
                         f"expected {nx + 1}x{ny + 1}")
    i = np.searchsorted(xs, np.round(points[:, 0], 12))
    j = np.searchsorted(ys, np.round(points[:, 1], 12))
    shape = (ny + 1, nx + 1)
    out = np.full(shape + values.shape[1:], np.nan)
    out[j, i] = values
    X = np.full(shape, np.nan)
    Y = np.full(shape, np.nan)
    X[j, i] = points[:, 0]
    Y[j, i] = points[:, 1]
    return X, Y, out


def load_case(name: str):
    d = HERE / name
    pts, _, vel = read_xdmf(d / "velocity.xdmf")
    _, _, pre = read_xdmf(d / "pressure.xdmf")
    pkey = "p" if "p" in pre else "p_"
    X, Y, U = to_grid(pts, np.asarray(vel["f"], dtype=float))
    _, _, P = to_grid(pts, np.asarray(pre[pkey], dtype=float).reshape(-1, 1))
    return X, Y, np.linalg.norm(U, axis=2), P[:, :, 0], U


def panel(ax, X, Y, F, title, cmap, label):
    pcm = ax.pcolormesh(X, Y, F, cmap=cmap, shading="gouraud")
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x (m)", fontsize=9)
    ax.set_ylabel("y (m)", fontsize=9)
    ax.tick_params(labelsize=8)
    cb = plt.colorbar(pcm, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label(label, fontsize=9)
    cb.ax.tick_params(labelsize=8)


def figure_case(name, X, Y, speed, pres, U):
    out = HERE / name / "example_figures"
    out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(12, 8.5))
    panel(axes[0], X, Y, speed, f"{name}:  |u| (m/s)", "turbo", "|u| (m/s)")
    panel(axes[1], X, Y, pres, f"{name}:  p - mean (Pa)", "coolwarm",
          "p - mean (Pa)")
    panel(axes[2], X, Y, U[:, :, 0], f"{name}:  u_x (m/s)", "RdBu_r",
          "u_x (m/s)")
    fig.suptitle(f"demo_424 {name}   t = 0.4 s", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    p = out / "00_fields_matplotlib.png"
    fig.savefig(p, dpi=180)
    plt.close(fig)
    print("wrote", p)

    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    panel(axes[0], X, Y, speed, f"{name}:  |u| linear", "turbo", "|u| (m/s)")
    panel(axes[1], X, Y, np.log10(np.maximum(speed, 1e-12)),
          f"{name}:  log10 |u|  (flat gap drag => near-zero outer channels)",
          "turbo", "log10 |u|")
    fig.suptitle(f"demo_424 {name}   t = 0.4 s", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    p = out / "01_speed_log_matplotlib.png"
    fig.savefig(p, dpi=180)
    plt.close(fig)
    print("wrote", p)


def figure_compare(loaded):
    out = HERE / "compare"
    out.mkdir(parents=True, exist_ok=True)
    for field, cmap, label, fname in (
            ("speed", "turbo", "|u| (m/s)", "compare_speed.png"),
            ("pres", "coolwarm", "p - mean (Pa)", "compare_pressure.png")):
        names = list(loaded)
        fig, axes = plt.subplots(len(names), 1, figsize=(12, 3.6 * len(names)))
        axes = np.atleast_1d(axes)
        for ax, name in zip(axes, names):
            X, Y, speed, pres, _ = loaded[name]
            panel(ax, X, Y, speed if field == "speed" else pres, name, cmap,
                  label)
        fig.suptitle(f"demo_424 {label}   t = 0.4 s", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        p = out / fname
        fig.savefig(p, dpi=180)
        plt.close(fig)
        print("wrote", p)


def figure_flow_history():
    names = [n for n in CASES if (HERE / n / "flow_history.csv").exists()]
    if not names:
        return
    fig, axes = plt.subplots(2, len(names), figsize=(6.5 * len(names), 7),
                             squeeze=False, sharex=True)
    for c, name in enumerate(names):
        rows = list(csv.DictReader(open(HERE / name / "flow_history.csv")))
        t = np.array([float(r["t"]) for r in rows])
        ax = axes[0][c]
        for key, color, mk in (("Q_gap", "tab:blue", "o"),
                               ("Q_leak_up", "tab:red", "s"),
                               ("Q_leak_dn", "tab:orange", "^")):
            ax.plot(t, np.abs([float(r[key]) for r in rows]), mk + "-", ms=3,
                    color=color, label=f"|{key}|")
        ax.set_yscale("log")
        ax.set_ylabel("|Q| (m$^2$/s)")
        ax.set_title(name, fontsize=10)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8)
        ax = axes[1][c]
        ax.plot(t, [float(r["max_u"]) for r in rows], "o-", ms=3,
                color="tab:green")
        ax.set_xlabel("t (s)")
        ax.set_ylabel("max|u| (m/s)")
        ax.grid(True, alpha=0.25)
    fig.suptitle("demo_424 flow history", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    p = HERE / "compare" / "flow_history.png"
    fig.savefig(p, dpi=180)
    plt.close(fig)
    print("wrote", p)


def main():
    names = sys.argv[1:] or CASES
    loaded = {}
    for name in names:
        if not (HERE / name / "velocity.xdmf").exists():
            print(f"skip {name}: no velocity.xdmf")
            continue
        X, Y, speed, pres, U = load_case(name)
        print(f"{name}: |u|_max={np.nanmax(speed):.4e}  "
              f"p range {np.nanmin(pres):.4f}..{np.nanmax(pres):.4f}")
        loaded[name] = (X, Y, speed, pres, U)
        figure_case(name, X, Y, speed, pres, U)
    if len(loaded) > 1:
        figure_compare(loaded)
    figure_flow_history()


if __name__ == "__main__":
    main()
