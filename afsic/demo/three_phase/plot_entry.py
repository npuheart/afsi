"""Create a water-entry sequence, history plot and optional GIF from saved states."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.tri as mtri
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--gif", action="store_true")
    args = parser.parse_args()
    p = json.loads((args.output/"parameters.json").read_text())
    if p["case"] != "water-entry":
        parser.error("This plotter requires a water-entry run")
    paths = sorted(args.output.glob("snapshot_*.npz"))
    if not paths:
        parser.error("No snapshots found; run main.py with --snapshots")
    states = []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            states.append({key: data[key].copy() for key in ("time", "phi", "solid_positions", "solid_cells", "nominal_entry_time")})
    times = np.array([float(s["time"]) for s in states])
    entry = float(states[-1]["nominal_entry_time"])
    x, y = np.meshgrid((np.arange(p["nx"])+.5)*p["lx"]/p["nx"],
                       (np.arange(p["ny"])+.5)*p["ly"]/p["ny"])
    cmap = ListedColormap(["#f4f6f8", "#55b9e8"])

    def panel(ax, state):
        ax.clear()
        ax.pcolormesh(x, y, (state["phi"] < 0).astype(float), cmap=cmap, vmin=0, vmax=1, shading="nearest")
        ax.contour(x, y, state["phi"], levels=[0.], colors=["#147bb6"], linewidths=1.)
        nodes = state["solid_positions"]
        tri = mtri.Triangulation(nodes[:, 0], nodes[:, 1], state["solid_cells"])
        ax.tripcolor(tri, np.ones(len(nodes)), cmap=ListedColormap(["#ce6a31"]), shading="flat")
        ax.triplot(tri, color="#703410", lw=.18, alpha=.7)
        ax.axhline(p["water_level"], color="#666666", ls="--", lw=.7)
        ax.set(xlim=(0, p["lx"]), ylim=(0, p["ly"]), aspect="equal",
               xlabel="x (m)", ylabel="y (m)", title=f"t = {float(state['time']):.4f} s")

    targets = [0., .8*entry, entry, times[-1]] if entry >= 0 else np.linspace(0, times[-1], 4)
    indices = [int(np.argmin(np.abs(times-t))) for t in targets]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.8), constrained_layout=True)
    for ax, index in zip(axes, indices):
        panel(ax, states[index])
    fig.suptitle("Elastic block released from rest into water · experimental 2D simulation")
    fig.savefig(args.output/"water_entry_sequence.png", dpi=180)
    plt.close(fig)

    with (args.output/"metrics.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    values = {key: np.array([float(row[key]) for row in rows]) for key in rows[0]}
    t = values["time"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    axes[0, 0].plot(t, values["solid_com_y"], label="Centre of mass")
    axes[0, 0].plot(t, values["solid_bottom_y"], label="Lowest solid node")
    axes[0, 0].axhline(p["water_level"], ls="--", c="gray", label="Initial waterline")
    axes[0, 0].set(ylabel="Height (m)")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].plot(t, values["solid_com_vy"], label="Computed COM velocity")
    # A vacuum reference, not an exact solution for an immersed body.
    free_t = t[t <= entry] if entry >= 0 else t
    axes[0, 1].plot(free_t, -9.81*free_t, "--", c="gray", label="Vacuum reference -gt")
    axes[0, 1].set(ylabel="Vertical velocity (m/s)")
    axes[0, 1].legend(fontsize=8)
    axes[1, 0].plot(t, values["solid_J_min"], label="Min J")
    axes[1, 0].plot(t, values["solid_J_max"], label="Max J")
    axes[1, 0].set(ylabel="Element Jacobian")
    axes[1, 0].legend(fontsize=8)
    axes[1, 1].plot(t, 100*values["density_geometry_relative_l1"], label="Density/geometry L1")
    axes[1, 1].plot(t, 100*values["solid_area_relative_error"], label="Solid total area change")
    axes[1, 1].set(ylabel="Error (%)")
    axes[1, 1].legend(fontsize=8)
    for ax in axes.flat:
        ax.set(xlabel="Time (s)")
        ax.grid(alpha=.2)
        if entry >= 0:
            ax.axvline(entry, color="#ce6a31", ls=":", lw=1.)
    fig.suptitle("Water-entry diagnostics · dotted line: first crossing of the initial waterline")
    fig.savefig(args.output/"water_entry_history.png", dpi=180)
    plt.close(fig)
    if args.gif:
        fig, ax = plt.subplots(figsize=(5, 5), constrained_layout=True)
        animation = FuncAnimation(fig, lambda i: panel(ax, states[i]), frames=len(states), interval=180)
        animation.save(args.output/"water_entry.gif", writer=PillowWriter(fps=6), dpi=110)
        plt.close(fig)


if __name__ == "__main__":
    main()
