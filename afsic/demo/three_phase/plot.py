"""Render the last checkpoint without running another simulation."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.tri as mtri
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    p = json.loads((args.output/"parameters.json").read_text())
    with np.load(args.output/"checkpoint.npz", allow_pickle=False) as data:
        x = (np.arange(p["nx"])+.5)*p["lx"]/p["nx"]
        y = (np.arange(p["ny"])+.5)*p["ly"]/p["ny"]
        xx, yy = np.meshgrid(x, y)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        view = axes[0].pcolormesh(xx, yy, data["rho"], shading="nearest", cmap="viridis", norm=LogNorm())
        fig.colorbar(view, ax=axes[0], label="Transported density (kg/m³)")
        speed = np.hypot(.5*(data["u"][:, :-1]+data["u"][:, 1:]), .5*(data["v"][:-1]+data["v"][1:]))
        view = axes[1].pcolormesh(xx, yy, speed, shading="nearest", cmap="magma")
        fig.colorbar(view, ax=axes[1], label="Speed (m/s)")
        nodes = data["solid_positions"]
        tri = mtri.Triangulation(nodes[:, 0], nodes[:, 1], data["solid_cells"])
        for ax in axes:
            ax.contour(xx, yy, data["phi"], levels=[0.], colors=["cyan"], linewidths=1.4)
            ax.triplot(tri, color="white", lw=.35)
            ax.set(xlim=(0, p["lx"]), ylim=(0, p["ly"]), xlabel="x (m)", ylabel="y (m)", aspect="equal")
        fig.suptitle(f"Experimental three-phase FSI · t = {float(data['time']):.5f} s · step {int(data['step'])}")
        fig.savefig(args.output/"checkpoint.png", dpi=180)
        plt.close(fig)


if __name__ == "__main__":
    main()
