#!/usr/bin/env python
"""220x41 / dt=5e-5 / T=2 s（出厂密度、ramp 走完）Chorin 结果图。

四个面板：
  (a) 尾尖流向位移 Δx(t)
  (b) 尾尖横向位移 Δy(t)（Turek 基准观测量）
  (c) max|u|/scale 与 u_L2/scale²（有界 = 没发散），并标出 88x17 那次冻结的时段
  (d) t = 2 s 的 |u| 场 + 变形后的尾巴/圆柱位置

用法：
    cd afsic/demo/demo_402
    ../../../.tools/afsi-run.sh python -B plot/fig_T2_220x41.py [运行目录名]
"""
import csv
import os
import sys

import numpy as np
import h5py

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = sys.argv[1] if len(sys.argv) > 1 else "chorin_220x41_T2"
D = os.path.join(HERE, RUN)

rows = [r for r in csv.DictReader(open(os.path.join(D, "history.csv")))
        if r.get("max_u")]
h = {k: np.array([float(r[k]) for r in rows])
     for k in ("t", "max_u", "inlet_scale", "tip_dx", "tip_dy", "cyl_dx",
               "cyl_dy", "u_L2", "p_L2")}
m = h["t"] > 0


def last_field(path):
    with h5py.File(path, "r") as f:
        geom = f["Mesh/mesh/geometry"][:]
        fn = list(f["Function"].keys())[0]
        ks = list(f[f"Function/{fn}"].keys())

        def kt(k):
            try:
                return float(k.replace("_", ".", 1))
            except ValueError:
                return -1.0
        ks.sort(key=kt)
        return geom[:, :2], f[f"Function/{fn}/{ks[-1]}"][:]


def main():
    import matplotlib
    matplotlib.use("Agg")
    from _cjk import setup_cjk
    setup_cjk()
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(14, 9))
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)

    ax1.plot(h["t"], h["tip_dx"], "-o", ms=2, color="tab:blue")
    ax1.set_ylabel(r"尾尖 $\Delta x$ [cm]")
    ax1.set_title("尾尖流向位移：全程单调演化，无冻结")
    ax2.plot(h["t"], h["tip_dy"], "-o", ms=2, color="tab:red")
    ax2.set_ylabel(r"尾尖 $\Delta y$ [cm]")
    ax2.set_title("尾尖横向位移（Turek 观测量）：先单调偏转、后小幅振荡")

    ax3.semilogy(h["t"][m], h["max_u"][m] / h["inlet_scale"][m], "-o", ms=2,
                 color="tab:blue", label=r"max|u| / scale")
    ax3.semilogy(h["t"][m], h["u_L2"][m] / h["inlet_scale"][m] ** 2, "-s", ms=2,
                 color="tab:green", label=r"$u_{L2}$ / scale$^2$")
    ax3.axhline(1.5, color="k", ls=":", lw=1.2)
    ax3.axvspan(0.5, 0.62, color="orange", alpha=.18)
    ax3.text(0.63, 1.62, "88×17 那次在此冻结", fontsize=8, color="darkorange")
    ax3.set_xlabel("t [s]")
    ax3.set_title("归一化指标全程有界 ⟹ 无发散 / 无解耦")
    ax3.grid(alpha=.3, which="both"); ax3.legend(fontsize=8)

    ax1.set_xlabel("t [s]"); ax2.set_xlabel("t [s]")
    for a in (ax1, ax2):
        a.grid(alpha=.3)

    # ---- t = T_end 的速度场 + 固体位置 ----
    gu, u = last_field(os.path.join(D, "velocity.h5"))
    mag = np.linalg.norm(u[:, :2], axis=1)
    gs, sc = last_field(os.path.join(D, "solid.h5"))
    nx = len(np.unique(np.round(gu[:, 0], 6)))
    ny = len(np.unique(np.round(gu[:, 1], 6)))
    order = np.lexsort((gu[:, 1], gu[:, 0]))
    U = mag[order].reshape(ny, nx)
    X = np.sort(np.unique(np.round(gu[:, 0], 6)))
    Y = np.sort(np.unique(np.round(gu[:, 1], 6)))
    cs = ax4.contourf(X, Y, U, levels=40, cmap="viridis")
    ax4.plot(sc[:, 0], sc[:, 1], ".", ms=1, color="w", alpha=.7,
             label="固体（圆柱+尾巴，t=T_end）")
    ax4.plot(20 + 5 * np.cos(np.linspace(0, 2 * np.pi, 100)),
             20 + 5 * np.sin(np.linspace(0, 2 * np.pi, 100)), "-", color="w", lw=.8)
    ax4.set_aspect("equal")
    ax4.set_xlim(0, 220); ax4.set_ylim(0, 41)
    ax4.set_xlabel("x [cm]"); ax4.set_ylabel("y [cm]")
    ax4.set_title(f"|u| 场（t = {h['t'][-1]:.2f} s）：流动已充满通道")
    ax4.legend(fontsize=8, loc="upper right")
    fig.colorbar(cs, ax=ax4, label="|u| [cm/s]", shrink=.9)

    fig.suptitle(f"demo_402 Chorin，出厂密度 {RUN}（220×41, dt=5e-5, "
                 f"T={h['t'][-1]:.2f} s）", fontsize=13)
    fig.tight_layout()
    out = os.path.join(HERE, f"fig_{RUN}.png")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"wrote {out}")

    print(f"t_end={h['t'][-1]:.4f}s  max|u|={h['max_u'].max():.1f} cm/s  "
          f"(max|u|/scale)_{'{'}max{'}'}={np.nanmax(h['max_u'][m]/h['inlet_scale'][m]):.3f}  "
          f"(uL2/scale^2) 范围={np.nanmin(h['u_L2'][m]/h['inlet_scale'][m]**2):.0f}"
          f"–{np.nanmax(h['u_L2'][m]/h['inlet_scale'][m]**2):.0f}  "
          f"|tip_dx|max={np.abs(h['tip_dx']).max():.4f}cm  "
          f"|tip_dy|max={np.abs(h['tip_dy']).max():.4f}cm  "
          f"|cyl_dx|max={np.abs(h['cyl_dx']).max():.4f}cm")


if __name__ == "__main__":
    main()
