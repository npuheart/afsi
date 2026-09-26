#!/usr/bin/env python
"""Chorin 单方法结果图（demo_402 Turek FSI2：圆柱 + 弹性尾巴）。

只画 Chorin 两组运行：
  coarse_chorin   110x21, dt=1e-4, T=1.0 s（完整跑完 10000 步）
  compare_chorin  220x41, dt=5e-5, 记录到 t≈0.23 s（细网格，中途停止）

四联：尾尖流向位移、尾尖横向位移（benchmark 观测量）、max|u| 与入口斜坡、
圆柱区平均漂移（惩罚约束质量）。

用法：
    cd afsic/demo/demo_402
    ../../../.tools/afsi-run.sh python -B plot/fig_chorin.py
"""
import csv
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = [
    ("coarse_chorin", "Chorin 110x21  dt=1e-4  T=1.0 s（完整）", "-", "tab:blue"),
    ("compare_chorin", "Chorin 220x41  dt=5e-5  t<=0.23 s（细网格）", "--", "tab:cyan"),
]


def load(name):
    path = os.path.join(HERE, name, "history.csv")
    if not os.path.exists(path):
        return None
    rows = [r for r in csv.DictReader(open(path)) if r.get("max_u")]
    if not rows:
        return None
    return {k: np.array([float(r[k]) for r in rows])
            for k in ("t", "max_u", "inlet_scale", "tip_dx", "tip_dy",
                      "cyl_dx", "cyl_dy", "u_L2")}


def main():
    import matplotlib
    matplotlib.use("Agg")
    from _cjk import setup_cjk
    setup_cjk()
    import matplotlib.pyplot as plt

    data = {n: load(n) for n, *_ in RUNS}
    fig, ax = plt.subplots(2, 2, figsize=(13, 8.5))

    for name, label, ls, color in RUNS:
        d = data[name]
        if d is None:
            continue
        ax[0, 0].plot(d["t"], d["tip_dx"], ls, color=color, lw=1.8, label=label)
        ax[0, 1].plot(d["t"], d["tip_dy"], ls, color=color, lw=1.8, label=label)
        ax[1, 0].plot(d["t"], d["max_u"], ls, color=color, lw=1.8, label="max|u| " + label)
        ax[1, 1].plot(d["t"], d["cyl_dx"], ls, color=color, lw=1.8, label=label)

    d = data["coarse_chorin"]
    ax[1, 0].plot(d["t"], d["inlet_scale"], ":", color="k", lw=1.6,
                  label="入口斜坡速度 scale")
    ax[1, 0].plot(d["t"], 1.5 * d["inlet_scale"], "-.", color="k", lw=1.0,
                  label=r"$1.5\times$scale（入口剖面峰值）")

    ax[0, 0].set_ylabel(r"尾尖 $\Delta x$ [cm]")
    ax[0, 0].set_title("尾巴末端流向位移：随流动逐渐伸出（无异常跳跃）")
    ax[0, 1].set_ylabel(r"尾尖 $\Delta y$ [cm]")
    ax[0, 1].set_title("尾巴末端横向位移（Turek 基准观测量）：t≈0.7 起大幅拍动")
    ax[1, 0].set_ylabel("速度 [cm/s]")
    ax[1, 0].set_title("max|u| 始终跟随入口斜坡（≈1.9–2.3×scale），无发散")
    ax[1, 1].set_ylabel(r"圆柱区平均 $\Delta x$ [cm]")
    ax[1, 1].set_title("圆柱漂移 ≤1.2e-2 cm：惩罚约束守住")
    for a in ax.ravel():
        a.set_xlabel("t [s]")
        a.grid(alpha=.3)
        a.legend(fontsize=8)

    fig.suptitle("demo_402 Turek FSI2（圆柱 + 弹性尾巴）：Chorin 投影法结果",
                 fontsize=13)
    fig.tight_layout()
    out = os.path.join(HERE, "fig_chorin.png")
    fig.savefig(out, dpi=170, bbox_inches="tight")
    print(f"wrote {out}")

    dr = data["coarse_chorin"]
    print(f"coarse_chorin 关键指标："
          f"t_end={dr['t'][-1]:.3f}s  |tip_dx|max={np.abs(dr['tip_dx']).max():.4f}cm  "
          f"|tip_dy|max={np.abs(dr['tip_dy']).max():.4f}cm  "
          f"max|u|={dr['max_u'].max():.1f}cm/s  "
          f"圆柱漂移max={np.abs(dr['cyl_dx']).max():.4f}cm")


if __name__ == "__main__":
    main()
