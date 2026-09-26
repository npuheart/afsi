#!/usr/bin/env python
"""一张图看清 demo_402 的求解器差异（Chorin 稳定 / IPCS 发散）。

读四份 history.csv（粗/细网格 × 两个求解器），画：
  1) max|u| 随时间（对数纵轴）+ 入口斜坡速度
  2) max|u|/入口速度 —— 发散就是这条线脱离 1.5~2 的平台
  3) 尾尖流向位移（粗网格两法）
  4) 圆柱区平均漂移（检验惩罚约束是否守住）

用法：
    cd afsic/demo/demo_402
    ../../../.tools/afsi-run.sh python -B plot/fig_divergence.py
"""
import csv
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = [
    ("coarse_chorin", "Chorin 110x21 dt=1e-4", "tab:blue", "-"),
    ("coarse_ipcs", "IPCS 110x21 dt=1e-4", "tab:red", "-"),
    ("compare_chorin", "Chorin 220x41 dt=5e-5", "tab:cyan", "--"),
    ("compare_ipcs", "IPCS 220x41 dt=5e-5", "tab:orange", "--"),
]


def load(name):
    path = os.path.join(HERE, name, "history.csv")
    if not os.path.exists(path):
        return None
    rows = [r for r in csv.DictReader(open(path))
            if r.get("max_u") not in (None, "")]
    if not rows:
        return None
    d = {k: np.array([float(r[k]) for r in rows])
         for k in ("t", "max_u", "inlet_scale", "tip_dx", "tip_dy", "cyl_dx")}
    return d


def main():
    import matplotlib
    matplotlib.use("Agg")
    from _cjk import setup_cjk
    setup_cjk()
    import matplotlib.pyplot as plt

    data = {n: load(n) for n, *_ in RUNS}
    fig, ax = plt.subplots(2, 2, figsize=(14, 9))

    for name, label, color, ls in RUNS:
        d = data[name]
        if d is None:
            continue
        m = d["t"] > 0
        ax[0, 0].semilogy(d["t"][m], d["max_u"][m], ls, color=color, lw=1.8,
                          label=label)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = d["max_u"] / d["inlet_scale"]
        ax[0, 1].semilogy(d["t"][m], ratio[m], ls, color=color, lw=1.8,
                          label=label)
    d = data["coarse_chorin"]
    m = d["t"] > 0
    ax[0, 0].semilogy(d["t"][m], d["inlet_scale"][m], ":", color="k", lw=1.5,
                      label="入口斜坡速度 scale")
    ax[0, 1].axhline(1.5, color="k", ls=":", lw=1.2)
    ax[0, 1].text(0.02, 1.62, r"入口剖面峰值 $1.5\times$scale", fontsize=8)

    ax[0, 0].set_xlabel("t [s]"); ax[0, 0].set_ylabel("max|u| [cm/s]")
    ax[0, 0].set_title("最大速度：IPCS 在 t≈0.04–0.12 s 指数发散")
    ax[0, 0].grid(alpha=.3, which="both"); ax[0, 0].legend(fontsize=8)
    ax[0, 1].set_xlabel("t [s]"); ax[0, 1].set_ylabel("max|u| / scale [-]")
    ax[0, 1].set_title("归一化后：高于 1.5 的部分就是数值尖峰")
    ax[0, 1].grid(alpha=.3, which="both"); ax[0, 1].legend(fontsize=8)

    for name, label, color, ls in RUNS[:2]:
        d = data[name]
        if d is None:
            continue
        ax[1, 0].plot(d["t"], d["tip_dx"], ls, color=color, lw=1.8, label=label)
        ax[1, 1].plot(d["t"], d["cyl_dx"], ls, color=color, lw=1.8, label=label)
    ax[1, 0].set_xlabel("t [s]"); ax[1, 0].set_ylabel(r"尾尖 $\Delta x$ [cm]")
    ax[1, 0].set_title("尾巴末端流向位移（IPCS 在 t>0.13 冻结）")
    ax[1, 0].grid(alpha=.3); ax[1, 0].legend(fontsize=8)
    ax[1, 1].set_xlabel("t [s]"); ax[1, 1].set_ylabel(r"圆柱区平均 $\Delta x$ [cm]")
    ax[1, 1].set_title("圆柱漂移（惩罚约束是否守住；IPCS 冻结在 9.4e-3 cm）")
    ax[1, 1].grid(alpha=.3); ax[1, 1].legend(fontsize=8)

    fig.suptitle("demo_402 Turek FSI2（圆柱 + 弹性尾巴）：Chorin vs IPCS",
                 fontsize=13)
    fig.tight_layout()
    out = os.path.join(HERE, "fig_divergence.png")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
