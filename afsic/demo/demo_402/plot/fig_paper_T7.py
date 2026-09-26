#!/usr/bin/env python
"""论文对齐算例（L=6H, SVK, paper 网格/时间步）的 Δy(t) 结果图。

只画有效窗口（发散前），并把发散时刻标出来。
用法: python -B plot/fig_paper_T7.py [运行目录]
"""
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = sys.argv[1] if len(sys.argv) > 1 else "paper_T7_dx1"
D = os.path.join(HERE, RUN)

rows = [r for r in csv.DictReader(open(os.path.join(D, "history.csv")))
        if r.get("max_u")]
h = {k: np.array([float(r[k]) for r in rows])
     for k in ("t", "max_u", "inlet_scale", "tip_dx", "tip_dy", "cyl_dx",
               "cyl_dy", "u_L2")}
m = h["t"] > 0
ratio = np.where(m, h["max_u"] / np.maximum(h["inlet_scale"], 1e-30), np.nan)
bad = np.where(m & (np.abs(ratio) > 5))[0]
t_div = h["t"][bad[0]] if len(bad) else None
t_end = t_div if t_div else h["t"][-1]
ok = h["t"] <= t_end

print(f"发散时刻 t≈{t_div:.4f} s" if t_div else "未出现发散")
print(f"有效窗口 0–{t_end:.3f} s：|Δy|max={np.abs(h['tip_dy'][ok]).max():.4f} cm, "
      f"|Δx|max={np.abs(h['tip_dx'][ok]).max():.4f} cm, "
      f"圆柱漂移max={np.abs(h['cyl_dx'][ok]).max():.4f} cm, "
      f"max|u|/scale max={np.nanmax(ratio[ok]):.3f}")


def main():
    import matplotlib
    matplotlib.use("Agg")
    from _cjk import setup_cjk
    setup_cjk()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    for a in ax.ravel():
        a.axvspan(t_end, h["t"][-1], color="red", alpha=.10)
        a.grid(alpha=.3)
        a.set_xlabel("t [s]")
    if t_div:
        for a in ax.ravel():
            a.axvline(t_div, color="red", ls="--", lw=1.4)
        ax[0, 0].text(t_div, np.nanmax(h["tip_dy"][ok]) * .6,
                      f"  t≈{t_div:.2f} s 发散\n（斜坡结束、入流满速后）",
                      color="red", fontsize=8)

    ax[0, 0].plot(h["t"][ok], h["tip_dy"][ok], "-o", ms=1.5, color="tab:red")
    ax[0, 0].set_ylabel(r"尾尖 A 竖向位移 $\Delta y$ [cm]")
    ax[0, 0].set_title(r"点 A(0.6,0.2) 的 $\Delta y(t)$ —— 论文 Fig.26 的观测量")
    ax[0, 1].plot(h["t"][ok], h["tip_dx"][ok], "-o", ms=1.5, color="tab:blue")
    ax[0, 1].set_ylabel(r"尾尖 $\Delta x$ [cm]")
    ax[0, 1].set_title("尾尖流向位移")
    ax[1, 0].semilogy(h["t"][m], ratio[m], "-o", ms=1.5, color="tab:blue",
                      label=r"max|u|/scale")
    ax[1, 0].semilogy(h["t"][m], h["u_L2"][m] / np.maximum(h["inlet_scale"][m], 1e-30) ** 2,
                      "-s", ms=1.5, color="tab:green", label=r"$u_{L2}$/scale$^2$")
    ax[1, 0].axhline(1.5, color="k", ls=":", lw=1.0)
    ax[1, 0].set_ylabel("[-]")
    ax[1, 0].set_title("归一化指标：0–0.56 s 有界，之后爆掉")
    ax[1, 0].legend(fontsize=8)
    ax[1, 1].plot(h["t"][ok], h["cyl_dx"][ok], "-o", ms=1.5, color="tab:orange")
    ax[1, 1].set_ylabel(r"圆柱区平均 $\Delta x$ [cm]")
    ax[1, 1].set_title("圆柱漂移（系绳罚约束）")
    for a_ in ax.ravel():
        a_.set_xlim(0.0, min(t_end * 1.25, 1.0))
    fig.suptitle(f"demo_402 论文对齐算例（L=6H, SVK, 246×41, dt=5e-5）"
                 f"  —— 只显示有效窗口 0–{t_end:.2f} s（程序继续裸跑到 t=7 s，"
                 f"那一段是 NaN 冻结态，未画）", fontsize=12)
    fig.tight_layout()
    out = os.path.join(HERE, f"fig_{RUN}.png")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
