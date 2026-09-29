#!/usr/bin/env python
"""demo_426 系绳符号修复 — 验证图表。

读取本目录下各运行的 *_hist.csv（step,t,max_u,plate_disp），输出
figures/signfix_histories.png（四联图）：
  左上 板漂移曲线（修复后、各配置；含改前基准虚线，逐位重合）
  右上 r7（β=-8e3，旧符号等效）的发散曲线
  左下 max|u| 曲线（稳定运行 + 解析 u_max 参考线）
  右下 各运行终了时的通道内相对 L2 误差柱状图
英文标签（避免 CJK 字体问题）。运行：python plot_signfix.py
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, "figures")
os.makedirs(FIGDIR, exist_ok=True)


def load(fn):
    rows = [r for r in list(csv.reader(open(os.path.join(HERE, fn))))[1:] if r]
    return ([float(r[1]) for r in rows],
            [float(r[2]) for r in rows],
            [float(r[3]) for r in rows])


runs = [
    ("before (pre-edit)",     "before_tether_ipcs_hist.csv", "k",          "--", 1.2),
    ("r1 tether default",     "r1_hist.csv",                 "tab:blue",   "-",  1.8),
    ("r2 tether + direct",    "r2_hist.csv",                 "tab:green",  "-",  1.4),
    ("r3 tether chorin",      "r3_hist.csv",                 "tab:orange", "-",  1.2),
    ("r5 direct, long t=10s", "r5_hist.csv",                 "tab:purple", "-",  1.4),
    ("r6 beta=1.2e4",         "r6_hist.csv",                 "tab:brown",  "-",  1.2),
]

fig, axs = plt.subplots(2, 2, figsize=(12.5, 8.6))

ax = axs[0, 0]
for lab, fn, c, ls, lw in runs:
    t, _, d = load(fn)
    ax.plot(t, d, ls, color=c, lw=lw, label=lab)
ax.set_title("plate drift, tether route (correct sign)")
ax.set_xlabel("t [s]")
ax.set_ylabel("plate max |disp| [m]")
ax.legend(fontsize=7.5, loc="upper left")
ax.grid(alpha=0.3)

ax = axs[0, 1]
t, mu, d = load("r7_hist.csv")
ax.plot(t, d, "-", color="crimson", lw=1.8)
ax.set_title(r"$\beta=-8{\times}10^3$ (== old inverted sign): divergence")
ax.set_xlabel("t [s]")
ax.set_ylabel("plate max |disp| [m]")
ax.annotate(f"final drift {d[-1]:.2f} m = {d[-1] / 0.03125:.0f} dx\n"
            f"max|u| peaked at {max(mu):.1f} m/s",
            xy=(0.05, 0.72), xycoords="axes fraction",
            fontsize=9, color="crimson")
ax.grid(alpha=0.3)

ax = axs[1, 0]
for lab, fn, c, ls, lw in runs:
    t, mu, _ = load(fn)
    ax.plot(t, mu, ls, color=c, lw=lw, label=lab)
ax.axhline(1 / 6, color="k", ls=":", lw=1)
ax.text(0.02, 1 / 6 + 0.0004, "analytic u_max = 1/6", fontsize=8)
ax.set_title("max |u| (stable runs)")
ax.set_xlabel("t [s]")
ax.set_ylabel("max |u| [m/s]")
ax.legend(fontsize=7.5, loc="upper right")
ax.grid(alpha=0.3)

ax = axs[1, 1]
errs = [("r1\ntether dflt",   0.6277, "tab:blue"),
        ("r2\ntether direct", 0.7569, "tab:green"),
        ("r6\nbeta 1.2e4",    0.8929, "tab:brown"),
        ("r5\nlong t=10s",    1.1435, "tab:purple"),
        ("r3\ntether chorin", 2.0443, "tab:orange"),
        ("r4b\ndrag band",    5.3347, "tab:red")]
vals = [v for _, v, _ in errs]
cols = [c for _, _, c in errs]
ax.bar(range(len(errs)), vals, color=cols, alpha=0.85)
ax.set_xticks(range(len(errs)))
ax.set_xticklabels([n for n, _, _ in errs], fontsize=8)
for i, v in enumerate(vals):
    ax.text(i, v + 0.08, f"{v:.2f}%", ha="center", fontsize=8.5)
ax.set_title("channel interior relative L2 (end of run)")
ax.set_ylabel("err [%]")
ax.grid(alpha=0.3, axis="y")

fig.suptitle("demo_426 tether sign-fix verification  "
             "(N=32, ipcs, dt=0.2dx, 2026-09-28)", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.96))
out = os.path.join(FIGDIR, "signfix_histories.png")
fig.savefig(out, dpi=150)
print("written:", out)
