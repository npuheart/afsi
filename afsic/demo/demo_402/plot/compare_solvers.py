#!/usr/bin/env python
"""比较 demo_402（Turek FSI2：圆柱 + 弹性尾巴）在两套流体求解器下的结果。

读取 `run_compare.py` 生成的 `<前缀>/history.csv`，输出终端对比表、对齐后的
CSV 与一张 PNG。

用法：
    cd afsic/demo/demo_402
    ../../../.tools/afsi-run.sh python -B plot/compare_solvers.py
    # 自定义两档（默认 compare_chorin vs compare_ipcs）：
    ... plot/compare_solvers.py coarse_chorin coarse_ipcs "chorin(110x21)" "ipcs(110x21)"
"""
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PA = sys.argv[1] if len(sys.argv) > 1 else "compare_chorin"
PB = sys.argv[2] if len(sys.argv) > 2 else "compare_ipcs"
LA = sys.argv[3] if len(sys.argv) > 3 else "chorin"
LB = sys.argv[4] if len(sys.argv) > 4 else "ipcs"
FIELDS = ["tip_dx", "tip_dy", "cyl_dx", "cyl_dy", "max_u", "u_L2", "p_L2",
          "umax_x", "umax_y"]


def load(prefix):
    path = os.path.join(HERE, prefix, "history.csv")
    if not os.path.exists(path):
        return None
    rows = list(csv.DictReader(open(path)))
    if not rows:
        return None
    d = {"t": np.array([float(r["t"]) for r in rows])}
    for f in FIELDS + ["inlet_scale", "elapsed_s"]:
        d[f] = np.array([float(r[f]) for r in rows])
    return d


def rel(x, y):
    return (x - y) / max(abs(x), abs(y), 1e-300)


def fmt(v, w=13, p=5):
    return f"{v:>{w}.{p}e}"


def main():
    a, b = load(PA), load(PB)
    if a is None or b is None:
        print(f"缺少 history.csv：{PA}={a is not None}, {PB}={b is not None}")
        return
    n = min(len(a["t"]), len(b["t"]))
    same = np.isclose(a["t"][:n], b["t"][:n], rtol=1e-9, atol=1e-15)
    ia_all, ib_all = np.where(same)[0], np.where(same)[0]
    t = a["t"][:n][same]
    if len(t) == 0:
        print("两个 run 没有对齐的输出时刻")
        return

    print("=" * 112)
    print(f"demo_402 (Turek FSI2：圆柱 + 弹性尾巴)  {LA} vs {LB}   "
          f"对齐输出时刻 {len(t)} 个")
    print("=" * 112)
    print(f"{'t [s]':>8} | {'scale':>9} | {'tip_dx A':>11} {'tip_dx B':>11} "
          f"{'Δ%':>8} | {'max|u| A':>11} {'max|u| B':>11} {'Δ%':>8} | "
          f"{'u_L2 Δ%':>9} | {'argmax|u| A':>15} | {'argmax|u| B':>15}")
    print("-" * 112)
    step = max(1, len(t) // 18)
    for k in list(range(0, len(t), step)) + [len(t) - 1]:
        i, j = ia_all[k], ib_all[k]
        print(f"{t[k]:>8.4f} | {a['inlet_scale'][i]:>9.4f} | "
              f"{fmt(a['tip_dx'][i], 11)} {fmt(b['tip_dx'][j], 11)} "
              f"{rel(a['tip_dx'][i], b['tip_dx'][j])*100:>8.3f} | "
              f"{fmt(a['max_u'][i], 11)} {fmt(b['max_u'][j], 11)} "
              f"{rel(a['max_u'][i], b['max_u'][j])*100:>8.3f} | "
              f"{rel(a['u_L2'][i], b['u_L2'][j])*100:>9.3f} | "
              f"{a['umax_x'][i]:>7.2f},{a['umax_y'][i]:>6.2f} | "
              f"{b['umax_x'][j]:>7.2f},{b['umax_y'][j]:>6.2f}")

    print("-" * 112)
    la, lb = a["elapsed_s"][ia_all[-1]], b["elapsed_s"][ib_all[-1]]
    print(f"末态 (t = {t[-1]:.4f} s, inlet scale = {a['inlet_scale'][ia_all[-1]]:.4f} cm/s):")
    for f in ["tip_dx", "tip_dy", "cyl_dx", "cyl_dy", "max_u", "u_L2", "p_L2"]:
        va, vb = a[f][ia_all[-1]], b[f][ib_all[-1]]
        print(f"  {f:<8} {LA}={fmt(va)}  {LB}={fmt(vb)}  "
              f"相对差={rel(va, vb)*100:+.4f} %")
    print(f"累计耗时：{LA} {la:.1f} s，{LB} {lb:.1f} s，"
          f"比值 {lb/la:.2f}×（同一物理时刻、同一时间步）")

    out_csv = os.path.join(HERE, f"summary_{PA}_vs_{PB}.csv")
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["t"] + [f"{f}_{s}" for s in (LA, LB) for f in FIELDS])
        for k in range(len(t)):
            i, j = ia_all[k], ib_all[k]
            w.writerow([f"{t[k]:.8e}"] +
                       [f"{a[f][i]:.10e}" for f in FIELDS] +
                       [f"{b[f][j]:.10e}" for f in FIELDS])
    print(f"wrote {out_csv}")

    try:
        import matplotlib
        from _cjk import setup_cjk
        setup_cjk()
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 4, figsize=(19, 4.2))
        ax[0].plot(t, a["tip_dx"][ia_all], "-o", ms=2.5, label=LA)
        ax[0].plot(t, b["tip_dx"][ib_all], "--s", ms=2.5, label=LB)
        ax[0].set_xlabel("t [s]"); ax[0].set_ylabel(r"flag tip $\Delta x$ [cm]")
        ax[0].set_title("尾尖流向位移"); ax[0].grid(alpha=.3); ax[0].legend()
        ax[1].plot(t, a["tip_dy"][ia_all], "-o", ms=2.5, label=LA)
        ax[1].plot(t, b["tip_dy"][ib_all], "--s", ms=2.5, label=LB)
        ax[1].set_xlabel("t [s]"); ax[1].set_ylabel(r"flag tip $\Delta y$ [cm]")
        ax[1].set_title("尾尖横向位移（Turek 基准观测量）"); ax[1].grid(alpha=.3)
        ax[1].legend()
        ax[2].semilogy(t, a["max_u"][ia_all], "-o", ms=2.5, label=LA + " max|u|")
        ax[2].semilogy(t, b["max_u"][ib_all], "--s", ms=2.5, label=LB + " max|u|")
        ax[2].semilogy(t, a["inlet_scale"][ia_all], ":", label="inlet scale")
        ax[2].set_xlabel("t [s]"); ax[2].set_ylabel("[cm/s]")
        ax[2].set_title("最大速度 vs 入口斜坡"); ax[2].grid(alpha=.3, which="both")
        ax[2].legend(fontsize=8)
        ra = np.array([rel(a["u_L2"][i], b["u_L2"][j])
                       for i, j in zip(ia_all, ib_all)]) * 100
        ax[3].plot(t, ra, "-o", ms=2.5)
        ax[3].axhline(0, color="k", lw=.8)
        ax[3].set_xlabel("t [s]"); ax[3].set_ylabel("[%]")
        ax[3].set_title(r"$u_{L2}$ 相对差 (" + LA + " vs " + LB + ")")
        ax[3].grid(alpha=.3)
        fig.suptitle(f"demo_402 Turek FSI2：{LA} vs {LB}", y=1.02)
        fig.tight_layout()
        out_png = os.path.join(HERE, f"compare_{PA}_vs_{PB}.png")
        fig.savefig(out_png, dpi=160, bbox_inches="tight")
        print(f"wrote {out_png}")
    except Exception as exc:
        print(f"(出图跳过: {type(exc).__name__}: {exc})")


if __name__ == "__main__":
    main()
