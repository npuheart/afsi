#!/usr/bin/env python
"""从 demo_402 的输出文件重建时间序列并对比两套求解器。

`run_compare.py` 的 history.csv 在某些情况下会被并发写坏，这个脚本直接从
`velocity.h5` / `pressure.h5` / `solid.h5`（每个输出时刻一个数据集）重建指标，
更稳。

用法：
    cd afsic/demo/demo_402
    ../../../.tools/afsi-run.sh python -B plot/compare_from_output.py \
        plot/compare_chorin plot/compare_ipcs "chorin(220x41)" "ipcs(220x41)"
    # 参数可省略，默认同上；粗网格用 coarse_chorin / coarse_ipcs
"""
import os
import re
import sys
import csv

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEMO = os.path.dirname(HERE)


def _times(xdmf_path):
    """XDMF 里按顺序的 <Time Value="..."/> 列表。"""
    txt = open(xdmf_path, encoding="utf-8", errors="ignore").read()
    return [float(m) for m in re.findall(r'<Time\s+Value="([^"]+)"', txt)]


def _series(h5_path, xdmf_path):
    """返回 {time: ndarray(nodes, ncomp)}。"""
    times = _times(xdmf_path)
    with h5py.File(h5_path, "r") as h:
        geom = h["Mesh/mesh/geometry"][:]
        fn = list(h["Function"].keys())[0]
        keys = list(h[f"Function/{fn}"].keys())
        # dolfinx 用时间值给数据集命名（'.' -> '_'），按数值排序
        def key_time(k):
            try:
                return float(k.replace("_", ".", 1))
            except ValueError:
                return np.inf
        keys.sort(key=key_time)
        data = [h[f"Function/{fn}/{k}"][:] for k in keys]
    n = min(len(times), len(data))
    return geom, times[:n], data[:n]


def load(run_dir):
    run_dir = run_dir if os.path.isabs(run_dir) else os.path.join(DEMO, run_dir)
    g_u, t_u, u_all = _series(os.path.join(run_dir, "velocity.h5"),
                              os.path.join(run_dir, "velocity.xdmf"))
    g_p, t_p, p_all = _series(os.path.join(run_dir, "pressure.h5"),
                              os.path.join(run_dir, "pressure.xdmf"))
    g_s, t_s, s_all = _series(os.path.join(run_dir, "solid.h5"),
                              os.path.join(run_dir, "solid.xdmf"))
    n = min(len(t_u), len(t_p), len(t_s))
    # P1 节点梯形积分权重：每个节点代表一个单元的面积（2D 四边形）
    cell_area = (g_u[:, 0].max() - g_u[:, 0].min()) * \
                (g_u[:, 1].max() - g_u[:, 1].min()) / (len(g_u) * 1.0)
    tip = g_s[:, 0] >= g_s[:, 0].max() - 1e-9
    cyl = g_s[:, 0] <= 25.0 + 1e-9
    rows = []
    for k in range(n):
        uk = u_all[k][:, :2]
        mag = np.linalg.norm(uk, axis=1)
        i = int(np.argmax(mag))
        pk = p_all[k][:, 0]
        sk = s_all[k][:, :2]
        d = sk - g_s[:, :2]
        rows.append(dict(
            t=t_u[k],
            u_L2=float((mag ** 2).sum() * cell_area),
            p_L2=float((pk ** 2).sum() * cell_area),
            max_u=float(mag[i]),
            umax_x=float(g_u[i, 0]), umax_y=float(g_u[i, 1]),
            tip_dx=float(d[tip, 0].mean()), tip_dy=float(d[tip, 1].mean()),
            cyl_dx=float(d[cyl, 0].mean()), cyl_dy=float(d[cyl, 1].mean()),
        ))
    return rows


def rel(x, y):
    return (x - y) / max(abs(x), abs(y), 1e-300)


def fmt(v, w=12, p=5):
    return f"{v:>{w}.{p}e}"


def main():
    a_dir = sys.argv[1] if len(sys.argv) > 1 else "plot/compare_chorin"
    b_dir = sys.argv[2] if len(sys.argv) > 2 else "plot/compare_ipcs"
    la = sys.argv[3] if len(sys.argv) > 3 else "chorin"
    lb = sys.argv[4] if len(sys.argv) > 4 else "ipcs"
    A, B = load(a_dir), load(b_dir)
    n = min(len(A), len(B))
    print("=" * 116)
    print(f"demo_402 (Turek FSI2：圆柱 + 弹性尾巴)  {la} vs {lb}   "
          f"共同输出时刻 {n} 个")
    print("=" * 116)
    print(f"{'t [s]':>8} | {'tip_dx A':>12} {'tip_dx B':>12} {'Δ%':>7} | "
          f"{'max|u| A':>12} {'max|u| B':>12} {'Δ%':>7} | {'u_L2 Δ%':>8} | "
          f"{'argmax|u| A':>15} | {'argmax|u| B':>15}")
    print("-" * 116)
    step = max(1, n // 16)
    for k in list(range(0, n, step)) + [n - 1]:
        ra, rb = A[k], B[k]
        print(f"{ra['t']:>8.4f} | {fmt(ra['tip_dx'])} {fmt(rb['tip_dx'])} "
              f"{rel(ra['tip_dx'], rb['tip_dx'])*100:>7.2f} | "
              f"{fmt(ra['max_u'])} {fmt(rb['max_u'])} "
              f"{rel(ra['max_u'], rb['max_u'])*100:>7.2f} | "
              f"{rel(ra['u_L2'], rb['u_L2'])*100:>8.2f} | "
              f"{ra['umax_x']:>6.2f},{ra['umax_y']:>6.2f} | "
              f"{rb['umax_x']:>6.2f},{rb['umax_y']:>6.2f}")
    print("-" * 116)
    ra, rb = A[n - 1], B[n - 1]
    print(f"末态 t = {ra['t']:.4f} s：")
    for f in ["tip_dx", "tip_dy", "cyl_dx", "cyl_dy", "max_u", "u_L2", "p_L2"]:
        print(f"  {f:<8} {la}={fmt(ra[f])}  {lb}={fmt(rb[f])}  "
              f"相对差={rel(ra[f], rb[f])*100:+.4f} %")

    out_csv = os.path.join(HERE, f"rebuild_{os.path.basename(a_dir)}_vs_"
                                 f"{os.path.basename(b_dir)}.csv")
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        fields = ["t", "tip_dx", "tip_dy", "cyl_dx", "cyl_dy", "max_u",
                  "umax_x", "umax_y", "u_L2", "p_L2"]
        w.writerow([f"{f}_{la}" for f in fields] + [f"{f}_{lb}" for f in fields])
        for k in range(n):
            w.writerow([f"{A[k][f]:.10e}" for f in fields] +
                       [f"{B[k][f]:.10e}" for f in fields])
    print(f"wrote {out_csv}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        from _cjk import setup_cjk
        setup_cjk()
        import matplotlib.pyplot as plt
        t = np.array([r["t"] for r in A[:n]])
        fig, ax = plt.subplots(1, 4, figsize=(19, 4.2))
        for d, lab, st in ((A, la, "-o"), (B, lb, "--s")):
            ax[0].plot(t, [r["tip_dx"] for r in d[:n]], st, ms=2.5, label=lab)
            ax[1].plot(t, [r["tip_dy"] for r in d[:n]], st, ms=2.5, label=lab)
            ax[2].plot(t, [r["max_u"] for r in d[:n]], st, ms=2.5, label=lab)
        ax[0].set_xlabel("t [s]"); ax[0].set_ylabel(r"tip $\Delta x$ [cm]")
        ax[0].set_title("尾尖流向位移"); ax[0].grid(alpha=.3); ax[0].legend()
        ax[1].set_xlabel("t [s]"); ax[1].set_ylabel(r"tip $\Delta y$ [cm]")
        ax[1].set_title("尾尖横向位移"); ax[1].grid(alpha=.3); ax[1].legend()
        ax[2].set_xlabel("t [s]"); ax[2].set_ylabel("max|u| [cm/s]")
        ax[2].set_title("最大速度"); ax[2].grid(alpha=.3); ax[2].legend()
        rr = np.array([rel(A[k]["u_L2"], B[k]["u_L2"]) for k in range(n)]) * 100
        ax[3].plot(t, rr, "-o", ms=2.5)
        ax[3].axhline(0, color="k", lw=.8)
        ax[3].set_xlabel("t [s]"); ax[3].set_ylabel("[%]")
        ax[3].set_title(r"$u_{L2}$ 相对差"); ax[3].grid(alpha=.3)
        fig.suptitle(f"demo_402 Turek FSI2：{la} vs {lb}", y=1.02)
        fig.tight_layout()
        out_png = os.path.join(HERE, f"rebuild_{os.path.basename(a_dir)}_vs_"
                                      f"{os.path.basename(b_dir)}.png")
        fig.savefig(out_png, dpi=160, bbox_inches="tight")
        print(f"wrote {out_png}")
    except Exception as exc:
        print(f"(出图跳过: {type(exc).__name__}: {exc})")


if __name__ == "__main__":
    main()
