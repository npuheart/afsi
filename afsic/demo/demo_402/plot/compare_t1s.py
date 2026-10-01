#!/usr/bin/env python
"""对比 demo_402 的 T=1 s 运行（chorin vs IPCS，论文对齐配置）。

用法：python compare_t1s.py [run_dir] [--tag 名字]
默认读取 <demo>/plot/t1s/{chorin,ipcs}/history.csv，输出终端表格 + PNG。
"""
import csv
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "t1s")

RUNS = {"chorin": os.path.join(BASE, "chorin", "history.csv"),
        "ipcs": os.path.join(BASE, "ipcs", "history.csv")}


def load(path):
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    data = {k: [] for k in rows[0]}
    for r in rows:
        for k, v in r.items():
            data[k].append(float(v))
    import numpy as np
    return {k: np.array(v) for k, v in data.items()}


def metrics(d):
    """t=1s 运行的关键指标。"""
    m = {}
    m["tip_dy_max_abs"] = float(abs(d["tip_dy"]).max())
    m["tip_dy_min"] = float(d["tip_dy"].min())
    m["tip_dy_max"] = float(d["tip_dy"].max())
    m["u_L2_final"] = float(d["u_L2"][-1])
    m["max_u_final"] = float(d["max_u"][-1])
    m["cyl_dx_max_abs"] = float(abs(d["cyl_dx"]).max())
    m["cyl_dy_max_abs"] = float(abs(d["cyl_dy"]).max())
    # 后半段（0.5-1.0 s，起振后）尾尖振荡的峰峰值与过零频率
    import numpy as np
    sel = d["t"] > 0.5
    ty = d["tip_dy"][sel]
    t = d["t"][sel]
    m["pp_after_0.5"] = float(ty.max() - ty.min())
    sign = np.sign(ty - ty.mean())
    cross = np.sum(np.abs(np.diff(sign)) > 0)
    m["freq_hz_est"] = float(cross / 2.0 / (t[-1] - t[0])) if t[-1] > t[0] else float("nan")
    return m


def main():
    data = {}
    for name, path in RUNS.items():
        if not os.path.exists(path):
            print(f"[skip] {name}: {path} 不存在")
            continue
        data[name] = load(path)

    keys = ["tip_dy_max_abs", "tip_dy_min", "tip_dy_max", "pp_after_0.5",
            "freq_hz_est", "u_L2_final", "max_u_final",
            "cyl_dx_max_abs", "cyl_dy_max_abs"]
    print(f"{'指标':>16} | " + " | ".join(f"{n:>14}" for n in data))
    for k in keys:
        row = " | ".join(f"{metrics(d)[k]:14.6g}" for d in data.values())
        print(f"{k:>16} | {row}")

    # 场量差异（如果有共同的时间网格）
    if len(data) == 2:
        n1, n2 = list(data)
        d1, d2 = data[n1], data[n2]
        m = min(len(d1["t"]), len(d2["t"]))
        rel = abs(d1["u_L2"][:m] - d2["u_L2"][:m]) / d1["u_L2"][:m]
        print(f"\nu_L2 逐帧相对差: 中位 {100*rel.mean():.4f}%  最大 {100*rel.max():.4f}%")

    # 图：tip_dy(t) + max_u(t)
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    for n, d in data.items():
        axes[0].plot(d["t"], d["tip_dy"], label=n)
        axes[1].plot(d["t"], d["u_L2"] / 1e8, label=n)
        axes[2].plot(d["t"], d["max_u"], label=n)
    axes[0].set_ylabel("tip $\\Delta y$ [cm]")
    axes[1].set_ylabel("$\\int u^2\\,dx$ [$10^8$]")
    axes[2].set_ylabel("$\\max|u|$ [cm/s]")
    axes[2].set_xlabel("t [s]")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle("demo_402 Turek(modified) T=1s: chorin vs IPCS")
    fig.tight_layout()
    out = os.path.join(BASE, "compare_t1s.png")
    fig.savefig(out, dpi=130)
    print(f"\nfigure -> {out}")


if __name__ == "__main__":
    main()
