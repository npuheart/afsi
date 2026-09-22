#!/usr/bin/env python3
"""Calibrate C for demo_426 -- run to the benchmark's T_final, not a transient.

200-step runs (t=1.25 of a 2.0 s case) only sampled the startup transient of
the stiff spring, which is why the displacement appeared to GROW with kappa.
Here each kappa runs the full T_final = 2.0 (= 20*lambda of the benchmark).
"""
from __future__ import annotations
import csv, json, os, subprocess
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
RUNNER = "/home/deepseek-harness/afsi/.tools/afsi-run.sh"
N, RHO, DTF = 32, 1.0, 0.2
H = 1.0 / N
DT = DTF * H
T_FINAL = float(os.environ.get("T_FINAL", "2.0"))
STEPS = int(round(T_FINAL / DT))
BETAS = [float(x) for x in os.environ.get(
    "TUNE_BETAS", "64,200,600,2000").split(",")]
OUT = HERE / "plot" / "N32_ipcs_smoke" / "history.csv"


def run(beta):
    env = dict(os.environ)
    env.update(USE_IMPLICIT_DRAG="0", BETA=repr(beta), DT_FACTOR=repr(DTF),
               SMOKE="1", SMOKE_STEPS=str(STEPS), T_END=repr(T_FINAL),
               MPLCONFIGDIR="/tmp/mplcache",
               HOME="/home/deepseek-harness/afsi/.home")
    log = HERE / f"_tune2_b{beta:g}.log"
    with open(log, "w") as fh:
        subprocess.run([RUNNER, "python", "-B", "-u", "main.py"], cwd=str(HERE),
                       env=env, stdout=fh, stderr=subprocess.STDOUT, check=False)
    rec = {"beta": beta, "diverged": True}
    if not OUT.exists():
        return rec, None
    # keep this beta's history before the next run overwrites it
    import shutil
    shutil.copy(OUT, HERE / f"hist_beta{beta:g}.csv")
    rows = list(csv.DictReader(open(OUT)))
    t = np.array([float(r["t"]) for r in rows])
    d = np.array([float(r["plate_disp"]) for r in rows])
    u = np.array([float(r["max_u"]) for r in rows])
    if not (np.isfinite(d).all() and np.isfinite(u).all()) or \
            d.max() > 1e3 or u.max() > 1e3:
        return rec, (t, d, u)
    rec.update(diverged=False, d_final=float(d[-1]), d_max=float(d.max()),
               u_final=float(u[-1]))
    q = max(len(d) // 10, 1)
    rec["d_tail_mean"] = float(d[-q:].mean())
    rec["d_tail_std"] = float(d[-q:].std())
    try:
        v = json.load(open(OUT.parent / "verify.json"))
        rec["L2_channel"] = v.get("err_L2_rel_channel")
        rec["plate_u"] = v.get("plate_u_max")
    except Exception:
        pass
    return rec, (t, d, u)


recs, hists = [], []
print(f"h={H:g} dt={DT:g} h/2={H/2:g} T_final={T_FINAL} steps={STEPS}")
for b in BETAS:
    r, h = run(b)
    recs.append(r)
    hists.append((b, h))
    if r["diverged"]:
        print(f"  beta={b:<8g} DIVERGED")
    else:
        print(f"  beta={b:<8g} d_final={r['d_final']:.5g} "
              f"({r['d_final']/(H/2):.2f} x h/2)  d_max={r['d_max']:.4g}  "
              f"tail {r['d_tail_mean']:.4g}+-{r['d_tail_std']:.2g}  "
              f"L2={r.get('L2_channel', float('nan')):.4g}  "
              f"plate|u|={r.get('plate_u', float('nan')):.4g}")

with open(HERE / "tune_C2.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["beta", "diverged", "d_final", "d_max",
                                       "d_tail_mean", "d_tail_std", "u_final",
                                       "L2_channel", "plate_u"],
                       extrasaction="ignore")
    w.writeheader()
    for r in recs:
        w.writerow(r)
print("wrote tune_C2.csv")

ok = [r for r in recs if not r["diverged"] and
      r.get("d_tail_mean", 1e9) <= H / 2]
if ok:
    k = min(r["beta"] for r in ok)
    C = k * DT**2 / (RHO * H)
    print(f"\n--- calibration for THIS scheme (steady state) ---")
    print(f"  smallest kappa with d <= h/2 : {k:g}")
    print(f"  C = kappa*dt^2/(rho*h)       : {C:.6g}   (benchmark 7.5e-2)")
    print(f"  ratio to benchmark           : {C/7.5e-2:.4g}")
else:
    print("\n  none met the h/2 criterion in this sweep")

# plot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(11, 6))
import glob as _glob
for f in sorted(_glob.glob(str(HERE / "hist_beta*.csv")),
                key=lambda p: float(p.split("beta")[-1].split(".csv")[0])):
    b = float(f.split("beta")[-1].split(".csv")[0])
    rows = list(csv.DictReader(open(f)))
    t = np.array([float(r["t"]) for r in rows])
    d = np.array([float(r["plate_disp"]) for r in rows])
    u = np.array([float(r["max_u"]) for r in rows])
    ok = np.isfinite(d) & (d < 1e3)
    if ok.sum() < 2:
        ax.plot([t[0]], [max(d[np.isfinite(d)].max(), 1e-3)], "x", ms=12,
                color="k", label=f"kappa={b:g} DIVERGED")
        continue
    ax.semilogy(t[ok], d[ok], lw=1.8, label=f"kappa={b:g}")
ax.axhline(H / 2, color="k", ls="--", lw=1.3, label=r"criterion $\delta \leq h/2$")
ax.set_xlabel("t (s)"); ax.set_ylabel("max plate displacement (m)")
ax.set_title("demo_426 tether calibration: full T_final = 2.0")
ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=9)
fig.tight_layout(); fig.savefig(HERE / "tune_C2.png", dpi=170)
print("wrote tune_C2.png")
