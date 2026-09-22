#!/usr/bin/env python3
"""Sweep the pure-tether penalty (BETA, DAMP=0) for demo_426 and plot it.

Motivation
----------
The benchmark's Eq. (14) has three parts -- penalty stiffness (a tether), a
penalty body force and damping.  AFSI's tether is the stiffness part, so the
question is whether a PURE tether (DAMP = 0), with the plate advected by the
interpolated fluid velocity, can hold the plates still enough.

This script runs the case for several BETA (and optionally DAMP) values,
collects the plate displacement history and the error metrics from verify.json,
and writes a summary figure plus a companion CSV.

Run:
    python sweep_tether.py
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RUNNER = "/home/deepseek-harness/afsi/.tools/afsi-run.sh"
CASE_DIR = HERE.parent

STEPS = int(os.environ.get("SWEEP_STEPS", "600"))
T_END = float(os.environ.get("SWEEP_T_END", "0.5"))

# (BETA, DAMP) pairs: the pure tether is DAMP = 0
CONFIGS = [
    (1.0e-4, 0.0),
    (1.0e-2, 0.0),
    (1.0e-1, 0.0),
    (1.0, 0.0),
    (1.0e2, 0.0),
    (1.0e4, 0.0),
]


def run(beta, damp):
    outdir = HERE / f"_sweep_b{beta:g}_d{damp:g}"
    outdir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update(
        USE_IMPLICIT_DRAG="0",
        BETA=repr(beta),
        DAMP=repr(damp),
        SMOKE="1",
        SMOKE_STEPS=str(STEPS),
        T_END=repr(T_END),
        MPLCONFIGDIR="/tmp/mplcache",
        HOME="/home/deepseek-harness/afsi/.home",
    )
    log = outdir / "run.log"
    with open(log, "w") as fh:
        subprocess.run([RUNNER, "python", "-B", "-u", "main.py"],
                       cwd=str(CASE_DIR), env=env, stdout=fh,
                       stderr=subprocess.STDOUT, check=False)

    text = log.read_text()
    rec = {"beta": beta, "damp": damp, "diverged": True}
    # history.csv lives in the case output dir chosen by configuration.py
    hist = CASE_DIR / "plot" / "N32_ipcs_smoke" / "history.csv"
    vjson = CASE_DIR / "plot" / "N32_ipcs_smoke" / "verify.json"
    if hist.exists():
        rows = list(csv.DictReader(open(hist)))
        t = np.array([float(r["t"]) for r in rows])
        pmax = np.array([float(r.get("plate_disp", "nan")) for r in rows])
        umax = np.array([float(r["max_u"]) for r in rows])
        finite = np.isfinite(pmax).all() and np.isfinite(umax).all()
        if finite and pmax.max() < 1e3 and umax.max() < 1e3:
            rec["diverged"] = False
            rec["disp_final"] = float(pmax[-1])
            rec["disp_max"] = float(pmax.max())
            # linear drift rate over the second half of the run
            h = len(t) // 2
            rec["drift_rate"] = float(
                np.polyfit(t[h:], pmax[h:], 1)[0]) if h > 1 else np.nan
            rec["u_max_final"] = float(umax[-1])
    if vjson.exists():
        d = json.load(open(vjson))
        rec["err_L2_channel"] = d.get("err_L2_rel_channel", np.nan)
        rec["u_max_rel_err"] = d.get("u_max_rel_err", np.nan)
        rec["plate_u_max"] = d.get("plate_u_max", np.nan)
    # save the history for replotting
    if hist.exists():
        np.savetxt(outdir / "history.csv",
                   np.genfromtxt(hist, delimiter=",", names=True,
                                 dtype=None, encoding="utf-8"),
                   delimiter=",", header="step,t,max_u,plate_disp",
                   comments="")
    print(f"  BETA={beta:<9g} DAMP={damp:<5g} -> "
          f"{'DIVERGED' if rec['diverged'] else 'stable'}"
          + ("" if rec["diverged"] else
             f"  disp_final={rec['disp_final']:.4g}"
             f"  drift={rec.get('drift_rate', float('nan')):.4g}/s"
             f"  L2={rec.get('err_L2_channel', float('nan')):.4g}"
             f"  plate|u|={rec.get('plate_u_max', float('nan')):.4g}"))
    return rec, outdir


def main():
    print(f"sweeping {len(CONFIGS)} tether configs, {STEPS} steps each")
    recs, dirs = [], []
    for beta, damp in CONFIGS:
        r, d = run(beta, damp)
        recs.append(r)
        dirs.append(d)

    with open(HERE / "sweep_tether.csv", "w", newline="") as fh:
        keys = ["beta", "damp", "diverged", "disp_final", "disp_max",
                "drift_rate", "u_max_final", "err_L2_channel", "u_max_rel_err",
                "plate_u_max"]
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in recs:
            w.writerow(r)
    print("wrote sweep_tether.csv")

    betas = np.array([r["beta"] for r in recs])
    stab = np.array([not r["diverged"] for r in recs])

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0))

    # (a) plate displacement history
    ax = axes[0]
    for r, d in zip(recs, dirs):
        f = d / "history.csv"
        if not f.exists():
            continue
        dat = np.genfromtxt(f, delimiter=",", names=True)
        t = np.atleast_1d(dat["t"])
        p = np.atleast_1d(dat["plate_disp"])
        if not np.isfinite(p).all() or p.max() > 1e3:
            ax.plot(t[:1], p[:1], "x", ms=10, color="k",
                    label=f"b={r['beta']:g} DIVERGED")
            continue
        ax.plot(t, p, lw=1.8, label=f"b={r['beta']:g}")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("max plate displacement (m)")
    ax.set_title("(a) pure tether: plate drift\n(channel width = 1.155)")
    ax.axhline(1.155, color="grey", ls=":", lw=1)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    # (b) feasibility: drift rate and resulting channel error
    ax = axes[1]
    ok = stab
    ax.loglog(betas[ok], [recs[i]["disp_final"] for i in np.where(ok)[0]],
              "o-", color="tab:red", label="plate displacement (final)")
    ax.loglog(betas[ok], [abs(recs[i].get("err_L2_channel", np.nan))
                          for i in np.where(ok)[0]],
              "s-", color="tab:blue", label="channel relative L2 error")
    ax.loglog(betas[~ok], np.full((~ok).sum(), 1.0), "x", ms=12, color="k",
              label="diverged")
    ax.set_xlabel(r"$\beta$ (tether stiffness)")
    ax.set_ylabel("magnitude")
    ax.set_title("(b) accuracy vs stiffness")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)

    # (c) the constraint requirement: beta*delta must balance the pressure load
    ax = axes[2]
    dpD = 1.0 * (1.0 / np.cos(np.pi / 6))     # dP/L * D  ~ 1.155
    ax.loglog(betas[ok], [recs[i]["disp_final"] for i in np.where(ok)[0]],
              "o-", color="tab:red", label=r"measured $\delta$")
    bb = np.array([b for b in betas[ok]])
    ax.loglog(bb, dpD / bb, "k--", lw=1.4,
              label=r"needed for balance: $\Delta p\,D/\beta$")
    ax.loglog(betas[~ok], np.full((~ok).sum(), 1.155), "x", ms=12, color="k",
              label="diverged")
    ax.set_xlabel(r"$\beta$ (tether stiffness)")
    ax.set_ylabel(r"plate displacement $\delta$ (m)")
    ax.set_title("(c) does the tether carry the load?\n"
                 r"($\beta\,\delta \approx \Delta p\,D = 1.155$ ?)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)

    fig.suptitle("demo_426 — pure tether penalty on the plates (DAMP = 0)",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = HERE / "sweep_tether.png"
    fig.savefig(out, dpi=170)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
