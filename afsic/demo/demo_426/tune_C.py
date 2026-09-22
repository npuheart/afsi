#!/usr/bin/env python3
"""Calibrate the tether constant C for demo_426.

The benchmark defines kappa as the SMALLEST stiffness that keeps the Lagrangian
markers within h/2 of their initial position, and then scales it as

    kappa = C * rho * h / dt^2

with C fixed once, from the coarsest discretisation.  This script sweeps kappa,
records the plate displacement history, and reports the resulting C together
with the stability ceiling (the largest kappa that does not blow up).

Run:
    python tune_C.py
"""
from __future__ import annotations

import csv, json, os, subprocess
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RUNNER = "/home/deepseek-harness/afsi/.tools/afsi-run.sh"
STEPS = int(os.environ.get("TUNE_STEPS", "200"))
BETAS = [float(x) for x in os.environ.get(
    "TUNE_BETAS", "64,200,600,2000,6000").split(",")]
OUTHIST = HERE / "plot" / "N32_ipcs_smoke" / "history.csv"

# grid / time-step, must match configuration defaults
N = 32
H = 1.0 / N
DTF = float(os.environ.get("DT_FACTOR", "0.2"))
DT = DTF * H
RHO = 1.0


def run(beta):
    env = dict(os.environ)
    env.update(USE_IMPLICIT_DRAG="0", BETA=repr(beta), DT_FACTOR=repr(DTF),
               SMOKE="1", SMOKE_STEPS=str(STEPS), T_END=repr(STEPS * DT),
               MPLCONFIGDIR="/tmp/mplcache",
               HOME="/home/deepseek-harness/afsi/.home")
    log = HERE / f"_tune_b{beta:g}.log"
    with open(log, "w") as fh:
        subprocess.run([RUNNER, "python", "-B", "-u", "main.py"],
                       cwd=str(HERE), env=env, stdout=fh,
                       stderr=subprocess.STDOUT, check=False)
    rec = {"beta": beta}
    if not OUTHIST.exists():
        return rec
    rows = list(csv.DictReader(open(OUTHIST)))
    d = np.array([float(r["plate_disp"]) for r in rows])
    t = np.array([float(r["t"]) for r in rows])
    u = np.array([float(r["max_u"]) for r in rows])
    finite = np.isfinite(d).all() and np.isfinite(u).all()
    rec["diverged"] = not (finite and d.max() < 1e3 and u.max() < 1e3)
    if not rec["diverged"]:
        rec["d_final"] = float(d[-1])
        rec["d_max"] = float(d.max())
        # steady? compare last quarter mean vs previous quarter mean
        q = len(d) // 4
        rec["d_lastq"] = float(d[-q:].mean())
        rec["d_prevq"] = float(d[-2 * q:-q].mean())
        h4 = len(t) // 2
        rec["drift"] = float(np.polyfit(t[h4:], d[h4:], 1)[0])
    try:
        v = json.load(open(OUTHIST.parent / "verify.json"))
        rec["L2_channel"] = v.get("err_L2_rel_channel")
        rec["plate_u"] = v.get("plate_u_max")
    except Exception:
        pass
    return rec


def main():
    print(f"h={H:g}  dt={DT:g}  h/2={H/2:g}  steps={STEPS}  betas={BETAS}")
    recs = []
    for b in BETAS:
        r = run(b)
        recs.append(r)
        if r.get("diverged", True):
            print(f"  beta={b:<8g} DIVERGED")
        else:
            print(f"  beta={b:<8g} d_final={r['d_final']:.5g} "
                  f"({r['d_final']/(H/2):.2f} x h/2)  drift={r['drift']:+.3g}/s"
                  f"  L2={r.get('L2_channel', float('nan')):.4g}"
                  f"  plate|u|={r.get('plate_u', float('nan')):.4g}")

    with open(HERE / "tune_C.csv", "w", newline="") as fh:
        keys = ["beta", "diverged", "d_final", "d_max", "d_lastq", "d_prevq",
                "drift", "L2_channel", "plate_u"]
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in recs:
            w.writerow(r)
    print("wrote tune_C.csv")

    # smallest kappa meeting d <= h/2
    ok = [r for r in recs if not r.get("diverged", True)
          and r.get("d_final", 1e9) <= H / 2]
    if ok:
        kappa = min(r["beta"] for r in ok)
        C = kappa * DT**2 / (RHO * H)
        print(f"\n--- calibration for THIS scheme ---")
        print(f"  smallest kappa meeting d <= h/2 : {kappa:g}")
        print(f"  C = kappa*dt^2/(rho*h)          : {C:.6g}")
        print(f"  benchmark value                 : 7.5e-2")
        print(f"  ratio (ours / benchmark)        : {C/7.5e-2:.3g}")
    else:
        print("\n  no kappa in the sweep met d <= h/2 while staying stable")


if __name__ == "__main__":
    main()
