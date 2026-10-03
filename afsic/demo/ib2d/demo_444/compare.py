"""Compare the AFSI runs with the original IB2d run (6 panels x 6 curves).

    python compare.py [ib2d_reference.npz] [out_dir]

Loads every run present in ``plot/``:

* ``afsi_result.npz``             — rk2 (Peskin two-stage, γ = 100)
* ``afsi_result_chorin_g0.npz``   — fiber + Chorin, γ = 0   (collapse)
* ``afsi_result_chorin_g100.npz`` — fiber + Chorin, γ = 100 (stabilised)
* ``afsi_result_ipcs_g0.npz``     — fiber + IPCS,   γ = 0
* ``afsi_result_ipcs_g100.npz``   — fiber + IPCS,   γ = 100

Writes ``figures/compare_shapes.png`` (6 dump times, 6 curves each),
``figures/compare_history.png``, ``figures/compare_fields.png`` and
``figures/compare_table.csv``.  Metrics per run: ``|dX|max`` (max distance to
the IB2d markers), enclosed ``area`` (IB leakage), relative discrete L2
``e_u`` / ``e_p`` (velocity / pressure on the IB2d grid).
"""
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

here = os.path.dirname(os.path.abspath(__file__))
f_ref = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "ib2d_reference.npz")
out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(here, "figures")
os.makedirs(out, exist_ok=True)

IB2D_STYLE = dict(color="k", ls="-", lw=2.2)
# (label, file, style); only the runs that exist are plotted / tabulated
RUNS = [
    ("AFSI rk2 (Peskin)", os.path.join(here, "plot", "afsi_result.npz"),
     dict(color="tab:red", ls=(0, (5, 2)), lw=1.4)),
    ("fiber Chorin, γ=0", os.path.join(here, "plot", "afsi_result_chorin_g0.npz"),
     dict(color="tab:orange", ls=(0, (1.5, 1.5)), lw=1.1)),
    ("fiber Chorin, γ=100", os.path.join(here, "plot", "afsi_result_chorin_g100.npz"),
     dict(color="tab:orange", ls="--", lw=1.4)),
    ("fiber IPCS, γ=0", os.path.join(here, "plot", "afsi_result_ipcs_g0.npz"),
     dict(color="tab:blue", ls=(0, (1.5, 1.5)), lw=1.1)),
    ("fiber IPCS, γ=100", os.path.join(here, "plot", "afsi_result_ipcs_g100.npz"),
     dict(color="tab:blue", ls="--", lw=1.4)),
]

R = np.load(f_ref)
runs = []
for label, path, style in RUNS:
    if os.path.exists(path):
        runs.append(dict(label=label, A=np.load(path), style=style))
    else:
        print(f"[compare] missing {os.path.basename(path)} — skipped")
if not runs:
    sys.exit("no AFSI run found in plot/ (run main.py first)")


def area(X):
    return 0.5 * abs(np.sum(X[:, 0] * np.roll(X[:, 1], -1) - np.roll(X[:, 0], -1) * X[:, 1]))


# per-run metrics against IB2d at the common dump times ---------------------
for r in runs:
    A = r["A"]
    ms = []                       # [(index in run, index in reference)]
    for iaa, ta in enumerate(A["t"]):
        irr = int(np.argmin(abs(R["t"] - ta)))
        if abs(R["t"][irr] - ta) < 1e-8:
            ms.append((iaa, irr))
    assert len(ms) >= 2, f"no common output times for {r['label']}"
    r["ms"] = ms
    r["t"] = A["t"][[m[0] for m in ms]]
    r["area"] = np.array([area(A["X"][iaa]) for iaa, _ in ms])
    r["area_ib2d"] = np.array([area(R["X"][irr]) for _, irr in ms])
    r["dX"] = np.array([np.linalg.norm(A["X"][iaa] - R["X"][irr], axis=1).max()
                        for iaa, irr in ms])
    if "u" in A and "u" in R:
        eu, ep = [], []
        for iaa, irr in ms:
            ua, ur = A["u"][iaa], R["u"][irr]
            nr = np.sqrt((ur ** 2).sum())
            eu.append(np.sqrt(((ua - ur) ** 2).sum()) / nr if nr > 0 else np.nan)
            pa, pr = A["p"][iaa], R["p"][irr]
            npr = np.sqrt((pr ** 2).sum())
            ep.append(np.sqrt(((pa - pr) ** 2).sum()) / npr if npr > 0 else np.nan)
        r["e_u"], r["e_p"] = np.array(eu), np.array(ep)

# table: one block per run + a combined csv
print(f"{'run':>22} {'t':>6} {'|dX|max':>10} {'area':>9} {'area IB2d':>10} "
      f"{'e_u':>9} {'e_p':>9} {'max|u|':>9}")
csv_rows = []
for r in runs:
    A = r["A"]
    for k, (iaa, irr) in enumerate(r["ms"]):
        eu = r["e_u"][k] if "e_u" in r else np.nan
        ep = r["e_p"][k] if "e_p" in r else np.nan
        umax = np.abs(A["u"][iaa]).max() if "u" in A else np.nan
        if k % 5 == 0 or k == len(r["ms"]) - 1:   # print every 0.1 s
            print(f"{r['label']:>22} {A['t'][iaa]:6.3f} {r['dX'][k]:10.3e} "
                  f"{r['area'][k]:9.6f} {r['area_ib2d'][k]:10.6f} "
                  f"{eu:9.3e} {ep:9.3e} {umax:9.4e}")
        csv_rows.append((r["label"].replace(",", " "), A["t"][iaa], r["dX"][k],
                         r["area"][k], r["area_ib2d"][k], eu, ep, umax))
with open(os.path.join(out, "compare_table.csv"), "w") as fh:
    fh.write("run,t,dXmax,area,area_ib2d,e_u,e_p,umax\n")
    for row in csv_rows:
        fh.write(",".join([str(row[0])] + [f"{v:.10g}" for v in row[1:]]) + "\n")

# --- 1. band shapes: 6 dump times, 6 curves each ---------------------------
sel_times = [0.0, 0.04, 0.1, 0.2, 0.5, 1.5]
fig, axes = plt.subplots(1, len(sel_times), figsize=(2.9 * len(sel_times), 3.2),
                         sharey=True)
axes = np.atleast_1d(axes)
for ax, tt in zip(axes, sel_times):
    irr = int(np.argmin(abs(R["t"] - tt)))
    Xr = R["X"][irr]
    ax.plot(*np.vstack([Xr, Xr[:1]]).T, label="IB2d", **IB2D_STYLE)
    for r in runs:
        iaa = int(np.argmin(abs(r["A"]["t"] - tt)))
        Xa = r["A"]["X"][iaa]
        ax.plot(*np.vstack([Xa, Xa[:1]]).T, label=r["label"], **r["style"])
    ax.set_title(f"t = {tt:.2f}")
    ax.set_aspect("equal")
    ax.set_xlim(0.05, 0.95)
    ax.set_ylim(0.05, 0.95)
axes[0].legend(loc="lower left", fontsize=6.0, framealpha=0.9)
fig.tight_layout()
fig.savefig(os.path.join(out, "compare_shapes.png"), dpi=150)

# --- 2. time histories: enclosed area, |dX|max, e_u ------------------------
fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
ax[0].plot(R["t"], [area(X) for X in R["X"]], label="IB2d", **IB2D_STYLE)
for r in runs:
    ax[0].plot(r["t"], r["area"], label=r["label"], **r["style"])
ax[0].set_xlabel("t")
ax[0].set_ylabel("enclosed area")
ax[0].set_title("IB volume leakage")
ax[0].legend(fontsize=7)
for r in runs:
    ax[1].semilogy(r["t"][1:], r["dX"][1:], label=r["label"], **r["style"])
ax[1].axhline(R["dx"] if "dx" in R else 1 / 32, color="gray", ls=":",
              label="grid spacing h")
ax[1].set_xlabel("t")
ax[1].set_title(r"$\max_k|X_k^{AFSI}-X_k^{IB2d}|$")
ax[1].legend(fontsize=7)
for r in runs:
    if "e_u" in r:
        ax[2].semilogy(r["t"][1:], r["e_u"][1:], label=r["label"], **r["style"])
ax[2].set_xlabel("t")
ax[2].set_title(r"$e_u$ (rel. $L^2$, IB2d grid)")
ax[2].legend(fontsize=7)
fig.tight_layout()
fig.savefig(os.path.join(out, "compare_history.png"), dpi=150)

# --- 3. velocity / pressure fields at t = 0.1 (rk2 vs IB2d) -----------------
rk2 = next((r for r in runs if "rk2" in r["label"]), None)
if rk2 is not None and "u" in R:
    A = rk2["A"]
    k = int(np.argmin(abs(A["t"] - 0.1)))
    irr = int(np.argmin(abs(R["t"] - A["t"][k])))
    ua, ur = A["u"][k], R["u"][irr]
    pa, pr = A["p"][k], R["p"][irr]
    ny, nx = pr.shape
    xg = np.arange(nx) * float(R["dx"])
    yg = np.arange(ny) * float(R["dy"])
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    vmin, vmax = pr.min(), pr.max()
    for a_, pp, uu, title in ((ax[0], pr, ur, "IB2d"),
                              (ax[1], pa, ua, "AFSI rk2")):
        c = a_.contourf(xg, yg, pp, 30, cmap="RdBu_r", vmin=vmin, vmax=vmax)
        a_.quiver(xg[::2], yg[::2], uu[::2, ::2, 0], uu[::2, ::2, 1], scale=40)
        a_.set_title(f"{title}: p and u, t = {A['t'][k]:.2f}")
        a_.set_aspect("equal")
        fig.colorbar(c, ax=a_, shrink=0.8)
    c = ax[2].contourf(xg, yg, np.linalg.norm(ua - ur, axis=-1), 30, cmap="viridis")
    ax[2].set_title(r"$|u^{AFSI}-u^{IB2d}|$")
    ax[2].set_aspect("equal")
    fig.colorbar(c, ax=ax[2], shrink=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "compare_fields.png"), dpi=150)

print(f"figures written to {out}")
