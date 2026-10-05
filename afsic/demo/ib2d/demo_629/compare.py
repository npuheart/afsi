"""Compare AFSI's coagulation port with the pyIB2d reference.

    python compare.py [ib2d_reference.npz] [out_dir]

Inputs: ``plot/afsi_result.npz`` (main.py) and ``ib2d_reference.npz``
(run_reference.py).

Frame alignment: the pyIB2d driver dumps the initial state before its loop and
then when its 0-based step counter equals a positive multiple of ``print_dump``
-- i.e. after ``print_dump``, ``2*print_dump``, ... *completed* steps -- and
``ib2d_reference.py`` stamps those frames ``k*print_dump*dt``, which is the true
time of the dumped state.  ``main.py`` mirrors exactly that schedule, so frame
``k`` of both codes is the *same step*; the alignment is checked here by
``round(t/dt)`` equality (frames that do not match are skipped).

Reported per frame: marker deviation, velocity agreement, the **bond state**
(number of active bonds, how many of them both codes share, their resting
lengths and current lengths), the peak coagulation force density, the stirred
target anchors and -- as an independent geometry check that does not use either
code's state machine -- the number of cell pairs whose *centre* distance is
below the bond threshold, recomputed from the saved marker positions.

The bond model itself is validated by
``ib2d_input/coagulation_model.analytic_self_test`` (closed-form force for two
synthetic cells), whose result is printed first.

Writes ``figures/compare_*.png`` and ``figures/compare_table.csv``.
"""
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

here = os.path.dirname(os.path.abspath(__file__))
f_ref = (sys.argv[1] if len(sys.argv) > 1
         else os.path.join(here, "ib2d_reference.npz"))
out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(here, "figures")
os.makedirs(out, exist_ok=True)

f_afsi = os.path.join(here, "plot", "afsi_result.npz")

sys.path.insert(0, here)
sys.path.insert(0, os.path.join(here, "ib2d_input"))
import ib2d_io                                               # noqa: E402
from coagulation_model import Coagulation, analytic_self_test  # noqa: E402

P = ib2d_io.read_input2d(os.path.join(here, "ib2d_input", "input2d"))
Lx, Ly = float(P["Lx"]), float(P["Ly"])
NAME = P.get("string_name", "channel")
H = Lx / int(P["Nx"])
spec = ib2d_io.read_coagulation(
    os.path.join(here, "ib2d_input", f"{NAME}.coagulation"))
COAG_ON = bool(int(os.environ["COAGULATION"])) \
    if os.environ.get("COAGULATION") is not None \
    else bool(int(P.get("coagulation", 0)))
coag = Coagulation(spec, Lx, Ly, index_base=1)


def cell_centers(X):
    i1 = coag.first
    i2 = coag.first + coag.counts // 2
    return 0.5 * (X[i1] + X[i2])


def close_pairs(X):
    """Cell pairs below the bond threshold, recomputed from saved markers."""
    C = cell_centers(X)
    n = len(coag.counts)
    return {(i, j) for i in range(n) for j in range(i + 1, n)
            if np.hypot(C[i, 0] - C[j, 0], C[i, 1] - C[j, 1]) < coag.threshold}


def bond_keys(bonds):
    return {(int(b[0]), int(b[1])) for b in bonds}


def bond_len(X, b):
    mi, si = int(b[2]), int(b[3])
    return float(np.hypot(X[si, 0] - X[mi, 0], X[si, 1] - X[mi, 1]))


ok, detail = analytic_self_test()
print(f"coagulation model self-test: {'PASS' if ok else 'FAIL'}  {detail}")
print(f"upstream coagulation flag  : {'on' if COAG_ON else 'off'}   "
      f"({len(spec['counts'])} cells, threshold {spec['threshold']:.6g}, "
      f"bond_k {spec['bond_k']:g}, fracture {spec['fracture']:g})")

A = np.load(f_afsi)
R = np.load(f_ref)
for k in ("t", "X", "u", "f", "nbonds", "bonds", "fcoag", "Xa", "dt", "nx"):
    assert k in A, f"afsi_result.npz lacks {k}"
    assert k in R, f"reference npz lacks {k}"

dt = float(A["dt"])
steps_a = np.rint(A["t"] / dt).astype(int)
steps_r = np.rint(R["t"] / dt).astype(int)
finite_a = {k: bool(np.isfinite(A["X"][k]).all()) for k in range(len(steps_a))}
finite_r = {k: bool(np.isfinite(R["X"][k]).all()) for k in range(len(steps_r))}
keep = [k for k in range(min(len(steps_a), len(steps_r)))
        if steps_a[k] == steps_r[k] and finite_a[k] and finite_r[k]]
dropped = [k for k in range(min(len(steps_a), len(steps_r)))
           if steps_a[k] == steps_r[k] and not (finite_a[k] and finite_r[k])]
if dropped:
    print(f"note: dropped {len(dropped)} aligned frame(s) that are not finite "
          f"(divergence): t = "
          f"{[round(float(A['t'][k]), 6) for k in dropped]}")
assert keep, f"no aligned frames: AFSI {steps_a[:6]} vs reference {steps_r[:6]}"
if len(keep) < len(steps_a) or len(keep) < len(steps_r):
    print(f"note: {len(steps_a)} AFSI / {len(steps_r)} reference frames, "
          f"{len(keep)} aligned")

# bond rows are stored concatenated per frame
def frame_bonds(D, k):
    n0 = int(np.sum(D["nbonds"][:k]))
    return D["bonds"][n0:n0 + int(D["nbonds"][k])]


rows = []
for k in keep:
    Xa, Xr = A["X"][k], R["X"][k]
    du = np.linalg.norm(A["u"][k] - R["u"][k], axis=-1)
    ba, br = frame_bonds(A, k), frame_bonds(R, k)
    ka, kr = bond_keys(ba), bond_keys(br)
    both = ka & kr
    ia = {(int(b[0]), int(b[1])): i for i, b in enumerate(ba)}
    ir = {(int(b[0]), int(b[1])): i for i, b in enumerate(br)}
    dla = [np.abs(bond_len(Xa, ba[ia[key]]) - bond_len(Xr, br[ir[key]]))
           for key in both]
    # the bond resting length is frozen at formation: a small difference in the
    # formation instant makes |L - RL| (and hence the bond force) differ
    drl = [np.abs(float(ba[ia[key]][4]) - float(br[ir[key]][4])) for key in both]
    rows.append(dict(
        t=A["t"][k],
        dXmax=np.linalg.norm(Xa - Xr, axis=1).max(),
        umax_a=np.abs(A["u"][k]).max(), umax_r=np.abs(R["u"][k]).max(),
        dumax=du.max(),
        nb_a=len(ba), nb_r=len(br), nb_both=len(both), dLmax=max(dla) if dla else 0.0,
        dRLmax=max(drl) if drl else 0.0,
        dfmax=float(np.abs(A["f"][k] - R["f"][k]).max()),
        fmax_r=float(np.abs(R["f"][k]).max()),
        fcoag_a=float(A["fcoag"][k]), fcoag_r=float(R["fcoag"][k]),
        dXa=float(np.abs(A["Xa"][k] - R["Xa"][k]).max()),
        fL2=float(np.linalg.norm(A["f"][k] - R["f"][k])
                  / max(np.linalg.norm(R["f"][k]), 1e-300)),
        pairs_a=len(close_pairs(Xa)), pairs_r=len(close_pairs(Xr)),
        unmatched=len(ka ^ kr)))

hdr = (f"{'t':>8} {'dXmax/h':>9} {'umax a/r':>17} {'max|du|':>9} "
       f"{'bonds a/r':>11} {'shared':>7} {'dRLmax':>9} {'dLmax':>9} "
       f"{'f_coag a/r':>19} {'dXa':>8} {'df/|f|max':>10} {'pairs a/r':>10}")
print(f"\n=== coagulation comparison (h = {H:g}, {len(keep)} aligned frames, "
      f"t = {rows[0]['t']:.5f} .. {rows[-1]['t']:.5f})")
print(hdr)
for r in rows:
    print(f"{r['t']:8.5f} {r['dXmax'] / H:9.2e} "
          f"{r['umax_a']:8.3e}/{r['umax_r']:<8.3e} {r['dumax']:9.2e} "
          f"{r['nb_a']:5d}/{r['nb_r']:<5d} {r['nb_both']:7d} "
          f"{r['dRLmax']:9.2e} {r['dLmax']:9.2e} "
          f"{r['fcoag_a']:9.3e}/{r['fcoag_r']:<9.3e} "
          f"{r['dXa']:8.1e} {r['dfmax'] / max(r['fmax_r'], 1e-300):10.2e} "
          f"{r['pairs_a']:4d}/{r['pairs_r']:<4d}")

r = rows[-1]
print(f"\nmarkers      : max|dX| {r['dXmax']:.3e} m = {r['dXmax'] / H:.2e} h")
print(f"velocity     : max|u| AFSI {r['umax_a']:.4e} / IB2d {r['umax_r']:.4e}"
      f"  ({abs(r['umax_a'] - r['umax_r']) / max(r['umax_r'], 1e-300):.2e} rel),"
      f" max|du| {r['dumax']:.3e}")
print(f"Eulerian f   : relative L2 {r['fL2']:.3e}, "
      f"max|df| {r['dfmax']:.3e} = {r['dfmax'] / max(r['fmax_r'], 1e-300):.2e} "
      f"max|f_IB2d|")
print(f"bonds        : AFSI {r['nb_a']} / IB2d {r['nb_r']}, shared "
      f"{r['nb_both']}, differing {r['unmatched']}, max|dL| "
      f"{r['dLmax']:.3e}, max|dRL| {r['dRLmax']:.3e}")
print("               (the bond resting length RL is frozen at formation, so a "
      "small difference in the formation instant shows up in |L-RL| and in "
      "f_coag)")
print(f"coag force   : max|f_coag| AFSI {r['fcoag_a']:.4e} / IB2d "
      f"{r['fcoag_r']:.4e}")
print(f"targets      : max|dXa| {r['dXa']:.3e} (same .m in both codes)")
print(f"geometry chk : cell pairs below threshold AFSI {r['pairs_a']} / IB2d "
      f"{r['pairs_r']} (recomputed from the saved markers; a bonded pair that "
      f"is still below threshold is consistent)")

with open(os.path.join(out, "compare_table.csv"), "w") as fh:
    fh.write("t,dXmax_over_h,umax_afsi,umax_ib2d,max_du,nbonds_afsi,"
             "nbonds_ib2d,bonds_shared,bond_unmatched,max_dbond_len,max_dRL,"
             "fcoag_afsi,fcoag_ib2d,dXanchor,relL2f,max_df,pairs_afsi,"
             "pairs_ib2d\n")
    for x in rows:
        fh.write(f"{x['t']:.8e},{x['dXmax'] / H:.8e},{x['umax_a']:.8e},"
                 f"{x['umax_r']:.8e},{x['dumax']:.8e},{x['nb_a']},{x['nb_r']},"
                 f"{x['nb_both']},{x['unmatched']},{x['dLmax']:.8e},"
                 f"{x['dRLmax']:.8e},"
                 f"{x['fcoag_a']:.8e},{x['fcoag_r']:.8e},{x['dXa']:.8e},"
                 f"{x['fL2']:.8e},{x['dfmax']:.8e},{x['pairs_a']},"
                 f"{x['pairs_r']}\n")

# ------------------------------------------------------------------ figures
t = [x["t"] for x in rows]
fig, ax = plt.subplots(figsize=(9, 3))
Xa, Xr = A["X"][keep[-1]], R["X"][keep[-1]]
ax.plot(Xr[:, 0], Xr[:, 1], "k.", ms=1, label="IB2d")
ax.plot(Xa[:, 0], Xa[:, 1], "r.", ms=1, alpha=0.6, label="AFSI")
for b in frame_bonds(A, keep[-1]):
    mi, si = int(b[2]), int(b[3])
    ax.plot([Xa[mi, 0], Xa[si, 0]], [Xa[mi, 1], Xa[si, 1]], "-", color="b", lw=0.8)
ax.set(aspect="equal", xlabel="x", ylabel="y",
       title=f"markers + AFSI bonds at t = {t[-1]:.4f}")
ax.legend(loc="upper right", fontsize=8, markerscale=8)
fig.tight_layout()
fig.savefig(os.path.join(out, "compare_markers.png"), dpi=130)
plt.close(fig)

fig, ax = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
ax[0].plot(t, [x["nb_a"] for x in rows], "r.-", label="AFSI")
ax[0].plot(t, [x["nb_r"] for x in rows], "k.--", label="IB2d")
ax[0].set_ylabel("# bonds")
ax[0].legend(fontsize=8)
ax[1].plot(t, [x["dXmax"] / H for x in rows], "b.-", label="max|dX|/h")
ax[1].plot(t, [x["dumax"] for x in rows], "g.--", label="max|du|")
ax[1].set_xlabel("t")
ax[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(out, "compare_history.png"), dpi=130)
plt.close(fig)

# field comparison at the last aligned frame
k = keep[-1]
fig, ax = plt.subplots(2, 2, figsize=(10, 5))
for a_, D, tag in ((ax[0, 0], A, "AFSI"), (ax[0, 1], R, "IB2d")):
    m = a_.imshow(np.abs(D["f"][k]).max(-1), origin="lower",
                  extent=[0, Lx, 0, Ly], aspect="auto")
    a_.set_title(f"{tag} |f| t={A['t'][k]:.4f}")
    fig.colorbar(m, ax=a_, shrink=0.8)
mm = ax[1, 0].imshow(np.abs(A["f"][k] - R["f"][k]).max(-1), origin="lower",
                     extent=[0, Lx, 0, Ly], aspect="auto")
ax[1, 0].set_title("|df|")
fig.colorbar(mm, ax=ax[1, 0], shrink=0.8)
mm = ax[1, 1].imshow(np.linalg.norm(A["u"][k] - R["u"][k], axis=-1),
                     origin="lower", extent=[0, Lx, 0, Ly], aspect="auto")
ax[1, 1].set_title("|du|")
fig.colorbar(mm, ax=ax[1, 1], shrink=0.8)
fig.tight_layout()
fig.savefig(os.path.join(out, "compare_fields.png"), dpi=130)
plt.close(fig)
print(f"\nwrote {out}/compare_table.csv, compare_markers.png, "
      f"compare_history.png, compare_fields.png")
