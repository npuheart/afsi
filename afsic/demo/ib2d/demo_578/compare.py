"""Compare the AFSI Boussinesq run with the pyIB2d reference.

    python compare.py [ib2d_reference.npz] [out_dir]

Inputs: ``plot/afsi_result.npz`` (main.py) and ``ib2d_reference.npz``
(run_reference.py).  Both codes (a) evaluate the *same* ported
``please_Compute_External_Forcing.py`` and (b) call the *same* ported
``please_Form_Boussinesq_Forcing_Terms`` module, so the comparison isolates the
solver, the immersed-boundary machinery and the scalar scheme.

Frame alignment: the pyIB2d driver dumps when its 0-based step counter
``cter`` is a positive multiple of ``print_dump`` -- i.e. at the end of step
``cter + 1`` -- and stamps that frame with ``cter * dt``, one step behind the
state it writes (upstream labelling quirk).  ``main.py`` dumps on exactly those
steps and stamps the true time, so AFSI frame ``k`` and reference frame ``k``
are the same step of the simulation whenever
``round(t_afsi/dt) == round(t_ref/dt) + 1``; only matching frames are used.

Metrics reported per frame (and for the last frame in the CSV/figures):

* marker deviation ``max|X_afsi - X_ib2d|`` (markers are inert in this example,
  so this is an advection diagnostic);
* velocity ``relL2`` and ``max|du|``, plus the buoyancy-driven vertical
  velocity ``max|u_y|`` and the ``L2`` norm of ``u_y`` (the flow starts from
  rest and is driven only by the buoyancy term and the penalty force);
* concentration ``relL2``, ``max|dC|``, peak ``max C`` and total ``sum C``;
* a check that the buoyancy term actually added to AFSI's force field is
  ``rho * fBouss * C`` (median of ``f_y / C`` over the mask, outside the IB
  band, must equal ``rho * exp_Coeff * g * gy``).
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

f_afsi = os.environ.get("AFSI_NPZ", os.path.join(here, "plot", "afsi_result.npz"))

sys.path.insert(0, here)
sys.path.insert(0, os.path.join(here, "ib2d_input"))
import ib2d_io  # noqa: E402

P = ib2d_io.read_input2d(os.path.join(here, "ib2d_input", "input2d"))
Lx, Ly = float(P["Lx"]), float(P["Ly"])
rho = float(P["rho"])

A = np.load(f_afsi)
R = np.load(f_ref)
dt = float(A["dt"])
nx, ny = int(A["nx"]), int(A["ny"])
h = Lx / nx
assert (int(R["nx"]), int(R["ny"])) == (nx, ny), "grid mismatch"

steps_a = np.rint(A["t"] / dt).astype(int)
steps_r = np.rint(R["t"] / dt).astype(int)
keep = [k for k in range(min(len(steps_a), len(steps_r)))
        if steps_a[k] == steps_r[k] + (0 if k == 0 else 1)]
assert keep, f"no aligned frames: AFSI {steps_a} vs reference {steps_r}"


def _nrm(x):
    return float(np.linalg.norm(x))


# ---------------------------------------------------------------------------
# Buoyancy mask (from the demo's own ported module, on the same node grid)
# ---------------------------------------------------------------------------
bouss_Yes = int(os.environ.get("BOUSSINESQ", P.get("boussinesq", 0)))
exp_Coeff = float(P.get("expansion_coeff", 1.0))
mask = np.zeros((ny, nx), bool)
FBOUSS = None
if bouss_Yes:
    if exp_Coeff == 0:      # upstream quirk, see main.py
        exp_Coeff = 1.0
    xG, yG = float(P["x_gravity_vec_comp"]), float(P["y_gravity_vec_comp"])
    nG = np.hypot(xG, yG)
    sys.path.insert(0, os.path.join(here, "ib2d_input"))
    import please_Form_Boussinesq_Forcing_Terms as bmod  # noqa: E402
    _cwd = os.getcwd()
    os.chdir(os.path.join(here, "ib2d_input"))
    try:
        FBOUSS = bmod.please_Form_Boussinesq_Forcing_Terms(
            exp_Coeff, nx, ny, [int(P["gravity"]), xG / nG, yG / nG])
    finally:
        os.chdir(_cwd)
    mask = (np.abs(FBOUSS[0]) + np.abs(FBOUSS[1])) != 0

# the example's own arbitrary-external-force module (shared by both codes)
ext_mod = None
if os.path.exists(os.path.join(here, "ib2d_input",
                              "please_Compute_External_Forcing.py")):
    import please_Compute_External_Forcing as ext_mod      # noqa: E402

XN = np.arange(nx) * h
YN = np.arange(ny) * (Ly / ny)
GRID_INFO = [nx, ny, Lx, Ly, h, Ly / ny, int(P["supp"]),
             int(R["X"].shape[1]), Lx / (2 * nx)]

# IB band: four-node delta support of every marker of frame 0
band = np.zeros((ny, nx), bool)
for x0, y0 in np.atleast_2d(R["X"][0]):
    i0 = int(round(x0 / h))
    j0 = int(round(y0 / (Ly / ny)))
    band[max(0, j0 - 2):j0 + 3, max(0, i0 - 2):i0 + 3] = True
away = ~band
core = mask & away

rows = []
for k in keep:
    ua, ur = A["u"][k], R["u"][k]
    ca, cr = A["C"][k], R["C"][k]
    du = np.linalg.norm(ua - ur, axis=-1)
    # markers live in a periodic box and both codes wrap them into [0, L);
    # the difference must be taken with the minimum-image convention
    dXm = (A["X"][k] - R["X"][k])[..., 0]
    dYm = (A["X"][k] - R["X"][k])[..., 1]
    dXm = (dXm + Lx / 2) % Lx - Lx / 2
    dYm = (dYm + Ly / 2) % Ly - Ly / 2
    dx_m = np.hypot(dXm, dYm)
    row = dict(
        frame=k,
        t_afsi=float(A["t"][k]),
        t_ref=float(R["t"][k]),
        step=steps_a[k],
        marker_max=float(dx_m.max()),
        u_relL2_away=_nrm(du[away]) / (_nrm(ur[away]) or 1.0),
        u_relL2_all=_nrm(du) / (_nrm(ur) or 1.0),
        du_max=float(du.max()),
        uy_max_afsi=float(np.abs(ua[..., 1]).max()),
        uy_max_ref=float(np.abs(ur[..., 1]).max()),
        uy_L2_afsi=_nrm(ua[..., 1]),
        uy_L2_ref=_nrm(ur[..., 1]),
        C_relL2=_nrm(ca - cr) / (_nrm(cr) or 1.0),
        dC_max=float(np.abs(ca - cr).max()),
        C_max_afsi=float(ca.max()),
        C_max_ref=float(cr.max()),
        C_sum_afsi=float(ca.sum()),
        C_sum_ref=float(cr.sum()),
    )
    if FBOUSS is not None:
        # the force field dumped at this frame is the one assembled during the
        # last step: buoyancy (using C of the previous step) + the penalty
        # force (using u of the previous step).  Evaluating the module at the
        # current u costs O(dt*du/dt) ~ 1e-2, negligible against fBouss*C
        # ~ 1e3-1e4.
        fx_e, fy_e = 0.0, 0.0
        if ext_mod is not None:
            fx_e, fy_e = ext_mod.please_Compute_External_Forcing(
                dt, float(A["t"][k]), XN, YN, GRID_INFO, ua[..., 0].copy(),
                ua[..., 1].copy(), 1, None)[:2]
        f_exp = rho * FBOUSS[1] * ca + fy_e
        fa = A["f"][k][..., 1]
        row["f_resid_max"] = float(np.abs(fa - f_exp)[core].max())
        row["f_max"] = float(np.abs(fa).max())
        row["f_resid_rel"] = row["f_resid_max"] / (row["f_max"] or 1.0)
        # where the external force is identically zero the ratio f_y/C must be
        # exactly rho * exp_Coeff * g * gy
        zero_ext = core & (np.abs(fy_e) == 0)
        if zero_ext.sum():
            num = fa[zero_ext]
            den = ca[zero_ext]
            row["bouss_ratio_median"] = float(np.median(num[den != 0] / den[den != 0]))
        else:
            row["bouss_ratio_median"] = float("nan")
        nz = np.concatenate([FBOUSS[0][mask], FBOUSS[1][mask]])
        nz = nz[nz != 0]
        row["bouss_expected"] = rho * float(np.median(nz)) if nz.size else 0.0
        row["n_zero_ext"] = int(zero_ext.sum())
    rows.append(row)

last = rows[-1]
hdr = f"{'frame':>5} {'t_afsi':>9} {'t_ref':>9} {'|dX|max':>10} {'relL2(u)':>9} " \
      f"{'max|du|':>9} {'max|uy|a':>9} {'max|uy|r':>9} {'relL2(C)':>9} " \
      f"{'max|dC|':>9} {'Cmax a':>8} {'Cmax r':>8}"
print(hdr)
print("-" * len(hdr))
for r in rows:
    print(f"{r['frame']:5d} {r['t_afsi']:9.5f} {r['t_ref']:9.5f} "
          f"{r['marker_max']:10.3e} {r['u_relL2_away']:9.3e} {r['du_max']:9.3e} "
          f"{r['uy_max_afsi']:9.4f} {r['uy_max_ref']:9.4f} {r['C_relL2']:9.3e} "
          f"{r['dC_max']:9.3e} {r['C_max_afsi']:8.4f} {r['C_max_ref']:8.4f}")
if FBOUSS is not None:
    print(f"\nbuoyancy check (last frame): |f_y - (rho*fBouss_y*C + F_ext,y)|max"
          f" = {last['f_resid_max']:.3e}  ({100 * last['f_resid_rel']:.4f}% of "
          f"|f|max = {last['f_max']:.3e})")
    if last["n_zero_ext"]:
        print(f"    median f_y/C where F_ext,y = 0 ({last['n_zero_ext']} nodes) "
              f"= {last['bouss_ratio_median']:.6f}  "
              f"(rho*fBouss_y = {last['bouss_expected']:.6f})")
print(f"\nlast frame  t={last['t_afsi']:.5f}  sum(C) AFSI/ref = "
      f"{last['C_sum_afsi']:.6f} / {last['C_sum_ref']:.6f} "
      f"({100 * (last['C_sum_afsi'] / last['C_sum_ref'] - 1):+.3f}%)")
print(f"            L2|u_y| AFSI/ref = {last['uy_L2_afsi']:.4f} / "
      f"{last['uy_L2_ref']:.4f} "
      f"({100 * (last['uy_L2_afsi'] / last['uy_L2_ref'] - 1):+.3f}%)")

# CSV of every aligned frame
with open(os.path.join(out, "compare_table.csv"), "w") as fh:
    keys = list(rows[0].keys())
    fh.write(",".join(keys) + "\n")
    for r in rows:
        fh.write(",".join(str(r[k]) for k in keys) + "\n")

# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
ca, cr = A["C"][last["frame"]], R["C"][last["frame"]]
ua, ur = A["u"][last["frame"]], R["u"][last["frame"]]
xa = np.arange(nx) * h
ya = np.arange(ny) * (Ly / ny)

jrow = ny // 2
if FBOUSS is not None and mask.any():
    jrow = int(np.argmax(mask.sum(axis=1)))

fig, ax = plt.subplots(2, 2, figsize=(11, 7), sharex="col")
ax[0, 0].plot(xa, cr[jrow], "k-", lw=2, label="IB2d")
ax[0, 0].plot(xa, ca[jrow], "r--", lw=1.5, label="AFSI")
ax[0, 0].set_ylabel(f"C  (y = {ya[jrow]:.3f})")
ax[0, 0].legend()
ax[0, 1].plot(xa, np.abs(ca[jrow] - cr[jrow]), "b-")
ax[0, 1].set_ylabel("|dC|")
ax[1, 0].plot(xa, ur[jrow, :, 1], "k-", lw=2, label="IB2d")
ax[1, 0].plot(xa, ua[jrow, :, 1], "r--", lw=1.5, label="AFSI")
ax[1, 0].set_ylabel(f"$u_y$  (y = {ya[jrow]:.3f})")
ax[1, 0].set_xlabel("x")
ax[1, 0].legend()
ax[1, 1].plot(xa, np.linalg.norm(ua[jrow] - ur[jrow], axis=-1), "b-")
ax[1, 1].set_ylabel("|du|")
ax[1, 1].set_xlabel("x")
fig.suptitle(f"demo Boussinesq -- row profile at t = {last['t_afsi']:.5f}")
fig.tight_layout()
fig.savefig(os.path.join(out, "compare_profiles.png"), dpi=120)

fig, ax = plt.subplots(1, 3, figsize=(13, 4.2))
vmin, vmax = float(min(ca.min(), cr.min())), float(max(ca.max(), cr.max()))
for a, f, ttl in ((ax[0], cr, "IB2d"), (ax[1], ca, "AFSI"),
                  (ax[2], ca - cr, "AFSI - IB2d")):
    im = a.imshow(f, origin="lower", extent=[0, Lx, 0, Ly], vmin=vmin,
                  vmax=vmax, aspect="auto")
    a.set_title(ttl)
    a.set_xlabel("x")
fig.colorbar(im, ax=ax, shrink=0.85)
fig.suptitle(f"concentration at t = {last['t_afsi']:.5f}")
fig.savefig(os.path.join(out, "compare_concentration_fields.png"), dpi=120)

fig, ax = plt.subplots(1, 3, figsize=(13, 4.2))
sa = np.linalg.norm(ua, axis=-1)
sr = np.linalg.norm(ur, axis=-1)
vmin, vmax = 0.0, float(max(sa.max(), sr.max()))
for a, f, ttl in ((ax[0], sr, "IB2d"), (ax[1], sa, "AFSI"),
                  (ax[2], sa - sr, "AFSI - IB2d")):
    im = a.imshow(f, origin="lower", extent=[0, Lx, 0, Ly], vmin=vmin,
                  vmax=vmax, aspect="auto")
    a.set_title(ttl)
    a.set_xlabel("x")
fig.colorbar(im, ax=ax, shrink=0.85)
fig.suptitle(f"|u| at t = {last['t_afsi']:.5f}")
fig.savefig(os.path.join(out, "compare_fields.png"), dpi=120)

fig, ax = plt.subplots(1, 3, figsize=(13, 4))
tt = [r["t_afsi"] for r in rows]
ax[0].plot(tt, [r["u_relL2_away"] for r in rows], "o-")
ax[0].set_ylabel("relL2(u), outside IB band")
ax[1].plot(tt, [r["uy_max_afsi"] for r in rows], "r-o", label="AFSI")
ax[1].plot(tt, [r["uy_max_ref"] for r in rows], "k-s", label="IB2d")
ax[1].set_ylabel("max |u_y|  (buoyancy-driven)")
ax[1].legend()
ax[2].plot(tt, [r["C_relL2"] for r in rows], "o-", label="relL2(C)")
ax[2].plot(tt, [r["dC_max"] for r in rows], "s-", label="max|dC|")
ax[2].set_ylabel("concentration error")
ax[2].legend()
for a in ax:
    a.set_xlabel("t")
fig.tight_layout()
fig.savefig(os.path.join(out, "compare_history.png"), dpi=120)

print(f"\nwrote {out}/compare_profiles.png, compare_concentration_fields.png, "
      f"compare_fields.png, compare_history.png, compare_table.csv")
