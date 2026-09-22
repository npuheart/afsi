"""Analytic references and error metrics for demo_426.

The benchmark's purpose is to quantify how accurately a given immersed-boundary
kernel reproduces the exact solution inside a confined, stationary, slanted
channel.  The reference is known everywhere, so every metric here is a
discretisation error with a zero target:

    xi(x,y) = y*cos(theta) - x*sin(theta)
    u_exact = (dP/L)/(2*mu) * ((D/2)^2 - xi^2) * (cos(theta), sin(theta))

Metrics
-------
* relative L2 and Linf of the velocity error over the WHOLE box
* the same restricted to the CHANNEL (|xi| <= D/2), which is the region the
  benchmark actually cares about
* the across-channel profile at x = X_PROFILE (Fig. 23 of the benchmark)
* the peak velocity and where it occurs
* the residual velocity ON the plates, which measures the numerical boundary
  layer produced by the IB kernel's smearing
"""
import numpy as np

from dolfinx.geometry import (bb_tree, compute_colliding_cells,
                              compute_collisions_points)


# --------------------------------------------------------------------------
# sampling helpers
# --------------------------------------------------------------------------
def _sample(func, pts, mesh, bs=1):
    pts = np.asarray(pts, dtype=np.float64)
    tree = bb_tree(mesh, mesh.geometry.dim)
    cells = compute_colliding_cells(mesh, compute_collisions_points(tree, pts),
                                    pts)
    out = np.full((len(pts), bs), np.nan)
    for i, x in enumerate(pts):
        links = cells.links(i)
        if len(links):
            out[i] = func.eval(x, links[0])[:bs]
    return out


def _fmt(name, value, unit=""):
    if isinstance(value, str):
        return f"  {name:<46} {value:>14} {unit}"
    return f"  {name:<46} {value:>14.6e} {unit}"


def _grid_points(cfg, mesh, margin=0.0):
    """Node coordinates of the fluid mesh (optionally inset by ``margin``)."""
    x = mesh.geometry.x
    pad = margin
    keep = ((x[:, 0] >= cfg.X_MIN + pad) & (x[:, 0] <= cfg.X_MAX - pad)
            & (x[:, 1] >= cfg.Y_MIN + pad) & (x[:, 1] <= cfg.Y_MAX - pad))
    p = x[keep]
    return np.column_stack([p[:, 0], p[:, 1], np.zeros(len(p))])


def report(cfg, mesh, u, p, t_prof, u_line, u_exact_line, solid_coords,
           coords_ref):
    """Compute and print every verification metric; return a dict."""
    res = {}
    print("=" * 78)
    print(f"verification report  (demo_426, N={cfg.N}, dx={cfg.DX:g}, "
          f"theta={cfg.THETA_DEG:g} deg, {cfg.SOLVER})")
    print("=" * 78)
    print(_fmt("u_max analytic", cfg.U_MAX, "m/s"))
    print(_fmt("(benchmark text quotes)", cfg.U_MAX_PAPER, "m/s"))

    # ---- global and in-channel error ------------------------------------
    pts = _grid_points(cfg, mesh)
    un = _sample(u, pts, mesh, bs=2)
    ue = np.column_stack(cfg.analytic(pts[:, 0], pts[:, 1]))
    ok = np.isfinite(un).all(axis=1)
    pts, un, ue = pts[ok], un[ok], ue[ok]
    err = np.linalg.norm(un - ue, axis=1)

    e_l2 = float(np.sqrt(np.mean(err**2)))
    ref_l2 = float(np.sqrt(np.mean(np.linalg.norm(ue, axis=1) ** 2)))
    res["err_L2_rel_box"] = e_l2 / ref_l2
    res["err_Linf_box"] = float(err.max())
    print("-- velocity error ------------------------------------------------")
    print(_fmt("relative L2 error (whole box)", res["err_L2_rel_box"]))
    print(_fmt("Linf error (whole box)", res["err_Linf_box"], "m/s"))
    print(_fmt("mean |u_exact| (whole box)", ref_l2, "m/s"))
    print("    NOTE: the whole-box numbers are NOT an accuracy measure for this")
    print("    case.  The exact parabola grows without bound OUTSIDE the channel")
    print("    (it reaches 2.67 in this box) while the physical flow there is")
    print("    stagnant, so this metric is dominated by a spurious reference.")
    print("    The CHANNEL numbers below are the meaningful ones.")

    t = cfg.xi(pts[:, 0], pts[:, 1])
    inside = np.abs(t) <= cfg.R_HALF
    if inside.any():
        e_in = err[inside]
        ref_in = np.linalg.norm(ue[inside], axis=1)
        res["err_L2_rel_channel"] = float(
            np.sqrt(np.mean(e_in**2)) / np.sqrt(np.mean(ref_in**2)))
        res["err_Linf_channel"] = float(e_in.max())
        print(_fmt("relative L2 error (channel only)",
                   res["err_L2_rel_channel"]))
        print(_fmt("Linf error (channel only)", res["err_Linf_channel"], "m/s"))

    # ---- across-channel profile at x = X_PROFILE -------------------------
    print("-- profile across the channel (Fig. 23) ----------------------------")
    okp = np.isfinite(u_line).all(axis=1)
    mag_n = np.linalg.norm(u_line[okp], axis=1)
    mag_e = np.linalg.norm(u_exact_line[okp], axis=1)
    e_p = mag_n - mag_e
    res["profile_Linf"] = float(np.max(np.abs(e_p))) if okp.any() else np.nan
    res["profile_relL2"] = float(
        np.linalg.norm(e_p) / np.linalg.norm(mag_e)) if okp.any() else np.nan
    print(_fmt("|u| Linf error across the profile", res["profile_Linf"], "m/s"))
    print(_fmt("|u| relative L2 error across the profile",
               res["profile_relL2"]))
    if okp.any():
        i = int(np.argmax(mag_n))
        res["u_max_num"] = float(mag_n[i])
        res["u_max_rel_err"] = float(mag_n[i] / cfg.U_MAX - 1.0)
        res["u_max_xi"] = float(t_prof[okp][i])
        print(_fmt("numerical u_max", res["u_max_num"], "m/s"))
        print(_fmt("relative error in u_max", res["u_max_rel_err"]))
        print(_fmt("u_max location (xi)", res["u_max_xi"], "m"))

    # ---- residual velocity on the plates = numerical boundary layer ------
    n_lag = len(solid_coords.x.array) // 2
    sv = np.linalg.norm(solid_coords.x.array.reshape(n_lag, 2), axis=1)
    disp = solid_coords.x.array - coords_ref.x.array
    res["plate_max_disp"] = float(np.max(np.linalg.norm(
        disp.reshape(n_lag, 2), axis=1)))
    print("-- immersed plates -------------------------------------------------")
    print(_fmt("plate max |displacement| from reference", res["plate_max_disp"],
               "m"))
    print(_fmt("plate Lagrangian nodes", n_lag))

    # fluid velocity sampled exactly on the plate nodes: this is the IB
    # smearing error and is expected to be the largest error in the solution
    lag_pts = np.column_stack([solid_coords.x.array.reshape(n_lag, 2),
                               np.zeros(n_lag)])
    u_on_plate = _sample(u, lag_pts, mesh, bs=2)
    okw = np.isfinite(u_on_plate).all(axis=1)
    if okw.any():
        w = np.linalg.norm(u_on_plate[okw], axis=1)
        res["plate_u_max"] = float(w.max())
        res["plate_u_mean"] = float(w.mean())
        print(_fmt("fluid |u| on the plates (should be 0)", res["plate_u_max"],
                   "m/s"))
        print(_fmt("fluid mean |u| on the plates", res["plate_u_mean"], "m/s"))

    print("=" * 78)
    return res
