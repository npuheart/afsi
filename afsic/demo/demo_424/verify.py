"""Analytic references and error metrics for demo_424.

Open case
---------
The box contains three parallel plane channels that all see the same end
pressures and therefore, in the fully developed region, the same axial
pressure gradient G = -dp/dx:

    lumen   |y - Y_C| <= A_LUMEN            (half-width a)
    gap     the two outer layers of width GAP, bounded by the box wall
            (no-slip) and the aortic wall (essentially stationary)

For a plane channel of half-width A with no-slip walls,

    u(y') = G/(2 mu) (A^2 - y'^2),     Q = 2 G A^3 / (3 mu)

with y' measured from the channel centreline.  G is *fitted* from the
computed pressure field so that the comparison tests the solution shape and
the flux relation rather than the (entrance-affected) overall pressure drop.

Closed case
-----------
The membrane seals the lumen, so each lumen chamber is stagnant (u = 0) and
isobaric, with a jump of exactly DP across the membrane; the outer gaps still
carry the analytic plane-Poiseuille flow driven by DP/BOX_L.  The metrics
below therefore quantify (a) how much of DP the immersed membrane actually
holds, and (b) how much fluid leaks through it.
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
    cells = compute_colliding_cells(mesh, compute_collisions_points(tree, pts), pts)
    out = np.full((len(pts), bs), np.nan)
    for i, x in enumerate(pts):
        links = cells.links(i)
        if len(links):
            out[i] = func.eval(x, links[0])[:bs]
    return out


def _broadcast(x, y):
    x = np.atleast_1d(np.asarray(x, dtype=np.float64))
    y = np.atleast_1d(np.asarray(y, dtype=np.float64))
    if x.size == 1:
        x = np.full(y.shape, x[0])
    if y.size == 1:
        y = np.full(x.shape, y[0])
    return x, y


def sample_u(u, mesh, x, y):
    x, y = _broadcast(x, y)
    return _sample(u, np.column_stack([x, y, np.zeros_like(x)]), mesh, bs=2)


def sample_p(p, mesh, x, y):
    x, y = _broadcast(x, y)
    return _sample(p, np.column_stack([x, y, np.zeros_like(x)]), mesh, bs=1)[:, 0]


def trapz(y, f):
    return float(np.trapezoid(f, y))


def plane_channel_umax(G, half_width, mu):
    return G * half_width**2 / (2.0 * mu)


def plane_channel_q(G, half_width, mu):
    return 2.0 * G * half_width**3 / (3.0 * mu)


def fit_gradient(p, mesh, cfg, x_lo, x_hi, y=None):
    """Least-squares -dp/dx over [x_lo, x_hi] along y (default: centreline)."""
    y = cfg.Y_C if y is None else y
    xs = np.linspace(x_lo, x_hi, 101)
    ps = sample_p(p, mesh, xs, y)
    ok = np.isfinite(ps)
    coef = np.polyfit(xs[ok], ps[ok], 1)
    resid = float(np.max(np.abs(ps[ok] - np.polyval(coef, xs[ok]))))
    return -coef[0], resid


def _fmt(name, value, unit=""):
    if isinstance(value, str):
        return f"  {name:<44} {value:>14} {unit}"
    return f"  {name:<44} {value:>14.6e} {unit}"


# --------------------------------------------------------------------------
def run(cfg, mesh, u, p, solid):
    """Compute and print every verification metric; return a dict."""
    h, mu = cfg.H, cfg.MU
    y_c, a = cfg.Y_C, cfg.A_LUMEN
    res = {}

    y_nodes = np.arange(cfg.NY + 1) * h
    gap_half = 0.5 * cfg.GAP
    gap_centres = [0.5 * cfg.GAP, cfg.BOX_W - 0.5 * cfg.GAP]
    y_full = y_nodes[(y_nodes >= y_c - a) & (y_nodes <= y_c + a)]
    x_d = cfg.X_OFF + cfg.DISC_X

    print("=" * 78)
    print(f"verification report  (case={cfg.CASE}, NY={cfg.NY}, h={h:g} m, "
          f"DP={cfg.DP:.6g} Pa, {cfg.SOLVER})")
    print("=" * 78)

    G_ideal = cfg.DP / cfg.BOX_L
    res["G_ideal"] = G_ideal

    # =====================================================================
    # OPEN: pressure-driven parallel channels
    # =====================================================================
    if cfg.CASE == "open":
        x_lo = cfg.X_OFF + 0.15 * cfg.L_AORTA
        x_hi = cfg.X_OFF + 0.85 * cfg.L_AORTA
        x_mid = cfg.X_OFF + 0.5 * cfg.L_AORTA

        G, resid = fit_gradient(p, mesh, cfg, x_lo, x_hi)
        res["G_fit"] = G
        res["p_linearity_residual"] = resid
        print("-- driving gradient ------------------------------------------------")
        print(_fmt("fitted G = -dp/dx (centreline)", G, "Pa/m"))
        print(_fmt("DP / box length", G_ideal, "Pa/m"))
        print(_fmt("relative difference", (G - G_ideal) / G_ideal))
        print(_fmt("max deviation from linear p(x)", resid, "Pa"))

        u_max_an = plane_channel_umax(G, a, mu)
        Q_lumen_an = plane_channel_q(G, a, mu)

        # ---- lumen profile, interior window (exclude a 2h IB band) ----
        mask = np.abs(y_nodes - y_c) <= a - 2.0 * h
        ys = y_nodes[mask]
        u_an = u_max_an * (1.0 - ((ys - y_c) / a) ** 2)
        print("-- lumen axial velocity profile ------------------------------------")
        for f in (0.25, 0.5, 0.75):
            x_s = cfg.X_OFF + f * cfg.L_AORTA
            un = sample_u(u, mesh, x_s, ys)[:, 0]
            ok = np.isfinite(un)
            e = un[ok] - u_an[ok]
            l2 = float(np.sqrt(trapz(ys[ok], e**2)))
            ref = float(np.sqrt(trapz(ys[ok], u_an[ok] ** 2)))
            res[f"u_relL2_x{f}"] = l2 / ref
            res[f"u_Linf_x{f}"] = float(np.max(np.abs(e)))
            print(_fmt(f"x={x_s:.5f}  relative L2 error", l2 / ref))
            print(_fmt(f"x={x_s:.5f}  |e|_inf", float(np.max(np.abs(e))), "m/s"))

        un = sample_u(u, mesh, x_mid, ys)[:, 0]
        res["u_max_num"] = float(np.nanmax(un))
        res["u_max_analytic"] = u_max_an
        res["u_max_rel_err"] = res["u_max_num"] / u_max_an - 1.0
        print(_fmt("numerical u_max (interior window)", res["u_max_num"], "m/s"))
        print(_fmt("analytic  u_max (with fitted G)", u_max_an, "m/s"))
        print(_fmt("relative error in u_max", res["u_max_rel_err"]))

        # ---- fluxes ----
        print("-- fluxes per unit depth -------------------------------------------")
        for f in (0.25, 0.5, 0.75):
            x_s = cfg.X_OFF + f * cfg.L_AORTA
            uf = sample_u(u, mesh, x_s, y_full)[:, 0]
            ok = np.isfinite(uf)
            q = trapz(y_full[ok], uf[ok])
            res[f"Q_lumen_x{f}"] = q
            print(_fmt(f"numerical Q_lumen at x={x_s:.5f}", q, "m^2/s"))
        res["Q_lumen_analytic"] = Q_lumen_an
        print(_fmt("analytic  Q_lumen (with fitted G)", Q_lumen_an, "m^2/s"))

    # =====================================================================
    # CLOSED: sealed lumen + bypass gap
    # =====================================================================
    else:
        y_lo_probe = np.linspace(cfg.Y_OUT_LO + 0.15 * cfg.GAP,
                                 cfg.Y_IN_LO - 0.15 * cfg.GAP, 9)
        print("-- sealed lumen (targets: u = 0, isobaric chambers) ----------------")
        for tag, x_a, x_b in (("upstream", cfg.X_OFF + 0.01 * cfg.L_AORTA,
                               x_d - 0.06 * cfg.L_AORTA),
                              ("downstream", x_d + 0.06 * cfg.L_AORTA,
                               cfg.X_OFF + 0.99 * cfg.L_AORTA)):
            xs = np.linspace(x_a, x_b, 61)
            pc = sample_p(p, mesh, xs, y_c)
            ok = np.isfinite(pc)
            res[f"p_mean_{tag}"] = float(np.nanmean(pc[ok]))
            res[f"p_std_{tag}"] = float(np.nanstd(pc[ok]))
            print(_fmt(f"p mean ({tag} chamber)", res[f"p_mean_{tag}"], "Pa"))
            print(_fmt(f"p std  ({tag} chamber, isobaric?)",
                       res[f"p_std_{tag}"], "Pa"))
            # axial velocity in the chamber
            PX, PY = np.meshgrid(xs, y_full, indexing="xy")
            pts = np.column_stack([PX.ravel(), PY.ravel(), np.zeros(PX.size)])
            uu = _sample(u, pts, mesh, bs=2)
            mg = np.linalg.norm(uu, axis=1)
            res[f"u_max_{tag}"] = float(np.nanmax(mg))
            print(_fmt(f"max |u| ({tag} chamber)", res[f"u_max_{tag}"], "m/s"))

        p_up = np.nanmean([res["p_mean_upstream"], res["p_mean_downstream"]][0])
        res["p_jump"] = res["p_mean_upstream"] - res["p_mean_downstream"]
        print("-- membrane ---------------------------------------------------------")
        print(_fmt("held pressure jump (chamber means)", res["p_jump"], "Pa"))
        print(_fmt("imposed DP", cfg.DP, "Pa"))
        print(_fmt("fraction of DP held", res["p_jump"] / cfg.DP))
        print(_fmt("relative error", res["p_jump"] / cfg.DP - 1.0))

        for tag, dx_s in (("4h upstream", -4.0 * h), ("4h downstream", 4.0 * h)):
            uf = sample_u(u, mesh, x_d + dx_s, y_full)[:, 0]
            ok = np.isfinite(uf)
            q = trapz(y_full[ok], uf[ok])
            res[f"Q_leak_{tag}"] = q
            print(_fmt(f"lumen leakage flux at x_d{dx_s:+.0e}", q, "m^2/s"))

        print(_fmt("membrane mean x-displacement", solid["disc_disp"], "m"))
        print(_fmt("expected DP/(beta*t_wall)", cfg.DISC_DELTA, "m"))
        print(_fmt("membrane tether force per depth", solid["disc_tether"], "N/m"))
        print(_fmt("pressure force per depth DP*2a", cfg.DP * 2.0 * a, "N/m"))

        G, resid = fit_gradient(p, mesh, cfg,
                                cfg.Y_OUT_LO + 0.15 * cfg.GAP,
                                cfg.Y_IN_LO - 0.15 * cfg.GAP)
        res["G_gap_fit"] = G
        res["p_gap_linearity_residual"] = resid
        print("-- bypass gap (outer channel) --------------------------------------")
        print(_fmt("fitted G in the lower gap", G, "Pa/m"))
        print(_fmt("DP / box length", G_ideal, "Pa/m"))

    # =====================================================================
    # gap flux (identical reference in both cases: driven by DP/BOX_L)
    # =====================================================================
    x_mid = cfg.X_OFF + 0.5 * cfg.L_AORTA
    q_gap = 0.0
    for yc_g in gap_centres:
        y_g = y_nodes[(y_nodes >= yc_g - gap_half) & (y_nodes <= yc_g + gap_half)]
        ug = sample_u(u, mesh, x_mid, y_g)[:, 0]
        ok = np.isfinite(ug)
        q_gap += trapz(y_g[ok], ug[ok])
    res["Q_gap_num"] = q_gap
    res["Q_gap_analytic"] = 2.0 * plane_channel_q(G_ideal, gap_half, mu)
    print("-- outer gap flux ---------------------------------------------------")
    print(_fmt("numerical Q_gap (both sides)", q_gap, "m^2/s"))
    print(_fmt("analytic  Q_gap with G=DP/BOX_L", res["Q_gap_analytic"], "m^2/s"))
    print(_fmt("relative error", q_gap / res["Q_gap_analytic"] - 1.0))

    # =====================================================================
    # global quantities
    # =====================================================================
    print("-- global -----------------------------------------------------------")
    print(_fmt("max |u| over the whole box", solid["max_u"], "m/s"))
    print(_fmt("max |u_y| in lumen", float(np.nanmax(np.abs(
        sample_u(u, mesh, x_mid, y_nodes)[:, 1]))), "m/s"))
    print(_fmt("wall mean |displacement|", solid["wall_disp"], "m"))
    print(_fmt("wall tether force per depth", solid["wall_tether"], "N/m"))
    res["max_u"] = solid["max_u"]
    res["wall_disp"] = solid["wall_disp"]
    res["wall_tether"] = solid["wall_tether"]
    print("=" * 78)
    return res
