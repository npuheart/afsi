"""Generate the immersed solids of the semilunar-valve demo (serial; run before main.py).

    python generate_mesh.py                 # H=0.2 (smoke test)
    H=0.05 python generate_mesh.py          # finer fluid grid -> finer solids

Writes to configuration.MESH_DIR:
    wall.xdmf / .h5     rigid vessel wall (tetrahedra), held by a tether in main.py
    valve.xdmf / .h5    leaflets (tetrahedra)
    valve_part.npy      part id per valve cell, in file order (3 root, 4 body, 5 attachment extension)
    meta.json           sizes and quality numbers
Downloads the LS-DYNA example into configuration.LSDYNA_DIR if it is not there yet.
"""
import io
import json
import os
import sys
import urllib.request
import zipfile

import numpy as np
import meshio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import configuration as cfg          # noqa: E402
import mesh_tools as mt              # noqa: E402


def fetch_lsdyna():
    if os.path.exists(os.path.join(cfg.LSDYNA_DIR, "mesh_leaf.k")):
        return
    os.makedirs(cfg.LSDYNA_DIR, exist_ok=True)
    print(f"downloading {cfg.LSDYNA_URL}")
    data = urllib.request.urlopen(cfg.LSDYNA_URL, timeout=300).read()
    zipfile.ZipFile(io.BytesIO(data)).extractall(cfg.LSDYNA_DIR)


def attachment_check(Pv, tv, part, Y, T, Y0):
    """Where do the leaflet attachment and its extension sit relative to the (moved) vessel wall?
    Signed distance along the wall's outward normal: < 0 inside the lumen, 0..WALL_T in the wall."""
    from scipy.spatial import cKDTree
    fn = np.cross(Y[T[:, 1]] - Y[T[:, 0]], Y[T[:, 2]] - Y[T[:, 0]])
    N = np.zeros_like(Y)
    for j in range(3):
        np.add.at(N, T[:, j], fn)
    N /= np.linalg.norm(N, axis=1)[:, None]
    ext_nodes = np.unique(tv[part == 5])
    body_nodes = np.unique(tv[part != 5])
    line = np.intersect1d(ext_nodes, body_nodes)            # where the extension meets the leaflet
    tips = np.setdiff1d(ext_nodes, body_nodes)
    tree = cKDTree(Y)
    def signed(Q):
        _, i = tree.query(Q)
        return np.einsum("ij,ij->i", Q - Y[i], N[i])
    sl, st = signed(Pv[line]), signed(Pv[tips])
    return dict(attachment_line_min=float(sl.min()), attachment_line_median=float(np.median(sl)),
                attachment_line_max=float(sl.max()), attachment_line_inside_lumen_beyond_0p02=float((sl < -0.02).mean()),
                extension_in_wall_fraction=float(((st >= 0) & (st <= cfg.WALL_T)).mean()),
                extension_outside_wall_fraction=float((st > cfg.WALL_T).mean()),
                wall_max_move=float(np.linalg.norm(Y - Y0, axis=1).max()))


def main():
    if cfg.EXT > cfg.WALL_T - cfg.DELTA_IB + 1e-12:
        raise SystemExit(f"EXT = {cfg.EXT:g} cm reaches within one IB grid spacing ({cfg.DELTA_IB:g}) of the wall's "
                         f"outer surface (WALL_T = {cfg.WALL_T:g}); refine H, thicken the wall (WALL_T) or set EXT")
    if cfg.EXT < cfg.DELTA_IB:
        print(f"warning: EXT = {cfg.EXT:g} cm is less than one IB grid spacing ({cfg.DELTA_IB:g})")
    fetch_lsdyna()
    os.makedirs(cfg.MESH_DIR, exist_ok=True)
    h_root, h_far = cfg.H_SOLID, 2 * cfg.H_SOLID
    Y, T = mt.vessel_surface(cfg.LSDYNA_DIR, cfg.Z_IN, cfg.Z_OUT, h_root, h_far)
    Pv, tv, part, vq = mt.valve_mesh(cfg.LSDYNA_DIR, cfg.LEAFLET_T, cfg.VALVE_LAYERS, cfg.VALVE_REFINE,
                                     cfg.EXT, 2, Y, T)
    Y0 = Y.copy()
    shift_info = {}
    if cfg.VALVE_SHIFT:
        # each leaflet moves rigidly outwards; the vessel root follows sector by sector (mesh_tools)
        lab = mt.components(Pv, tv)
        trans, ang = mt.leaflet_translations(Pv, lab, cfg.VALVE_SHIFT)
        Pv = Pv + trans[lab]
        Y = Y + mt.vessel_morph(Y, trans, ang)
        shift_info = dict(shift=cfg.VALVE_SHIFT, leaflet_angles_deg=np.degrees(ang).round(2).tolist())
    P, tets, wq = mt.wall_mesh(Y, T, cfg.WALL_T, cfg.WALL_LAYERS, z_ends=(cfg.Z_IN, cfg.Z_OUT))
    assert wq["flipped"] == 0, f"wall extrusion folded: {wq}"
    meshio.write(os.path.join(cfg.MESH_DIR, "wall.xdmf"), meshio.Mesh(P, [("tetra", tets)]))
    print(f"wall: {len(P)} nodes, {len(tets)} tets, thickness median {wq['thickness_median']:.3f} "
          f"min {wq['thickness_min']:.3f} cm")

    if cfg.VALVE_SHIFT:
        shift_info.update(attachment_check(Pv, tv, part, Y, T, Y0))
        print("attachment vs moved wall:", {k: v for k, v in shift_info.items() if k != "leaflet_angles_deg"})
    meshio.write(os.path.join(cfg.MESH_DIR, "valve.xdmf"), meshio.Mesh(Pv, [("tetra", tv)]))
    np.save(os.path.join(cfg.MESH_DIR, "valve_part.npy"), part.astype(np.int32))
    print(f"valve: {len(Pv)} nodes, {len(tv)} tets, parts {dict(zip(*np.unique(part, return_counts=True)))}")

    # lumen radius at the inlet plane (for A in the plug inflow v = Q/A)
    ring = Y[np.abs(Y[:, 2] - cfg.Z_IN) < 1e-9]
    r_in = float(np.hypot(ring[:, 0], ring[:, 1]).mean())
    meta = dict(valve_shift=shift_info, EXT=cfg.EXT, WALL_T=cfg.WALL_T, H=cfg.H, H_SOLID=cfg.H_SOLID, wall_layers=cfg.WALL_LAYERS, valve_refine=cfg.VALVE_REFINE,
                r_inlet=r_in, wall_nodes=len(P), wall_tets=len(tets), valve_nodes=len(Pv), valve_tets=len(tv),
                wall_quality=wq, valve_quality={k: (int(v) if isinstance(v, (int, np.integer)) else v)
                                                for k, v in vq.items()})
    with open(os.path.join(cfg.MESH_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"inlet lumen radius {r_in:.4f} cm -> {cfg.MESH_DIR}")


if __name__ == "__main__":
    main()
