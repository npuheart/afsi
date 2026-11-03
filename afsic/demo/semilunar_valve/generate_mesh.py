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


def main():
    fetch_lsdyna()
    os.makedirs(cfg.MESH_DIR, exist_ok=True)
    h_root, h_far = cfg.H_SOLID, 2 * cfg.H_SOLID
    Y, T = mt.vessel_surface(cfg.LSDYNA_DIR, cfg.Z_IN, cfg.Z_OUT, h_root, h_far)
    P, tets, wq = mt.wall_mesh(Y, T, cfg.WALL_T, cfg.WALL_LAYERS, z_ends=(cfg.Z_IN, cfg.Z_OUT))
    assert wq["flipped"] == 0, f"wall extrusion folded: {wq}"
    meshio.write(os.path.join(cfg.MESH_DIR, "wall.xdmf"), meshio.Mesh(P, [("tetra", tets)]))
    print(f"wall: {len(P)} nodes, {len(tets)} tets, thickness median {wq['thickness_median']:.3f} "
          f"min {wq['thickness_min']:.3f} cm")

    Pv, tv, part, vq = mt.valve_mesh(cfg.LSDYNA_DIR, cfg.LEAFLET_T, cfg.VALVE_LAYERS, cfg.VALVE_REFINE,
                                     cfg.EXT, 2, Y, T)
    meshio.write(os.path.join(cfg.MESH_DIR, "valve.xdmf"), meshio.Mesh(Pv, [("tetra", tv)]))
    np.save(os.path.join(cfg.MESH_DIR, "valve_part.npy"), part.astype(np.int32))
    print(f"valve: {len(Pv)} nodes, {len(tv)} tets, parts {dict(zip(*np.unique(part, return_counts=True)))}")

    # lumen radius at the inlet plane (for A in the plug inflow v = Q/A)
    ring = Y[np.abs(Y[:, 2] - cfg.Z_IN) < 1e-9]
    r_in = float(np.hypot(ring[:, 0], ring[:, 1]).mean())
    meta = dict(H=cfg.H, H_SOLID=cfg.H_SOLID, wall_layers=cfg.WALL_LAYERS, valve_refine=cfg.VALVE_REFINE,
                r_inlet=r_in, wall_nodes=len(P), wall_tets=len(tets), valve_nodes=len(Pv), valve_tets=len(tv),
                wall_quality=wq, valve_quality={k: (int(v) if isinstance(v, (int, np.integer)) else v)
                                                for k, v in vq.items()})
    with open(os.path.join(cfg.MESH_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"inlet lumen radius {r_in:.4f} cm -> {cfg.MESH_DIR}")


if __name__ == "__main__":
    main()
