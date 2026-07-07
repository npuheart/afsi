#!/usr/bin/env python3
"""
Simple merge: combine leaflets_M2_tet + vessel_wall_wall_3d_tet into one mesh
with cell tags: 1 = vessel, 2 = leaflets.

Usage:
    python3 merge_meshes.py
"""

from mpi4py import MPI
import dolfinx
import numpy as np
from basix.ufl import element as basix_element

# ---------------------------------------------------------------------------
# 1. Read both tet meshes
# ---------------------------------------------------------------------------
print("Reading vessel mesh...")
with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "vessel_wall_wall_3d_tet.xdmf", "r") as f:
    vessel = f.read_mesh(name="Grid")

print("Reading leaflets mesh...")
with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "leaflets_M2_tet.xdmf", "r") as f:
    leaflets = f.read_mesh(name="Grid")

n_vessel_cells = vessel.topology.index_map(3).size_global
n_leaf_cells = leaflets.topology.index_map(3).size_global
print(f"  Vessel:   {n_vessel_cells} tets, {vessel.geometry.x.shape[0]} nodes")
print(f"  Leaflets: {n_leaf_cells} tets, {leaflets.geometry.x.shape[0]} nodes")

# ---------------------------------------------------------------------------
# 2. Extract connectivity (numpy for combine, AdjacencyList for centroids)
# ---------------------------------------------------------------------------
vessel_topo = vessel.topology
vessel_topo.create_connectivity(3, 0)
vessel_c2v = vessel_topo.connectivity(3, 0)
n_vessel_cells = vessel_topo.index_map(3).size_local
vessel_conn = np.array([vessel_c2v.links(c) for c in range(n_vessel_cells)], dtype=np.int64)
vessel_nodes = vessel.geometry.x.copy()
n_vessel_nodes = len(vessel_nodes)

leaf_topo = leaflets.topology
leaf_topo.create_connectivity(3, 0)
leaf_c2v = leaf_topo.connectivity(3, 0)
n_leaf_cells = leaf_topo.index_map(3).size_local
leaf_conn = np.array([leaf_c2v.links(c) for c in range(n_leaf_cells)], dtype=np.int64)
leaf_nodes = leaflets.geometry.x.copy()

# ---------------------------------------------------------------------------
# 3. Combine raw data
# ---------------------------------------------------------------------------
combined_conn = np.vstack([vessel_conn, leaf_conn + n_vessel_nodes])
combined_nodes = np.vstack([vessel_nodes, leaf_nodes])

# ---------------------------------------------------------------------------
# 4. Create merged mesh via create_mesh()
# ---------------------------------------------------------------------------
_e_tet = basix_element("Lagrange", "tetrahedron", 1, shape=(3,))
merged = dolfinx.mesh.create_mesh(MPI.COMM_WORLD, combined_conn, _e_tet, combined_nodes)

merged.topology.create_connectivity(3, 0)
merged_c2v = merged.topology.connectivity(3, 0)
n_merged = merged.topology.index_map(3).size_local

# ---------------------------------------------------------------------------
# 5. Tag cells by centroid-based matching (vectorized)
# ---------------------------------------------------------------------------
merged_nodes = merged.geometry.x

def centroids(conn_arr, nodes):
    """Compute cell centroids: mean of 4 vertices per tet."""
    return nodes[conn_arr].mean(axis=1)

vessel_centroids = centroids(vessel_conn, vessel_nodes)
leaf_centroids   = centroids(leaf_conn, leaf_nodes)
# merged_conn from merged_c2v
merged_conn = np.array([merged_c2v.links(c) for c in range(n_merged)], dtype=np.int64)
merged_centroids = centroids(merged_conn, merged_nodes)

# Use KDTree for efficient matching
from scipy.spatial import cKDTree
tree_v = cKDTree(vessel_centroids)
tree_l = cKDTree(leaf_centroids)

merged_tags = np.zeros(n_merged, dtype=np.int32)
for c in range(n_merged):
    dv, _ = tree_v.query(merged_centroids[c])
    dl, _ = tree_l.query(merged_centroids[c])
    if dv < dl:
        merged_tags[c] = 1  # vessel
    else:
        merged_tags[c] = 2  # leaflets

# ---------------------------------------------------------------------------
# 6. Write with cell tags
# ---------------------------------------------------------------------------
from dolfinx.mesh import meshtags
cell_tags = meshtags(merged, 3, np.arange(n_merged, dtype=np.int32), merged_tags)

output_path = "combined_vessel_leaflets.xdmf"
with dolfinx.io.XDMFFile(MPI.COMM_WORLD, output_path, "w") as f:
    f.write_mesh(merged)
    f.write_meshtags(cell_tags, merged.geometry)

print(f"  Merged: {n_merged} tets, {merged.geometry.x.shape[0]} nodes")
print(f"  Tags: vessel=1 ({np.sum(merged_tags==1)}), leaflets=2 ({np.sum(merged_tags==2)})")
print(f"=== Done! Written to {output_path} ===")
