"""Geometry and meshes for the semilunar-valve demo.

Source geometry: the LS-DYNA ICFD example "Advanced: Heart Valve"
(https://lsdyna.ansys.com/heartvalve/, ICFD_example_heartvalve.zip), which Black et al.
(arXiv:2601.08266, Sec. 4.5) name as the origin of their model. Units: cm.

* vessel_surface(): the vessel wall (LS-DYNA fluid-surface parts 1, 2, 6, 7: root with three sinuses,
  inflow tube, two small sinus patches) with straight extensions to z_in / z_out, re-meshed with gmsh.
* wall_mesh(): that surface extruded outwards by the wall thickness -> tetrahedra (rigid vessel wall,
  held by a tether in main.py).
* valve_mesh(): the leaflet shells refined, the attachment edges extended outwards into the wall,
  and extruded to the leaflet thickness -> tetrahedra, with a part id per cell.
"""
import collections
import numpy as np

from split import quad_to_tri, prism_to_tet, orient

WALL_PARTS = (1, 2, 6, 7)


# ---------------------------------------------------------------------------------------------------
# LS-DYNA keyword files
# ---------------------------------------------------------------------------------------------------
def read_k(path):
    nodes, elems, sec = {}, [], None
    for line in open(path):
        if line.startswith('*'):
            sec = line.strip().lower()
            continue
        if line.startswith('$') or not line.strip():
            continue
        if sec in ('*node', '*mesh_surface_node'):
            nodes[int(line[0:8])] = [float(line[8 + 16 * i:24 + 16 * i]) for i in range(3)]
        elif sec in ('*element_shell', '*mesh_surface_element'):
            pid = int(line[8:16])
            conn = [int(line[16 + 8 * i:24 + 8 * i]) for i in range(4) if line[16 + 8 * i:24 + 8 * i].strip()]
            elems.append((pid, conn))
    return nodes, elems


def read_node_set(path):
    ids, sec = [], None
    for line in open(path):
        if line.startswith('*'):
            sec = line.strip().lower()
            continue
        if line.startswith('$') or not line.strip() or sec != '*set_node_list' or '.' in line:
            continue
        ids += [int(line[i:i + 10]) for i in range(0, 80, 10) if line[i:i + 10].strip()]
    return {i for i in ids if i > 0}


def _arrays(nodes, elems):
    ids = sorted(nodes)
    idx = {n: i for i, n in enumerate(ids)}
    X = np.array([nodes[n] for n in ids], float)
    out = []
    for pid, conn in elems:
        c = [idx[n] for n in conn]
        if len(c) == 4 and c[2] == c[3]:
            c = c[:3]
        out.append((pid, c))
    return X, out, idx


def _triangulate(faces):
    tris = []
    for c in faces:
        tris += [c] if len(c) == 3 else [[c[0], c[1], c[2]], [c[0], c[2], c[3]]]
    return np.array(tris)


# ---------------------------------------------------------------------------------------------------
# vessel wall surface
# ---------------------------------------------------------------------------------------------------
def vessel_surface(dyna_dir, z_in, z_out, h_root, h_far, root_z=(-0.8, 4.0)):
    """Triangulated vessel wall from z_in to z_out, re-meshed with gmsh; faces point out of the lumen."""
    import gmsh
    X, E, _ = _arrays(*read_k(f'{dyna_dir}/mesh_cfd.k'))
    tris = _triangulate([c for p, c in E if p in WALL_PARTS])
    # open ends: boundary loops ordered as in the faces
    und = collections.Counter()
    di = set()
    for t in tris:
        for i in range(3):
            u, v = t[i], t[(i + 1) % 3]
            und[(min(u, v), max(u, v))] += 1
            di.add((u, v))
    nxt = {u: v for (u, v) in di if und[(min(u, v), max(u, v))] == 1}
    loops, seen = [], set()
    for s in nxt:
        if s in seen:
            continue
        L = [s]
        seen.add(s)
        while nxt[L[-1]] != s:
            L.append(nxt[L[-1]])
            seen.add(L[-1])
        loops.append(L)
    assert len(loops) == 2, f'expected two open ends, found {len(loops)}'
    X = list(map(list, X))
    tris = tris.tolist()
    for L in loops:
        P = np.array([X[i] for i in L])
        z1 = z_in if P[:, 2].mean() < 1.0 else z_out
        r = np.hypot(P[:, 0], P[:, 1])
        th = np.arctan2(P[:, 1], P[:, 0])
        R = r.mean()
        nseg = max(2, int(round(abs(z1 - P[:, 2].mean()) / (2 * np.pi * R / len(L)))))
        prev = L
        for k in range(1, nseg + 1):
            s = k / nseg
            z, rr = (1 - s) * P[:, 2] + s * z1, (1 - s) * r + s * R      # last ring planar and circular
            ring = []
            for j in range(len(L)):
                X.append([rr[j] * np.cos(th[j]), rr[j] * np.sin(th[j]), z[j]])
                ring.append(len(X) - 1)
            for j in range(len(L)):
                u, v = prev[j], prev[(j + 1) % len(L)]
                u2, v2 = ring[j], ring[(j + 1) % len(L)]
                tris += [[v, u, u2], [v, u2, v2]]
            prev = ring
    X, tris = np.array(X), np.array(tris)

    gmsh.initialize()
    gmsh.option.setNumber('General.Terminal', 0)
    gmsh.model.add('vessel')
    gmsh.model.addDiscreteEntity(2, 1)
    gmsh.model.mesh.addNodes(2, 1, np.arange(1, len(X) + 1), X.ravel())
    gmsh.model.mesh.addElementsByType(1, 2, [], (tris + 1).ravel())
    gmsh.model.mesh.classifySurfaces(40 * np.pi / 180, True, True, np.pi)
    gmsh.model.mesh.createGeometry()
    f = gmsh.model.mesh.field.add('Box')
    gmsh.model.mesh.field.setNumber(f, 'VIn', h_root)
    gmsh.model.mesh.field.setNumber(f, 'VOut', h_far)
    for k, v in zip(('XMin', 'XMax', 'YMin', 'YMax', 'ZMin', 'ZMax'), (-3, 3, -3, 3, root_z[0], root_z[1])):
        gmsh.model.mesh.field.setNumber(f, k, v)
    gmsh.model.mesh.field.setNumber(f, 'Thickness', 2.0)
    gmsh.model.mesh.field.setAsBackgroundMesh(f)
    for k in ('Mesh.MeshSizeExtendFromBoundary', 'Mesh.MeshSizeFromPoints', 'Mesh.MeshSizeFromCurvature'):
        gmsh.option.setNumber(k, 0)
    gmsh.option.setNumber('Mesh.Algorithm', 6)
    gmsh.model.mesh.generate(2)
    tags, xyz, _ = gmsh.model.mesh.getNodes()
    Y = xyz.reshape(-1, 3)
    pos = np.full(int(tags.max()) + 1, -1)
    pos[tags.astype(int)] = np.arange(len(tags))
    T = []
    for dim, tag in gmsh.model.getEntities(2):
        types, _, conn = gmsh.model.mesh.getElements(dim, tag)
        for et, c in zip(types, conn):
            if et == 2:
                T.append(pos[c.astype(int)].reshape(-1, 3))
    gmsh.finalize()
    T = np.vstack(T)
    used, inv = np.unique(T, return_inverse=True)
    Y, T = Y[used], inv.reshape(-1, 3)
    # snap the end rings onto their planes and orient every face away from the vessel axis
    Y[np.abs(Y[:, 2] - z_in) < 1e-2, 2] = z_in
    Y[np.abs(Y[:, 2] - z_out) < 1e-2, 2] = z_out
    n = np.cross(Y[T[:, 1]] - Y[T[:, 0]], Y[T[:, 2]] - Y[T[:, 0]])
    inward = np.einsum('ij,ij->i', n, Y[T].mean(1) * [1, 1, 0]) < 0
    T[inward] = T[inward][:, [0, 2, 1]]
    return Y, T


# ---------------------------------------------------------------------------------------------------
# rigid vessel wall: the surface extruded outwards
# ---------------------------------------------------------------------------------------------------
def wall_mesh(Y, T, thickness, layers, smooth=300, z_ends=()):
    """Extrude the vessel surface outwards by `thickness` in `layers` prism layers -> tetrahedra.

    The extrusion directions are vertex normals smoothed over the surface; this removes folding in
    the concave groove where the inflow tube meets the sinuses, at the price of a locally thinner
    wall there (reported by the caller). Nodes on the end planes keep their plane."""
    n = len(Y)
    fn = np.cross(Y[T[:, 1]] - Y[T[:, 0]], Y[T[:, 2]] - Y[T[:, 0]])
    N0 = np.zeros_like(Y)
    for j in range(3):
        np.add.at(N0, T[:, j], fn)
    N0 /= np.linalg.norm(N0, axis=1)[:, None]
    N = N0.copy()
    E = np.unique(np.sort(np.vstack([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), axis=1), axis=0)
    deg = np.bincount(E.ravel(), minlength=n)
    end = np.zeros(n, bool)
    for z in z_ends:
        end |= np.abs(Y[:, 2] - z) < 1e-9
    for _ in range(smooth):
        S = N.copy()
        np.add.at(S, E[:, 0], N[E[:, 1]])
        np.add.at(S, E[:, 1], N[E[:, 0]])
        N = S / (deg + 1)[:, None]
        N[end, 2] = 0
        N /= np.linalg.norm(N, axis=1)[:, None]
    P = np.vstack([Y + t * N for t in np.linspace(0, thickness, layers + 1)])
    prisms = np.vstack([np.c_[T + k * n, T + (k + 1) * n] for k in range(layers)])
    bot, top = P[prisms[:, :3]], P[prisms[:, 3:]]
    nb = np.cross(bot[:, 1] - bot[:, 0], bot[:, 2] - bot[:, 0])
    nt = np.cross(top[:, 1] - top[:, 0], top[:, 2] - top[:, 0])
    flipped = int((np.einsum('ij,ij->i', nb, nt) <= 0).sum())
    tets, vol = orient(P, prism_to_tet(prisms))
    normal_thickness = thickness * np.einsum('ij,ij->i', N, N0)
    return P, tets, dict(flipped=flipped, min_volume=float(vol.min()), volume=float(vol.sum()),
                         thickness_min=float(normal_thickness.min()),
                         thickness_p01=float(np.percentile(normal_thickness, 1)),
                         thickness_median=float(np.median(normal_thickness)))


# ---------------------------------------------------------------------------------------------------
# leaflets
# ---------------------------------------------------------------------------------------------------
PART_ROOT, PART_BODY, PART_EXT = 3, 4, 5


def valve_mesh(dyna_dir, thickness, layers, refine, ext, ext_rows, wall_Y, wall_T):
    """Leaflet solid: LS-DYNA shells (mid-surface) refined, attachment edges extended outwards by `ext`,
    extruded to `thickness` in `layers` layers -> tetrahedra. Parts: 3 root strip (LS-DYNA LF_root),
    4 leaflet body (LF_top), 5 attachment extension."""
    nodes, elems = read_k(f'{dyna_dir}/mesh_leaf.k')
    s_root = read_node_set(f'{dyna_dir}/set_node_leaf.k')
    s_com = read_node_set(f'{dyna_dir}/set_node_com.k')
    X, E, idx = _arrays(nodes, elems)
    faces = [c for _, c in E]
    part = [p for p, _ in E]
    attach = np.zeros(len(X), int)
    attach[[idx[i] for i in s_root if i in idx]] = 1
    attach[[idx[i] for i in s_com if i in idx]] = 2

    for _ in range(refine):                                  # uniform in-plane refinement
        Xl, att, mid, F, Pp = list(map(list, X)), list(attach), {}, [], []

        def m(u, v):
            k = (min(u, v), max(u, v))
            if k not in mid:
                Xl.append(list((np.array(Xl[u]) + np.array(Xl[v])) / 2))
                att.append(att[u] if (att[u] and att[v]) else 0)
                mid[k] = len(Xl) - 1
            return mid[k]
        for c, p in zip(faces, part):
            if len(c) == 3:
                a0, a1, a2 = c
                m01, m12, m20 = m(a0, a1), m(a1, a2), m(a2, a0)
                F += [[a0, m01, m20], [m01, a1, m12], [m20, m12, a2], [m01, m12, m20]]
            else:
                a0, a1, a2, a3 = c
                m01, m12, m23, m30 = m(a0, a1), m(a1, a2), m(a2, a3), m(a3, a0)
                Xl.append(list(np.mean([Xl[i] for i in c], 0)))
                att.append(0)
                cc = len(Xl) - 1
                F += [[a0, m01, cc, m30], [m01, a1, m12, cc], [cc, m12, a2, m23], [m30, cc, m23, a3]]
            Pp += [p] * 4
        X, attach, faces, part = np.array(Xl), np.array(att), F, Pp

    # attachment edges: boundary edges with both nodes on the LS-DYNA attachment / commissure sets
    und, owner = collections.Counter(), {}
    for fi, c in enumerate(faces):
        for i in range(len(c)):
            u, v = c[i], c[(i + 1) % len(c)]
            und[(min(u, v), max(u, v))] += 1
            owner[(u, v)] = fi
    att_edges = [(u, v) for (u, v), fi in owner.items()
                 if und[(min(u, v), max(u, v))] == 1 and attach[u] and attach[v]]

    def face_normal(c):
        return np.cross(X[c[1]] - X[c[0]], X[c[-1]] - X[c[0]])
    dirs = collections.defaultdict(lambda: np.zeros(3))
    for u, v in att_edges:
        c = faces[owner[(u, v)]]
        d = np.cross(X[v] - X[u], face_normal(c))
        d /= np.linalg.norm(d)
        if np.dot(d, (X[u] + X[v]) / 2 - X[c].mean(0)) < 0:
            d = -d
        dirs[u] += d
        dirs[v] += d
    # blend with the wall's outward vertex normal (smooth), then smooth along the attachment line
    fn = np.cross(wall_Y[wall_T[:, 1]] - wall_Y[wall_T[:, 0]], wall_Y[wall_T[:, 2]] - wall_Y[wall_T[:, 0]])
    Nv = np.zeros_like(wall_Y)
    for j in range(3):
        np.add.at(Nv, wall_T[:, j], fn)
    Nv /= np.linalg.norm(Nv, axis=1)[:, None]

    def wall_normal(q, k=6):
        d2 = ((wall_Y - q) ** 2).sum(1)
        i = np.argpartition(d2, k)[:k]
        w = 1 / (np.sqrt(d2[i]) + 1e-6)
        nn = (Nv[i] * w[:, None]).sum(0)
        return nn / np.linalg.norm(nn)
    D = {u: d / np.linalg.norm(d) + wall_normal(X[u]) for u, d in dirs.items()}
    D = {u: d / np.linalg.norm(d) for u, d in D.items()}
    nb = collections.defaultdict(set)
    for u, v in att_edges:
        nb[u].add(v)
        nb[v].add(u)
    for _ in range(10):
        D = {u: D[u] + sum(D[v] for v in nb[u]) for u in D}
        D = {u: d / np.linalg.norm(d) for u, d in D.items()}
    Xl, att, rings = list(map(list, X)), list(attach), {}
    for u, d in D.items():
        rings[u] = [u]
        for k in range(1, ext_rows + 1):
            Xl.append(list(X[u] + d * ext * k / ext_rows))
            att.append(3)
            rings[u].append(len(Xl) - 1)
    for u, v in att_edges:
        for k in range(ext_rows):
            faces.append([rings[v][k], rings[u][k], rings[u][k + 1], rings[v][k + 1]])
            part.append(PART_EXT)
    X, attach = np.array(Xl), np.array(att)

    # extrusion through the thickness (shell = mid-surface), conforming tetrahedra
    nrm = np.zeros_like(X)
    for c in faces:
        nrm[c] += face_normal(c)
    nrm /= np.linalg.norm(nrm, axis=1)[:, None]
    nN = len(X)
    P = np.vstack([X + o * nrm for o in np.linspace(-thickness / 2, thickness / 2, layers + 1)])
    tri, tpart = [], []
    for c, p in zip(faces, part):
        if len(c) == 3:
            tri.append(c)
            tpart.append(p)
        else:
            tri += quad_to_tri(np.array([c])).tolist()
            tpart += [p, p]
    tri = np.array(tri)
    prisms = np.vstack([np.c_[tri + k * nN, tri + (k + 1) * nN] for k in range(layers)])
    tets = prism_to_tet(prisms)
    tet_part = np.tile(np.tile(np.array(tpart), layers), 3)
    tets, vol = orient(P, tets)
    return P, tets, tet_part, dict(attachment_edges=len(att_edges), min_volume=float(vol.min()),
                                   volume=float(vol.sum()))
