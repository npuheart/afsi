"""Conforming splits used by the mesh generator: quadrilateral -> 2 triangles, prism -> 3 tetrahedra.

Every quadrilateral face is cut along the diagonal through its smallest global vertex index, so two
cells sharing that face always cut it the same way and the tetrahedral mesh stays conforming."""
import numpy as np


def quad_to_tri(q):
    q = np.asarray(q)
    k = np.argmin(q, axis=1)
    q = np.take_along_axis(q, (k[:, None] + np.arange(4)) % 4, axis=1)   # smallest vertex first
    return np.vstack([q[:, [0, 1, 2]], q[:, [0, 2, 3]]])


# prism: bottom (0,1,2), top (3,4,5), vertical edges 0-3, 1-4, 2-5
_PRISM_SYM = np.array([[0, 1, 2, 3, 4, 5], [1, 2, 0, 4, 5, 3], [2, 0, 1, 5, 3, 4],
                       [3, 5, 4, 0, 2, 1], [4, 3, 5, 1, 0, 2], [5, 4, 3, 2, 1, 0]])


def prism_to_tet(p):
    p = np.asarray(p)
    k = np.argmin(p, axis=1)
    p = p[np.arange(len(p))[:, None], _PRISM_SYM[k]]        # smallest vertex at position 0
    a, b, c, d, e, f = p.T
    use_bf = np.minimum(b, f) < np.minimum(c, e)
    t1 = np.where(use_bf[:, None], np.c_[a, b, c, f], np.c_[a, b, c, e])
    t2 = np.where(use_bf[:, None], np.c_[a, b, f, e], np.c_[a, e, c, f])
    t3 = np.c_[a, e, f, d]
    return np.vstack([t1, t2, t3])


def orient(X, tets):
    """Reorder tetrahedra to positive volume; returns (tets, volumes)."""
    e1, e2, e3 = (X[tets[:, i]] - X[tets[:, 0]] for i in (1, 2, 3))
    v = np.einsum('ij,ij->i', np.cross(e1, e2), e3)
    tets = tets.copy()
    tets[v < 0] = tets[v < 0][:, [0, 2, 1, 3]]
    return tets, np.abs(v) / 6
