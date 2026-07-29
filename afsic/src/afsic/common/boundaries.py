"""
边界标记工具 — 消除演示中 ~13 处重复的边界标记样板代码。

用法:
    from afsic.common import tag_boundaries

    boundaries = [(1, lambda x: np.isclose(x[0], 0)),     # left
                  (2, lambda x: np.isclose(x[0], Lx)),    # right
                  (3, lambda x: np.isclose(x[1], 0)),     # bottom
                  (4, lambda x: np.isclose(x[1], Ly))]    # top
    facet_tag = tag_boundaries(mesh, boundaries)
"""

import numpy as np
from dolfinx.mesh import locate_entities, meshtags


def tag_boundaries(mesh, boundaries):
    """
    为矩形网格的四个边界创建 meshtags。

    Parameters
    ----------
    mesh : dolfinx.mesh.Mesh
        已创建拓扑连通性的网格。
    boundaries : list of tuple
        [(marker_id, locator_fn), ...] 列表，
        locator_fn 签名为 lambda x: bool_array。

    Returns
    -------
    facet_tag : dolfinx.mesh.MeshTags
        面标签对象，可通过 facet_tag.find(marker) 获取对应面。
    """
    mesh.topology.create_connectivity(mesh.topology.dim - 1,
                                       mesh.topology.dim)

    fdim = mesh.topology.dim - 1
    facet_indices, facet_markers = [], []

    for marker, locator in boundaries:
        facets = locate_entities(mesh, fdim, locator)
        facet_indices.append(facets)
        facet_markers.append(np.full_like(facets, marker))

    facet_indices = np.hstack(facet_indices).astype(np.int32)
    facet_markers = np.hstack(facet_markers).astype(np.int32)
    sorted_facets = np.argsort(facet_indices)

    return meshtags(mesh, fdim,
                    facet_indices[sorted_facets],
                    facet_markers[sorted_facets])


# 常用边界标记常量，可用 marker_* 替代魔法数字
MARKER_LEFT   = 1
MARKER_RIGHT  = 2
MARKER_BOTTOM = 3
MARKER_TOP    = 4
MARKER_FRONT  = 5
MARKER_BACK   = 6


def rectangle_boundaries(Lx=1.0, Ly=1.0):
    """返回标准矩形域的四个边界 locator。"""
    return [
        (MARKER_LEFT,   lambda x: np.isclose(x[0], 0)),
        (MARKER_RIGHT,  lambda x: np.isclose(x[0], Lx)),
        (MARKER_BOTTOM, lambda x: np.isclose(x[1], 0)),
        (MARKER_TOP,    lambda x: np.isclose(x[1], Ly)),
    ]


def box_boundaries(Lx=1.0, Ly=1.0, Lz=1.0):
    """返回标准长方体域的六个边界 locator。"""
    return [
        (MARKER_LEFT,   lambda x: np.isclose(x[0], 0)),
        (MARKER_RIGHT,  lambda x: np.isclose(x[0], Lx)),
        (MARKER_BOTTOM, lambda x: np.isclose(x[1], 0)),
        (MARKER_TOP,    lambda x: np.isclose(x[1], Ly)),
        (MARKER_FRONT,  lambda x: np.isclose(x[2], 0)),
        (MARKER_BACK,   lambda x: np.isclose(x[2], Lz)),
    ]
