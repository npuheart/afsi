"""Serial nodal immersed coupling: U = E u, b = E.T L.

Basis values are evaluated by DOLFINx, including Piola mappings and DOF
transformations. A cell-conflict coloring evaluates independent unit basis
functions in batches. No regularized delta, mass inverse, or RT projection.
"""
import numpy as np
from scipy.sparse import coo_matrix
from dolfinx import fem, geometry


class RTNodalCoupling:
    def __init__(self, V):
        if V.mesh.comm.size != 1:
            raise NotImplementedError("RTNodalCoupling currently supports one MPI rank only")
        if V.dofmap.bs != 1 or V.element.value_shape != (V.mesh.geometry.dim,):
            raise ValueError("Expected a vector-valued RT space with scalar DOF storage")
        self.V = V
        self.mesh = V.mesh
        self.gdim = self.mesh.geometry.dim
        self.ndofs = V.dofmap.index_map.size_local
        self.tree = geometry.bb_tree(self.mesh, self.mesh.topology.dim)
        ncells = self.mesh.topology.index_map(self.mesh.topology.dim).size_local
        self.cell_dofs = np.asarray([V.dofmap.cell_dofs(c) for c in range(ncells)])
        neighbours = [set() for _ in range(self.ndofs)]
        for dofs in self.cell_dofs:
            for j in dofs:
                neighbours[j].update(int(k) for k in dofs if k != j)
        self.colors = np.full(self.ndofs, -1, dtype=np.int32)
        for j in sorted(range(self.ndofs), key=lambda j: len(neighbours[j]), reverse=True):
            used = {self.colors[k] for k in neighbours[j] if self.colors[k] >= 0}
            color = 0
            while color in used:
                color += 1
            self.colors[j] = color
        self.ncolors = int(self.colors.max()) + 1
        self.probe = fem.Function(V)
        self.E = None

    def update(self, positions):
        """Rebuild at CURRENT physical marker positions; no solid mesh movement.

        A point on an interelement facet uses the smallest local cell index.
        The same chosen trace is used in interpolation and spreading.
        Out-of-domain markers are an error, never silently discarded.
        """
        if isinstance(positions, fem.Function):
            xyz = positions.x.array.reshape(-1, self.gdim)
        else:
            xyz = np.asarray(positions, dtype=np.float64)
        if xyz.ndim != 2 or xyz.shape[1] not in (self.gdim, 3):
            raise ValueError("positions must have shape (number of markers, gdim) or (n, 3)")
        if not len(xyz) or not np.isfinite(xyz).all():
            raise ValueError("Expected nonempty finite marker coordinates")
        points = np.zeros((len(xyz), 3), dtype=self.mesh.geometry.x.dtype)
        points[:, :self.gdim] = xyz[:, :self.gdim]
        candidates = geometry.compute_collisions_points(self.tree, points)
        hits = geometry.compute_colliding_cells(self.mesh, candidates, points)
        missing = [a for a in range(len(points)) if len(hits.links(a)) == 0]
        if missing:
            raise ValueError(f"{len(missing)} markers outside fluid mesh; first indices {missing[:8]}")
        cells = np.asarray([min(hits.links(a)) for a in range(len(points))], dtype=np.int32)
        dofs = self.cell_dofs[cells]
        colors = self.colors[dofs]
        values = np.zeros((len(points), dofs.shape[1], self.gdim))
        for color in range(self.ncolors):
            self.probe.x.array[:] = (self.colors == color)
            evaluated = self.probe.eval(points, cells).reshape(len(points), self.gdim)
            a, j = np.nonzero(colors == color)
            values[a, j, :] = evaluated[a]
        rows = np.broadcast_to(
            (np.arange(len(points))[:, None, None] * self.gdim + np.arange(self.gdim)), values.shape)
        cols = np.broadcast_to(dofs[:, :, None], values.shape)
        self.E = coo_matrix((values.ravel(), (rows.ravel(), cols.ravel())),
                            shape=(len(points) * self.gdim, self.ndofs)).tocsr()
        self.points, self.cells = points, cells
        return self.E

    def interpolate(self, velocity, out=None):
        if self.E is None:
            raise RuntimeError("Call update before coupling")
        coeff = velocity.x.array if isinstance(velocity, fem.Function) else np.asarray(velocity)
        result = np.asarray(self.E @ coeff)
        if out is not None:
            if out.x.array.size != result.size:
                raise ValueError("Solid vector layout does not match marker layout")
            out.x.array[:] = result
            out.x.scatter_forward()
        return result

    def spread(self, solid_load, out=None):
        """Input is an ASSEMBLED nodal load, not force density. No weights added."""
        if self.E is None:
            raise RuntimeError("Call update before coupling")
        if isinstance(solid_load, fem.Function):
            loads = solid_load.x.array
        elif hasattr(solid_load, "getArray"):
            loads = solid_load.getArray(readonly=True)
        else:
            loads = np.asarray(solid_load)
        result = np.asarray(self.E.T @ loads)
        if out is not None:
            out.getArray()[:] = result
        return result
