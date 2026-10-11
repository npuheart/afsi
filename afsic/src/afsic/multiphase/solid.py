"""FEniCSx TRI3, plane-strain neo-Hookean solid for nodal IFED coupling."""
import numpy as np
from scipy.spatial import cKDTree


class ElasticSolid2D:
    """Serial CG1 solid; elastic/optional support loads per reference volume.

    This is the ordinary single-mesh IFED formulation. Solid inertia is already
    present in the Eulerian density; no second solid inertia or gravity is added.
    Support is a volumetric spring on selected reference cells, not an exact
    displacement boundary condition. The fluid interpolant advances every node.
    """
    def __init__(self, domain, shear_modulus, bulk_modulus, density,
                 anchor=None, anchor_stiffness=0., anchor_damping=0.):
        from dolfinx import fem, mesh
        from dolfinx.fem.petsc import create_vector
        from basix.ufl import element
        import ufl

        if domain.comm.size != 1:
            raise ValueError("The initial MAC/FE backend is serial only")
        if domain.topology.dim != 2 or domain.geometry.dim != 2 or domain.topology.cell_name() != "triangle":
            raise ValueError("A 2D triangular mesh is required")
        p = [shear_modulus, bulk_modulus, density, anchor_stiffness, anchor_damping]
        if not np.isfinite(p).all() or min(p[:3]) <= 0 or min(p[3:]) < 0:
            raise ValueError("Invalid solid material/support parameters")
        self.domain, self.shear_modulus, self.bulk_modulus, self.density = domain, *p[:3]
        self.anchor_stiffness, self.anchor_damping = p[3:]
        self.V = fem.functionspace(domain, element("Lagrange", "triangle", 1, shape=(2,)))
        self.chi = fem.Function(self.V, name="solid_position")
        self.chi.interpolate(lambda x: x[:2])
        self.velocity = fem.Function(self.V, name="solid_velocity")
        self.reference = self.V.tabulate_dof_coordinates()[:, :2].copy()
        self.cells = np.array([self.V.dofmap.cell_dofs(c) for c in range(domain.topology.index_map(2).size_local)])
        # Exterior mesh facets -> geometry nodes -> CG1 block dofs. Avoid assuming
        # the geometry and function-space node orderings are identical.
        domain.topology.create_connectivity(1, 2)
        facets = mesh.exterior_facet_indices(domain.topology)
        geometry_nodes = mesh.entities_to_geometry(domain, 1, facets, False)
        distances, self.boundary_dofs = cKDTree(self.reference).query(domain.geometry.x[geometry_nodes, :2])
        if np.max(distances) > 1e-10:
            raise ValueError("Cannot match boundary vertices to CG1 dofs")
        self.boundary_dofs = self.boundary_dofs.astype(int)
        q = fem.functionspace(domain, ("DG", 0))
        support = fem.Function(q)
        if anchor is not None:
            support.interpolate(lambda x: np.asarray(anchor(x), dtype=float))
        if np.any((support.x.array < 0) | (support.x.array > 1)):
            raise ValueError("anchor must return weights in [0,1]")
        F = ufl.variable(ufl.grad(self.chi))
        J = ufl.det(F)
        I1 = ufl.inner(F, F) + 1.  # Embedded 3D F=diag(F2,1), plane strain.
        W = .5*shear_modulus*(J**(-2/3)*I1 - 3) + .5*bulk_modulus*ufl.ln(J)**2
        P = ufl.diff(W, F)
        v = ufl.TestFunction(self.V)
        dx = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": 4})
        displacement = self.chi - ufl.SpatialCoordinate(domain)
        self.load_form = fem.form(-ufl.inner(P, ufl.grad(v))*dx
                                 - support*anchor_stiffness*ufl.inner(displacement, v)*dx
                                 - support*anchor_damping*ufl.inner(self.velocity, v)*dx)
        self.energy_form = fem.form((W + .5*support*anchor_stiffness*ufl.inner(displacement, displacement))*dx)
        self.volume_form = fem.form(J*dx)
        self.j_expression = fem.Expression(J, np.array([[1/3, 1/3]]))
        self.cell_ids = np.arange(len(self.cells), dtype=np.int32)
        self.load_vector = create_vector(self.V)
        self.volume0 = float(fem.assemble_scalar(self.volume_form))
        # A conservative user-visible estimate, not a proven coupled stability bound.
        tri = self.reference[self.cells]
        edges = [np.linalg.norm(tri[:, i] - tri[:, j], axis=1) for i, j in ((0,1), (1,2), (2,0))]
        e1, e2 = tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]
        area2 = np.abs(e1[:, 0]*e2[:, 1] - e1[:, 1]*e2[:, 0])
        self.h_min = float(np.min(area2 / np.maximum.reduce(edges)))

    @property
    def positions(self):
        return self.chi.x.array.reshape(-1, 2)

    def boundary_segments(self):
        return self.positions[self.boundary_dofs]

    def stable_dt(self, cfl=.1):
        if not 0 < cfl <= .5:
            raise ValueError("Solid cfl must be in (0,.5]")
        wave = np.sqrt((self.bulk_modulus + 4*self.shear_modulus/3) / self.density)
        rates = [wave / self.h_min, np.sqrt(self.anchor_stiffness / self.density), self.anchor_damping / self.density]
        return cfl / max(rates)

    def diagnostics(self):
        from dolfinx import fem
        j = self.j_expression.eval(self.domain, self.cell_ids).ravel()
        if not np.isfinite(j).all() or np.min(j) <= 0:
            raise RuntimeError("Solid element inversion/nonfinite Jacobian; reduce dt/refine the grid")
        return dict(solid_J_min=float(j.min()), solid_J_max=float(j.max()),
                    solid_volume=float(fem.assemble_scalar(self.volume_form)),
                    elastic_support_energy=float(fem.assemble_scalar(self.energy_form)),
                    solid_displacement_max=float(np.linalg.norm(self.positions-self.reference, axis=1).max()))

    def load(self):
        from dolfinx.fem.petsc import assemble_vector
        from petsc4py import PETSc
        self.diagnostics()  # Check J before evaluating log(J) in the load form.
        with self.load_vector.localForm() as local:
            local.set(0)
        assemble_vector(self.load_vector, self.load_form)
        self.load_vector.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        load = self.load_vector.array.copy().reshape(-1, 2)
        if not np.isfinite(load).all():
            raise RuntimeError("Nonfinite FE load")
        return load

    def advance(self, velocity, dt):
        velocity = np.asarray(velocity, float)
        if velocity.shape != self.positions.shape or not np.isfinite(velocity).all() or not np.isfinite(dt) or dt <= 0:
            raise ValueError("Invalid solid velocity or dt")
        self.velocity.x.array[:] = velocity.ravel()
        self.chi.x.array[:] += dt * velocity.ravel()
        self.velocity.x.scatter_forward()
        self.chi.x.scatter_forward()
        self.diagnostics()

    def close(self):
        self.load_vector.destroy()
