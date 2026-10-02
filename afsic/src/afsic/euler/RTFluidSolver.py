"""Serial exactly divergence-conforming RT/DG backward-Euler fluid solver.

SIP viscous terms and lagged-velocity upwind convection follow DOLFINx's
v0.10.0 demo_navier-stokes.py (LGPL-3.0-or-later):
https://github.com/FEniCS/dolfinx/blob/v0.10.0/python/demo/demo_navier-stokes.py
Normal Dirichlet data are strong; tangential data are weak (Nitsche).
Open facets carry zero natural traction. This is not the Chorin algorithm.
"""
import numpy as np
import ufl
from basix.ufl import element, mixed_element
from petsc4py import PETSc
from dolfinx import fem, mesh as dmesh
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, set_bc


class RTFluidSolver:
    def __init__(self, mesh, dt, rho=1., mu=1., degree=2, convection=True,
                 dirichlet_facets=None, penalty=None):
        if mesh.comm.size != 1:
            raise NotImplementedError("RTFluidSolver currently supports one MPI rank only")
        if mesh.geometry.dim != 2 or np.issubdtype(PETSc.ScalarType, np.complexfloating):
            raise NotImplementedError("This prototype requires 2D geometry and real PETSc scalars")
        if degree < 2:
            raise ValueError("Start at Basix RT degree 2 for immersed motion")
        if min(dt, rho, mu) <= 0:
            raise ValueError("dt, rho and mu must be positive")
        self.mesh, self.dt = mesh, dt
        cell = mesh.basix_cell()
        ve = element("RT", cell, degree)
        qe = element("Lagrange", cell, degree - 1, discontinuous=True)
        self.W = fem.functionspace(mesh, mixed_element([ve, qe]))
        self.V, self.u_map = self.W.sub(0).collapse()
        self.Q, self.p_map = self.W.sub(1).collapse()
        self.u_ = fem.Function(self.V, name="velocity")
        self.u_n = fem.Function(self.V, name="previous_velocity")
        self.p_ = fem.Function(self.Q, name="pressure")
        self.solution = fem.Function(self.W)
        self.u_D = fem.Function(self.V)
        self.f = fem.Function(fem.functionspace(mesh, ("DG", degree, (mesh.geometry.dim,))))
        self.ib_load = None
        tdim = mesh.topology.dim
        mesh.topology.create_connectivity(tdim - 1, tdim)
        exterior = dmesh.exterior_facet_indices(mesh.topology)
        facets = exterior if dirichlet_facets is None else np.asarray(dirichlet_facets, dtype=np.int32)
        facets = np.unique(facets).astype(np.int32)
        self.closed = len(facets) == len(exterior)
        tags = dmesh.meshtags(mesh, tdim - 1, facets, np.ones(len(facets), dtype=np.int32))
        ds = ufl.Measure("ds", domain=mesh, subdomain_data=tags)
        dofs = fem.locate_dofs_topological((self.W.sub(0), self.V), tdim - 1, facets)
        self.bcs = [fem.dirichletbc(self.u_D, dofs, self.W.sub(0))]
        u, p = ufl.TrialFunctions(self.W)
        v, q = ufl.TestFunctions(self.W)
        n, h = ufl.FacetNormal(mesh), ufl.CellDiameter(mesh)
        alpha = fem.Constant(mesh, PETSc.ScalarType(penalty or 20. * degree**2))
        def jump(w):
            return ufl.outer(w('+'), n('+')) + ufl.outer(w('-'), n('-'))
        a = rho / dt * ufl.inner(u, v) * ufl.dx
        a += mu * (
            ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
            - ufl.inner(ufl.avg(ufl.grad(u)), jump(v)) * ufl.dS
            - ufl.inner(jump(u), ufl.avg(ufl.grad(v))) * ufl.dS
            + alpha / ufl.avg(h) * ufl.inner(jump(u), jump(v)) * ufl.dS
            - ufl.inner(ufl.grad(u), ufl.outer(v, n)) * ds(1)
            - ufl.inner(ufl.outer(u, n), ufl.grad(v)) * ds(1)
            + alpha / h * ufl.inner(ufl.outer(u, n), ufl.outer(v, n)) * ds(1))
        a -= (p * ufl.div(v) + q * ufl.div(u)) * ufl.dx
        L = ufl.inner(rho / dt * self.u_n + self.f, v) * ufl.dx
        L += mu * (-ufl.inner(ufl.outer(self.u_D, n), ufl.grad(v)) * ds(1)
                   + alpha / h * ufl.inner(ufl.outer(self.u_D, n), ufl.outer(v, n)) * ds(1))
        if convection:
            w = self.u_n
            inflow_selector = ufl.conditional(ufl.gt(ufl.dot(w, n), 0), 1., 0.)
            upwind = inflow_selector('+') * u('+') + inflow_selector('-') * u('-')
            a += rho * (-ufl.inner(u, ufl.div(ufl.outer(v, w))) * ufl.dx
                        + ufl.inner(ufl.dot(w, n)('+') * upwind, v('+')) * ufl.dS
                        + ufl.inner(ufl.dot(w, n)('-') * upwind, v('-')) * ufl.dS
                        + ufl.inner(ufl.dot(w, n) * inflow_selector * u, v) * ufl.ds)
            L -= rho * ufl.inner(ufl.dot(w, n) * (1 - inflow_selector) * self.u_D, v) * ds(1)
        self.a, self.L = fem.form(a), fem.form(L)
        self.ksp = PETSc.KSP().create(mesh.comm)
        self.ksp.setType("preonly")
        self.ksp.getPC().setType("lu")
        self.ksp.getPC().setFactorSolverType("mumps")
        self.ksp.setOptionsPrefix("afsi_rt_")
        opts = PETSc.Options()
        opts["afsi_rt_mat_mumps_icntl_24"] = 1
        opts["afsi_rt_mat_mumps_icntl_25"] = 0
        self.ksp.setFromOptions()
        self.ksp.setErrorIfNotConverged(True)
        self.convection = convection
        self.A = None
        self.div_form = fem.form(ufl.div(self.u_)**2 * ufl.dx)
        self.pressure_form = fem.form(self.p_ * ufl.dx)
        self.volume = fem.assemble_scalar(fem.form(fem.Constant(mesh, PETSc.ScalarType(1.)) * ufl.dx))

    def solve_one_step(self, ib_load=None):
        if ib_load is None:
            ib_load = self.ib_load
        if self.A is None or self.convection:
            if self.A is not None:
                self.A.destroy()
            self.A = assemble_matrix(self.a, bcs=self.bcs)
            self.A.assemble()
            self.ksp.setOperators(self.A)
        b = assemble_vector(self.L)
        apply_lifting(b, [self.a], [self.bcs])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        if ib_load is not None:
            values = ib_load.getArray(readonly=True) if hasattr(ib_load, 'getArray') else np.asarray(ib_load)
            if values.shape != self.u_.x.array.shape:
                raise ValueError("IB load must use the collapsed RT velocity-space layout")
            b.getArray()[self.u_map] += values
        set_bc(b, self.bcs)
        self.ksp.solve(b, self.solution.x.petsc_vec)
        b.destroy()
        self.solution.x.scatter_forward()
        self.u_.x.array[:] = self.solution.x.array[self.u_map]
        self.p_.x.array[:] = self.solution.x.array[self.p_map]
        if self.closed:
            self.p_.x.array[:] -= fem.assemble_scalar(self.pressure_form) / self.volume
        self.u_n.x.array[:] = self.u_.x.array
        return self.u_

    def divergence_norm(self):
        return float(np.sqrt(max(0., fem.assemble_scalar(self.div_form))))
