"""Numerical invariants for the single-rank RT immersed coupling prototype."""
import numpy as np
import pytest
import ufl
from mpi4py import MPI
from dolfinx import fem, mesh
from afsic import RTFluidSolver, RTNodalCoupling


@pytest.mark.parametrize('cell_type', [mesh.CellType.triangle, mesh.CellType.quadrilateral])
def test_sampling_adjoint_force_and_torque(cell_type):
    msh = mesh.create_unit_square(MPI.COMM_WORLD, 5, 4, cell_type=cell_type)
    # Oblique physical elements exercise mapping and non-axis-aligned facets.
    msh.geometry.x[:, 0] += 0.17 * msh.geometry.x[:, 1]
    V = fem.functionspace(msh, ('RT', 2))
    coupling = RTNodalCoupling(V)
    rng = np.random.default_rng(104)
    points = rng.uniform(.1, .85, (47, 2))
    points[:, 0] += .17 * points[:, 1]
    # Include a vertex and facet points; use the stored trace consistently.
    points = np.vstack((points, [0., 0.], [.1 + .17 * .25, .25]))
    coupling.update(points)
    u = fem.Function(V)
    u.x.array[:] = rng.normal(size=len(u.x.array))
    sampled = coupling.interpolate(u).reshape(-1, 2)
    exact = u.eval(coupling.points, coupling.cells).reshape(-1, 2)
    np.testing.assert_allclose(sampled, exact, rtol=2e-13, atol=2e-13)
    loads = rng.normal(size=points.size)
    b = coupling.spread(loads)
    np.testing.assert_allclose(u.x.array @ b, sampled.ravel() @ loads, rtol=2e-13, atol=2e-12)
    for direction in range(2):
        u.interpolate(lambda x: np.vstack((np.full(x.shape[1], float(direction == 0)),
                                            np.full(x.shape[1], float(direction == 1)))))
        np.testing.assert_allclose(u.x.array @ b, loads.reshape(-1,2)[:,direction].sum(), atol=2e-12)
    u.interpolate(lambda x: np.vstack((-x[1], x[0])))
    torque = np.sum(points[:,0] * loads.reshape(-1,2)[:,1] - points[:,1] * loads.reshape(-1,2)[:,0])
    np.testing.assert_allclose(u.x.array @ b, torque, atol=2e-12)
    # Moving the markers updates E, rather than using stale positions.
    coupling.update(points[:-2] + .01)
    np.testing.assert_allclose(coupling.interpolate(u).reshape(-1,2),
                               np.column_stack((-(points[:-2,1]+.01),points[:-2,0]+.01)), atol=2e-12)
    with pytest.raises(ValueError, match='outside fluid mesh'):
        coupling.update(np.array([[3., 3.]]))


@pytest.mark.parametrize('cell_type', [mesh.CellType.triangle, mesh.CellType.quadrilateral])
@pytest.mark.parametrize('convection', [False, True])
def test_incompressibility_and_normal_continuity(cell_type, convection):
    msh = mesh.create_unit_square(MPI.COMM_WORLD, 4, 4, cell_type=cell_type)
    solver = RTFluidSolver(msh, .01, mu=.2, convection=convection)
    coupling = RTNodalCoupling(solver.V)
    coupling.update(np.array([[.31,.37],[.6,.7],[.7,.43]]))
    b = coupling.spread(np.array([1., .4, -.5, .7, -.3, -.8]))
    for _ in range(3):
        solver.solve_one_step(b)
        assert solver.divergence_norm() < 1.e-10
        n = ufl.FacetNormal(msh)
        jump_norm = fem.assemble_scalar(fem.form(ufl.jump(solver.u_, n)**2 * ufl.dS))
        assert abs(jump_norm) < 1.e-20
    assert np.linalg.norm(solver.u_.x.array) > 1.e-5


def test_gradient_force_has_zero_velocity():
    msh = mesh.create_unit_square(MPI.COMM_WORLD, 4, 4, cell_type=mesh.CellType.quadrilateral)
    solver = RTFluidSolver(msh, .01, convection=False)
    # Gradient of p=x^2+y^2. Even though pressure is not in Q1, exact
    # integration against the solenoidal discrete kernel annihilates it.
    solver.f.interpolate(lambda x: np.vstack((2*x[0], 2*x[1])))
    solver.solve_one_step()
    norm = np.sqrt(fem.assemble_scalar(fem.form(ufl.inner(solver.u_,solver.u_) * ufl.dx)))
    assert norm < 1.e-11
    assert solver.divergence_norm() < 1.e-11


def test_open_channel_flux_balance():
    msh = mesh.create_unit_square(MPI.COMM_WORLD, 6, 6, cell_type=mesh.CellType.quadrilateral)
    facets = mesh.locate_entities_boundary(msh, 1, lambda x: np.isclose(x[0],0) | np.isclose(x[1],0) | np.isclose(x[1],1))
    solver = RTFluidSolver(msh, .01, convection=True, dirichlet_facets=facets)
    solver.u_D.interpolate(lambda x: np.vstack((6*x[1]*(1-x[1]), np.zeros(x.shape[1]))))
    for _ in range(2):
        solver.solve_one_step()
    assert solver.divergence_norm() < 1.e-9
    flux = fem.assemble_scalar(fem.form(ufl.dot(solver.u_,ufl.FacetNormal(msh))*ufl.ds))
    assert abs(flux) < 1.e-10
