"""Numerical invariants for the experimental serial three-phase backend."""
import numpy as np
import pytest

from afsic.multiphase import (MACGrid, PhaseProperties, ConservativeMACSolver,
                              MACCoupling, ElasticSolid2D, ThreePhaseSimulation)
from afsic.multiphase.levelset import (heaviside, solid_distance, reinitialize,
                                      liquid_volume, correct_volume)


def test_three_material_limits():
    p = PhaseProperties()
    rho, mu = p.fields(np.array([-2., 2., -2., 2.]), np.array([2., 2., -2., -2.]), .1)
    np.testing.assert_allclose(rho, [p.rho_liquid, p.rho_gas, p.rho_solid, p.rho_solid])
    np.testing.assert_allclose(mu, [p.mu_liquid, p.mu_gas, p.mu_solid, p.mu_solid])
    np.testing.assert_allclose(heaviside(np.array([-1., 0., 1.]), .2), [0., .5, 1.])
    with pytest.raises(ValueError):
        PhaseProperties(rho_gas=0.)


def test_polygon_distance_orientation_and_translation():
    vertices = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    segments = np.stack([vertices, np.roll(vertices, -1, axis=0)], axis=1)
    points = np.array([[.5, .5], [2., .5], [0., .3], [2., 2.]])
    expected = [-.5, 1., 0., np.sqrt(2)]
    np.testing.assert_allclose(solid_distance(points, segments), expected)
    np.testing.assert_allclose(solid_distance(points + [3., 7.], segments[:, ::-1] + [3., 7.]), expected)


def test_reinitialization_and_liquid_volume_with_solid():
    g = MACGrid(30, 20, 1.5, 1.)
    x, y = g.coordinates()
    phi = 3*(x-.71)
    d = reinitialize(g, phi)
    np.testing.assert_allclose(d, x-.71, atol=2e-15)
    psi = np.sqrt((x-.8)**2 + (y-.5)**2) - .18
    eps = .075
    target = liquid_volume(g, x-.71, psi, eps)
    fixed, shift = correct_volume(g, d+.14, psi, eps, target)
    assert abs(shift+.14) < 1e-12
    assert abs(liquid_volume(g, fixed, psi, eps)-target) < 1e-12
    with pytest.raises(ValueError):
        correct_volume(g, phi, psi, eps, 2*g.lx*g.ly)


@pytest.mark.parametrize("ratio", [1., 1000.])
def test_variable_density_projection(ratio):
    g = MACGrid(14, 10, 1.4, 1.)
    rng = np.random.default_rng(42)
    rho = np.where(g.coordinates()[0] < .6, ratio, 1.)
    f = ConservativeMACSolver(g, rho, np.zeros(g.shape))
    u, v = [rng.normal(size=a.shape) for a in g.zeros()]
    u, v, _ = f.project(u, v, rho, .01)
    assert np.abs(g.divergence(u, v)).max() < 1e-8
    assert np.max(np.abs(u[:, [0, -1]])) == 0
    assert np.max(np.abs(v[[0, -1]])) == 0
    u2, v2, _ = f.project(u, v, rho, .01)
    np.testing.assert_allclose(u2, u, atol=1e-9)
    np.testing.assert_allclose(v2, v, atol=1e-9)


def test_hydrostatic_balance_at_air_water_density_ratio():
    g = MACGrid(12, 20, .6, 1.)
    rho = np.where(g.coordinates()[1] < .53, 1000., 1.2)
    f = ConservativeMACSolver(g, rho, np.full(g.shape, .001))
    for _ in range(4):
        f.step(.001)
    assert max(np.abs(f.u).max(), np.abs(f.v).max()) < 2e-10
    assert np.max(np.abs(g.divergence(f.u, f.v))) < 1e-8


def test_mass_momentum_use_consistent_dual_fluxes():
    g = MACGrid(20, 16)
    rng = np.random.default_rng(19)
    rho = 1 + 999*rng.random(g.shape)
    f = ConservativeMACSolver(g, rho, np.zeros(g.shape), gravity=(0, 0))
    # Open uniform field for this local stencil test; do not call the closed-wall
    # projection. All interior face momenta must preserve this uniform velocity.
    f.u[:] = .73
    f.v[:] = -.41
    transported, pu, pv = f.transport(.001)
    ru, rv = g.face_average(transported)
    np.testing.assert_allclose(pu[:, 1:-1]/ru[:, 1:-1], .73, atol=2e-14)
    np.testing.assert_allclose(pv[1:-1]/rv[1:-1], -.41, atol=2e-14)


def test_closed_tank_density_mass_and_positivity():
    g = MACGrid(16, 12)
    rng = np.random.default_rng(25)
    rho = np.where(g.coordinates()[0] < .43, 1000., 1.2)
    f = ConservativeMACSolver(g, rho, np.full(g.shape, .001), gravity=(0., 0.))
    u, v = [rng.normal(scale=.1, size=a.shape) for a in g.zeros()]
    f.u, f.v, _ = f.project(u, v, rho, .01)
    for _ in range(10):
        f.step(min(.001, f.stable_dt()))
    assert abs(np.sum(f.rho)-np.sum(rho))/np.sum(rho) < 1e-14
    assert f.rho.min() >= 1.2-1e-9
    assert f.rho.max() <= 1000.+1e-9


def test_viscous_stress_dissipates_energy():
    g = MACGrid(16, 12, 1.6, 1.2)
    rng = np.random.default_rng(91)
    f = ConservativeMACSolver(g, np.ones(g.shape), .1 + rng.random(g.shape))
    f.u[:] = rng.normal(size=f.u.shape)
    f.v[:] = rng.normal(size=f.v.shape)
    f.u[:, [0, -1]], f.v[[0, -1]] = 0., 0.
    fu, fv = f._viscous_force()
    assert np.sum(fu*f.u) + np.sum(fv*f.v) < 0


@pytest.mark.parametrize("kernel", ["bspline", "peskin4"])
def test_coupling_power_and_interior_partition_of_unity(kernel):
    g = MACGrid(20, 16)
    rng = np.random.default_rng(3)
    points = .25 + .5*rng.random((60, 2))
    ib = MACCoupling(g, points, kernel=kernel)
    u, v = [rng.normal(size=a.shape) for a in g.zeros()]
    load = rng.normal(size=points.shape)
    fu, fv = ib.spread(load)
    lhs = np.sum(load*ib.interpolate(u, v))
    rhs = g.dx*g.dy*(np.sum(fu*u)+np.sum(fv*v))
    assert abs(lhs-rhs) < 1e-12
    np.testing.assert_allclose(ib.interpolate(np.ones_like(u), np.ones_like(v)), 1., atol=1e-14)
    np.testing.assert_allclose([fu.sum()*g.dx*g.dy, fv.sum()*g.dx*g.dy], load.sum(axis=0), atol=1e-13)
    # The same transpose identity must hold for supports truncated at a wall.
    ib.update(points * .05)
    fu, fv = ib.spread(load)
    assert abs(np.sum(load*ib.interpolate(u, v))-g.dx*g.dy*(np.sum(fu*u)+np.sum(fv*v))) < 1e-12


def test_bspline_interpolant_is_continuously_divergence_free():
    g = MACGrid(20, 16)
    rng = np.random.default_rng(5)
    f = ConservativeMACSolver(g, np.ones(g.shape), np.zeros(g.shape))
    u, v = [rng.normal(size=a.shape) for a in g.zeros()]
    u, v, _ = f.project(u, v, f.rho, .01)
    points = .25 + .5*rng.random((100, 2))
    h = 1e-6
    derivatives = []
    for axis in range(2):
        delta = np.eye(2)[axis]*h
        plus = MACCoupling(g, points+delta).interpolate(u, v)[:, axis]
        minus = MACCoupling(g, points-delta).interpolate(u, v)[:, axis]
        derivatives.append((plus-minus)/(2*h))
    assert np.max(np.abs(derivatives[0]+derivatives[1])) < 1e-7


@pytest.fixture
def solid():
    from mpi4py import MPI
    from dolfinx import mesh
    domain = mesh.create_rectangle(MPI.COMM_SELF, [[.45, .3], [.55, .6]], [2, 6], cell_type=mesh.CellType.triangle)
    s = ElasticSolid2D(domain, 30., 100., 1000.)
    yield s
    s.close()


def test_solid_rigid_motion_and_energy_gradient(solid):
    s = solid
    assert np.max(np.abs(s.load())) < 1e-11
    angle = .3
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    s.positions[:] = (s.reference-[.5,.45]) @ rotation.T + [.5,.45]
    assert np.max(np.abs(s.load())) < 1e-10
    s.positions[:] = s.reference @ np.array([[1., .07], [0., 1.]]).T
    load = s.load()
    rng = np.random.default_rng(9)
    direction = rng.normal(size=s.positions.shape)
    base = s.positions.copy()
    h = 1e-7
    s.positions[:] = base + h*direction
    plus = s.diagnostics()["elastic_support_energy"]
    s.positions[:] = base - h*direction
    minus = s.diagnostics()["elastic_support_energy"]
    s.positions[:] = base
    np.testing.assert_allclose((plus-minus)/(2*h), -np.sum(load*direction), rtol=1e-6, atol=1e-7)


def test_coupled_three_phase_step(solid):
    g = MACGrid(24, 20, 1.2, 1.)
    # Predeformation ensures this checks elastic feedback, not merely a passive mesh.
    solid.positions[:, 0] += .02*(solid.reference[:, 1]-.3)
    p = PhaseProperties(rho_solid=solid.density)
    x, y = g.coordinates()
    sim = ThreePhaseSimulation(g, solid, np.maximum(x-.5, y-.7), p, reinitialize_every=2)
    initial = solid.positions.copy()
    for _ in range(6):
        row = sim.step(min(1e-4, sim.stable_dt()))
    assert row["solid_J_min"] > 0
    assert abs(row["mass_relative_error"]) < 1e-12
    assert abs(row["liquid_volume_error"]) < 1e-12
    assert row["divergence_max"] < 1e-8
    assert row["rho_min"] > 0
    assert np.linalg.norm(solid.positions-initial) > 1e-10
    with pytest.raises(ValueError):
        sim.step(-1)
