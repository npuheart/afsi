"""Water-entry setup and initial release checks (serial)."""
import numpy as np
import pytest

from afsic.multiphase.water_entry import create_water_entry, entry_diagnostics


def test_free_release_geometry_and_gravity():
    sim, info = create_water_entry(nx=32, ny=32)
    try:
        row = entry_diagnostics(sim)
        np.testing.assert_allclose([row["solid_com_x"], row["solid_com_y"], row["solid_bottom_y"]], [.4, .51, .46])
        assert row["nominal_waterline_gap"] == pytest.approx(.11)
        assert row["boundary_phi_min"] == pytest.approx(.11)
        assert sim.solid.anchor_stiffness == sim.solid.anchor_damping == 0.
        assert np.max(np.abs(sim.solid.load())) < 1e-9
        assert np.max(np.abs(sim.fluid.u)) == np.max(np.abs(sim.fluid.v)) == 0.
        assert info["initial_velocity"] == [0., 0.]
        y0 = row["solid_com_y"]
        for _ in range(10):
            sim.step(min(1e-4, sim.stable_dt()))
        row = entry_diagnostics(sim)
        assert row["solid_com_y"] < y0
        assert row["solid_com_vy"] < 0
        # The uniformly oriented TRI3 diagonals are not reflection-symmetric;
        # require lateral drift to be negligible relative to the fall.
        assert abs(row["solid_com_x"]-.4) < 1e-3*(y0-row["solid_com_y"])
        assert abs(row["solid_com_vx"]) < 1e-3*abs(row["solid_com_vy"])
        assert abs(sim.diagnostics()["mass_relative_error"]) < 1e-12
    finally:
        sim.solid.close()


def test_invalid_release_is_rejected():
    for parameters in [dict(release_gap=0.), dict(release_gap=.001),
                       dict(water_level=-.1), dict(water_level=.7), dict(release_gap=np.nan)]:
        with pytest.raises(ValueError):
            create_water_entry(**parameters)
