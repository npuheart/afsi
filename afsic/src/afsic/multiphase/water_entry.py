"""Rest-release of a free elastic rectangle above a quiescent water layer.

Illustrative 2D, closed-tank case; no rigid-body constraint, contact or capillarity.
"""
import numpy as np
from scipy.ndimage import map_coordinates

from .grid import MACGrid
from .levelset import PhaseProperties
from .solid import ElasticSolid2D
from .simulation import ThreePhaseSimulation


def create_water_entry(nx=64, ny=64, shear=2000., bulk=20000., rho_gas=1.2,
                       rho_solid=1200., water_level=.35, release_gap=.11,
                       kernel="bspline"):
    """Return a simulation and its geometry metadata, in SI units.

    All velocities start at zero. Gravity acts through the Eulerian three-phase
    density; do not add a separate solid gravitational load. The rectangle has
    width .12 m and height .10 m in an .8 x .8 m tank.
    """
    from mpi4py import MPI
    from dolfinx import mesh

    grid = MACGrid(nx, ny, .8, .8)
    h = min(grid.dx, grid.dy)
    if not np.isfinite([water_level, release_gap]).all() or water_level <= 0 or release_gap <= 0:
        raise ValueError("water_level and release_gap must be finite and positive")
    bottom = water_level + release_gap
    if bottom + .10 >= grid.ly - 2*max(grid.dx, grid.dy):
        raise ValueError("Released solid must remain at least two grid cells below the lid")
    if release_gap <= 3*max(grid.dx, grid.dy):
        raise ValueError("Release gap must exceed the combined smoothed-interface half-widths (3h)")
    counts = [max(2, int(np.ceil(length/(.5*h)))) for length in (.12, .10)]
    domain = mesh.create_rectangle(MPI.COMM_SELF, [[.34, bottom], [.46, bottom+.10]],
                                   counts, cell_type=mesh.CellType.triangle)
    solid = ElasticSolid2D(domain, shear, bulk, rho_solid)
    properties = PhaseProperties(rho_gas=rho_gas, rho_solid=rho_solid)
    _, y = grid.coordinates()
    sim = ThreePhaseSimulation(grid, solid, y-water_level, properties, kernel=kernel)
    sim.water_level = water_level
    info = dict(lx=grid.lx, ly=grid.ly, water_level=water_level, release_gap=release_gap,
                solid_bounds=[[.34, bottom], [.46, bottom+.10]], solid_cells=counts,
                anchor_stiffness=0., anchor_damping=0., initial_velocity=[0., 0.],
                phases=vars(properties))
    return sim, info


def entry_diagnostics(sim):
    """Reference-mass-weighted COM and sampled geometric entry indicators.

    Boundary phi is an interface proximity indicator; it is not a contact-force
    criterion. The initial-waterline crossing is a distinct, reproducible event.
    """
    solid = sim.solid
    ref = solid.reference[solid.cells]
    a, b = ref[:, 1]-ref[:, 0], ref[:, 2]-ref[:, 0]
    weights = np.abs(a[:, 0]*b[:, 1]-a[:, 1]*b[:, 0])
    weights /= weights.sum()
    centroid = np.sum(solid.positions[solid.cells].mean(axis=1)*weights[:, None], axis=0)
    velocity = solid.velocity.x.array.reshape(-1, 2)
    mean_velocity = np.sum(velocity[solid.cells].mean(axis=1)*weights[:, None], axis=0)
    segments = solid.boundary_segments()
    points = np.concatenate([segments[:, 0], segments.mean(axis=1)])
    coordinates = np.array([points[:, 1]/sim.grid.dy-.5, points[:, 0]/sim.grid.dx-.5])
    phi = map_coordinates(sim.phi, coordinates, order=1, mode="nearest")
    return dict(solid_com_x=float(centroid[0]), solid_com_y=float(centroid[1]),
                solid_com_vx=float(mean_velocity[0]), solid_com_vy=float(mean_velocity[1]),
                solid_bottom_y=float(solid.positions[:, 1].min()),
                boundary_phi_min=float(phi.min()),
                nominal_waterline_gap=float(solid.positions[:, 1].min()-sim.water_level),
                solid_area_relative_error=solid.diagnostics()["solid_volume"]/solid.volume0-1.)
