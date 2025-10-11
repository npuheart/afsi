# # Problem 3: inflation and active contraction of a ventricle

# In the third problem we will solve the inflation and active contraction of a ventricle. First we import the necessary libraries
#

from pathlib import Path
from mpi4py import MPI
from dolfinx import log
import dolfinx
import numpy as np
import math
import cardiac_geometries
import pulse
from dolfinx.geometry import bb_tree, compute_colliding_cells, compute_collisions_points


# Next we will create the geometry and save it in the folder called `lv_ellipsoid`. Now we will also generate fibers and use a sixth order quadrature space for the fibers

comm = MPI.COMM_WORLD
geodir = Path("lv_ellipsoid-problem3")
if not geodir.exists():
    comm.barrier()
    cardiac_geometries.mesh.lv_ellipsoid(
        outdir=geodir,
        r_short_endo=7.0,
        r_short_epi=10.0,
        r_long_endo=17.0,
        r_long_epi=20.0,
        mu_apex_endo=-math.pi,
        mu_base_endo=-math.acos(5 / 17),
        mu_apex_epi=-math.pi,
        mu_base_epi=-math.acos(5 / 20),
        fiber_space="Quadrature_6",
        create_fibers=True,
        fiber_angle_epi=-90,
        fiber_angle_endo=90,
        comm=comm,
    )
    print("Done creating geometry.")

# If the folder already exist, then we just load the geometry

geo = cardiac_geometries.geometry.Geometry.from_folder(
    comm=comm,
    folder=geodir,
)

# Now, lets convert the geometry to a `pulse.Geometry` object.

geometry = pulse.HeartGeometry.from_cardiac_geometries(geo, metadata={"quadrature_degree": 6})


# The material model used in this benchmark is the {py:class}`Guccione <pulse.material_models.guccione.Guccione>` model.

material_params = {
    "C": dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(2.0)),
    "bf": dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(8.0)),
    "bt": dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(2.0)),
    "bfs": dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(4.0)),
}
material = pulse.Guccione(f0=geo.f0, s0=geo.s0, n0=geo.n0, **material_params)


# We use an active stress approach with 60% transverse active stress

Ta = dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(0.0))
active_model = pulse.ActiveStress(geo.f0, activation=pulse.Variable(Ta, "kPa"))

# and the model should be incompressible

comp_model = pulse.Incompressible()


# and assembles the `CardiacModel`

model = pulse.CardiacModel(
    material=material,
    active=active_model,
    compressibility=comp_model,
)


# We apply a traction in endocardium

traction = dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(0.0))
neumann = pulse.NeumannBC(
    traction=pulse.Variable(traction, "kPa"),
    marker=geo.markers["ENDO"][0],
)

# and finally combine all the boundary conditions

bcs = pulse.BoundaryConditions(neumann=(neumann,))

# and create a Mixed problem

problem = pulse.StaticProblem(
    model=model, geometry=geometry, bcs=bcs, parameters={"base_bc": pulse.BaseBC.fixed},
)

# Now we can solve the problem
log.set_log_level(log.LogLevel.INFO)

problem.solve()

# Now we will solve the problem for a range of active contraction and traction values

target_pressure = 15.0
target_Ta = 60.0

# Let us just gradually increase the active contraction and traction with a linear ramp of 40 steps


class DomainCollisionChecker:
    def __init__(self, domain):
        self.tree = bb_tree(domain, domain.geometry.dim)
        self.domain = domain

    def eval_point(self, disp, x, y, z):
        x0 = np.array([x, y, z], dtype=dolfinx.default_scalar_type)
        cell_candidates = compute_collisions_points(self.tree, x0)
        cell = compute_colliding_cells(self.domain, cell_candidates, x0).array
        if len(cell) == 0:
            return [0.0,0.0,0.0]
        else:
            first_cell = cell[0]
            return disp.eval(x0, first_cell)[:3]

us = []
N = 40
dcc = DomainCollisionChecker(geometry.mesh)

for Ta_value, traction_value in zip(np.linspace(0, target_Ta, N), np.linspace(0, target_pressure, N)):
    print(f"Solving problem for traction={traction_value} and active contraction={Ta_value}")
    Ta.value = Ta_value
    traction.value = traction_value
    problem.solve()



for point in np.loadtxt('data/ideal_middle_wall.txt'):
    u = dcc.eval_point(problem.u, point[0], point[1], 0)
    us.append([u[0], u[1]])


us = np.array(us)
global_sum = np.zeros_like(us)
comm.Reduce(us, global_sum, op=MPI.SUM, root=0)


if comm.rank == 0:
    np.savetxt("data/systole-pulse-disp.txt", np.column_stack((global_sum[:, 0], global_sum[:, 1])))





