"""demo_425 — tethered aorta in a box, 3-D round tube.

demo_424 in three dimensions.  The geometry is a genuine round pipe instead of
the 2-D planar cut:

    fluid   a square duct, side BOX_SIDE, pipe axis along x
    solid   a cylindrical shell (the aortic wall) of radius a .. a+t, held to
            its reference position only by a tether beta*(X_ref - X); for
            CASE=closed a coaxial disc of radius a occludes the lumen at
            mid-length, so a cut through the axis reads as demo_424's "H"

The duct is driven by a pressure Dirichlet on both open ends, exactly as in
demo_424: p = p_in(t) at x = 0 and p = 0 at x = BOX_L, with no velocity
condition (the natural condition there is zero traction).  Prescribing p is an
essential condition on the pressure Poisson problem, i.e. the `bcp` argument.

Run (inside the afsi-dolfinx environment, from this directory):
    python generate_mesh.py && python main.py

Smoke test (a few steps only; this is what the readme documents):
    SMOKE=1 python main.py
"""
import os
import time

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

import dolfinx
from dolfinx import log
from dolfinx.fem import (Function, functionspace, dirichletbc,
                         locate_dofs_topological, form, assemble_scalar,
                         Constant)
from dolfinx.fem.petsc import create_vector, assemble_vector
from dolfinx.io import XDMFFile
from dolfinx.mesh import (CellType, GhostMode, locate_entities, meshtags)
from basix.ufl import element
from ufl import (Measure, TestFunction, SpatialCoordinate, as_vector,
                 inner, dx)

import configuration as cfg
from afsic import IBMesh3D, IBInterpolation3D, IPCSSolver, ChorinSolver

t_start = time.time()
t_last = t_start


def phase(label):
    """Report a setup phase; the 3-D IB map build dominates the run time."""
    global t_last
    now = time.time()
    if MPI.COMM_WORLD.rank == 0:
        print(f"  [setup] {label:<44} {now - t_last:8.1f} s "
              f"(total {now - t_start:7.1f} s)")
    t_last = now


if MPI.COMM_WORLD.rank == 0:
    print(cfg.summary())
    print("-" * 78)

# ==========================================================================
# Fluid domain: the square duct
# ==========================================================================
mesh = dolfinx.mesh.create_box(
    comm=MPI.COMM_WORLD,
    points=((0.0, 0.0, 0.0), (cfg.BOX_L, cfg.BOX_SIDE, cfg.BOX_SIDE)),
    n=(cfg.NX, cfg.NY, cfg.NZ),
    cell_type=CellType.hexahedron,
    ghost_mode=GhostMode.shared_facet,
)

MARKER_INLET, MARKER_OUTLET = 1, 2
MARKER_Y0, MARKER_Y1, MARKER_Z0, MARKER_Z1 = 3, 4, 5, 6
fdim = mesh.topology.dim - 1
mesh.topology.create_connectivity(fdim, mesh.topology.dim)

boundaries = [
    (MARKER_INLET, lambda x: np.isclose(x[0], 0.0)),
    (MARKER_OUTLET, lambda x: np.isclose(x[0], cfg.BOX_L)),
    (MARKER_Y0, lambda x: np.isclose(x[1], 0.0)),
    (MARKER_Y1, lambda x: np.isclose(x[1], cfg.BOX_SIDE)),
    (MARKER_Z0, lambda x: np.isclose(x[2], 0.0)),
    (MARKER_Z1, lambda x: np.isclose(x[2], cfg.BOX_SIDE)),
]
facet_indices, facet_markers = [], []
for marker, locator in boundaries:
    facets = locate_entities(mesh, fdim, locator)
    facet_indices.append(facets)
    facet_markers.append(np.full_like(facets, marker))
facet_indices = np.hstack(facet_indices).astype(np.int32)
facet_markers = np.hstack(facet_markers).astype(np.int32)
order = np.argsort(facet_indices)
facet_tag = meshtags(mesh, fdim, facet_indices[order], facet_markers[order])

v_cg2 = element("Lagrange", mesh.topology.cell_name(), cfg.VELOCITY_ORDER,
                shape=(mesh.geometry.dim,))
s_cg1 = element("Lagrange", mesh.topology.cell_name(), cfg.PRESSURE_ORDER)
V = functionspace(mesh, v_cg2)
Q = functionspace(mesh, s_cg1)

# --- optional implicit drag in the outer fluid layer ------------------------
# Same device as demo_424: a smooth porous-medium damping that can hold the
# outer gap nearly static without a sharp split of the end-face BCs.  Default
# 0 for the open case (all three passages are meant to carry flow).
GAP_DRAG = float(os.environ.get(
    "GAP_DRAG", "1.0e6" if cfg.CASE == "closed" else "0.0"))
EDGE_DRAG = float(os.environ.get("EDGE_DRAG", "0.0"))
DRAG_DELTA = 2.0 * cfg.H


def _smoothstep(s):
    s = np.clip(s, 0.0, 1.0)
    return s * s * (3.0 - 2.0 * s)


def drag_weight(y, z):
    """1 outside the wall (radius > a+t), 0 inside, smooth over DRAG_DELTA."""
    r = np.hypot(y - cfg.Y_C, z - cfg.Z_C)
    return _smoothstep((r - (cfg.A_LUMEN + cfg.T_WALL)) / DRAG_DELTA)


# --- velocity BCs: no-slip on the four duct side walls ----------------------
u_zero = np.zeros(mesh.geometry.dim, dtype=PETSc.ScalarType)
bcu = [dirichletbc(u_zero, locate_dofs_topological(V, fdim,
                                                   facet_tag.find(m)), V)
       for m in (MARKER_Y0, MARKER_Y1, MARKER_Z0, MARKER_Z1)]

# --- pressure BCs: uniform pressure on the two open ends --------------------
dofs_inlet = locate_dofs_topological(Q, fdim, facet_tag.find(MARKER_INLET))
dofs_outlet = locate_dofs_topological(Q, fdim, facet_tag.find(MARKER_OUTLET))
bcp_outlet = dirichletbc(PETSc.ScalarType(0.0), dofs_outlet, Q)


def make_bcp(value):
    return [dirichletbc(PETSc.ScalarType(value), dofs_inlet, Q), bcp_outlet]


bcp = make_bcp(0.0)
ds_inlet = Measure("ds", domain=mesh, subdomain_data=facet_tag)(MARKER_INLET)

if GAP_DRAG > 0.0 or EDGE_DRAG > 0.0:
    drag_coeff = Function(Q)
    drag_coeff.interpolate(
        lambda x: GAP_DRAG * drag_weight(x[1], x[2]))
    drag_coeff.x.scatter_forward()
else:
    drag_coeff = None

if MPI.COMM_WORLD.rank == 0 and drag_coeff is not None:
    print(f"gap drag coefficient: {GAP_DRAG:g} (outside the tube)")

if cfg.SOLVER == "chorin":
    ns_solver = ChorinSolver(V, Q, bcu, bcp, cfg.DT, cfg.RHO, cfg.MU,
                             drag=drag_coeff)
    force_scale = 1.0          # ChorinSolver uses -f
else:
    p_traction = Constant(mesh, PETSc.ScalarType(0.0))
    ns_solver = IPCSSolver(V, Q, bcu, bcp, cfg.DT, cfg.RHO, cfg.MU,
                           ds_p=ds_inlet, p_traction=p_traction,
                           drag=drag_coeff)
    force_scale = -1.0         # IPCSSolver uses +f
phase("fluid mesh, BCs, solver assembly")

# ==========================================================================
# Solid: tethered aortic wall (+ occluding disc when CASE=closed)
# ==========================================================================
with XDMFFile(MPI.COMM_WORLD, cfg.solid_mesh_path(), "r") as xdmf:
    structure = xdmf.read_mesh(name="mesh")

v_cg2_s = element("Lagrange", structure.topology.cell_name(),
                  cfg.FORCE_ORDER, shape=(structure.geometry.dim,))
Vs = functionspace(structure, v_cg2_s)

solid_coords = Function(Vs, name="solid_coords")
solid_force = Function(Vs, name="solid_force")
solid_velocity = Function(Vs, name="solid_velocity")
coords_ref = Function(Vs, name="coords_ref")
displacement = Function(Vs, name="displacement")

for f in (solid_coords, coords_ref):
    f.interpolate(lambda x: np.array([x[0], x[1], x[2]]))
phase(f"solid mesh load ({structure.topology.index_map(3).size_local} tets), "
      f"function spaces")

# --- tether (volumetric spring) is the ONLY solid force --------------------
dVs = TestFunction(Vs)
X0 = SpatialCoordinate(structure)
spring = solid_coords - as_vector([X0[0], X0[1], X0[2]])
dx_s = Measure("dx", domain=structure)
L_hat = form(-cfg.BETA * inner(spring, dVs) * dx_s)
b1 = create_vector(Vs)

# ==========================================================================
# Immersed-boundary coupling
# ==========================================================================
ibmesh = IBMesh3D(0.0, cfg.BOX_L, 0.0, cfg.BOX_SIDE, 0.0, cfg.BOX_SIDE,
                  cfg.NX, cfg.NY, cfg.NZ, cfg.VELOCITY_ORDER)
ib_interpolation = IBInterpolation3D(ibmesh)
coords_bg = Function(V)
coords_bg.interpolate(lambda x: np.array([x[0], x[1], x[2]]))
phase("IB mesh + background coordinate field")
ibmesh.build_map(coords_bg._cpp_object)
phase("IB build_map")
ib_interpolation.evaluate_current_points(solid_coords._cpp_object)
phase("IB evaluate_current_points")

if MPI.COMM_WORLD.rank == 0:
    print(f"IB grid: {cfg.NX}x{cfg.NY}x{cfg.NZ} cells, order "
          f"{cfg.VELOCITY_ORDER} -> {(cfg.VELOCITY_ORDER * cfg.NX + 1)} x "
          f"{(cfg.VELOCITY_ORDER * cfg.NY + 1)} x "
          f"{(cfg.VELOCITY_ORDER * cfg.NZ + 1)} nodes")
    print(f"solid: {structure.topology.index_map(3).size_local} tets, "
          f"{structure.topology.index_map(0).size_local} nodes")

# ==========================================================================
# Time loop
# ==========================================================================
log.set_log_level(log.LogLevel.WARNING)
p_prev = 0.0
n_steps = cfg.SMOKE_STEPS if cfg.SMOKE else cfg.NSTEPS

# diagnostic probes: a line of fluid points across the duct at mid-length, and
# the disc's centre if there is one
x_mid = cfg.X_OFF + 0.5 * cfg.L_AORTA
y_probe = np.linspace(cfg.H, cfg.BOX_SIDE - cfg.H, cfg.NY - 1)

history = []
for step in range(n_steps):
    t = step * cfg.DT

    p_target = cfg.p_inlet(t)
    if cfg.SOLVER == "chorin":
        ns_solver.bcp = make_bcp(p_target)
    else:
        # IPCS accumulates the pressure increment into p_ and uses the full
        # pressure as the inlet traction.
        ns_solver.bcp = make_bcp(p_target - p_prev)
        ns_solver.p_traction.value = p_target
    p_prev = p_target

    ns_solver.solve_one_step()

    # advect the Lagrangian solid with the interpolated fluid velocity
    ib_interpolation.fluid_to_solid(ns_solver.u_._cpp_object,
                                    solid_velocity._cpp_object)
    solid_coords.x.array[:] += solid_velocity.x.array[:] * cfg.DT
    solid_coords.x.scatter_forward()

    # tether force from the current configuration
    ib_interpolation.evaluate_current_points(solid_coords._cpp_object)
    with b1.localForm() as loc:
        loc.set(0)
    assemble_vector(b1, L_hat)
    b1.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    with b1.getBuffer() as arr:
        solid_force.x.array[: len(arr)] = force_scale * arr[:]

    ib_interpolation.solid_to_fluid(ns_solver.f._cpp_object,
                                    solid_force._cpp_object)
    ns_solver.f.x.scatter_forward()

    u_mag = np.linalg.norm(ns_solver.u_.x.array.reshape(-1, 3), axis=1)
    umax = float(np.max(u_mag))
    pv = ns_solver.p_.x.array
    history.append((step + 1, t + cfg.DT, umax, float(pv.min()),
                    float(pv.max())))
    if MPI.COMM_WORLD.rank == 0:
        print(f"  step {step + 1:>5}/{n_steps}  t={t + cfg.DT:.4g}  "
              f"p_in={p_target:9.4f}  max|u|={umax:.4e}  "
              f"p range [{pv.min(): .4e}, {pv.max(): .4e}]")

# ==========================================================================
# Smoke-test report
# ==========================================================================
displacement.x.array[:] = solid_coords.x.array[:] - coords_ref.x.array[:]
disp = displacement.x.array.reshape(-1, 3)
disp_mag = np.linalg.norm(disp, axis=1)

disc_speed = None
if cfg.CASE == "closed":
    # solid nodes on the occluding disc (mid-length, inside the lumen)
    xs = solid_coords.x.array.reshape(-1, 3)[:, 0]
    on_disc = np.abs(xs - (cfg.X_OFF + cfg.DISC_X)) < 0.5 * cfg.DISC_T
    if on_disc.any():
        disc_speed = float(np.max(np.linalg.norm(
            solid_velocity.x.array.reshape(-1, 3)[on_disc], axis=1)))

if MPI.COMM_WORLD.rank == 0:
    print("=" * 78)
    print(f"smoke test report  (case={cfg.CASE}, NY={cfg.NY}, "
          f"NX={cfg.NX}, dt={cfg.DT:g}, steps={n_steps}, "
          f"solver={cfg.SOLVER})")
    print("=" * 78)
    print(f"  fluid cells                        {mesh.topology.index_map(3).size_local}")
    print(f"  velocity dofs                      {ns_solver.u_.x.array.size // 3}")
    print(f"  IB nodes                           "
          f"{(cfg.VELOCITY_ORDER * cfg.NX + 1) * (cfg.VELOCITY_ORDER * cfg.NY + 1) * (cfg.VELOCITY_ORDER * cfg.NZ + 1)}")
    print(f"  finite max|u|                      "
          f"{np.isfinite(u_mag).all()}")
    print(f"  finite p                           {np.isfinite(pv).all()}")
    print(f"  max|u| at last step                {umax:.6e} m/s")
    print(f"  p at last step min/max             "
          f"{pv.min(): .6e} / {pv.max(): .6e} Pa")
    print(f"  imposed p_in at last step          {p_target:.6e} Pa")
    print(f"  inlet dofs hold p_in?              "
          f"max|p_in - p_inlet| = "
          f"{float(np.max(np.abs(pv[dofs_inlet] - p_target))):.3e} Pa")
    print(f"  outlet dofs hold 0?                "
          f"max|p_out| = {float(np.max(np.abs(pv[dofs_outlet]))):.3e} Pa")
    print(f"  solid max |displacement|           {float(disp_mag.max()):.6e} m")
    print(f"  solid max |velocity|               "
          f"{float(np.max(np.linalg.norm(solid_velocity.x.array.reshape(-1, 3), axis=1))):.6e} m/s")
    if disc_speed is not None:
        print(f"  disc max |velocity|                {disc_speed:.6e} m/s")
        print(f"  DP/(beta*t_wall) scale             "
              f"{cfg.DISC_DELTA_SCALE:.6e} m")
    print("=" * 78)
    print(f"elapsed {time.time() - t_start:.1f} s")

    out = cfg.output_path()
    with open(os.path.join(out, "smoke_history.csv"), "w") as fh:
        fh.write("step,t,max_u,p_min,p_max\n")
        for row in history:
            fh.write(f"{row[0]},{row[1]:.8e},{row[2]:.16e},"
                     f"{row[3]:.16e},{row[4]:.16e}\n")
    print(f"history written to {out}smoke_history.csv")
