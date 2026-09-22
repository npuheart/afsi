"""demo_426 — flow through a slanted channel (2-D immersed-boundary benchmark).

A steady plane-Poiseuille flow through a channel inclined at theta = pi/6,
in a rectangular box, with the two channel walls represented as immersed
boundary plates so that they are deliberately NOT grid-aligned.  This is the
two-dimensional slanted-channel benchmark used to compare how different IB
kernels reproduce the exact solution inside a confined stationary geometry.

Exact solution (see configuration.py for the derivation and for the note on the
self-consistency of the profile amplitude):

    xi(x,y) = y*cos(theta) - x*sin(theta)                 (across the channel)
    u       = (dP/L)/(2*mu) * ((D/2)^2 - xi^2) * (cos, sin)

driven by the constant body force f = -(dP/L)*(cos(theta), sin(theta)).

Boundary conditions
-------------------
* inlet  (x = X_MIN, channel cross-section):  velocity Dirichlet, analytic
* outlet (x = X_MAX, channel cross-section):  velocity Dirichlet, analytic
* every other boundary segment:              no-slip u = 0
* the two channel plates INSIDE the box:     immersed-boundary penalty
  f_ib = -BETA*(X - X_ref) - DAMP*u_interp   (Eq. (14) of the benchmark)

The flow is driven by the body force, not by the pressure boundary conditions,
so no pressure BC is imposed and the pressure field is determined only by
incompressibility.

Run (inside the afsi-dolfinx environment, from this directory):
    python main.py
    SMOKE=1 SMOKE_STEPS=200 python main.py
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
from ufl import Measure, TestFunction, SpatialCoordinate, as_vector, inner, dx

import configuration as cfg
import verify as vf
from afsic import IBMesh, IBInterpolation, ChorinSolver, IPCSSolver

t_start = time.time()
t_last = t_start


def phase(label):
    global t_last
    now = time.time()
    if MPI.COMM_WORLD.rank == 0:
        print(f"  [setup] {label:<46} {now - t_last:7.2f} s "
              f"(total {now - t_start:7.2f} s)")
    t_last = now


if MPI.COMM_WORLD.rank == 0:
    print(cfg.summary())
    print("-" * 78)

# ==========================================================================
# Fluid mesh: the box (padded to a whole number of cells)
# ==========================================================================
NX = cfg.NX
NY = int(np.ceil((cfg.Y_MAX - cfg.Y_MIN) / cfg.DX))
mesh = dolfinx.mesh.create_rectangle(
    comm=MPI.COMM_WORLD,
    points=((cfg.X_MIN, cfg.Y_MIN), (cfg.X_MIN + NX * cfg.DX,
                                     cfg.Y_MIN + NY * cfg.DX)),
    n=(NX, NY),
    cell_type=CellType.quadrilateral,
    ghost_mode=GhostMode.shared_facet,
)
BOX_TOP = cfg.Y_MIN + NY * cfg.DX

MARKER_INLET, MARKER_OUTLET, MARKER_BOTTOM, MARKER_TOP = 1, 2, 3, 4
fdim = mesh.topology.dim - 1
mesh.topology.create_connectivity(fdim, mesh.topology.dim)

boundaries = [
    (MARKER_INLET, lambda x: np.isclose(x[0], cfg.X_MIN)),
    (MARKER_OUTLET, lambda x: np.isclose(x[0], cfg.X_MIN + NX * cfg.DX)),
    (MARKER_BOTTOM, lambda x: np.isclose(x[1], cfg.Y_MIN)),
    (MARKER_TOP, lambda x: np.isclose(x[1], BOX_TOP)),
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

V = functionspace(mesh, element("Lagrange", mesh.topology.cell_name(),
                                cfg.VELOCITY_ORDER, shape=(mesh.geometry.dim,)))
Q = functionspace(mesh, element("Lagrange", mesh.topology.cell_name(),
                                cfg.PRESSURE_ORDER))

# Assign dof values directly: Function.interpolate with a lambda is brittle for
# vector-valued spaces in dolfinx 0.10 ("interpolation data has the wrong
# shape/size").  The layout for a blocked vector space is [x0,y0,x1,y1,...],
# matching V.tabulate_dof_coordinates().
def _set_scalar(fn, values):
    coords = fn.function_space.tabulate_dof_coordinates()
    fn.x.array[:] = values(coords[:, 0], coords[:, 1])
    fn.x.scatter_forward()


def _set_vector(fn, values):
    coords = fn.function_space.tabulate_dof_coordinates()
    ux, uy = values(coords[:, 0], coords[:, 1])
    fn.x.array[:] = np.column_stack([ux, uy]).reshape(-1)
    fn.x.scatter_forward()



# --- generic no-slip on ALL four faces, then override the channel openings --
u_zero = np.zeros(mesh.geometry.dim, dtype=PETSc.ScalarType)
bcu_all = [dirichletbc(u_zero, locate_dofs_topological(V, fdim,
                                                       facet_tag.find(m)), V)
           for m in (MARKER_INLET, MARKER_OUTLET, MARKER_BOTTOM, MARKER_TOP)]

# Interior dofs of the inlet / outlet interval where the channel crosses the
# face.  Nodes at the interval ends sit exactly on a plate, where the analytic
# solution is zero, so it makes no difference whether they are included.
dofs_inlet = locate_dofs_topological(V, fdim, facet_tag.find(MARKER_INLET))
dofs_outlet = locate_dofs_topological(V, fdim, facet_tag.find(MARKER_OUTLET))
coords = V.tabulate_dof_coordinates()
x_in = coords[dofs_inlet]
x_out = coords[dofs_outlet]

y_lo_in, y_hi_in = cfg.inlet_interval()
y_lo_out, y_hi_out = cfg.outlet_interval()
sel_in = (x_in[:, 1] > y_lo_in + 1e-9) & (x_in[:, 1] < y_hi_in - 1e-9)
sel_out = (x_out[:, 1] > y_lo_out + 1e-9) & (x_out[:, 1] < y_hi_out - 1e-9)
dofs_in_channel = dofs_inlet[sel_in]
dofs_out_channel = dofs_outlet[sel_out]

# Non-constant Dirichlet data.  dolfinx 0.10 requires a fem.Function for this
# (a raw ndarray is interpreted as a Constant and rejected unless its size
# equals the block size), and the working call form is
#     dirichletbc(function, dofs)
# with NO explicit function space -- the value function's own space is used.
# Passing V explicitly, or wrapping dofs in a tuple/list, fails overload
# resolution in this version.
# The benchmark prescribes the STEADY ANALYTIC solution as the inflow condition
# and starts the channel from rest, so the boundary data are the analytic
# profile scaled by a ramp factor r(t): 0 at t=0, 1 for t >= RAMP_T.
u_bc = Function(V)


def set_inflow(scale):
    """Set the inlet/outlet Dirichlet data to scale * analytic profile."""
    _set_vector(u_bc, lambda x, y: tuple(scale * v
                                         for v in cfg.analytic(x, y)))


set_inflow(0.0)
bc_inlet = dirichletbc(u_bc, dofs_in_channel, None)
bc_outlet = dirichletbc(u_bc, dofs_out_channel, None)


def make_bcu():
    return bcu_all + [bc_inlet, bc_outlet]


bcu = make_bcu()
if MPI.COMM_WORLD.rank == 0:
    print(f"channel opening dofs: inlet {len(dofs_in_channel)}, "
          f"outlet {len(dofs_out_channel)}")

# --- constant body force along the channel axis ----------------------------
f_body = Function(V)
_set_vector(f_body, lambda x, y: (-cfg.DP_DL * cfg.COS_T + 0.0 * x,
                                  -cfg.DP_DL * cfg.SIN_T + 0.0 * y))
phase("fluid mesh, boundaries, BC dofs")

# ==========================================================================
# Solver
# ==========================================================================
# The flow is driven by the body force, so the pressure has NO Dirichlet data
# anywhere: bcp is empty and the pressure null space is handled by the
# projection step (the pressure is fixed up to a constant, which is all that is
# needed when no pressure datum is prescribed).
# Implicit plate penalty: `drag` is assembled into the momentum LHS, so unlike
# the body-force route it carries no time-step restriction.
drag_coeff = None
if cfg.USE_IMPLICIT_DRAG:
    drag_coeff = Function(Q)
    _set_scalar(drag_coeff, cfg.plate_drag_coefficient)

if cfg.SOLVER == "chorin":
    ns_solver = ChorinSolver(V, Q, bcu, cfg.DT, cfg.RHO, cfg.MU,
                             drag=drag_coeff)
    force_sign = -1.0    # ChorinSolver carries -f in the momentum equation
else:
    ns_solver = IPCSSolver(V, Q, bcu, [], cfg.DT, cfg.RHO, cfg.MU,
                           drag=drag_coeff)
    force_sign = +1.0    # IPCSSolver carries +f in the momentum equation
phase(f"solver assembly ({cfg.SOLVER})")

# ==========================================================================
# Solid: the two plates as 2-D triangular strips (built by generate_mesh.py)
# ==========================================================================
# 2-D rather than 1-D on purpose: AFSI's immersed-boundary distributor takes an
# integration weight per Lagrangian point, and the weak-form tether assembly
# below carries the AREA measure dx_s.  With 1-D line segments the assembled
# force would be a force per unit LENGTH while a 2-D fluid needs a force per
# unit AREA -- the two differ by the plate thickness.  demo_424 works because
# its solid is 2-D.
with XDMFFile(MPI.COMM_WORLD, cfg.solid_mesh_path(), "r") as xdmf:
    structure = xdmf.read_mesh(name="mesh")

Vs = functionspace(structure, element("Lagrange",
                                      structure.topology.cell_name(),
                                      cfg.FORCE_ORDER, shape=(2,)))
solid_coords = Function(Vs, name="solid_coords")
coords_ref = Function(Vs, name="coords_ref")
solid_velocity = Function(Vs, name="solid_velocity")
solid_force = Function(Vs, name="solid_force")


def _set_vec(fn, values):
    c = fn.function_space.tabulate_dof_coordinates()
    ux, uy = values(c[:, 0], c[:, 1])
    fn.x.array[:] = np.column_stack([ux, uy]).reshape(-1)
    fn.x.scatter_forward()


for f in (solid_coords, coords_ref):
    _set_vec(f, lambda x, y: (x, y))
phase(f"solid mesh ({structure.topology.index_map(2).size_local} triangles)")

# --- tether (volumetric spring) is the ONLY solid force, as in demo_424 ----
dVs = TestFunction(Vs)
X0 = SpatialCoordinate(structure)
spring = solid_coords - as_vector([X0[0], X0[1]])
dx_s = Measure("dx", domain=structure)
L_hat = form(-cfg.BETA * inner(spring, dVs) * dx_s)
b1 = create_vector(Vs)

ibmesh = IBMesh(cfg.X_MIN, cfg.X_MIN + NX * cfg.DX, cfg.Y_MIN, BOX_TOP,
                NX, NY, cfg.VELOCITY_ORDER)
ib_interpolation = IBInterpolation(ibmesh)
coords_bg = Function(V)
_set_vector(coords_bg, lambda x, y: (x, y))
ibmesh.build_map(coords_bg._cpp_object)
ib_interpolation.evaluate_current_points(solid_coords._cpp_object)
phase("IB mesh + map on the 2-D plates")

# ==========================================================================
# Time loop
# ==========================================================================
log.set_log_level(log.LogLevel.WARNING)
n_steps = cfg.SMOKE_STEPS if cfg.SMOKE else cfg.NSTEPS

# a normal line across the channel at x=0.5, for the profile comparison
X_PROFILE = float(os.environ.get("X_PROFILE", "0.5"))
s_ax = X_PROFILE * cfg.COS_T
s_perp = X_PROFILE * cfg.SIN_T


def profile_line(npts=201):
    """Points across the channel at x = X_PROFILE (xi from -D/2 to +D/2)."""
    t = np.linspace(-cfg.R_HALF, cfg.R_HALF, npts)
    x = s_ax - t * cfg.SIN_T
    y = s_perp + t * cfg.COS_T
    return t, np.column_stack([x, y, np.zeros_like(x)])


_, pts_prof = profile_line()
t_prof = np.linspace(-cfg.R_HALF, cfg.R_HALF, len(pts_prof))

def ramp_factor(t):
    """Linear ramp of the DRIVING from rest, 0 -> 1 over RAMP_T.

    The driving is the prescribed inflow: the inlet/outlet Dirichlet data are
    the analytic steady profile scaled by this factor, so the channel starts
    from rest and settles to the exact solution.  RAMP_T = 0 disables the ramp.
    """
    if cfg.RAMP_T <= 0.0:
        return 1.0
    return min(max(t / cfg.RAMP_T, 0.0), 1.0)


history = []
for step in range(n_steps):
    t = step * cfg.DT
    rfac = ramp_factor(t)
    # ramp the prescribed inflow (Dirichlet data are time dependent)
    set_inflow(rfac)

    # --- solve the fluid step with the force assembled at the end of the
    #     previous step (the body force plus the IB penalty) --------------
    ns_solver.solve_one_step()

    # --- plates: time-centered (trapezoidal) penalty ----------------------
    # The benchmark evaluates the spring at the TIME-CENTERED position
    #
    #     F^{n+1/2} = kappa * ( chi^0 - (chi_tilde^{n+1} + chi^n)/2 )
    #
    # where chi_tilde^{n+1} = chi^n + dt*U^{n+1/2} is the predicted marker
    # position.  The force therefore depends on the position at the SAME time
    # level, which makes it self-limiting: if the markers overshoot, the
    # averaged position pulls the force back the other way.  The earlier
    # explicit form F = -kappa*(chi^n - chi^0) used a lagged position and had no
    # such restoring property, which is why the markers drifted.
    if not cfg.USE_IMPLICIT_DRAG:
        # 1) fluid velocity at the markers, from the tentatively updated field
        ib_interpolation.fluid_to_solid(ns_solver.u_s._cpp_object,
                                        solid_velocity._cpp_object)
        solid_coords_prev = solid_coords.x.array.copy()
        # 2) predict the marker position with the same velocity
        solid_coords.x.array[:] = solid_coords_prev + \
            solid_velocity.x.array[:] * cfg.DT
        solid_coords.x.scatter_forward()

        # 3) assemble the spring at the time-centered position and distribute.
        #    Re-evaluating the map at the predicted position makes this the
        #    implicit (mid-point) form in practice.
        if cfg.MAP_AT_REFERENCE:
            ib_interpolation.evaluate_current_points(coords_ref._cpp_object)
        else:
            ib_interpolation.evaluate_current_points(solid_coords._cpp_object)
        # overwrite the marker position used by the assemble with the average
        solid_coords.x.array[:] = 0.5 * (solid_coords_prev
                                        + solid_coords.x.array[:])
        solid_coords.x.scatter_forward()
        with b1.localForm() as loc:
            loc.set(0)
        assemble_vector(b1, L_hat)
        b1.ghostUpdate(addv=PETSc.InsertMode.ADD,
                       mode=PETSc.ScatterMode.REVERSE)
        with b1.getBuffer() as arr:
            solid_force.x.array[: len(arr)] = force_sign * arr[:]
        solid_force.x.scatter_forward()
        # restore the (predicted) marker position as the new configuration
        solid_coords.x.array[:] = 2.0 * solid_coords.x.array[:] \
            - solid_coords_prev
        solid_coords.x.scatter_forward()

        ns_solver.f.x.array[:] = 0.0
        ib_interpolation.solid_to_fluid(ns_solver.f._cpp_object,
                                        solid_force._cpp_object)
        if cfg.USE_BODY_FORCE:
            ns_solver.f.x.array[:] += force_sign * rfac * f_body.x.array[:]
        ns_solver.f.x.scatter_forward()

    if (step + 1) % max(n_steps // 10, 1) == 0 or step == 0:
        umax = float(np.max(np.linalg.norm(
            ns_solver.u_.x.array.reshape(-1, 2), axis=1)))
        if MPI.COMM_WORLD.rank == 0:
            print(f"  step {step + 1:>6}/{n_steps}  t={t + cfg.DT:.4g}  "
                  f"max|u|={umax:.6e}")
    _nm = len(solid_coords.x.array) // 2
    _pmax = float(np.max(np.linalg.norm(
        (solid_coords.x.array - coords_ref.x.array).reshape(_nm, 2), axis=1)))
    history.append((step + 1, t + cfg.DT,
                    float(np.max(np.linalg.norm(
                        ns_solver.u_.x.array.reshape(-1, 2), axis=1))),
                    _pmax))

# ==========================================================================
# Report
# ==========================================================================
u_line = vf._sample(ns_solver.u_, pts_prof, mesh, bs=2)
u_exact = np.column_stack(cfg.analytic(pts_prof[:, 0], pts_prof[:, 1]))
res = vf.report(cfg, mesh, ns_solver.u_, ns_solver.p_, t_prof, u_line,
                u_exact, solid_coords, coords_ref)
res["elapsed_s"] = time.time() - t_start

if MPI.COMM_WORLD.rank == 0:
    import json
    out = cfg.output_path()
    with open(os.path.join(out, "history.csv"), "w") as fh:
        fh.write("step,t,max_u,plate_disp\n")
        for row in history:
            fh.write(f"{row[0]},{row[1]:.8e},{row[2]:.16e},{row[3]:.16e}\n")
    print(f"history written to {out}history.csv")

    # machine-readable metrics (same convention as demo_424)
    res["N"] = cfg.N
    res["DX"] = cfg.DX
    res["DT"] = cfg.DT
    res["theta_deg"] = cfg.THETA_DEG
    res["DP_DL"] = cfg.DP_DL
    res["MU"] = cfg.MU
    res["RHO"] = cfg.RHO
    res["U_MAX_analytic"] = cfg.U_MAX
    res["U_MAX_paper"] = cfg.U_MAX_PAPER
    res["plate_drag"] = cfg.PLATE_DRAG
    res["plate_drag_band"] = cfg.PLATE_DRAG_BAND
    res["use_implicit_drag"] = cfg.USE_IMPLICIT_DRAG
    with open(os.path.join(out, "verify.json"), "w") as fh:
        json.dump({k: (float(v) if isinstance(v, (int, float, np.floating))
                       else v) for k, v in res.items()}, fh, indent=2)
    print(f"verify.json written to {out}verify.json")

u_io = Function(functionspace(mesh, element("Lagrange",
                                            mesh.topology.cell_name(), 1,
                                            shape=(mesh.geometry.dim,))))
u_io.interpolate(ns_solver.u_)
with XDMFFile(MPI.COMM_WORLD, cfg.output_path() + "velocity.xdmf", "w") as f:
    f.write_mesh(mesh)
    f.write_function(u_io, cfg.DT * n_steps)
with XDMFFile(MPI.COMM_WORLD, cfg.output_path() + "pressure.xdmf", "w") as f:
    f.write_mesh(mesh)
    f.write_function(ns_solver.p_, cfg.DT * n_steps)

# --- solid output: displaced coordinates and the displacement field ---------
# Written on the LAGRANGIAN mesh, so the deformed plates can be compared with
# their reference position (the tether's X_ref).
displacement = Function(Vs, name="displacement")
displacement.x.array[:] = solid_coords.x.array[:] - coords_ref.x.array[:]
displacement.x.scatter_forward()
# XDMF requires the output Function degree to match the mesh geometry degree
# (the solid mesh is P1), so interpolate the P2 fields down to P1 for output.
Vs_io = functionspace(structure, element("Lagrange",
                                         structure.topology.cell_name(), 1,
                                         shape=(2,)))
solid_coords_io = Function(Vs_io, name="solid_coords_io")
disp_io = Function(Vs_io, name="displacement_io")
solid_coords_io.interpolate(solid_coords)
disp_io.interpolate(displacement)
with XDMFFile(MPI.COMM_WORLD, cfg.output_path() + "solid_coords.xdmf",
              "w") as f:
    f.write_mesh(structure)
    f.write_function(solid_coords_io, cfg.DT * n_steps)
with XDMFFile(MPI.COMM_WORLD, cfg.output_path() + "solid_displacement.xdmf",
              "w") as f:
    f.write_mesh(structure)
    f.write_function(disp_io, cfg.DT * n_steps)
if MPI.COMM_WORLD.rank == 0:
    _n = len(solid_coords.x.array) // 2
    _d = np.linalg.norm(
        (solid_coords.x.array - coords_ref.x.array).reshape(_n, 2), axis=1)
    print(f"solid displacement: max {_d.max():.6e}  mean {_d.mean():.6e}")
    print(f"solid output written to {cfg.output_path()}solid_*.xdmf")

if MPI.COMM_WORLD.rank == 0:
    print(f"elapsed {res['elapsed_s']:.1f} s")
