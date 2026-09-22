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
from ufl import Measure, TestFunction, SpatialCoordinate, as_vector, inner, dx, dot

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
u_inlet = Function(V)
_set_vector(u_inlet, lambda x, y: cfg.analytic(x, y))

bc_inlet = dirichletbc(u_inlet, dofs_in_channel, None)
bc_outlet = dirichletbc(u_inlet, dofs_out_channel, None)


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
# Lagrangian structure: the two channel plates (1-D, disconnected)
# ==========================================================================
pts, cells = [], []
for side in (-1, +1):
    (x0, y0), (x1, y1) = cfg.wall_endpoints(side)
    length = float(np.hypot(x1 - x0, y1 - y0))
    n = max(int(np.ceil(length / cfg.DS_LAG)), 1)
    base = len(pts)
    for i in range(n + 1):
        s = i / n
        pts.append((x0 + s * (x1 - x0), y0 + s * (y1 - y0)))
    for i in range(n):
        cells.append((base + i, base + i + 1))
pts = np.asarray(pts, dtype=np.float64)
cells = np.asarray(cells, dtype=np.int64)

structure = dolfinx.mesh.create_mesh(
    MPI.COMM_WORLD, cells,
    element("Lagrange", "interval", 1, shape=(2,)), pts)

Vs = functionspace(structure, element("Lagrange", "interval",
                                      cfg.FORCE_ORDER, shape=(2,)))
solid_coords = Function(Vs, name="solid_coords")
coords_ref = Function(Vs, name="coords_ref")
solid_velocity = Function(Vs, name="solid_velocity")
solid_force = Function(Vs, name="solid_force")
for f in (solid_coords, coords_ref):
    c = f.function_space.tabulate_dof_coordinates()
    f.x.array[:] = np.column_stack([c[:, 0], c[:, 1]]).reshape(-1)
    f.x.scatter_forward()

ibmesh = IBMesh(cfg.X_MIN, cfg.X_MIN + NX * cfg.DX, cfg.Y_MIN, BOX_TOP,
                NX, NY, cfg.VELOCITY_ORDER)
ib_interpolation = IBInterpolation(ibmesh)
coords_bg = Function(V)
_set_vector(coords_bg, lambda x, y: (x, y))
ibmesh.build_map(coords_bg._cpp_object)
ib_interpolation.evaluate_current_points(solid_coords._cpp_object)
phase(f"IB mesh + Lagrangian plates ({len(pts)} nodes, {len(cells)} cells)")

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

history = []
for step in range(n_steps):
    t = step * cfg.DT

    # --- solve the fluid step with the force assembled at the end of the
    #     previous step (the body force plus the IB penalty) --------------
    ns_solver.solve_one_step()

    # --- plate penalty ---------------------------------------------------
    # IMPLICIT route (default): the drag term assembled into the momentum LHS
    # already imposes no-slip in a thin band around each plate, so nothing more
    # is done here.
    #
    # EXPLICIT route (USE_IMPLICIT_DRAG=0): interpolate the fluid velocity to
    # the plates and spread the penalty force f_ib = -DAMP*u_ib back to the
    # grid.  This is the classic Peskin Lagrangian coupling, but because the
    # force enters ns_solver.f it is frozen during the momentum solve, which
    # caps the usable DAMP at rho*dx*dy/dt (= 0.208 at dx=1/32, dt=0.15dx).
    # Kept because the benchmark is fundamentally about this kernel coupling;
    # see the readme for what that limit means for reproducing Fig. 23.
    if not cfg.USE_IMPLICIT_DRAG:
        ib_interpolation.fluid_to_solid(ns_solver.u_._cpp_object,
                                       solid_velocity._cpp_object)
        # plates are rigid and stationary: X == X_ref identically, u_struct == 0
        solid_force.x.array[:] = -cfg.DAMP * solid_velocity.x.array[:]
        solid_force.x.scatter_forward()
        ns_solver.f.x.array[:] = 0.0
        ib_interpolation.solid_to_fluid(ns_solver.f._cpp_object,
                                        solid_force._cpp_object)
        ns_solver.f.x.array[:] += force_sign * f_body.x.array[:]
        ns_solver.f.x.scatter_forward()

    if (step + 1) % max(n_steps // 10, 1) == 0 or step == 0:
        umax = float(np.max(np.linalg.norm(
            ns_solver.u_.x.array.reshape(-1, 2), axis=1)))
        if MPI.COMM_WORLD.rank == 0:
            print(f"  step {step + 1:>6}/{n_steps}  t={t + cfg.DT:.4g}  "
                  f"max|u|={umax:.6e}")
    history.append((step + 1, t + cfg.DT,
                    float(np.max(np.linalg.norm(
                        ns_solver.u_.x.array.reshape(-1, 2), axis=1)))))

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
        fh.write("step,t,max_u\n")
        for row in history:
            fh.write(f"{row[0]},{row[1]:.8e},{row[2]:.16e}\n")
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

if MPI.COMM_WORLD.rank == 0:
    print(f"elapsed {res['elapsed_s']:.1f} s")
