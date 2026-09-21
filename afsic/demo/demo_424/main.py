"""demo_424 — tethered aorta in a box (2-D planar), two solver-verification cases.

CASE=open    patent aorta: two tethered wall strips in a pressure-driven box
CASE=closed  occluded aorta: the strips plus a full-occlusion membrane at
             mid-length, so the solid outline reads as an "H"

The wall carries no constitutive law; each Lagrangian point is held only by a
tether (volumetric spring) beta*(X_ref - X).  The pressure difference between
the two open ends of the box is imposed as a Dirichlet condition on the
pressure Poisson problem, which is the projection-method realisation of a
prescribed-normal-traction (pressure-driven) open boundary.

Run (inside the afsi-dolfinx environment, from this directory):
    python generate_mesh.py && python main.py
    CASE=closed NY=90 python generate_mesh.py && CASE=closed NY=90 python main.py
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
from dolfinx.io import VTKFile, XDMFFile
from dolfinx.mesh import (CellType, GhostMode, locate_entities, meshtags)
from basix.ufl import element
from ufl import (Measure, TestFunction, SpatialCoordinate, as_vector,
                 inner, dx)

import configuration as cfg
import verify as vf
from afsic import ChorinSolver, IPCSSolver, IBMesh, IBInterpolation

t_start = time.time()

if MPI.COMM_WORLD.rank == 0:
    print(cfg.summary())
    print("-" * 78)

# ==========================================================================
# Fluid domain: the box
# ==========================================================================
mesh = dolfinx.mesh.create_rectangle(
    comm=MPI.COMM_WORLD,
    points=((0.0, 0.0), (cfg.BOX_L, cfg.BOX_W)),
    n=(cfg.NX, cfg.NY),
    cell_type=CellType.quadrilateral,
    ghost_mode=GhostMode.shared_facet,
)

MARKER_INLET, MARKER_OUTLET, MARKER_BOTTOM, MARKER_TOP = 1, 2, 3, 4
fdim = mesh.topology.dim - 1
mesh.topology.create_connectivity(fdim, mesh.topology.dim)

boundaries = [
    (MARKER_INLET, lambda x: np.isclose(x[0], 0.0)),
    (MARKER_OUTLET, lambda x: np.isclose(x[0], cfg.BOX_L)),
    (MARKER_BOTTOM, lambda x: np.isclose(x[1], 0.0)),
    (MARKER_TOP, lambda x: np.isclose(x[1], cfg.BOX_W)),
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

# --- gap damping instead of a sharp pressure/velocity BC split -------------
# Uniform pressure on the whole end face keeps the boundary-data analysis
# smooth and avoids the pressure/no-slip corner singularity.  The outer gap
# is then made effectively static by an implicit linear damping term
# ``drag * u`` in the momentum equation.  This is a smooth porous-medium
# regularisation: the lumen (drag = 0) is unaffected, while the gap is damped
# to a small velocity.
GAP_DRAG = float(os.environ.get(
    "GAP_DRAG", "1.0e6" if cfg.CASE == "closed" else "0.0"))
# Damping can be applied to the full outer gap (GAP_DRAG_LENGTH <= 0) or
# only to a short plug near each open end (positive length).
GAP_DRAG_LENGTH = float(os.environ.get("GAP_DRAG_LENGTH", "-1.0"))
# Optional extra volumetric drag in a thin band around all immersed solid
# surfaces.  This is a numerical experiment to see whether the near-wall
# high-speed layer is a fluid boundary-layer artifact.
EDGE_DRAG = float(os.environ.get("EDGE_DRAG", "0.0"))
EDGE_BAND = float(os.environ.get("EDGE_BAND", str(2.0 * cfg.H)))
DRAG_DELTA = 2.0 * cfg.H


def _smoothstep(s):
    s = np.clip(s, 0.0, 1.0)
    return s * s * (3.0 - 2.0 * s)


def drag_weight(y):
    # 1 in the outer gap, 0 in the wall/lumen, smooth over DRAG_DELTA.
    lo = _smoothstep((cfg.Y_OUT_LO - y) / DRAG_DELTA)
    hi = _smoothstep((y - cfg.Y_OUT_HI) / DRAG_DELTA)
    return np.maximum(lo, hi)


def end_weight(x):
    # 1 near the inlet/outlet plugs, 0 in the middle of the tube.
    if GAP_DRAG_LENGTH <= 0.0:
        return np.ones_like(x)
    L = GAP_DRAG_LENGTH
    left = _smoothstep((L - x) / L)
    right = _smoothstep((x - (cfg.BOX_L - L)) / L)
    return np.maximum(left, right)


def edge_weight(x, y):
    """Smooth band around the wall/membrane surfaces (diagnostic test)."""
    if EDGE_DRAG <= 0.0:
        return np.zeros_like(x)
    w = np.zeros_like(x)
    xw = (x >= cfg.X_OFF) & (x <= cfg.X_OFF + cfg.L_AORTA)
    for y0 in (cfg.Y_OUT_LO, cfg.Y_IN_LO, cfg.Y_IN_HI, cfg.Y_OUT_HI):
        w = w + np.exp(-((y - y0) / EDGE_BAND) ** 2) * xw
    xd = cfg.X_OFF + cfg.DISC_X
    ym = (y >= cfg.Y_IN_LO) & (y <= cfg.Y_IN_HI)
    for x0 in (xd - 0.5 * cfg.DISC_T, xd + 0.5 * cfg.DISC_T):
        w = w + np.exp(-((x - x0) / EDGE_BAND) ** 2) * ym
    return np.clip(w, 0.0, 1.0)


# --- velocity BCs: no-slip on the two side walls only ----------------------
u_zero = np.zeros(mesh.geometry.dim, dtype=PETSc.ScalarType)
bcu = [dirichletbc(u_zero, locate_dofs_topological(V, fdim, facet_tag.find(m)),
                   V)
       for m in (MARKER_BOTTOM, MARKER_TOP)]

# --- pressure BCs: uniform traction on the whole inlet/outlet --------------
dofs_inlet = locate_dofs_topological(Q, fdim, facet_tag.find(MARKER_INLET))
dofs_outlet = locate_dofs_topological(Q, fdim, facet_tag.find(MARKER_OUTLET))
bcp_outlet = dirichletbc(PETSc.ScalarType(0.0), dofs_outlet, Q)


def make_bcp(value):
    return [dirichletbc(PETSc.ScalarType(value), dofs_inlet, Q), bcp_outlet]


bcp = make_bcp(0.0)
ds_inlet = Measure("ds", domain=mesh, subdomain_data=facet_tag)(MARKER_INLET)

# Implicit damping coefficient.  It enters the momentum LHS, so it can be
# made large without an explicit time-step stability restriction.
if GAP_DRAG > 0.0:
    drag_coeff = Function(Q)
    drag_coeff.interpolate(
        lambda x: (GAP_DRAG * drag_weight(x[1]) * end_weight(x[0])
                   + EDGE_DRAG * edge_weight(x[0], x[1])))
    drag_coeff.x.scatter_forward()
else:
    drag_coeff = None

if MPI.COMM_WORLD.rank == 0:
    if GAP_DRAG_LENGTH > 0.0:
        print(f"gap drag coefficient: {GAP_DRAG:g} "
              f"(end plugs only, length {GAP_DRAG_LENGTH:g} m)")
    else:
        print(f"gap drag coefficient: {GAP_DRAG:g} (full gap)")
    if EDGE_DRAG > 0.0:
        print(f"edge drag coefficient: {EDGE_DRAG:g} "
              f"(band {EDGE_BAND:g} m)")

if cfg.SOLVER == "chorin":
    ns_solver = ChorinSolver(V, Q, bcu, bcp, cfg.DT, cfg.RHO, cfg.MU,
                             drag=drag_coeff)
    # ChorinSolver uses -f in the momentum equation, IPCSSolver uses +f
    force_scale = 1.0
else:
    p_traction = Constant(mesh, PETSc.ScalarType(0.0))
    ns_solver = IPCSSolver(V, Q, bcu, bcp, cfg.DT, cfg.RHO, cfg.MU,
                           ds_p=ds_inlet, p_traction=p_traction,
                           drag=drag_coeff)
    force_scale = -1.0

# ==========================================================================
# Solid: tethered aortic wall (+ occluding membrane when CASE=closed)
# ==========================================================================
with XDMFFile(MPI.COMM_WORLD, cfg.solid_mesh_path(), "r") as xdmf:
    structure = xdmf.read_mesh(name="mesh")

v_cg2_s = element("Lagrange", structure.topology.cell_name(),
                  cfg.FORCE_ORDER, shape=(structure.geometry.dim,))
Vs = functionspace(structure, v_cg2_s)
Vs_io = functionspace(structure, element("Lagrange",
                                         structure.topology.cell_name(), 1,
                                         shape=(structure.geometry.dim,)))

solid_coords = Function(Vs, name="solid_coords")
solid_coords_io = Function(Vs_io, name="solid_coords_io")
solid_force = Function(Vs, name="solid_force")
solid_force_io = Function(Vs_io, name="solid_force_io")
solid_velocity = Function(Vs, name="solid_velocity")
coords_ref = Function(Vs, name="coords_ref")
displacement = Function(Vs, name="displacement")

for f in (solid_coords, coords_ref):
    f.interpolate(lambda x: np.array([x[0], x[1]]))

# --- tether (volumetric spring) is the ONLY solid force --------------------
dVs = TestFunction(Vs)
X0 = SpatialCoordinate(structure)
spring = solid_coords - as_vector([X0[0], X0[1]])
dx_s = Measure("dx", domain=structure)
L_hat = form(-cfg.BETA * inner(spring, dVs) * dx_s)
b1 = create_vector(Vs)

# ==========================================================================
# Immersed-boundary coupling
# ==========================================================================
ibmesh = IBMesh(0.0, cfg.BOX_L, 0.0, cfg.BOX_W, cfg.NX, cfg.NY,
                cfg.VELOCITY_ORDER)
ib_interpolation = IBInterpolation(ibmesh)
coords_bg = Function(V)
coords_bg.interpolate(lambda x: np.array([x[0], x[1]]))
ibmesh.build_map(coords_bg._cpp_object)
ib_interpolation.evaluate_current_points(solid_coords._cpp_object)

OUT = cfg.output_path()

# ==========================================================================
# Time loop
# ==========================================================================
log.set_log_level(log.LogLevel.WARNING)
p_prev = 0.0
wall_area = 2.0 * cfg.L_AORTA * cfg.T_WALL
disc_area = 2.0 * cfg.A_LUMEN * cfg.DISC_T
y_wall_probe = np.array([0.5 * (cfg.Y_OUT_LO + cfg.Y_IN_LO),
                         0.5 * (cfg.Y_IN_HI + cfg.Y_OUT_HI)])
x_probe = np.linspace(cfg.X_OFF + 0.05 * cfg.L_AORTA,
                      cfg.X_OFF + 0.95 * cfg.L_AORTA, 41)
x_disc = cfg.X_OFF + cfg.DISC_X
y_disc_probe = np.linspace(cfg.Y_IN_LO + 0.2 * cfg.A_LUMEN,
                           cfg.Y_IN_HI - 0.2 * cfg.A_LUMEN, 21)

# ---- lightweight flow-history diagnostics ---------------------------------
flow_diag_every = int(os.environ.get(
    "FLOW_DIAG_EVERY", max(cfg.NSTEPS // 20, 1)))
y_flow = np.arange(cfg.NY + 1) * cfg.H
gap_half_flow = 0.5 * cfg.GAP
gap_centres_flow = [0.5 * cfg.GAP, cfg.BOX_W - 0.5 * cfg.GAP]
y_lumen_flow = y_flow[(y_flow >= cfg.Y_C - cfg.A_LUMEN)
                      & (y_flow <= cfg.Y_C + cfg.A_LUMEN)]
x_mid_flow = cfg.X_OFF + 0.5 * cfg.L_AORTA
x_d_flow = cfg.X_OFF + cfg.DISC_X


def compute_flow_rates(u):
    q_gap = 0.0
    for yc_g in gap_centres_flow:
        y_g = y_flow[(y_flow >= yc_g - gap_half_flow)
                     & (y_flow <= yc_g + gap_half_flow)]
        ug = vf.sample_u(u, mesh, x_mid_flow, y_g)[:, 0]
        ok = np.isfinite(ug)
        q_gap += vf.trapz(y_g[ok], ug[ok])
    q_leak = []
    for dx_s in (-4.0 * cfg.H, 4.0 * cfg.H):
        uf = vf.sample_u(u, mesh, x_d_flow + dx_s, y_lumen_flow)[:, 0]
        ok = np.isfinite(uf)
        q_leak.append(vf.trapz(y_lumen_flow[ok], uf[ok]))
    return q_gap, q_leak[0], q_leak[1]


flow_file = None
if MPI.COMM_WORLD.rank == 0:
    flow_file = open(OUT + "flow_history.csv", "w")
    flow_file.write("step,t,Q_gap,Q_leak_up,Q_leak_dn,max_u\n")

for step in range(cfg.NSTEPS):
    t = step * cfg.DT

    p_target = cfg.p_inlet(t)
    if cfg.SOLVER == "chorin":
        ns_solver.bcp = make_bcp(p_target)
    else:
        # IPCS accumulates the pressure increment phi into p_ and uses the
        # full pressure as the inlet traction.
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

    if (step + 1) % max(cfg.NSTEPS // 10, 1) == 0:
        umax = np.max(np.linalg.norm(
            ns_solver.u_.x.array.reshape(-1, 2), axis=1))
        if MPI.COMM_WORLD.rank == 0:
            print(f"  step {step + 1:>6}/{cfg.NSTEPS}  t={t + cfg.DT:.4g}  "
                  f"p_in={p_target:9.4f}  max|u|={umax:.4e}")

    if (step + 1) % flow_diag_every == 0 or step == cfg.NSTEPS - 1:
        q_gap, q_leak_up, q_leak_dn = compute_flow_rates(ns_solver.u_)
        umax_flow = float(np.max(np.linalg.norm(
            ns_solver.u_.x.array.reshape(-1, 2), axis=1)))
        if MPI.COMM_WORLD.rank == 0:
            print(f"  FLOW step {step + 1:>6}/{cfg.NSTEPS}  "
                  f"t={t + cfg.DT:8.4f}  Q_gap={q_gap: .6e}  "
                  f"Q_leak_up={q_leak_up: .6e}  "
                  f"Q_leak_dn={q_leak_dn: .6e}  max|u|={umax_flow:.4e}")
            if flow_file is not None:
                flow_file.write(
                    f"{step + 1},{t + cfg.DT:.8e},{q_gap:.16e},"
                    f"{q_leak_up:.16e},{q_leak_dn:.16e},{umax_flow:.16e}\n")
                flow_file.flush()

if flow_file is not None:
    flow_file.close()

# ==========================================================================
# Solid diagnostics
# ==========================================================================
displacement.x.array[:] = solid_coords.x.array[:] - coords_ref.x.array[:]

pts_wall = np.column_stack([
    np.repeat(x_probe, 2),
    np.tile(y_wall_probe, len(x_probe)),
])
dw = vf._sample(displacement, np.column_stack(
    [pts_wall, np.zeros(len(pts_wall))]), structure, bs=2)
dw = dw[np.isfinite(dw).all(axis=1)]
wall_disp = float(np.mean(np.linalg.norm(dw, axis=1)))
wall_tether = cfg.BETA * float(np.mean(dw[:, 0])) * wall_area

solid_info = {
    "wall_disp": wall_disp,
    "wall_tether": wall_tether,
    "disc_disp": 0.0,
    "disc_tether": 0.0,
    "max_u": float(np.max(np.linalg.norm(
        ns_solver.u_.x.array.reshape(-1, 2), axis=1))),
}

if cfg.CASE == "closed":
    pts_disc = np.column_stack([np.full_like(y_disc_probe, x_disc), y_disc_probe])
    dd = vf._sample(displacement, np.column_stack(
        [pts_disc, np.zeros(len(pts_disc))]), structure, bs=2)
    dd = dd[np.isfinite(dd).all(axis=1)]
    if len(dd):
        solid_info["disc_disp"] = float(np.mean(dd[:, 0]))
        solid_info["disc_tether"] = (cfg.BETA * float(np.mean(dd[:, 0]))
                                     * disc_area)

# ==========================================================================
# Diagnostics
# ==========================================================================
if os.environ.get("DIAG"):
    print("\n" + "#" * 78)
    print("# DIAG")
    print("#" * 78)
    pv = ns_solver.p_.x.array
    print(f"inlet  dofs: {len(dofs_inlet):5d}   p on them: "
          f"min={pv[dofs_inlet].min(): .6e} max={pv[dofs_inlet].max(): .6e}")
    print(f"outlet dofs: {len(dofs_outlet):5d}   p on them: "
          f"min={pv[dofs_outlet].min(): .6e} max={pv[dofs_outlet].max(): .6e}")
    print(f"p global: min={pv.min(): .6e} max={pv.max(): .6e}")

    xs_d = np.array([0.0, 0.00025, 0.0005, 0.005, 0.025, 0.0505, 0.075,
                     0.1005, 0.10075, 0.101])
    print("\np along y=Y_C:")
    for v in vf.sample_p(ns_solver.p_, mesh, xs_d, cfg.Y_C):
        print(f"    {v: .6e}")
    print("\np(y) on the inlet face (x=0.00005):")
    ys_d = np.linspace(0.0002, cfg.BOX_W - 0.0002, 12)
    for yv, v in zip(ys_d, vf.sample_p(ns_solver.p_, mesh, 0.00005, ys_d)):
        print(f"    y={yv:8.5f}  p={v: .6e}")

    # locate max |u|
    gx = np.arange(cfg.NX + 1) * cfg.H
    gy = np.arange(cfg.NY + 1) * cfg.H
    GX, GY = np.meshgrid(gx, gy, indexing="xy")
    pts = np.column_stack([GX.ravel(), GY.ravel(), np.zeros(GX.size)])
    uu = vf._sample(ns_solver.u_, pts, mesh, bs=2)
    mag = np.linalg.norm(uu, axis=1)
    i = int(np.nanargmax(mag))
    print(f"\nmax |u| on nodes = {mag[i]:.6e} m/s at "
          f"({pts[i, 0]:.6f}, {pts[i, 1]:.6f})")
    order_idx = np.argsort(-np.nan_to_num(mag))[:10]
    for j in order_idx:
        print(f"    |u|={mag[j]:.4e} at ({pts[j, 0]:.6f}, {pts[j, 1]:.6f})")

    print("\nu profile across the full box at x=0.0505:")
    ys_full = np.arange(cfg.NY + 1) * cfg.H
    uf = vf.sample_u(ns_solver.u_, mesh, 0.0505, ys_full)
    for yv, (ux, uy) in zip(ys_full, uf):
        if not np.isfinite(ux):
            continue
        print(f"    y={yv:8.5f}  ux={ux: .5e}  uy={uy: .5e}")
    print("#" * 78 + "\n")

# ==========================================================================
# Verification / error report
# ==========================================================================
results = vf.run(cfg, mesh, ns_solver.u_, ns_solver.p_, solid_info)
results["wall_disp"] = solid_info["wall_disp"]
results["wall_tether"] = solid_info["wall_tether"]
results["disc_disp"] = solid_info["disc_disp"]
results["disc_tether"] = solid_info["disc_tether"]
results["elapsed_s"] = time.time() - t_start

# ==========================================================================
# Field output
# ==========================================================================
u_io = Function(functionspace(mesh, element("Lagrange",
                                            mesh.topology.cell_name(), 1,
                                            shape=(mesh.geometry.dim,))))
u_io.interpolate(ns_solver.u_)
solid_coords_io.interpolate(solid_coords)

# Solid displacement for ParaView/PyVista.  The .pvd collection stores
# the current snapshot and can later be extended to a full time series.
displacement_io = Function(Vs_io, name="displacement")
displacement_io.interpolate(displacement)
# The .pvd collection references the actual .vtu pieces, e.g.
# solid_displacement_p0_000000.vtu.  Open the .pvd in ParaView/PyVista.
with VTKFile(MPI.COMM_WORLD, OUT + "solid_displacement.pvd", "w") as file_d:
    file_d.write_function(displacement_io, cfg.T_END)

# Open the field files only now: holding them open for the whole time loop
# leaves stale HDF5 locks behind if the run is interrupted.
with XDMFFile(MPI.COMM_WORLD, OUT + "velocity.xdmf", "w") as file_u:
    file_u.write_mesh(mesh)
    file_u.write_function(u_io, cfg.T_END)
with XDMFFile(MPI.COMM_WORLD, OUT + "pressure.xdmf", "w") as file_p:
    file_p.write_mesh(mesh)
    file_p.write_function(ns_solver.p_, cfg.T_END)

if MPI.COMM_WORLD.rank == 0:
    import json
    with open(OUT + "verify.json", "w") as fh:
        json.dump({k: (float(v) if isinstance(v, (int, float, np.floating))
                       else v) for k, v in results.items()}, fh, indent=2)
    print(f"elapsed {results['elapsed_s']:.1f} s, results in {OUT}")
