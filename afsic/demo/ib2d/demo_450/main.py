"""IB2d example "Gravity Cellular Race" (pyIB2d ``Examples/Gravity_Cellular_Race``)
in AFSI.

Two elastic cell networks (springs + invariant beams, 81 markers each) sit in a
1 x 1 periodic box of viscous fluid (mu = 0.01, rho = 1).  *Every* marker is
additionally coupled -- by its own "mass-spring" (k = 1e6) -- to a ghost
particle with mass M (0.05, 0.2 or 1, varying per point) that **does not move
with the fluid**; gravity (0, -g) acts on the ghosts, which then drag the
markers down through the springs.  The result: the two cells "race" downward
with deformation set by their (different) mass layouts.

The mass model is easy to get wrong from the docs, so here is exactly what
IB2d (both matIB2d and pyIB2d) implements -- reproduced verbatim here:

* each mass row (id, k, M): the marker at ``id`` (which moves with the fluid)
  feels ``F = k (X_ghost - X_marker)``.  There is **no** anchor to the initial
  position; the ghost's position is a separate state variable,
* the ghost obeys ``M dV/dt = -F + M g`` (gravity acts on the ghost only),
* the ghost position is *not* synced back into the marker list -- the marker
  keeps following the fluid (the two are connected only through F).

Per step (driver order): markers take the usual fluid half-step; the ghosts
step ``dt/2`` with their velocity; forces are evaluated (markers at X_h,
ghosts at X_h_mass); the fluid is solved; markers are advanced with the fluid
velocity; the ghosts are restored and advanced by ``dt`` with the new
half-step velocity (``please_Move_Massive_Boundary`` /
``please_Update_Massive_Boundary_Velocity``).

Run::

    conda activate afsi-dolfinx
    python main.py                      # full example (T = 0.35 s, 64x64)
    TFINAL=0.005 python main.py         # smoke test

Output: ``plot/afsi_result_g<GRAD_DIV>.npz`` (t, X markers, Xmass ghosts,
u/p on the IB2d grid) + XDMF.
"""
import os
import time

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

import basix.ufl
import dolfinx
import ufl
from dolfinx import fem
from dolfinx.mesh import CellType, GhostMode

from afsic import IBMesh, IBInterpolation
from afsic.euler.PeskinRK2Solver import PeskinRK2Solver

import ib2d_io

_here = os.path.dirname(os.path.abspath(__file__))
IB2D_EXAMPLE = os.environ.get("IB2D_EXAMPLE", os.path.join(_here, "ib2d_input"))

comm = MPI.COMM_WORLD
assert comm.size == 1, "this demo is serial (IB coupling of afsic gathers on rank 0)"

ex = ib2d_io.load_example(IB2D_EXAMPLE)
P = ex["params"]
rho, mu = float(P["rho"]), float(P["mu"])
Lx, Ly = float(P["Lx"]), float(P["Ly"])
Nx = int(os.environ.get("NX", P["Nx"]))
Ny = int(os.environ.get("NY", round(Nx * Ly / Lx)))
dt = float(os.environ.get("DT", P["dt"]))
T = float(os.environ.get("TFINAL", P["Tfinal"]))
print_dump = int(os.environ.get("PRINT_DUMP", P["print_dump"]))
num_steps = int(round(T / dt))
out_dir = os.environ.get("OUTPUT_PATH", os.path.join(_here, "plot"))
os.makedirs(out_dir, exist_ok=True)

GRAD_DIV = float(os.environ.get("GRAD_DIV", "0"))

X_ib2d = ex["X"]
Nb = X_ib2d.shape[0]
sp = ex["springs"]
bm = ex["beams"]
mass = ex["mass"]
assert "targets" not in ex and "noninv_beams" not in ex
ds_ib2d = min(Lx / (2 * Nx), Ly / (2 * Ny))

# gravity (as the driver: normalized direction; g = 9.80665 on the ghosts)
GRAVITY = 9.80665
g_dir = np.array([float(P["x_gravity_vec_comp"]), float(P["y_gravity_vec_comp"])])
g_dir = g_dir / np.linalg.norm(g_dir)
g_vec = GRAVITY * g_dir
if comm.rank == 0:
    print(f"IB2d example : {IB2D_EXAMPLE}")
    print(f"rho={rho} mu={mu} box={Lx}x{Ly} grid={Nx}x{Ny} dt={dt} T={T} "
          f"steps={num_steps} ds={ds_ib2d}")
    print(f"structure    : {Nb} markers, {len(sp['conn'])} springs, "
          f"{len(bm['p1'])} beams, {len(mass['ids'])} mass points "
          f"(k={mass['k'][0]:.2g}, M in [{mass['M'].min():g}, {mass['M'].max():g}])"
          f", gravity dir {g_dir}")

# ---------------------------------------------------------------------------
# Fluid + structure (as demo_449)
# ---------------------------------------------------------------------------
mesh = dolfinx.mesh.create_rectangle(
    comm, ((0.0, 0.0), (Lx, Ly)), (Nx, Ny),
    cell_type=CellType.quadrilateral, ghost_mode=GhostMode.shared_facet)
solver = PeskinRK2Solver(mesh, (0.0, Lx, 0.0, Ly), dt, rho, mu, grad_div=GRAD_DIV)
Vc = solver.V

ibmesh = IBMesh(0.0, Lx, 0.0, Ly, Nx, Ny, 1)
V_ib = fem.functionspace(mesh, basix.ufl.element("Lagrange", "quadrilateral", 1,
                                                 shape=(2,)))
u_ib, f_ib = fem.Function(V_ib), fem.Function(V_ib)
coords_bg = fem.Function(V_ib)
coords_bg.interpolate(lambda x: np.array([x[0], x[1]]))
ibmesh.build_map(coords_bg._cpp_object)
ib = IBInterpolation(ibmesh)


def ib_velocity(u):
    u_ib.interpolate(u)
    return u_ib


coord_el = basix.ufl.element("Lagrange", "interval", 1, shape=(2,))
structure = dolfinx.mesh.create_mesh(comm, sp["conn"].astype(np.int64),
                                     coord_el, X_ib2d)
Vs = fem.functionspace(structure, coord_el)
assert Vs.dofmap.index_map.size_local == Nb

X = fem.Function(Vs, name="position")
X.interpolate(lambda x: np.array([x[0], x[1]]))
X_h = fem.Function(Vs, name="position_half")
X_h.x.array[:] = X.x.array
U_s = fem.Function(Vs, name="velocity")
F_s = fem.Function(Vs, name="force")

dof_rows = Vs.tabulate_dof_coordinates()[:, :2]
dof_of_ib2d = np.array([int(np.argmin(np.linalg.norm(dof_rows - p, axis=1)))
                        for p in X_ib2d])
assert len(np.unique(dof_of_ib2d)) == Nb
assert np.allclose(dof_rows[dof_of_ib2d], X_ib2d, atol=1e-12)


def marker_array(fn):
    return fn.x.array.reshape(-1, 2)[dof_of_ib2d]


# ---------------------------------------------------------------------------
# Forces: springs + invariant beams (at X_h) + mass-spring coupling (X_h, Xmass)
# ---------------------------------------------------------------------------
sp_a, sp_b = sp["conn"][:, 0], sp["conn"][:, 1]
sp_k, sp_L = sp["k"], sp["L"]
bm_p1, bm_p2, bm_p3 = bm["p1"], bm["p2"], bm["p3"]
bm_k, bm_C = bm["kb"], bm["C"]
m_ids, m_k, m_M = mass["ids"], mass["k"], mass["M"]

Xmass = X_ib2d[m_ids].copy()             # ghost-particle positions
Vmass = np.zeros_like(Xmass)             # ghost-particle velocities


def spring_forces(Xm, Fm):
    d = Xm[sp_b] - Xm[sp_a]
    L = np.hypot(d[:, 0], d[:, 1])
    sF = sp_k * (L - sp_L) / L
    np.add.at(Fm[:, 0], sp_a, sF * d[:, 0])
    np.add.at(Fm[:, 1], sp_a, sF * d[:, 1])
    np.add.at(Fm[:, 0], sp_b, -sF * d[:, 0])
    np.add.at(Fm[:, 1], sp_b, -sF * d[:, 1])


def beam_forces(Xm, Fm):
    Xp, Xq, Xr = Xm[bm_p1], Xm[bm_p2], Xm[bm_p3]
    S = ((Xr[:, 0] - Xq[:, 0]) * (Xq[:, 1] - Xp[:, 1])
         - (Xr[:, 1] - Xq[:, 1]) * (Xq[:, 0] - Xp[:, 0]))
    K = bm_k * (S - bm_C)
    np.add.at(Fm[:, 0], bm_p1, K * (Xr[:, 1] - Xq[:, 1]))
    np.add.at(Fm[:, 1], bm_p1, -K * (Xr[:, 0] - Xq[:, 0]))
    np.add.at(Fm[:, 0], bm_p2, K * ((Xq[:, 1] - Xp[:, 1]) + (Xr[:, 1] - Xq[:, 1])))
    np.add.at(Fm[:, 1], bm_p2, -K * ((Xr[:, 0] - Xq[:, 0]) + (Xq[:, 0] - Xp[:, 0])))
    np.add.at(Fm[:, 0], bm_p3, K * (Xq[:, 1] - Xp[:, 1]))
    np.add.at(Fm[:, 1], bm_p3, -K * (Xq[:, 0] - Xp[:, 0]))


def spread_forces_at_half(Xmass_h):
    """Springs + beams + mass-marker tethers at (X_h, Xmass_h), spread x ds.

    Returns the per-mass tether force F = k (X_ghost - X_marker) (evaluated
    at the half-step), which both acts on the marker and drives the ghost
    ODE."""
    Fm = np.zeros((Nb, 2))
    Xm = marker_array(X_h)
    spring_forces(Xm, Fm)
    beam_forces(Xm, Fm)
    F_mass = m_k[:, None] * (Xmass_h - Xm[m_ids])
    np.add.at(Fm[:, 0], m_ids, F_mass[:, 0])
    np.add.at(Fm[:, 1], m_ids, F_mass[:, 1])
    Fdof = np.empty_like(Fm)
    Fdof[dof_of_ib2d] = Fm
    F_s.x.array[:] = Fdof.ravel()
    F_s.x.array[:] *= ds_ib2d
    ib.evaluate_current_points(X_h._cpp_object)
    ib.solid_to_fluid(f_ib._cpp_object, F_s._cpp_object)
    solver.periodic_fold(f_ib)
    if f_ib is not solver.f:
        solver.f.interpolate(f_ib)
    return F_mass


# sanity checks of the vectorised forces against plain loops
rng = np.random.default_rng(0)
Xp = X_ib2d + 1e-3 * rng.standard_normal(X_ib2d.shape)
F_check = np.zeros((Nb, 2))
spring_forces(Xp, F_check)
F_loop = np.zeros((Nb, 2))
for a, b, kk, LL in zip(sp_a, sp_b, sp_k, sp_L):
    d = Xp[b] - Xp[a]
    L = np.hypot(*d)
    sF = kk * (L - LL) / L
    F_loop[a] += sF * d
    F_loop[b] -= sF * d
err_sp = np.abs(F_check - F_loop).max() / np.abs(F_loop).max()
F_check = np.zeros((Nb, 2))
beam_forces(Xp, F_check)
F_loop = np.zeros((Nb, 2))
for i1, i2, i3, kk, CC in zip(bm_p1, bm_p2, bm_p3, bm_k, bm_C):
    ax, ay = Xp[i1]
    bx, by = Xp[i2]
    cx, cy = Xp[i3]
    S = (cx - bx) * (by - ay) - (cy - by) * (bx - ax)
    K = kk * (S - CC)
    F_loop[i1] += (K * (cy - by), -K * (cx - bx))
    F_loop[i2] += (K * ((by - ay) + (cy - by)), -K * ((cx - bx) + (bx - ax)))
    F_loop[i3] += (K * (by - ay), -K * (bx - ax))
err_bm = np.abs(F_check - F_loop).max() / np.abs(F_loop).max()
if comm.rank == 0:
    print(f"force checks (perturbed config): springs {err_sp:.2e}, beams {err_bm:.2e}")
assert err_sp < 1e-12 and err_bm < 1e-12

# ---------------------------------------------------------------------------
# Output sampling on the IB2d grid (as demo_445)
# ---------------------------------------------------------------------------
gx = np.arange(Nx) * (Lx / Nx)
gy = np.arange(Ny) * (Ly / Ny)
Vdof = Vc.tabulate_dof_coordinates()[:, :2]
iu = np.rint(Vdof[:, 0] / (Lx / Nx)).astype(int)
ju = np.rint(Vdof[:, 1] / (Ly / Ny)).astype(int)
on_grid = lambda Q, i, j: (np.isclose(Q[:, 0], i * (Lx / Nx))
                           & np.isclose(Q[:, 1], j * (Ly / Ny)))
keep = (iu < Nx) & (ju < Ny) & on_grid(Vdof, iu, ju)
Q2s = fem.functionspace(mesh, ("Lagrange", 2))
p_grid_fn = fem.Function(Q2s)
Pdof = Q2s.tabulate_dof_coordinates()[:, :2]
ip = np.rint(Pdof[:, 0] / (Lx / Nx)).astype(int)
jp = np.rint(Pdof[:, 1] / (Ly / Ny)).astype(int)
keep_p = (ip < Nx) & (jp < Ny) & on_grid(Pdof, ip, jp)


def grid_fields():
    u = np.zeros((Ny, Nx, 2))
    u[ju[keep], iu[keep]] = solver.u_.x.array.reshape(-1, 2)[keep]
    p_grid_fn.interpolate(solver.p_)
    p = np.zeros((Ny, Nx))
    p[jp[keep_p], ip[keep_p]] = p_grid_fn.x.array[keep_p]
    return u, p


V1 = fem.functionspace(mesh, basix.ufl.element("Lagrange", "quadrilateral", 1,
                                               shape=(2,)))
u_io, f_io = fem.Function(V1, name="u"), fem.Function(V1, name="f")
p_io = fem.Function(fem.functionspace(mesh, ("Lagrange", 1)), name="p")
xf = dolfinx.io.XDMFFile(comm, os.path.join(out_dir, "fluid.xdmf"), "w")
xs = dolfinx.io.XDMFFile(comm, os.path.join(out_dir, "structure.xdmf"), "w")
xf.write_mesh(mesh)
xs.write_mesh(structure)
disp = fem.Function(Vs, name="displacement")
X0 = X.x.array.copy()


def write_output(t):
    u_io.interpolate(solver.u_)
    f_io.interpolate(solver.f)
    p_io.interpolate(solver.p_)
    for fn in (u_io, p_io, f_io):
        xf.write_function(fn, t)
    disp.x.array[:] = X.x.array - X0
    xs.write_function(disp, t)


hist = {"t": [0.0], "X": [marker_array(X).copy()], "Xmass": [Xmass.copy()]}
u0, p0 = grid_fields()
hist["u"], hist["p"] = [u0], [p0]
write_output(0.0)

# cell split (from the example's generator: two 81-point cells)
cellA = np.arange(0, 81)
cellB = np.arange(81, 162)

# ---------------------------------------------------------------------------
# Time loop -- Peskin (2002) / IB2d IBM_Driver ordering (+ mass bookkeeping)
# ---------------------------------------------------------------------------
tic = time.perf_counter()
for step in range(num_steps):
    # (1) marker half-step X_h = X + dt/2 U^n(X)
    ib.evaluate_current_points(X._cpp_object)
    if step == 0:
        X_h.x.array[:] = X.x.array
    else:
        ib.fluid_to_solid(ib_velocity(solver.u_n)._cpp_object, U_s._cpp_object)
        X_h.x.array[:] = X.x.array + 0.5 * dt * U_s.x.array

    # (1b) ghost masses: half-step move; remember their step-start positions
    massOld = Xmass.copy()
    Xmass_h = Xmass + 0.5 * dt * Vmass

    # (2) forces at (X_h, Xmass_h), spread with the 4-point kernel
    F_mass = spread_forces_at_half(Xmass_h)

    # (3) two-stage fluid solve
    solver.solve_one_step()

    # (4) markers: X^{n+1} = X^n + dt U_h(X_h)   (mass-ids included: the
    #     marker follows the fluid; the ghost is a separate state)
    ib.fluid_to_solid(ib_velocity(solver.u_h)._cpp_object, U_s._cpp_object)
    X.x.array[:] += dt * U_s.x.array
    X.x.scatter_forward()

    # (4b) ghosts: v_half = v - dt/2 (F/M - g);  X_mass = X_mass^n + dt v_half;
    #      v^{n+1} = v - dt (F/M - g)   (driver's Move/Update pair)
    acc = F_mass / m_M[:, None] - g_vec
    Vmass_h = Vmass - 0.5 * dt * acc
    Xmass = massOld + dt * Vmass_h
    Vmass = Vmass - dt * acc

    t = (step + 1) * dt
    if (step + 1) % print_dump == 0 or step + 1 == num_steps:
        Xm = marker_array(X).copy()
        hist["t"].append(t)
        hist["X"].append(Xm)
        hist["Xmass"].append(Xmass.copy())
        ug, pg = grid_fields()
        hist["u"].append(ug)
        hist["p"].append(pg)
        write_output(t)
        if comm.rank == 0:
            print(f"step {step+1:6d}  t={t:.4f}  comA=({Xm[cellA,0].mean():.4f},"
                  f"{Xm[cellA,1].mean():.4f})  comB=({Xm[cellB,0].mean():.4f},"
                  f"{Xm[cellB,1].mean():.4f})  |F|max={np.abs(F_s.x.array).max():.3e}"
                  f"  wall={time.perf_counter()-tic:.1f}s", flush=True)

xf.close()
xs.close()
np.savez_compressed(os.path.join(out_dir, f"afsi_result_g{GRAD_DIV:g}.npz"),
                    t=np.array(hist["t"]), X=np.array(hist["X"]),
                    Xmass=np.array(hist["Xmass"]),
                    u=np.array(hist["u"]), p=np.array(hist["p"]),
                    n_markers=Nb, n_cellA=len(cellA),
                    dx=Lx / Nx, dy=Ly / Ny)
if comm.rank == 0:
    print(f"done in {time.perf_counter()-tic:.1f}s -> "
          f"{out_dir}/afsi_result_g{GRAD_DIV:g}.npz")
