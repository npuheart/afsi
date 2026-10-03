"""IB2d example "Tracers in Impedance Pump" (pyIB2d
``Examples/Tracers_In_Impedance_Pump``) in AFSI.

A flexible heart tube (156 markers: springs k=1e7 + invariant beams + 4
corner target points) sits in a 5 x 5 box of fluid (mu = 1, rho = 1,
64 x 64).  Eleven cross-diameter "pump" springs (k = 1e5) have their rest
length driven in time

    RL = d - |0.9 d sin(2 pi f t)|,   f = 10 Hz,  d = 1,

(example's ``update_Springs.py``; in pyIB2d these are spring rows
``N-3+10 : N-2+20`` with ``N`` the marker count) so the tube pinches
periodically.  110 passive *tracers* ride the flow -- they exert no force
and are simply advected like markers: ``X_t^{n+1} = X_t^n + dt U_h(X_t^n)``
(the driver moves them right after the markers, with the same kernel).

Run::

    conda activate afsi-dolfinx
    python main.py                      # full example (T = 2 s, 64x64)
    TFINAL=0.02 python main.py          # smoke test

Output: ``plot/afsi_result.npz`` (t, X markers, Xt tracers; u/p on a
coarser sampling) + XDMF.
"""
import os
import time
from math import pi

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
FIELD_EVERY = int(os.environ.get("FIELD_EVERY", "50"))   # u/p sampling (steps)

X_ib2d = ex["X"]
Nb = X_ib2d.shape[0]
sp = ex["springs"]
bm = ex["beams"]
tg = ex["targets"]
tracers0 = ex["tracers"]
Nt = tracers0.shape[0]
assert "mass" not in ex and "muscles" not in ex and "noninv_beams" not in ex
ds_ib2d = min(Lx / (2 * Nx), Ly / (2 * Ny))

if comm.rank == 0:
    print(f"IB2d example : {IB2D_EXAMPLE}")
    print(f"rho={rho} mu={mu} box={Lx}x{Ly} grid={Nx}x{Ny} dt={dt} T={T} "
          f"steps={num_steps} ds={ds_ib2d}")
    print(f"structure    : {Nb} markers, {len(sp['conn'])} springs, "
          f"{len(bm['p1'])} beams, {len(tg['ids'])} targets, {Nt} tracers")

# ---------------------------------------------------------------------------
# Fluid + structure
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


# tracers: same machinery, passive (no forces, never spread back)
tracer_cells = np.column_stack(
    [np.arange(Nt - 1), np.arange(1, Nt)]).astype(np.int64)
tracer_mesh = dolfinx.mesh.create_mesh(comm, tracer_cells, coord_el, tracers0)
Vt = fem.functionspace(tracer_mesh, coord_el)
Xt = fem.Function(Vt, name="tracer_position")
Ut = fem.Function(Vt, name="tracer_velocity")
dof_rows_t = Vt.tabulate_dof_coordinates()[:, :2]
dof_of_tracer = np.array([int(np.argmin(np.linalg.norm(dof_rows_t - p, axis=1)))
                          for p in tracers0])
assert np.allclose(dof_rows_t[dof_of_tracer], tracers0, atol=1e-12)
_tmp = np.empty_like(tracers0)
_tmp[dof_of_tracer] = tracers0
Xt.x.array[:] = _tmp.ravel()


def tracer_array():
    return Xt.x.array.reshape(-1, 2)[dof_of_tracer]


# ---------------------------------------------------------------------------
# Forces: springs (RL driven) + invariant beams + target points (at X_h)
# ---------------------------------------------------------------------------
sp_a, sp_b = sp["conn"][:, 0], sp["conn"][:, 1]
sp_k = sp["k"]
sp_L = sp["L"].copy()                    # working copy: pump rest lengths move
bm_p1, bm_p2, bm_p3 = bm["p1"], bm["p2"], bm["p3"]
bm_k, bm_C = bm["kb"], bm["C"]
tg_ids, tg_k = tg["ids"], tg["k"]
tg_anchor = X_ib2d[tg_ids].copy()        # update_target = 0: anchors fixed

Pump = dict(d=1.0, freq=10.0)            # update_Springs.py constants
N_xlag = Nb                              # xLag.size in the driver


def update_rest_lengths(t):
    """pyIB2d ``update_Springs``: relax the pump springs (rows
    ``N-3+10 : N-2+20`` of the .spring file) to ``d - |0.9 d sin(2 pi f t)|``."""
    sp_L[N_xlag - 3 + 10:N_xlag - 2 + 20] = Pump["d"] - np.abs(
        0.9 * Pump["d"] * np.sin(2 * pi * Pump["freq"] * t))


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


def target_forces(Xm, Fm):
    Ft = tg_k[:, None] * (tg_anchor - Xm[tg_ids])
    np.add.at(Fm[:, 0], tg_ids, Ft[:, 0])
    np.add.at(Fm[:, 1], tg_ids, Ft[:, 1])


def spread_forces_at_half():
    """Springs + beams + targets at X_h, spread with the 4-point kernel."""
    Fm = np.zeros((Nb, 2))
    Xm = marker_array(X_h)
    spring_forces(Xm, Fm)
    beam_forces(Xm, Fm)
    target_forces(Xm, Fm)
    Fdof = np.empty_like(Fm)
    Fdof[dof_of_ib2d] = Fm
    F_s.x.array[:] = Fdof.ravel()
    F_s.x.array[:] *= ds_ib2d
    ib.evaluate_current_points(X_h._cpp_object)
    ib.solid_to_fluid(f_ib._cpp_object, F_s._cpp_object)
    solver.periodic_fold(f_ib)
    if f_ib is not solver.f:
        solver.f.interpolate(f_ib)


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
F_check = np.zeros((Nb, 2))
target_forces(Xp, F_check)
F_loop = np.zeros((Nb, 2))
for i1, kk in zip(tg_ids, tg_k):
    F_loop[i1] += kk * (X_ib2d[i1] - Xp[i1])
err_tg = np.abs(F_check - F_loop).max() / np.abs(F_loop).max()
if comm.rank == 0:
    print(f"force checks (perturbed config): springs {err_sp:.2e}, "
          f"beams {err_bm:.2e}, targets {err_tg:.2e}")
assert err_sp < 1e-12 and err_bm < 1e-12 and err_tg < 1e-12

# ---------------------------------------------------------------------------
# Output sampling on the IB2d grid
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


hist = {"t": [0.0], "X": [marker_array(X).copy()], "Xt": [tracer_array().copy()],
        "t_u": [0.0]}
u0, p0 = grid_fields()
hist["u"], hist["p"] = [u0], [p0]
write_output(0.0)

# ---------------------------------------------------------------------------
# Time loop -- Peskin (2002) / IB2d IBM_Driver ordering (+ pump + tracers)
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

    # (1b) pump: relax the cross-diameter springs at the *start* of the step
    #      (driver: update_Springs runs before the force computation)
    update_rest_lengths(step * dt)

    # (2) forces at X_h, spread with the 4-point kernel
    spread_forces_at_half()

    # (3) two-stage fluid solve
    solver.solve_one_step()

    # (4) markers follow the fluid: X^{n+1} = X^n + dt U_h(X_h)
    #     (fluid_to_solid reuses the X_h coordinates from (2))
    ib.fluid_to_solid(ib_velocity(solver.u_h)._cpp_object, U_s._cpp_object)
    X.x.array[:] += dt * U_s.x.array
    X.x.scatter_forward()

    # (5) tracers: passive advection X_t += dt U_h(X_t)  (driver: tracers
    #     move right after the markers, same kernel; no force feedback)
    ib.evaluate_current_points(Xt._cpp_object)
    ib.fluid_to_solid(ib_velocity(solver.u_h)._cpp_object, Ut._cpp_object)
    Xt.x.array[:] += dt * Ut.x.array
    Xt.x.scatter_forward()

    if (step + 1) % print_dump == 0 or step == num_steps - 1:
        t = (step + 1) * dt
        hist["t"].append(t)
        hist["X"].append(marker_array(X).copy())
        hist["Xt"].append(tracer_array().copy())
        if (step + 1) % FIELD_EVERY == 0 or step == num_steps - 1:
            hist["t_u"].append(t)
            uf, pf = grid_fields()
            hist["u"].append(uf)
            hist["p"].append(pf)
        write_output(t)
        if comm.rank == 0 and ((step + 1) % (print_dump * 20) == 0
                               or step == num_steps - 1):
            wall = time.perf_counter() - tic
            print(f"step {step+1:6d}  t={t:.3f}  "
                  f"y-range=[{hist['X'][-1][:, 1].min():.3f},"
                  f"{hist['X'][-1][:, 1].max():.3f}]  "
                  f"tracer-y-mean={hist['Xt'][-1][:, 1].mean():.3f}  "
                  f"wall={wall:.1f}s")

elapsed = time.perf_counter() - tic
xf.close()
xs.close()
fname = os.path.join(out_dir, f"afsi_result_g{GRAD_DIV:g}.npz")
np.savez_compressed(fname,
                    t=np.array(hist["t"]), X=np.array(hist["X"]),
                    Xt=np.array(hist["Xt"]),
                    t_u=np.array(hist["t_u"]),
                    u=np.array(hist["u"]), p=np.array(hist["p"]),
                    dx=Lx / Nx, dy=Ly / Ny, n_tracers=Nt)
if comm.rank == 0:
    print(f"done in {elapsed:.1f}s -> {fname}")
