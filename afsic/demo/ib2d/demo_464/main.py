"""IB2d example "Update Springs Peristaltic Pumping" (matIB2d
``Examples/Example_HeartTube/Update_Springs_Peristaltic_Pumping``) in AFSI.

The same flexible heart tube as demo_451 (156 markers: springs k = 1e7 +
invariant beams + 4 corner target points, 5 x 5 box, mu = 1, rho = 1, 64 x 64,
dt = 1e-3, T = 2 s), but driven by a *peristaltic* wave instead of the
impedance pump.  The example's ``update_Springs.m`` moves the resting lengths of
the 76 cross-diameter "pump" springs (rows 156..231 of the .spring file, each
joining marker i to marker i + 78) with a travelling sine wave

    RL_i = d - 0.95 sin( (2 pi / L) (x_i - x_M - v t) )   if the sine >= 0,
    RL_i = d                                              otherwise,

where ``d = 1``, ``L = 2``, ``freq = 4`` Hz, ``v = L freq = 8``, the wave time
is ``t = rem(current_time, 1/freq)``, and ``x_M = xLag(39)`` (MATLAB index, i.e.
the 39th marker) is the wave's reference point.  Unlike the impedance pump
(demo_451), which pinches a fixed set of springs in phase, the reflex of this
law is a *spatially propagating* constriction, and it is the first
``update_Springs`` example in the port whose driving depends on the marker
positions themselves (``x_i`` is read from the current configuration).

The example is one of the ``matIB2d`` ones (1-based identifiers), so
``ib2d_io.load_example(..., index_base=1)`` rebases the ids and
``run_reference.py`` rewrites the ``.spring``/``.beam``/``.target`` files for the
pyIB2d reference, importing the python port of the .m file from the run
directory.

Run::

    conda activate afsi-dolfinx
    python main.py                      # full example (T = 2 s, 64x64)
    TFINAL=0.05 python main.py          # smoke test

Output: ``plot/afsi_result_g{GRAD_DIV}.npz`` (t, X markers, u/p on the IB2d
grid) + XDMF.
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

# the matIB2d example files number the Lagrangian points from 1
ex = ib2d_io.load_example(IB2D_EXAMPLE, index_base=1)
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

# gamma = 0 lets the tube collapse numerically (NaN); 2.5 is the value used for
# the shared HeartTube geometry in demo_451
GRAD_DIV = float(os.environ.get("GRAD_DIV", "2.5"))
FIELD_EVERY = int(os.environ.get("FIELD_EVERY", "50"))   # u/p sampling (steps)

X_ib2d = ex["X"]
Nb = X_ib2d.shape[0]
sp = ex["springs"]
bm = ex["beams"]
tg = ex["targets"]
assert "mass" not in ex and "muscles" not in ex and "noninv_beams" not in ex
ds_ib2d = min(Lx / (2 * Nx), Ly / (2 * Ny))

if comm.rank == 0:
    print(f"IB2d example : {IB2D_EXAMPLE}")
    print(f"rho={rho} mu={mu} box={Lx}x{Ly} grid={Nx}x{Ny} dt={dt} T={T} "
          f"steps={num_steps} ds={ds_ib2d}")
    print(f"structure    : {Nb} markers, {len(sp['conn'])} springs, "
          f"{len(bm['p1'])} beams, {len(tg['ids'])} targets")

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

# update_Springs.m constants: a travelling wave of period 1/freq over the tube
Pump = dict(d=1.0, L=2.0, freq=4.0, rows=np.arange(155, 231),
            probe=sp["conn"][155:231, 0], xM_marker=38)
Pump["v"] = Pump["L"] * Pump["freq"]
N_xlag = Nb                              # xLag.size in the driver


def update_rest_lengths(Xm, t):
    """Port of the example's ``update_Springs.m`` (peristaltic wave).

    The 76 cross-diameter pump springs are the last rows of the .spring file.
    The wave reads the *current* configuration (the driver passes the full-step
    ``xLag``, before the forces are evaluated at the half step): the master
    markers ``x_i`` of the pump springs and the reference point ``x_M =
    xLag(39)``.
    """
    x_here = Xm[Pump["probe"], 0]             # x_i of the pump springs
    x_M = Xm[Pump["xM_marker"], 0]            # xLag(39) in the .m file
    tt = np.remainder(t, 1.0 / Pump["freq"])
    delta = np.sin((2 * pi / Pump["L"]) * ((x_here - x_M) - Pump["v"] * tt))
    sp_L[Pump["rows"]] = np.where(delta >= 0.0,
                                  Pump["d"] - 0.95 * delta, Pump["d"])


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


hist = {"t": [0.0], "X": [marker_array(X).copy()], "t_u": [0.0]}
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

    # (1b) pump: advance the peristaltic wave (driver: update_Springs runs
    #      before the force computation and sees the full-step positions)
    update_rest_lengths(marker_array(X), step * dt)

    # (2) forces at X_h, spread with the 4-point kernel
    spread_forces_at_half()

    # (3) two-stage fluid solve
    solver.solve_one_step()

    # (4) markers follow the fluid: X^{n+1} = X^n + dt U_h(X_h)
    #     (fluid_to_solid reuses the X_h coordinates from (2))
    ib.fluid_to_solid(ib_velocity(solver.u_h)._cpp_object, U_s._cpp_object)
    X.x.array[:] += dt * U_s.x.array
    X.x.scatter_forward()

    # dump on exactly the steps the pyIB2d driver writes frames on (its 0-based
    # counter ``cter`` is tested before being incremented), so that AFSI frame k
    # and reference frame k are the same step of the simulation
    if (step % print_dump == 0 and step >= print_dump) or step + 1 == num_steps:
        t = (step + 1) * dt
        hist["t"].append(t)
        hist["X"].append(marker_array(X).copy())
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
                  f"wall={wall:.1f}s")

elapsed = time.perf_counter() - tic
xf.close()
xs.close()
fname = os.path.join(out_dir, f"afsi_result_g{GRAD_DIV:g}.npz")
np.savez_compressed(fname,
                    t=np.array(hist["t"]), X=np.array(hist["X"]),
                    t_u=np.array(hist["t_u"]),
                    u=np.array(hist["u"]), p=np.array(hist["p"]),
                    dx=Lx / Nx, dy=Ly / Ny)
if comm.rank == 0:
    print(f"done in {elapsed:.1f}s -> {fname}")
