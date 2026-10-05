"""IB2d example "Poroelastic Rubberband -- Restart Protocol" (matIB2d
``Examples/Example_Test_Restart_Protocol/Poroelastic_Rubberband_Restart``) in
AFSI, including the AFSI restart mechanism.

Like demo_453/demo_454: a 64-point rubberband (linear springs ``k = 1e6``,
rest length 0 -- it wants to collapse) where every marker is **poroelastic**
(``.poroelastic`` file, coefficient ``c = 2.5e5``): at every move the point
gets an extra displacement

    dX_pe = dt * F_spring / (mu * c)

(``F_spring`` = spring force only, evaluated at ``X_h``; the half-step move
uses the previous step's value -- the driver's ``F_Poro`` bookkeeping).
Unlike demo_454 this example has ``porous_media = 0`` (no ``.porous`` file):
the point creeps through the fluid but does not slip along the curve normal.
The box is 1 x 1 periodic (``mu = 0.01``, ``rho = 1``, 32 x 32,
``dt = 1e-4``, ``T = 0.2`` as shipped).

The upstream example exists to demonstrate IB2d's *restart protocol*
(``Restart_Flag = 1`` plus a hand-written ``help_Me_Restart.m``); see
``readme.md`` for the protocol and its quirks.  AFSI restart is env driven:

* ``RESTART_AT=<k>``   - dump the complete solver state after ``k`` steps;
* ``RESTART_DUMP=<f>`` - where to write that state (``.npz``);
* ``RESTART_FROM=<f>`` - warm-start from a state written by ``RESTART_DUMP``.

The dumped state is the *entire* integrator state (marker dofs ``X``, the
fluid velocity dofs ``u_n = u_``, the pressure dofs ``p_`` and the poroelastic
force memory ``F_Poro_prev`` of the previous step), so a run ``0..t1`` followed
by a restart ``t1..T`` reproduces the single continuous run ``0..T`` to
round-off (the linear solves are not bitwise reproducible across processes, so
the residual sits at the run-to-run noise level; ``compare.py`` measures both).

Run::

    conda activate afsi-dolfinx
    python main.py                      # full example (T = 0.2 s, 32x32)
    TFINAL=0.01 python main.py          # smoke test
    RESTART_AT=1001 RESTART_DUMP=state.npz TFINAL=0.1001 python main.py
    RESTART_FROM=state.npz python main.py

Output (``OUTPUT_PATH``, default ``./plot``): XDMF fields +
``afsi_result.npz`` (t, X markers, u/p on the IB2d grid).
"""
import os
import time

import numpy as np
from mpi4py import MPI

import basix.ufl
import dolfinx
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

# restart protocol (see module docstring)
RESTART_FROM = os.environ.get("RESTART_FROM") or None
RESTART_DUMP = os.environ.get("RESTART_DUMP") or None
RESTART_AT = int(os.environ.get("RESTART_AT", "0"))
if RESTART_DUMP and RESTART_AT <= 0:            # default: mid-run frame
    RESTART_AT = min(print_dump * max(1, (num_steps // (2 * print_dump))) + 1,
                     num_steps)
assert RESTART_AT <= num_steps, "RESTART_AT beyond Tfinal"
assert not (RESTART_FROM and RESTART_DUMP), \
    "RESTART_FROM and RESTART_DUMP are mutually exclusive"

# gamma = 0 leaks area for a zero-rest-length closed ring (the Q2/Q1 pressure
# space cannot represent the transmural jump, see demo_444/demo_447/demo_454);
# the poroelastic example ships no stabilisation, so the validated value of
# demo_454 is the default
GRAD_DIV = float(os.environ.get("GRAD_DIV", "2.5"))

X_ib2d = ex["X"]
Nb = X_ib2d.shape[0]
sp = ex["springs"]
por = ex.get("porous")            # this example has porous_media = 0
pe = ex["poroelastic"]
assert "mass" not in ex and "beams" not in ex and "noninv_beams" not in ex
assert "targets" not in ex and "muscles" not in ex and "tracers" not in ex
ds_ib2d = min(Lx / (2 * Nx), Ly / (2 * Ny))

if comm.rank == 0:
    print(f"IB2d example : {IB2D_EXAMPLE}")
    print(f"rho={rho} mu={mu} box={Lx}x{Ly} grid={Nx}x{Ny} dt={dt} T={T} "
          f"steps={num_steps} ds={ds_ib2d}")
    print(f"structure    : {Nb} markers, {len(sp['conn'])} springs, "
          f"{len(pe['ids'])} poroelastic points (c={pe['c'][0]:.2g}), "
          f"porous media off, grad-div gamma={GRAD_DIV:g}")
    if RESTART_FROM:
        print(f"restart      : from {RESTART_FROM}")
    if RESTART_DUMP:
        print(f"restart dump : step {RESTART_AT} (t={RESTART_AT * dt:g}) "
              f"-> {RESTART_DUMP}")

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


def write_fun(fn, Xm):
    tmp = np.empty_like(Xm)
    tmp[dof_of_ib2d] = Xm
    fn.x.array[:] = tmp.ravel()
    fn.x.scatter_forward()


# ---------------------------------------------------------------------------
# Forces: springs at X_h, plus the poroelastic displacement
# ---------------------------------------------------------------------------
sp_a, sp_b = sp["conn"][:, 0], sp["conn"][:, 1]
sp_k, sp_L = sp["k"], sp["L"]
pe_ids = pe["ids"]
pe_c = pe["c"]
F_Poro_prev = np.zeros((len(pe_ids), 2))    # driver keeps F_Poro across steps


def poroelastic_disp(F_Poro):
    """Extra marker displacement dX = dt * F_Poro / (mu c) (driver:
    ``please_Move_Lagrangian_Point_Positions``' poroelastic branch)."""
    return F_Poro / (mu * pe_c[:, None])


def spring_forces(Xm, Fm):
    d = Xm[sp_b] - Xm[sp_a]
    L = np.hypot(d[:, 0], d[:, 1])
    sF = sp_k * (L - sp_L) / L
    np.add.at(Fm[:, 0], sp_a, sF * d[:, 0])
    np.add.at(Fm[:, 1], sp_a, sF * d[:, 1])
    np.add.at(Fm[:, 0], sp_b, -sF * d[:, 0])
    np.add.at(Fm[:, 1], sp_b, -sF * d[:, 1])


def spread_forces_at_half():
    """Spring forces at X_h, spread with the 4-point kernel.  Returns the
    raw per-marker forces (un-scaled by ds), which the poroelastic block
    needs (driver: ``F_Lag``)."""
    Fm = np.zeros((Nb, 2))
    spring_forces(marker_array(X_h), Fm)
    Fdof = np.empty_like(Fm)
    Fdof[dof_of_ib2d] = Fm
    F_s.x.array[:] = Fdof.ravel()
    F_s.x.array[:] *= ds_ib2d
    ib.evaluate_current_points(X_h._cpp_object)
    ib.solid_to_fluid(f_ib._cpp_object, F_s._cpp_object)
    solver.periodic_fold(f_ib)
    if f_ib is not solver.f:
        solver.f.interpolate(f_ib)
    return Fm


# porous slip is off in this example (porous_media = 0); the implementation
# below is kept so that the module also runs a porous+poroelastic variant
p_ids = por["ids"] if por is not None else np.zeros(0, dtype=np.int64)
p_kappa = (por["kappa"] * float(os.environ.get("POROUS_SCALE", "1"))
           if por is not None else np.zeros(0))
p_stencil = por["stencil"] if por is not None else np.zeros(0, dtype=np.int64)
Np = len(p_ids)


def porous_slip_update(Xm, Fm, dt):
    """pyIB2d ``please_Compute_Porous_Slip_Velocity`` + driver update."""
    xL, yL = Xm[p_ids, 0], Xm[p_ids, 1]
    xL_s = np.zeros(Np)
    yL_s = np.zeros(Np)
    for i in range(Np):
        c = p_stencil[i]
        if c == -2:
            xL_s[i] = (-25 / 12 * xL[i] + 4 * xL[i + 1] - 3 * xL[i + 2]
                       + 4 / 3 * xL[i + 3] - 0.25 * xL[i + 4]) / ds_ib2d
            yL_s[i] = (-25 / 12 * yL[i] + 4 * yL[i + 1] - 3 * yL[i + 2]
                       + 4 / 3 * yL[i + 3] - 0.25 * yL[i + 4]) / ds_ib2d
        elif c == -1:
            xL_s[i] = (-0.25 * xL[i - 1] - 5 / 6 * xL[i] + 1.5 * xL[i + 1]
                       - 0.5 * xL[i + 2] + 1 / 12 * xL[i + 3]) / ds_ib2d
            yL_s[i] = (-0.25 * yL[i - 1] - 5 / 6 * yL[i] + 1.5 * yL[i + 1]
                       - 0.5 * yL[i + 2] + 1 / 12 * yL[i + 3]) / ds_ib2d
        elif c == 0:
            xL_s[i] = (1 / 12 * xL[i - 2] - 2 / 3 * xL[i - 1] + 2 / 3 * xL[i + 1]
                       - 1 / 12 * xL[i + 2]) / ds_ib2d
            yL_s[i] = (1 / 12 * yL[i - 2] - 2 / 3 * yL[i - 1] + 2 / 3 * yL[i + 1]
                       - 1 / 12 * yL[i + 2]) / ds_ib2d
        elif c == 1:
            xL_s[i] = (-1 / 12 * xL[i - 3] + 0.5 * xL[i - 2] - 1.5 * xL[i - 1]
                       + 5 / 6 * xL[i] + 0.25 * xL[i + 1]) / ds_ib2d
            yL_s[i] = (-1 / 12 * yL[i - 3] + 0.5 * yL[i - 2] - 1.5 * yL[i - 1]
                       + 5 / 6 * yL[i] + 0.25 * yL[i + 1]) / ds_ib2d
        elif c == 2:
            xL_s[i] = (0.25 * xL[i - 4] - 4 / 3 * xL[i - 3] + 3 * xL[i - 2]
                       - 4 * xL[i - 1] + 25 / 12 * xL[i]) / ds_ib2d
            yL_s[i] = (0.25 * yL[i - 4] - 4 / 3 * yL[i - 3] + 3 * yL[i - 2]
                       - 4 * yL[i - 1] + 25 / 12 * yL[i]) / ds_ib2d

    sqrtN = np.hypot(xL_s, yL_s)
    nX = yL_s / sqrtN
    nY = -xL_s / sqrtN
    Up_X = -p_kappa * Fm[p_ids, 0] * nX / sqrtN
    Up_Y = -p_kappa * Fm[p_ids, 1] * nY / sqrtN
    Xm[p_ids, 0] -= dt * Up_X * nX
    Xm[p_ids, 1] -= dt * Up_Y * nY


# sanity check of the vectorised spring force against a plain loop
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
if comm.rank == 0:
    print(f"force checks (perturbed config): springs {err_sp:.2e}")
assert err_sp < 1e-12

# ---------------------------------------------------------------------------
# Output sampling on the IB2d grid
# ---------------------------------------------------------------------------
Vdof = Vc.tabulate_dof_coordinates()[:, :2]
iu = np.rint(Vdof[:, 0] / (Lx / Nx)).astype(int)
ju = np.rint(Vdof[:, 1] / (Ly / Ny)).astype(int)
on_grid = lambda Q, i, j: (np.isclose(Q[:, 0], i * (Lx / Nx))    # noqa: E731
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


def ring_area(Xm):
    x, y = Xm[:, 0], Xm[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def save_restart(path, step_done, t):
    """Write the complete integrator state (float64 npz)."""
    np.savez(path, step=np.int64(step_done), t=np.float64(t),
             dt=np.float64(dt),
             X_dof=X.x.array.copy(),
             u_dof=solver.u_n.x.array.copy(),
             p_dof=solver.p_.x.array.copy(),
             F_Poro_prev=F_Poro_prev.copy(),
             X_markers=marker_array(X).copy())
    if comm.rank == 0:
        print(f"restart dump : step {step_done} (t={t:g}) -> {path}", flush=True)


# ---------------------------------------------------------------------------
# Restart: warm-start the whole integrator state, or start from rest
# ---------------------------------------------------------------------------
if RESTART_FROM:
    st = np.load(RESTART_FROM)
    step0 = int(st["step"])
    t0 = float(st["t"])
    assert abs(t0 - step0 * dt) < 1e-12, "restart state dt/step mismatch"
    assert step0 < num_steps, "restart state already at/after Tfinal"
    X.x.array[:] = st["X_dof"]
    X.x.scatter_forward()
    solver.u_n.x.array[:] = st["u_dof"]
    solver.u_.x.array[:] = st["u_dof"]
    solver.p_.x.array[:] = st["p_dof"]
    F_Poro_prev = st["F_Poro_prev"].copy()
    if comm.rank == 0:
        print(f"restart      : resumed at step {step0} (t={t0:g}), "
              f"{num_steps - step0} steps left", flush=True)
else:
    step0, t0 = 0, 0.0

hist = {"t": [t0], "X": [marker_array(X).copy()]}
u0, p0 = grid_fields()
hist["u"], hist["p"] = [u0], [p0]
write_output(t0)
if comm.rank == 0 and step0 == 0:
    print(f"initial      : area={ring_area(X_ib2d):.6f}")

# ---------------------------------------------------------------------------
# Time loop -- Peskin (2002) / IB2d IBM_Driver ordering (+ poroelastic
# displacement; this example has no porous slip)
# ---------------------------------------------------------------------------
tic = time.perf_counter()
for step in range(step0, num_steps):
    # (1) marker half-step X_h = X + dt/2 (U^n(X) + F_Poro_prev/(mu c))
    ib.evaluate_current_points(X._cpp_object)
    if step == 0:
        X_h.x.array[:] = X.x.array
    else:
        ib.fluid_to_solid(ib_velocity(solver.u_n)._cpp_object, U_s._cpp_object)
        X_h.x.array[:] = X.x.array + 0.5 * dt * U_s.x.array
        Xh = marker_array(X_h).copy()
        Xh[pe_ids] += 0.5 * dt * poroelastic_disp(F_Poro_prev)
        write_fun(X_h, Xh)

    # (2) forces at X_h, spread with the 4-point kernel; the poroelastic
    #     points also store the (spring-only) force for their displacement
    Fm = spread_forces_at_half()
    F_Poro = Fm[pe_ids].copy()

    # (3) two-stage fluid solve
    solver.solve_one_step()

    # (4) markers: X += dt U_h(X_h), then the poroelastic displacement
    ib.fluid_to_solid(ib_velocity(solver.u_h)._cpp_object, U_s._cpp_object)
    Xm = marker_array(X).copy()
    Xm += dt * U_s.x.array.reshape(-1, 2)[dof_of_ib2d]
    Xm[pe_ids] += dt * poroelastic_disp(F_Poro)

    # (5) porous slip (driver's porous branch, after the marker update):
    #     off in this example
    if por is not None:
        porous_slip_update(Xm, Fm, dt)
    Xm[:, 0] %= Lx
    Xm[:, 1] %= Ly
    write_fun(X, Xm)
    F_Poro_prev = F_Poro

    # Dump schedule: the same steps on which the pyIB2d driver writes its VTK
    # frames (see demo_478/compare.py); t below is the true time of the state.
    dumped = (step % print_dump == 0 and step >= print_dump) \
        or (step + 1 == num_steps)
    if dumped:
        t = (step + 1) * dt
        hist["t"].append(t)
        Xm = marker_array(X).copy()
        hist["X"].append(Xm)
        uf, pf = grid_fields()
        hist["u"].append(uf)
        hist["p"].append(pf)
        write_output(t)
        if comm.rank == 0:
            print(f"step {step+1:6d}  t={t:.4f}  area={ring_area(Xm):.6f}  "
                  f"wall={time.perf_counter()-tic:.1f}s", flush=True)

    if RESTART_DUMP and step + 1 == RESTART_AT:
        save_restart(RESTART_DUMP, step + 1, (step + 1) * dt)

elapsed = time.perf_counter() - tic
xf.close()
xs.close()
fname = os.path.join(out_dir, "afsi_result.npz")
np.savez_compressed(fname,
                    t=np.array(hist["t"]), X=np.array(hist["X"]),
                    u=np.array(hist["u"]), p=np.array(hist["p"]),
                    dx=Lx / Nx, dy=Ly / Ny)
if comm.rank == 0:
    print(f"done in {elapsed:.1f}s -> {fname}")
