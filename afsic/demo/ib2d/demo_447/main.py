"""IB2d example "Rubberband with Damped Springs"
(pyIB2d ``Examples/Rubberband_with_Damped_Springs``) in AFSI.

A closed rubberband (64 points on a circle of radius 0.2 around (0.5, 0.5))
made of 64 *damped* springs (kappa = 2.5e4, rest length 0, damping b = 5) is
released in a 1 x 1 periodic box of viscous fluid (mu = 0.01, rho = 1).  The
band contracts and relaxes -- the damped-spring variant of the classic
rubberband example (the damping bleeds off the elastic oscillation).

Structure ingredients (all read from ``ib2d_input/``, pyIB2d conventions,
0-based indices):

* **damped springs**: 64 rows ``p1 p2 kappa L_rest b`` (0-1, 1-2, ..., 63-0),
  kappa = 2.5e4, L_rest = 0, b = 5.

Fluid / coupling: identical pipeline to the other ib2d demos -- periodic
Q2/Q1 Taylor-Hood, Peskin (2002) two-stage scheme (``PeskinRK2Solver``),
Peskin 4-point kernel, Lagrangian weight ds = min(Lx/2Nx, Ly/2Ny),
``IBMesh(order=1)`` so the mesh vertices are IB2d's Cartesian grid.  The
spring forces are evaluated in numpy with IB2d's own formula at the
half-step positions X^{n+1/2}, using the step-start positions X^n for the
damping velocities (a zero-stiffness FE ring only carries the marker dofs):

    sF = kappa (L - L_rest) * dhat - b * dV_leader,   dV = (X_h - X_h_prev)/dt

where ``X_h_prev`` is the previous step's half-step configuration (the
driver's ``xLag_P`` bookkeeping, IBM_Driver.py line ~792).

Run::

    conda activate afsi-dolfinx
    python main.py                      # full example (T = 1.5 s, 32x32)
    TFINAL=0.01 python main.py          # smoke test

Output (``OUTPUT_PATH``, default ``./plot``): XDMF fields +
``afsi_result_g<GRAD_DIV>.npz`` (t, X markers, u/p on the IB2d grid).
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

# ---------------------------------------------------------------------------
# Parameters: read from the IB2d example, optionally overridden
# ---------------------------------------------------------------------------
ex = ib2d_io.load_example(IB2D_EXAMPLE)
P = ex["params"]
rho, mu = float(P["rho"]), float(P["mu"])
Lx, Ly = float(P["Lx"]), float(P["Ly"])
Nx = int(os.environ.get("NX", P["Nx"]))
Ny = int(os.environ.get("NY", round(Nx * Ly / Lx)))
dt = float(os.environ.get("DT", P["dt"]))
T = float(os.environ.get("TFINAL", P["Tfinal"]))
print_dump = int(os.environ.get("PRINT_DUMP", P["print_dump"]))
assert Nx % 2 == 0 and Ny % 2 == 0, "Nx, Ny must be even (as in IB2d)"
num_steps = int(round(T / dt))
out_dir = os.environ.get("OUTPUT_PATH", os.path.join(_here, "plot"))
os.makedirs(out_dir, exist_ok=True)

GRAD_DIV = float(os.environ.get("GRAD_DIV", "0"))

X_ib2d = ex["X"]                               # (Nb, 2) IB2d ordering
Nb = X_ib2d.shape[0]
dsp = ex["d_springs"]                          # damped springs
assert "springs" not in ex and "targets" not in ex and "beams" not in ex, \
    "this demo implements the damped-spring example only"
ds_ib2d = min(Lx / (2 * Nx), Ly / (2 * Ny))   # IB2d's Lagrangian weight
if comm.rank == 0:
    print(f"IB2d example : {IB2D_EXAMPLE}")
    print(f"rho={rho} mu={mu} box={Lx}x{Ly} grid={Nx}x{Ny} dt={dt} T={T} "
          f"steps={num_steps} ds={ds_ib2d}")
    print(f"structure    : {Nb} markers, {len(dsp['conn'])} damped springs "
          f"(kappa={dsp['k'][0]:.2g}, L_rest={dsp['L'][0]:g}, b={dsp['b'][0]:g})")

# ---------------------------------------------------------------------------
# Fluid: periodic Taylor-Hood, Peskin two-stage scheme (as in demo_444/445)
# ---------------------------------------------------------------------------
mesh = dolfinx.mesh.create_rectangle(
    comm, ((0.0, 0.0), (Lx, Ly)), (Nx, Ny),
    cell_type=CellType.quadrilateral, ghost_mode=GhostMode.shared_facet)
solver = PeskinRK2Solver(mesh, (0.0, Lx, 0.0, Ly), dt, rho, mu, grad_div=GRAD_DIV)
Vc = solver.V

ibmesh = IBMesh(0.0, Lx, 0.0, Ly, Nx, Ny, 1)   # order 1: vertices = IB2d grid
V_ib = fem.functionspace(mesh, basix.ufl.element("Lagrange", "quadrilateral", 1,
                                                 shape=(2,)))
u_ib, f_ib = fem.Function(V_ib), fem.Function(V_ib)
coords_bg = fem.Function(V_ib)
coords_bg.interpolate(lambda x: np.array([x[0], x[1]]))
ibmesh.build_map(coords_bg._cpp_object)
ib = IBInterpolation(ibmesh)


def ib_velocity(u):
    """Eulerian velocity on the IB (vertex) grid."""
    u_ib.interpolate(u)
    return u_ib


# ---------------------------------------------------------------------------
# Structure: I-point FE ring carrying the marker dofs (zero stiffness)
# ---------------------------------------------------------------------------
cells = dsp["conn"].astype(np.int64)           # the closed ring 0-1-...-63-0
coord_el = basix.ufl.element("Lagrange", "interval", 1, shape=(2,))
structure = dolfinx.mesh.create_mesh(comm, cells, coord_el, X_ib2d)
Vs = fem.functionspace(structure, coord_el)
assert Vs.dofmap.index_map.size_local == Nb, \
    "the structure FE mesh must have one dof per IB2d marker"

X = fem.Function(Vs, name="position")          # current configuration X(t)
X.interpolate(lambda x: np.array([x[0], x[1]]))
X_h = fem.Function(Vs, name="position_half")   # force configuration X^{n+1/2}
X_h.x.array[:] = X.x.array
U_s = fem.Function(Vs, name="velocity")
F_s = fem.Function(Vs, name="force")

# map FE dof -> IB2d point index (and back) by coordinate matching
dof_rows = Vs.tabulate_dof_coordinates()[:, :2]
dof_of_ib2d = np.array([int(np.argmin(np.linalg.norm(dof_rows - p, axis=1)))
                        for p in X_ib2d])
assert len(np.unique(dof_of_ib2d)) == Nb, "dof matching is not injective"
assert np.allclose(dof_rows[dof_of_ib2d], X_ib2d, atol=1e-12), \
    "dof coordinates do not recover the IB2d marker list"


def marker_array(fn):
    """Function on the structure space in IB2d marker order."""
    return fn.x.array.reshape(-1, 2)[dof_of_ib2d]


# ---------------------------------------------------------------------------
# Damped-spring forces (IB2d formula, evaluated at X_h with X^n for dV)
# ---------------------------------------------------------------------------
dsp_a, dsp_b = dsp["conn"][:, 0], dsp["conn"][:, 1]
dsp_k, dsp_L, dsp_beta = dsp["k"], dsp["L"], dsp["b"]


def _wrap(d, Lbox):
    """IB2d's nearest-image correction: |d| > Lx/2 -> d +- Lbox."""
    return np.where(np.abs(d) > Lx / 2, np.sign(d) * (Lbox - np.sign(d) * d), d)


def damped_spring_forces(Xm, Xm_prev, Fm):
    """IB2d ``give_Me_Damped_Springs_Lagrangian_Force_Densities`` at X_h.

    sF = kappa (L - L_rest) * dhat - b * dV_leader, applied as +sF to the
    leader node and -sF to the follower node.  ``dV`` is the *leader's*
    displacement velocity between the previous half step and the current one,
    (X_h - X_h_prev)/dt (the driver's ``xLag_P`` bookkeeping).
    """
    dx = _wrap(Xm[dsp_b, 0] - Xm[dsp_a, 0], Lx)
    dy = _wrap(Xm[dsp_b, 1] - Xm[dsp_a, 1], Ly)
    L = np.hypot(dx, dy)
    Ls = np.where(L > 0, L, 1.0)               # guard the measure-zero L = 0
    dVx = _wrap(Xm[dsp_a, 0] - Xm_prev[dsp_a, 0], Lx) / dt
    dVy = _wrap(Xm[dsp_a, 1] - Xm_prev[dsp_a, 1], Ly) / dt
    sFx = dsp_k * (L - dsp_L) * dx / Ls - dsp_beta * dVx
    sFy = dsp_k * (L - dsp_L) * dy / Ls - dsp_beta * dVy
    np.add.at(Fm[:, 0], dsp_a, sFx)
    np.add.at(Fm[:, 1], dsp_a, sFy)
    np.add.at(Fm[:, 0], dsp_b, -sFx)
    np.add.at(Fm[:, 1], dsp_b, -sFy)


def spread_forces_at_half(Xh_prev):
    """Total Lagrangian force at X_h (damped springs) x ds."""
    Fm = np.zeros((Nb, 2))
    damped_spring_forces(marker_array(X_h), Xh_prev, Fm)
    Fdof = np.empty_like(Fm)
    Fdof[dof_of_ib2d] = Fm
    F_s.x.array[:] = Fdof.ravel()
    F_s.x.array[:] *= ds_ib2d
    ib.evaluate_current_points(X_h._cpp_object)
    ib.solid_to_fluid(f_ib._cpp_object, F_s._cpp_object)
    solver.periodic_fold(f_ib)
    if f_ib is not solver.f:
        solver.f.interpolate(f_ib)


# sanity check of the vectorised force against a plain per-spring loop (at a
# perturbed configuration with a random "previous" state)
rng = np.random.default_rng(0)
Xp = X_ib2d + 1e-3 * rng.standard_normal(X_ib2d.shape)
Xq = Xp - 1e-3 * rng.standard_normal(X_ib2d.shape)
F_vec = np.zeros((Nb, 2))
damped_spring_forces(Xp, Xq, F_vec)
F_loop = np.zeros((Nb, 2))
for a, b, kk, LL, bb in zip(dsp_a, dsp_b, dsp_k, dsp_L, dsp_beta):
    dx = _wrap(np.array([Xp[b, 0] - Xp[a, 0]]), Lx)[0]
    dy = _wrap(np.array([Xp[b, 1] - Xp[a, 1]]), Ly)[0]
    Lg = np.hypot(dx, dy)
    dVx = _wrap(np.array([Xp[a, 0] - Xq[a, 0]]), Lx)[0] / dt
    dVy = _wrap(np.array([Xp[a, 1] - Xq[a, 1]]), Ly)[0] / dt
    sFx = kk * (Lg - LL) * dx / Lg - bb * dVx
    sFy = kk * (Lg - LL) * dy / Lg - bb * dVy
    F_loop[a] += (sFx, sFy)
    F_loop[b] -= (sFx, sFy)
err = np.abs(F_vec - F_loop).max() / np.abs(F_loop).max()
if comm.rank == 0:
    print(f"damped-spring check (perturbed config): rel. max err = {err:.2e} "
          f"(scale {np.abs(F_loop).max():.3g})")
assert err < 1e-12

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


def ring_area(Xm):
    """Shoelace area of the closed ring."""
    x, y = Xm[:, 0], Xm[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - y * np.roll(x, -1))


hist = {"t": [0.0], "X": [marker_array(X).copy()]}
u0, p0 = grid_fields()
hist["u"], hist["p"] = [u0], [p0]
write_output(0.0)

# ---------------------------------------------------------------------------
# Time loop -- Peskin (2002) / IB2d IBM_Driver ordering
# ---------------------------------------------------------------------------
tic = time.perf_counter()
Xh_prev = X_ib2d.copy()             # driver's xLag_P (previous half step)
for step in range(num_steps):
    # (1) half-step Lagrangian positions X_h = X + dt/2 U^n(X)
    ib.evaluate_current_points(X._cpp_object)
    if step == 0:
        X_h.x.array[:] = X.x.array           # u = 0 initially (as IB2d)
    else:
        ib.fluid_to_solid(ib_velocity(solver.u_n)._cpp_object, U_s._cpp_object)
        X_h.x.array[:] = X.x.array + 0.5 * dt * U_s.x.array

    # (2) damped-spring forces at X_h, spread with the 4-point kernel
    spread_forces_at_half(Xh_prev)
    Xh_prev = marker_array(X_h).copy()

    # (3) two-stage fluid solve: u_h, u^{n+1}, p^{n+1/2}
    solver.solve_one_step()

    # (4) X^{n+1} = X^n + dt U_h(X_h)
    ib.fluid_to_solid(ib_velocity(solver.u_h)._cpp_object, U_s._cpp_object)
    X.x.array[:] += dt * U_s.x.array
    X.x.scatter_forward()

    t = (step + 1) * dt
    if (step + 1) % print_dump == 0 or step + 1 == num_steps:
        Xm = marker_array(X).copy()
        hist["t"].append(t)
        hist["X"].append(Xm)
        ug, pg = grid_fields()
        hist["u"].append(ug)
        hist["p"].append(pg)
        write_output(t)
        if comm.rank == 0:
            print(f"step {step+1:6d}  t={t:.4f}  area={ring_area(Xm):.6f}  "
                  f"|F|max={np.abs(F_s.x.array).max():.3e}  "
                  f"wall={time.perf_counter()-tic:.1f}s", flush=True)

xf.close()
xs.close()
np.savez_compressed(os.path.join(out_dir, f"afsi_result_g{GRAD_DIV:g}.npz"),
                    t=np.array(hist["t"]), X=np.array(hist["X"]),
                    u=np.array(hist["u"]), p=np.array(hist["p"]),
                    n_markers=Nb, dx=Lx / Nx, dy=Ly / Ny)
if comm.rank == 0:
    print(f"done in {time.perf_counter()-tic:.1f}s -> "
          f"{out_dir}/afsi_result_g{GRAD_DIV:g}.npz")
