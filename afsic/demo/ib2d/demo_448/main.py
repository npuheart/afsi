"""IB2d example "HeartTube Muscle" (pyIB2d ``Examples/HeartTube_Muscle``) in AFSI.

A side-view model of a heart tube: two parallel elastic walls (y = 2 and
y = 3, x in [1, 4], 155 points each) pinned at all four corners, connected by
153 *Hill force-velocity + length-tension muscle* bands (one per interior
point pair).  A traveling activation wave (10 Hz, square profile) squeezes
the bands sequentially -> peristaltic pumping of the surrounding fluid
(mu = 0.1, rho = 1, 5 x 5 periodic box).

Structure ingredients (all read from ``ib2d_input/``, pyIB2d conventions,
0-based indices):

* **springs**: the two wall chains (308 edges, kappa = 1e7, L_rest = ds),
* **invariant beams**: 306 triples along the walls (kappa_beam = 7.5e7),
* **target points**: the four corners (ids 0, 154, 155, 309), k = 1e6,
* **FV_LT muscles**: 153 bands ``i -- i+155`` (Fmax = 1e5, L_opt = 1,
  Hill a = 0.25, b = 4, length-tension SK = 0.3) driven by the *example's
  own* activation ``give_Muscle_Activation`` (imported from
  ``ib2d_input/``): a square wave of width (1/10 of the activation region)
  traveling along the tube at 10 Hz.

Fluid / coupling: identical pipeline to the other ib2d demos -- periodic
Q2/Q1 Taylor-Hood, Peskin (2002) two-stage scheme (``PeskinRK2Solver``),
Peskin 4-point kernel, Lagrangian weight ds = min(Lx/2Nx, Ly/2Ny),
``IBMesh(order=1)`` so the mesh vertices are IB2d's Cartesian grid.  All
structure forces are evaluated in numpy with IB2d's own formulas at the
half-step positions X^{n+1/2}; the muscle contraction speed uses the previous
half-step (the driver's ``xLag_P`` bookkeeping).  A zero-stiffness FE graph
(one cell per IB2d spring) only carries the marker degrees of freedom.

Run::

    conda activate afsi-dolfinx
    python main.py                      # full example (T = 0.25 s, 128x128)
    TFINAL=0.002 python main.py         # smoke test

Output (``OUTPUT_PATH``, default ``./plot``): XDMF fields +
``afsi_result_g<GRAD_DIV>.npz`` (t, X markers, u/p on the IB2d grid).
"""
import os
import sys
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
sys.path.insert(0, IB2D_EXAMPLE)
from give_Muscle_Activation import give_Muscle_Activation   # noqa: E402

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
sp = ex["springs"]                             # wall springs
bm = ex["beams"]                               # invariant beams
tgt = ex["targets"]                            # pinned corners
mus = ex["muscles"]                            # FV_LT muscle bands
ds_ib2d = min(Lx / (2 * Nx), Ly / (2 * Ny))   # IB2d's Lagrangian weight
if comm.rank == 0:
    print(f"IB2d example : {IB2D_EXAMPLE}")
    print(f"rho={rho} mu={mu} box={Lx}x{Ly} grid={Nx}x{Ny} dt={dt} T={T} "
          f"steps={num_steps} ds={ds_ib2d}")
    print(f"structure    : {Nb} markers, {len(sp['conn'])} springs "
          f"(k={sp['k'][0]:.2g}), {len(bm['p1'])} beams, "
          f"{len(mus['conn'])} muscles (Fmax={mus['Fmax'][0]:.2g}), "
          f"{len(tgt['ids'])} targets")

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
# Structure: zero-stiffness FE graph (one cell per IB2d spring) for the dofs
# ---------------------------------------------------------------------------
coord_el = basix.ufl.element("Lagrange", "interval", 1, shape=(2,))
structure = dolfinx.mesh.create_mesh(comm, sp["conn"].astype(np.int64),
                                     coord_el, X_ib2d)
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
# Structure forces (IB2d formulas, evaluated at X_h)
# ---------------------------------------------------------------------------
sp_a, sp_b = sp["conn"][:, 0], sp["conn"][:, 1]
sp_k, sp_L = sp["k"], sp["L"]
bm_p1, bm_p2, bm_p3 = bm["p1"], bm["p2"], bm["p3"]
bm_k, bm_C = bm["kb"], bm["C"]
tgt_ids, tgt_k = tgt["ids"], tgt["k"]
tgt_anchor = X_ib2d[tgt_ids].copy()
ms_a, ms_b = mus["conn"][:, 0], mus["conn"][:, 1]
ms_LFO, ms_SK = mus["LFO"], mus["SK"]
ms_ha, ms_hb, ms_Fmax = mus["a"], mus["b"], mus["Fmax"]


def spring_forces(Xm, Fm):
    """Linear springs, F = k (L - L0) dhat  (IB2d give_Me_Spring_...)."""
    d = Xm[sp_b] - Xm[sp_a]
    L = np.hypot(d[:, 0], d[:, 1])
    sF = sp_k * (L - sp_L) / L
    np.add.at(Fm[:, 0], sp_a, sF * d[:, 0])
    np.add.at(Fm[:, 1], sp_a, sF * d[:, 1])
    np.add.at(Fm[:, 0], sp_b, -sF * d[:, 0])
    np.add.at(Fm[:, 1], sp_b, -sF * d[:, 1])


def beam_forces(Xm, Fm):
    """Invariant (cross-product) beams, as in demo_446."""
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
    """Target points (tethered to the initial positions)."""
    d = tgt_anchor - Xm[tgt_ids]
    d -= np.round(d / np.array([Lx, Ly])) * np.array([Lx, Ly])   # min. image
    np.add.at(Fm, tgt_ids, tgt_k[:, None] * d)


def muscle_forces(Xm, Xm_prev, t, Fm):
    """FV_LT muscle bands, IB2d ``give_Muscle_Force_Densities``.

    Fm_i = a_f(t, x_i) * Fmax * exp(-((Q-1)/SK)^2) * (1/Fmax)(b*Fmax-a*v)/(v+b)
    with Q = LF/L_opt and v = |LF(X_h) - LF(X_h_prev)|/dt; the activation is
    the example's own ``give_Muscle_Activation`` (traveling square wave).
    """
    dx = Xm[ms_b, 0] - Xm[ms_a, 0]
    dy = Xm[ms_b, 1] - Xm[ms_a, 1]
    LF = np.hypot(dx, dy)
    dx_P = Xm_prev[ms_b, 0] - Xm_prev[ms_a, 0]
    dy_P = Xm_prev[ms_b, 1] - Xm_prev[ms_a, 1]
    LF_P = np.hypot(dx_P, dy_P)
    v = np.abs(LF - LF_P) / dt
    # NB: the example's activation takes *1-D coordinate arrays* (the driver
    # stores xLag, yLag as separate 1-D arrays): xPt = leader x's, xLag = all x's
    Fmag = give_Muscle_Activation(v, LF, ms_LFO, ms_SK, ms_ha, ms_hb, ms_Fmax,
                                  t, Xm[ms_a, 0].copy(), Xm[:, 0].copy())
    mFx = Fmag * dx / LF
    mFy = Fmag * dy / LF
    np.add.at(Fm[:, 0], ms_a, mFx)
    np.add.at(Fm[:, 1], ms_a, mFy)
    np.add.at(Fm[:, 0], ms_b, -mFx)
    np.add.at(Fm[:, 1], ms_b, -mFy)


def spread_forces_at_half(Xh_prev, t):
    """Total Lagrangian force at X_h (springs+beams+targets+muscles) x ds."""
    Xm = marker_array(X_h)
    Fm = np.zeros((Nb, 2))
    spring_forces(Xm, Fm)
    beam_forces(Xm, Fm)
    target_forces(Xm, Fm)
    muscle_forces(Xm, Xh_prev, t, Fm)
    Fdof = np.empty_like(Fm)
    Fdof[dof_of_ib2d] = Fm
    F_s.x.array[:] = Fdof.ravel()
    F_s.x.array[:] *= ds_ib2d
    ib.evaluate_current_points(X_h._cpp_object)
    ib.solid_to_fluid(f_ib._cpp_object, F_s._cpp_object)
    solver.periodic_fold(f_ib)
    if f_ib is not solver.f:
        solver.f.interpolate(f_ib)


# sanity check of the vectorised spring/beam forces against plain loops
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
    print(f"force checks (perturbed config): springs rel. err {err_sp:.2e}, "
          f"beams rel. err {err_bm:.2e}")
assert err_sp < 1e-12 and err_bm < 1e-12

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


hist = {"t": [0.0], "X": [marker_array(X).copy()]}
u0, p0 = grid_fields()
hist["u"], hist["p"] = [u0], [p0]
write_output(0.0)

# ---------------------------------------------------------------------------
# Time loop -- Peskin (2002) / IB2d IBM_Driver ordering
# ---------------------------------------------------------------------------
Xh_prev = X_ib2d.copy()             # driver's xLag_P (previous half step)
tic = time.perf_counter()
for step in range(num_steps):
    # (1) half-step Lagrangian positions X_h = X + dt/2 U^n(X)
    ib.evaluate_current_points(X._cpp_object)
    if step == 0:
        X_h.x.array[:] = X.x.array           # u = 0 initially (as IB2d)
    else:
        ib.fluid_to_solid(ib_velocity(solver.u_n)._cpp_object, U_s._cpp_object)
        X_h.x.array[:] = X.x.array + 0.5 * dt * U_s.x.array

    # (2) structure forces at X_h, spread with the 4-point kernel; current
    #     time = step * dt (as the driver, before the update)
    spread_forces_at_half(Xh_prev, step * dt)
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
        width = np.hypot(Xm[155:309, 0] - Xm[1:155, 0],
                         Xm[155:309, 1] - Xm[1:155, 1]).mean()
        hist["t"].append(t)
        hist["X"].append(Xm)
        ug, pg = grid_fields()
        hist["u"].append(ug)
        hist["p"].append(pg)
        write_output(t)
        if comm.rank == 0:
            print(f"step {step+1:6d}  t={t:.4f}  mean_width={width:.5f}  "
                  f"|F|max={np.abs(F_s.x.array).max():.3e}  "
                  f"wall={time.perf_counter()-tic:.1f}s", flush=True)

xf.close()
xs.close()
np.savez_compressed(os.path.join(out_dir, f"afsi_result_g{GRAD_DIV:g}.npz"),
                    t=np.array(hist["t"]), X=np.array(hist["X"]),
                    u=np.array(hist["u"]), p=np.array(hist["p"]),
                    n_markers=Nb, half=155, dx=Lx / Nx, dy=Ly / Ny)
if comm.rank == 0:
    print(f"done in {time.perf_counter()-tic:.1f}s -> "
          f"{out_dir}/afsi_result_g{GRAD_DIV:g}.npz")
