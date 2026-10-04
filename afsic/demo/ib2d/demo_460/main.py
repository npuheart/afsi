"""IB2d example "Pulsing Heart" (matIB2d ``Examples/Example_Pulsing_Heart/32x32``)
in AFSI.

A 384-marker **heart outline** fills much of a 1 x 1 periodic box of fluid
(mu = 0.5, rho = 1, 32 x 32).  There are no springs and no beams: every marker
is a target point with k = 1e8, and its anchor is *prescribed in time* by the
example's ``update_Target_Point_Positions.m``, which interpolates between the
two heart shapes stored in ``All_Positions.txt`` (columns ``xP1, yP1`` = the
relaxed heart, ``xP2, yP2`` = the contracted heart) with a two-phase
piecewise-polynomial schedule of period ``tP1 + tP2 = 0.05`` s:

    phase 1->2, t <= tP1:   X = P1 + g(t/tP1) (P2 - P1)
    phase 2->1, t >  tP1:   X = P2 + g((t-tP1)/tP2) (P1 - P2)
    g(s) = a s^2  (s<0.1) | c s^3+d s^2+g s+h  (0.1<=s<0.9) | -b(s-1)^2+1 (s>=0.9)

(the three polynomial pieces match in value at the knots, not in slope --
reproduced verbatim).  The heart therefore contracts and expands twice over
the example's ``Tfinal = 0.1`` s (dt = 1e-5, 10000 steps): a prescribed-motion
pump, and the reference case for **file-driven, large-amplitude moving target
points** -- the anchors travel up to 0.119 m (3.8 h at h = 1/32) and, unlike
demo_459, the anchors of every marker move on their own trajectory, so the
port must interpolate all 384 of them.

The example is one of the ``matIB2d`` ones: its ``.target`` file is 1-based,
so ``ib2d_io.load_example(..., index_base=1)`` rebases the identifiers, and
``run_reference.py`` writes a rebased copy of the ``.target`` file (plus
``All_Positions.txt`` and the python port of the .m driver) for the pyIB2d
reference.

AFSI ingredients: periodic Q2/Q1 Taylor-Hood + Peskin (2002) two-stage scheme
(``PeskinRK2Solver``), a zero-stiffness closed FE chain carrying the 384 marker
dofs, the IB2d target-point force ``k (X_anchor - X^{n+1/2})`` at the half-step
positions, and the anchors advanced by the same recurrence pyIB2d applies, at
the same place in the step.

Run::

    conda activate afsi-dolfinx
    python main.py                      # full example (T = 0.1 s, 32x32)
    TFINAL=0.001 python main.py         # smoke test

Output (``OUTPUT_PATH``, default ``./plot``): XDMF fields +
``afsi_result.npz`` (t, X markers, X_anchor, u/p on the IB2d grid) with the
same layout as the pyIB2d reference from ``run_reference.py``.
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

GRAD_DIV = float(os.environ.get("GRAD_DIV", "0"))
FIELD_EVERY = int(os.environ.get("FIELD_EVERY", "50"))   # u/p sampling (steps)

# the moving-target law (port of update_Target_Point_Positions.m): a two-phase
# piecewise polynomial in the time within the 0.05 s period that interpolates
# the marker anchors between the two shapes of All_Positions.txt
TP1 = 0.025                       # contraction time
TP2 = 0.025                       # expansion time
PERIOD = TP1 + TP2
A_COEF = 2.739726027397260
B_COEF = 2.739726027397260
C_COEF = -2.029426686960933
D_COEF = 3.044140030441400
G_COEF = -0.015220700152207
H_COEF = 0.000253678335870
_shapes = np.loadtxt(os.path.join(IB2D_EXAMPLE, "All_Positions.txt"), ndmin=2)
P1_IB2D = _shapes[:, 0:2].copy()          # relaxed heart (All_Positions cols 1-2)
P2_IB2D = _shapes[:, 2:4].copy()          # contracted heart (cols 3-4)

X_ib2d = ex["X"]
Nb = X_ib2d.shape[0]
tg = ex["targets"]
ds_ib2d = min(Lx / (2 * Nx), Ly / (2 * Ny))

assert "springs" not in ex and "beams" not in ex and "noninv_beams" not in ex \
    and "mass" not in ex and "muscles" not in ex and "d_springs" not in ex, \
    "this demo implements the moving-target example only"
assert tg["ids"].size == Nb and np.array_equal(tg["ids"], np.arange(Nb)), \
    "the anchor law of this example moves every marker of the heart outline"
# the .target file stores the same shape as All_Positions.txt column pair 1
assert np.allclose(tg["ids"], np.arange(Nb)) and np.allclose(X_ib2d, P1_IB2D), \
    "heart.vertex and All_Positions.txt columns 1-2 disagree"

if comm.rank == 0:
    print(f"IB2d example : {IB2D_EXAMPLE}")
    print(f"rho={rho} mu={mu} box={Lx}x{Ly} grid={Nx}x{Ny} dt={dt} T={T} "
          f"steps={num_steps} ds={ds_ib2d}")
    print(f"structure    : {Nb} markers, {len(tg['ids'])} target points "
          f"(k={tg['k'][0]:.2g}), anchor stroke "
          f"{np.abs(P2_IB2D - P1_IB2D).max():.4f} m = "
          f"{np.abs(P2_IB2D - P1_IB2D).max() / (Lx / Nx):.2f} h, "
          f"period {PERIOD:g} s")
    print(f"grad-div     : gamma={GRAD_DIV:g}")

# ---------------------------------------------------------------------------
# Fluid: periodic Taylor-Hood, Peskin two-stage scheme
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
# Structure: zero-stiffness FE chain carrying the marker dofs
# ---------------------------------------------------------------------------
cells = np.vstack([np.column_stack([np.arange(Nb - 1), np.arange(Nb - 1) + 1]),
                   [[Nb - 1, 0]]]).astype(np.int64)   # closed chain
coord_el = basix.ufl.element("Lagrange", "interval", 1, shape=(2,))
structure = dolfinx.mesh.create_mesh(comm, cells, coord_el, X_ib2d)
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


# optional uniform rescaling of the target stiffnesses for the stiff-penalty
# study documented in readme.md (1 = the example's k)
tgt_ids, tgt_k = tg["ids"], tg["k"] * float(os.environ.get("K_SCALE", "1"))
tgt_anchor = X_ib2d[tgt_ids].copy()      # moving: updated by update_targets()


def _phase(s):
    """The piecewise polynomial g(s) of the .m file (value-continuous, not
    slope-continuous, at s = 0.1 and 0.9)."""
    if s < 0.1:
        return A_COEF * s ** 2
    if s < 0.9:
        return C_COEF * s ** 3 + D_COEF * s ** 2 + G_COEF * s + H_COEF
    return -B_COEF * (s - 1.0) ** 2 + 1.0


def update_targets(t):
    """Port of the example's ``update_Target_Point_Positions.m``.

    Called once per step with the time at the *start* of the step, i.e. at the
    same place pyIB2d calls the ``.py`` twin of the ``.m`` file (after the
    half-step marker positions are formed and before the forces are spread).
    The law is stateless: the anchors depend only on the phase of
    ``t mod period`` -- the heart pulses between the two prescribed shapes.
    """
    tt = np.remainder(t, PERIOD)
    if tt <= TP1:
        g = _phase(tt / TP1)
        tgt_anchor[:] = P1_IB2D[tgt_ids] + g * (P2_IB2D[tgt_ids] - P1_IB2D[tgt_ids])
    else:
        g = _phase((tt - TP1) / TP2)
        tgt_anchor[:] = P2_IB2D[tgt_ids] + g * (P1_IB2D[tgt_ids] - P2_IB2D[tgt_ids])


def target_forces(Xm, Fm):
    """IB2d target-point force densities ``k (X_anchor - X)``.

    IB2d folds the difference of a point that wrapped around the periodic box
    back into the box (``|d| > L/2`` branch of
    ``give_Me_Target_Lagrangian_Force_Densities``); reproduced here verbatim
    (it never fires in this example: the anchors track the markers to within
    a few 1e-6 m).
    """
    d = tgt_anchor - Xm[tgt_ids]
    for c, L in enumerate((Lx, Ly)):
        over = np.abs(d[:, c]) > L / 2
        sgn = np.sign(d[over, c])
        d[over, c] = sgn * (L - sgn * d[over, c])
    np.add.at(Fm, tgt_ids, tgt_k[:, None] * d)


def spread_forces_at_half():
    Fm = np.zeros((Nb, 2))
    target_forces(marker_array(X_h), Fm)
    Fdof = np.empty_like(Fm)
    Fdof[dof_of_ib2d] = Fm
    F_s.x.array[:] = Fdof.ravel()
    F_s.x.array[:] *= ds_ib2d
    ib.evaluate_current_points(X_h._cpp_object)
    ib.solid_to_fluid(f_ib._cpp_object, F_s._cpp_object)
    solver.periodic_fold(f_ib)
    if f_ib is not solver.f:
        solver.f.interpolate(f_ib)


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


def area(Xm):
    """Enclosed area of the closed marker chain (shoelace)."""
    x, y = Xm[:, 0], Xm[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


hist = {"t": [0.0], "X": [marker_array(X).copy()],
        "Xa": [tgt_anchor.copy()]}
u0, p0 = grid_fields()
hist["u"], hist["p"] = [u0], [p0]
write_output(0.0)
if comm.rank == 0:
    print(f"initial      : enclosed area={area(X_ib2d):.6f}, "
          f"max|X-X_anchor|="
          f"{np.abs(marker_array(X) - tgt_anchor).max():.2e}")

# ---------------------------------------------------------------------------
# Time loop -- Peskin (2002) / IB2d IBM_Driver ordering
# ---------------------------------------------------------------------------
tic = time.perf_counter()
ctime = 0.0
for step in range(num_steps):
    # (1) half-step Lagrangian positions X_h = X + dt/2 U^n(X)
    ib.evaluate_current_points(X._cpp_object)
    if step == 0:
        X_h.x.array[:] = X.x.array
    else:
        ib.fluid_to_solid(ib_velocity(solver.u_n)._cpp_object, U_s._cpp_object)
        X_h.x.array[:] = X.x.array + 0.5 * dt * U_s.x.array

    # (2) move the target points (the driver does this right after X_h)
    update_targets(ctime)

    # (3) target forces at X_h, spread with the 4-point kernel
    spread_forces_at_half()

    # (4) two-stage fluid solve: u_h, u^{n+1}, p^{n+1/2}
    solver.solve_one_step()

    # (5) markers
    ib.fluid_to_solid(ib_velocity(solver.u_h)._cpp_object, U_s._cpp_object)
    X.x.array[:] += dt * U_s.x.array
    X.x.scatter_forward()

    ctime += dt
    # Dump schedule: exactly the steps on which the pyIB2d driver writes its
    # VTK frames.  Its loop dumps when its step counter ``cter`` (0-based,
    # checked before it is incremented) is a positive multiple of
    # ``print_dump``, i.e. at the *end* of step ``cter + 1``; mirroring that
    # here makes AFSI frame k and reference frame k the same step of the
    # simulation (the driver's own time stamp for frame k, ``k*print_dump*dt``,
    # is one step behind the state it wrote -- an upstream labelling quirk).
    # t below is the true time of the dumped state.
    dumped = (step % print_dump == 0 and step >= print_dump) \
        or (step + 1 == num_steps)
    if dumped:
        t = (step + 1) * dt
        hist["t"].append(t)
        hist["X"].append(marker_array(X).copy())
        hist["Xa"].append(tgt_anchor.copy())
        ug, pg = grid_fields()
        hist["u"].append(ug)
        hist["p"].append(pg)
        write_output(t)
        if comm.rank == 0:
            Xm = hist["X"][-1]
            print(f"step {step+1:5d}  t={t:.5f}  "
                  f"Xc=({Xm[:, 0].mean():.4f},{Xm[:, 1].mean():.4f})  "
                  f"area={area(Xm):.6f}  "
                  f"|X-Xa|max={np.abs(Xm - tgt_anchor).max():.2e}  "
                  f"umax={np.abs(ug).max():.3f}  "
                  f"wall={time.perf_counter()-tic:.1f}s", flush=True)

elapsed = time.perf_counter() - tic
xf.close()
xs.close()
fname = os.path.join(out_dir, "afsi_result.npz")
np.savez_compressed(fname,
                    t=np.array(hist["t"]), X=np.array(hist["X"]),
                    Xa=np.array(hist["Xa"]),
                    u=np.array(hist["u"]), p=np.array(hist["p"]),
                    dx=Lx / Nx, dy=Ly / Ny)
if comm.rank == 0:
    print(f"done in {elapsed:.1f}s -> {fname}")
