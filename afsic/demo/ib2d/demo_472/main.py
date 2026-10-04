"""IB2d example "Flow Around Cylinder" (matIB2d
``Examples/Example_Channel_Flow/Example_Flow_Around_Cylinder``) in AFSI.

An `arb_ext_force` example in a long strip: 2273 softly pinned target points
(``k = 1e5``) trace the two walls (``y = 0.025`` and ``y = 0.225``,
``x in [0.05, 0.949]``) and a small cylinder inside the strip, in a
``1 x 0.25`` periodic box (``mu = 0.001``, ``rho = 1``, ``512 x 128``,
``dt = 5e-5``).

The example's ``please_Compute_External_Forcing.m`` (ported in ``ib2d_input/``)
imposes a channel-wide parabolic jet over ``x in [0.1, 0.16]``,
``y in [0.035, 0.215]``:

    Fx -= kStiff (uX - uX_tar),  uX_tar = -uMax 5 tanh(t) (Ly/2 + w/2 - y)(Ly/2 - w/2 - y)

with ``kStiff = 1e4``, ``W = 0.2``, ``u_max = 250``.

> The shipped run is ``T = 1.25`` = 25 000 steps on 512 x 128; the default here is
> ``T = 0.02`` (400 steps) with ``print_dump = 50`` so that both codes finish in
> minutes.  ``TFINAL``/``PRINT_DUMP`` restore the shipped values.

Run::

    conda activate afsi-dolfinx
    python run_reference.py             # pyIB2d reference
    python main.py                      # AFSI
    python compare.py

Output (``OUTPUT_PATH``, default ``./plot``): XDMF fields +
``afsi_result.npz`` (t, X markers, u/p/f on the IB2d grid).
"""
import os
import sys
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

# the external forcing module is the one shipped with the example: pyIB2d
# imports it from the run directory, so the reference and this port run the
# very same forcing code
sys.path.insert(0, IB2D_EXAMPLE)
import please_Compute_External_Forcing as ext_mod      # noqa: E402

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
# the pyIB2d driver advances while (current_time < Tfinal) with
# current_time += dt, so rounding noise can add one last step; replicate it so
# that the dump schedules of the two codes end on the same step
_n, _t = 0, 0.0
while _t < T:
    _t += dt
    _n += 1
num_steps = _n
out_dir = os.environ.get("OUTPUT_PATH", os.path.join(_here, "plot"))
os.makedirs(out_dir, exist_ok=True)

GRAD_DIV = float(os.environ.get("GRAD_DIV", "0"))

X_ib2d = ex["X"]
Nb = X_ib2d.shape[0]
sp = ex.get("springs")                     # optional linear springs
bm = ex.get("beams")                       # optional invariant beams
tg = ex["targets"]                         # pinned target points
ds_ib2d = Lx / (2 * Nx)   # IB2d sets ds = Lx/(2Nx)

assert "mass" not in ex and "muscles" not in ex and "d_springs" not in ex \
    and "noninv_beams" not in ex, \
    "this template implements springs, invariant beams, target points and the "\
    "arbitrary external force only"

if comm.rank == 0:
    desc = [f"{len(tg['ids'])} target points (k={tg['k'][0]:.2g})"]
    if sp is not None:
        desc.insert(0, f"{len(sp['conn'])} springs (k={sp['k'][0]:.2g})")
    if bm is not None:
        desc.append(f"{len(bm['p1'])} invariant beams (kb={bm['kb'][0]:.2g})")
    print(f"IB2d example : {IB2D_EXAMPLE}")
    print(f"rho={rho} mu={mu} box={Lx}x{Ly} grid={Nx}x{Ny} dt={dt} T={T} "
          f"steps={num_steps} ds={ds_ib2d}")
    print(f"structure    : {Nb} markers: " + ", ".join(desc))
    print(f"forcing      : {os.path.basename(ext_mod.__file__)}: "
          f"x in [{getattr(ext_mod, 'X_MIN', '?')}, "
          f"{getattr(ext_mod, 'X_MAX', '?')}], "
          f"y in [{getattr(ext_mod, 'Y_MIN', '?')}, "
          f"{getattr(ext_mod, 'Y_MAX', '?')}], "
          f"k={getattr(ext_mod, 'K_STIFF', '?')}, "
          f"uMax={getattr(ext_mod, 'U_MAX', '?')}")
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
cells = np.column_stack([np.arange(Nb - 1), np.arange(Nb - 1) + 1]).astype(np.int64)
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


tgt_ids, tgt_k = tg["ids"], tg["k"]
tgt_anchor = X_ib2d[tgt_ids].copy()      # pinned: update_target = 0
if sp is not None:
    sp_a, sp_b = sp["conn"][:, 0], sp["conn"][:, 1]
    sp_k, sp_L = sp["k"], sp["L"]
if bm is not None:
    bm_p1, bm_p2, bm_p3 = bm["p1"], bm["p2"], bm["p3"]
    bm_k, bm_C = bm["kb"], bm["C"]


def _min_image(d, L):
    """IB2d's minimum-image branch for a leader-follower difference."""
    over = np.abs(d) > L / 2
    d[over] = np.sign(d[over]) * (L - np.sign(d[over]) * d[over])
    return d


def _beam_boundary(Xp, Xq, Xr, L):
    """IB2d ``check_If_Beam_Points_Pass_Through_Boundary`` along one axis."""
    dX_pq, dX_qr = Xp - Xq, Xq - Xr
    over = np.abs(dX_pq) > 5 * ds_ib2d
    Xp = np.where(over, Xp + np.where(dX_pq < 0, L, -L), Xp)
    over = np.abs(dX_qr) > 5 * ds_ib2d
    Xr = np.where(over, Xr + np.where(dX_qr < 0, -L, L), Xr)
    return Xp, Xq, Xr


def spring_forces(Xm, Fm):
    """Linear springs, F = k (L - L0) dhat (IB2d give_Me_Spring_...)."""
    d = Xm[sp_b] - Xm[sp_a]
    d[:, 0] = _min_image(d[:, 0], Lx)
    d[:, 1] = _min_image(d[:, 1], Ly)
    L = np.hypot(d[:, 0], d[:, 1])
    sF = sp_k * (L - sp_L) / L
    np.add.at(Fm[:, 0], sp_a, sF * d[:, 0])
    np.add.at(Fm[:, 1], sp_a, sF * d[:, 1])
    np.add.at(Fm[:, 0], sp_b, -sF * d[:, 0])
    np.add.at(Fm[:, 1], sp_b, -sF * d[:, 1])


def beam_forces(Xm, Fm):
    """Invariant (cross-product) beams, as in demo_446/455."""
    Xp = Xm[bm_p1].copy()
    Xq = Xm[bm_p2].copy()
    Xr = Xm[bm_p3].copy()
    Xp, Xq, Xr = _beam_boundary(Xp, Xq, Xr, Lx)
    for c in (0, 1):
        Xp[:, c], Xq[:, c], Xr[:, c] = _beam_boundary(Xp[:, c], Xq[:, c],
                                                      Xr[:, c], Ly)
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


# ---------------------------------------------------------------------------
# Arbitrary external force (IB2d's arb_ext_force): the same ported module the
# pyIB2d reference imports, evaluated at the Q1 node dofs of this mesh (which
# coincide with IB2d's collocated fluid grid) and added to solver.f (Q2)
# through its Q2 interpolant
# ---------------------------------------------------------------------------
GRID_INFO = [Nx, Ny, Lx, Ly, Lx / Nx, Ly / Ny, int(P["supp"]), Nb, ds_ib2d]
XNODES = np.arange(Nx) * (Lx / Nx)
YNODES = np.arange(Ny) * (Ly / Ny)
# map every dof of the Q1 velocity space onto the node grid (wrapped images of
# the periodic boundary included, so that the force field is periodic)
Qcoords = V_ib.tabulate_dof_coordinates()[:, :2]
gi = np.rint(Qcoords[:, 0] / (Lx / Nx)).astype(int) % Nx
gj = np.rint(Qcoords[:, 1] / (Ly / Ny)).astype(int) % Ny
assert np.allclose(Qcoords[:, 0], np.rint(Qcoords[:, 0] / (Lx / Nx)) * (Lx / Nx),
                   atol=1e-12)
f_ext1 = fem.Function(V_ib, name="f_ext_nodal")
f_ext2 = fem.Function(solver.V, name="f_ext_q2")
ext_state = {"first": 1, "inds": None}


def add_external_force(t):
    """Add (Fx, Fy) from please_Compute_External_Forcing to solver.f."""
    u_ib.interpolate(solver.u_n)
    u_dof = u_ib.x.array.reshape(-1, 2)
    uX = np.zeros((Ny, Nx))
    uY = np.zeros((Ny, Nx))
    uX[gj, gi] = u_dof[:, 0]
    uY[gj, gi] = u_dof[:, 1]
    Fx, Fy, first, inds = ext_mod.please_Compute_External_Forcing(
        dt, t, XNODES, YNODES, GRID_INFO, uX, uY, ext_state["first"],
        ext_state["inds"])
    ext_state["first"], ext_state["inds"] = first, inds
    f_ext1.x.array[:] = np.column_stack([Fx[gj, gi], Fy[gj, gi]]).ravel()
    f_ext2.interpolate(f_ext1)          # Q1 nodal field -> Q2 interpolant
    solver.f.x.array[:] += f_ext2.x.array[:]


def spread_forces_at_half():
    Fm = np.zeros((Nb, 2))
    Xm = marker_array(X_h)
    if sp is not None:
        spring_forces(Xm, Fm)
    if bm is not None:
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
    f_io.interpolate(solver.f)
    f = np.zeros((Ny, Nx, 2))
    f[gj, gi] = f_io.x.array.reshape(-1, 2)
    return u, p, f


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


hist = {"t": [0.0], "X": [marker_array(X).copy()],
        "Xa": [tgt_anchor.copy()]}
u0, p0, f0 = grid_fields()
hist["u"], hist["p"], hist["f"] = [u0], [p0], [f0]
write_output(0.0)
if comm.rank == 0:
    print(f"initial      : max|X_target - anchor|="
          f"{np.abs(marker_array(X)[tgt_ids] - tgt_anchor).max():.2e}")

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

    # (3) target forces at X_h, spread with the 4-point kernel
    spread_forces_at_half()

    # (3b) arbitrary external force, added to the IB force density
    add_external_force(ctime)

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
        ug, pg, fg = grid_fields()
        hist["u"].append(ug)
        hist["p"].append(pg)
        hist["f"].append(fg)
        write_output(t)
        if comm.rank == 0:
            Xm = hist["X"][-1]
            print(f"step {step+1:5d}  t={t:.5f}  "
                  f"|X_tgt-Xa|max="
                  f"{np.abs(Xm[tgt_ids] - tgt_anchor).max():.2e}  "
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
                    f=np.array(hist["f"]),
                    dx=Lx / Nx, dy=Ly / Ny, dt=dt, nx=Nx, ny=Ny)
if comm.rank == 0:
    print(f"done in {elapsed:.1f}s -> {fname}")
