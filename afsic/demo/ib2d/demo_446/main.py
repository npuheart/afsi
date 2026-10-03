"""IB2d example "Wobbly Beam" (pyIB2d ``Examples/Wobbly_Beam``) in AFSI.

A flexible beam ("bump" curve from (0.25,0.5) to (0.75,0.5), both ends held by
stiff target points) is immersed in a 1 x 1 periodic box of viscous fluid
(mu = 0.1, rho = 1) and released; the beam wobbles and sheds vorticity.

Structure ingredients (all read from ``ib2d_input/``, pyIB2d conventions,
0-based indices):

* **invariant beams**: 62 torsional springs over consecutive point triples,
  kappa_beam = 7.5e9 (IB2d's cross-product formulation, reproduced exactly;
  reference cross product C = 0),
* **target points**: ids 0 and 63 (the two ends of the beam), k = 2e8,
  tethered to their initial positions -- the beam is pinned at both ends.

Fluid / coupling: identical pipeline to the other ib2d demos -- periodic
Q2/Q1 Taylor-Hood, Peskin (2002) two-stage scheme (``PeskinRK2Solver``),
Peskin 4-point kernel, Lagrangian weight ds = min(Lx/2Nx, Ly/2Ny),
``IBMesh(order=1)`` so the mesh vertices are IB2d's Cartesian grid.  The
structure forces are evaluated in numpy with IB2d's own formulas at the
half-step positions X^{n+1/2} (a zero-stiffness FE chain only carries the
marker degrees of freedom; the beams/targets contribute no FE energy).

Run::

    conda activate afsi-dolfinx
    python main.py                      # full example (T = 0.05 s, 32x32)
    TFINAL=0.001 python main.py         # smoke test
    GRAD_DIV=100 python main.py         # optional grad-div stabilisation

Output (``OUTPUT_PATH``, default ``./plot``): XDMF fields +
``afsi_result_g<GRAD_DIV>.npz`` (t, X markers, u/p on the IB2d grid) with the
same layout as the pyIB2d reference produced by ``run_reference.py``.
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
bm = ex["beams"]                               # invariant beams
tgt = ex.get("targets", dict(ids=np.zeros(0, np.int64), k=np.zeros(0)))
assert "springs" not in ex and "noninv_beams" not in ex, \
    "this demo implements the beam + target example only"
# IB2d's Lagrangian weight (IBM_Driver.py: ds = min(Lx/(2Nx), Ly/(2Ny)))
ds_ib2d = min(Lx / (2 * Nx), Ly / (2 * Ny))
if comm.rank == 0:
    print(f"IB2d example : {IB2D_EXAMPLE}")
    print(f"rho={rho} mu={mu} box={Lx}x{Ly} grid={Nx}x{Ny} dt={dt} T={T} "
          f"steps={num_steps} ds={ds_ib2d}")
    print(f"structure    : {Nb} markers, {len(bm['p1'])} beams "
          f"(k={bm['kb'][0]:.2g}), {len(tgt['ids'])} targets "
          f"(k={tgt['k'][0]:.2g} at ids {tgt['ids'].tolist()})")

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
# Structure: zero-stiffness FE chain carrying the marker dofs
# ---------------------------------------------------------------------------
# The beams/targets are evaluated with IB2d's numpy formulas only; the FE mesh
# exists so that the markers have dofs (X update and force spreading).
cells = np.column_stack([np.arange(Nb - 1), np.arange(Nb - 1) + 1]).astype(np.int64)
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
# Beam + target forces (IB2d formulas, evaluated at X_h)
# ---------------------------------------------------------------------------
bm_p1, bm_p2, bm_p3 = bm["p1"], bm["p2"], bm["p3"]
bm_k, bm_C = bm["kb"], bm["C"]
tgt_ids, tgt_k = tgt["ids"], tgt["k"]
tgt_anchor = X_ib2d[tgt_ids].copy() if len(tgt_ids) else np.zeros((0, 2))


def beam_forces(Xm, Fm):
    """Invariant beam force densities at markers Xm, added to Fm.

    IB2d ``give_Me_Beam_Lagrangian_Force_Densities`` (pyIB2d): with the cross
    product  S = (xr-xq)(yq-yp) - (yr-yq)(xq-xp)  and K = k (S - C),

        node p:  fx += K (yr-yq)      fy -= K (xr-xq)
        node q:  fx += K [(yq-yp)+(yr-yq)]   fy -= K [(xr-xq)+(xq-xp)]
        node r:  fx += K (yq-yp)      fy -= K (xq-xp)
    """
    Xp, Xq, Xr = Xm[bm_p1], Xm[bm_p2], Xm[bm_p3]
    S = ((Xr[:, 0] - Xq[:, 0]) * (Xq[:, 1] - Xp[:, 1])
         - (Xr[:, 1] - Xq[:, 1]) * (Xq[:, 0] - Xp[:, 0]))
    K = bm_k * (S - bm_C)
    np.add.at(Fm[:, 0], bm_p1, K * (Xr[:, 1] - Xq[:, 1]))
    np.add.at(Fm[:, 1], bm_p1, -K * (Xr[:, 0] - Xq[:, 0]))
    np.add.at(Fm[:, 0], bm_p2, K * ((Xq[:, 1] - Xp[:, 1])
                                    + (Xr[:, 1] - Xq[:, 1])))
    np.add.at(Fm[:, 1], bm_p2, -K * ((Xr[:, 0] - Xq[:, 0])
                                     + (Xq[:, 0] - Xp[:, 0])))
    np.add.at(Fm[:, 0], bm_p3, K * (Xq[:, 1] - Xp[:, 1]))
    np.add.at(Fm[:, 1], bm_p3, -K * (Xq[:, 0] - Xp[:, 0]))


def target_forces(Xm, Fm):
    """Target-point forces (tethered to the initial positions)."""
    if len(tgt_ids):
        d = tgt_anchor - Xm[tgt_ids]
        d -= np.round(d / np.array([Lx, Ly])) * np.array([Lx, Ly])   # min. image
        np.add.at(Fm, tgt_ids, tgt_k[:, None] * d)


def extra_forces(Xm, Fm):
    """Total structure force densities at markers Xm, added to Fm."""
    beam_forces(Xm, Fm)
    target_forces(Xm, Fm)


def spread_forces_at_half():
    """Total Lagrangian force at X_h (beams + targets) x ds."""
    Fm = np.zeros((Nb, 2))
    extra_forces(marker_array(X_h), Fm)
    Fdof = np.empty_like(Fm)
    Fdof[dof_of_ib2d] = Fm
    F_s.x.array[:] = Fdof.ravel()
    F_s.x.array[:] *= ds_ib2d
    ib.evaluate_current_points(X_h._cpp_object)
    ib.solid_to_fluid(f_ib._cpp_object, F_s._cpp_object)
    solver.periodic_fold(f_ib)
    if f_ib is not solver.f:
        solver.f.interpolate(f_ib)


# sanity check of the vectorised beam force against a plain per-beam loop
rng = np.random.default_rng(0)
Xp = X_ib2d + 1e-3 * rng.standard_normal(X_ib2d.shape)
F_vec = np.zeros((Nb, 2))
beam_forces(Xp, F_vec)
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
err = np.abs(F_vec - F_loop).max() / np.abs(F_loop).max()
if comm.rank == 0:
    print(f"beam-force check (perturbed config): rel. max err = {err:.2e} "
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


hist = {"t": [0.0], "X": [marker_array(X).copy()]}
u0, p0 = grid_fields()
hist["u"], hist["p"] = [u0], [p0]
write_output(0.0)
mid = Nb // 2

# ---------------------------------------------------------------------------
# Time loop -- Peskin (2002) / IB2d IBM_Driver ordering
# ---------------------------------------------------------------------------
tic = time.perf_counter()
for step in range(num_steps):
    # (1) half-step Lagrangian positions X_h = X + dt/2 U^n(X)
    ib.evaluate_current_points(X._cpp_object)
    if step == 0:
        X_h.x.array[:] = X.x.array           # u = 0 initially (as IB2d)
    else:
        ib.fluid_to_solid(ib_velocity(solver.u_n)._cpp_object, U_s._cpp_object)
        X_h.x.array[:] = X.x.array + 0.5 * dt * U_s.x.array

    # (2) beam + target forces at X_h, spread with the 4-point kernel
    spread_forces_at_half()

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
            print(f"step {step+1:6d}  t={t:.5f}  y_mid={Xm[mid, 1]:.6f}  "
                  f"|F|max={np.abs(F_s.x.array).max():.3e}  "
                  f"wall={time.perf_counter()-tic:.1f}s", flush=True)

xf.close()
xs.close()
np.savez_compressed(os.path.join(out_dir, f"afsi_result_g{GRAD_DIV:g}.npz"),
                    t=np.array(hist["t"]), X=np.array(hist["X"]),
                    u=np.array(hist["u"]), p=np.array(hist["p"]),
                    n_markers=Nb, mid_index=mid,
                    dx=Lx / Nx, dy=Ly / Ny)
if comm.rank == 0:
    print(f"done in {time.perf_counter()-tic:.1f}s -> "
          f"{out_dir}/afsi_result_g{GRAD_DIV:g}.npz")
