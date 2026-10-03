"""IB2d example "Wobbly Non-Invariant Beam" (pyIB2d ``Examples/Wobbly_NonInv_Beam``)
in AFSI.

The *non-invariant* sibling of demo_446: the same elastic arch (from
(0.25, 0.5) to (0.75, 0.5), apex y ~ 0.625) pinned at both ends, but the
torsional springs are now stored with a **reference second difference**
``C = (p1 + p3 - 2 p2)`` taken at generation time (here C = 0 for every
triple, so the beam is released from a stressed configuration); kappa_beam =
1e10.  Box 1 x 1, mu = 0.1, rho = 1, 32x32 grid, dt = 1e-5 (as shipped).

Structure (0-based, read from ``ib2d_input/``):

* **non-invariant beams**: 62 triples with F = -k r on the outer nodes and
  +2 k r on the middle node, r = (X_p + X_r - 2 X_q) - C   (IB2d
  ``give_Me_nonInv_Beam_Lagrangian_Force_Densities``),
* **target points**: ids 0 and 63 (beam ends), k = 2e8.

Fluid / coupling: identical pipeline to the other ib2d demos (periodic Q2/Q1
Taylor-Hood + Peskin two-stage + 4-point kernel + zero-stiffness FE chain for
the marker dofs).

Run::

    conda activate afsi-dolfinx
    python main.py                      # full example (T = 0.05 s)
    TFINAL=0.001 python main.py         # smoke test

Output: ``plot/afsi_result_g<GRAD_DIV>.npz`` + XDMF, same layout as the
pyIB2d reference from ``run_reference.py``.
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
nb = ex["noninv_beams"]                        # non-invariant beams
tgt = ex.get("targets", dict(ids=np.zeros(0, np.int64), k=np.zeros(0)))
assert "springs" not in ex and "beams" not in ex, \
    "this demo implements the non-invariant beam + target example only"
ds_ib2d = min(Lx / (2 * Nx), Ly / (2 * Ny))
if comm.rank == 0:
    print(f"IB2d example : {IB2D_EXAMPLE}")
    print(f"rho={rho} mu={mu} box={Lx}x{Ly} grid={Nx}x{Ny} dt={dt} T={T} "
          f"steps={num_steps} ds={ds_ib2d}")
    print(f"structure    : {Nb} markers, {len(nb['p1'])} non-invariant beams "
          f"(k={nb['kb'][0]:.2g}), {len(tgt['ids'])} targets "
          f"(k={tgt['k'][0]:.2g} at ids {tgt['ids'].tolist()})")

# ---------------------------------------------------------------------------
# Fluid + structure (identical to demo_446)
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


# ---------------------------------------------------------------------------
# Non-invariant beam + target forces (IB2d formulas at X_h)
# ---------------------------------------------------------------------------
nb_p1, nb_p2, nb_p3 = nb["p1"], nb["p2"], nb["p3"]
nb_k, nb_C = nb["kb"], nb["C"]
tgt_ids, tgt_k = tgt["ids"], tgt["k"]
tgt_anchor = X_ib2d[tgt_ids].copy() if len(tgt_ids) else np.zeros((0, 2))


def noninv_beam_forces(Xm, Fm):
    """r = (X_p + X_r - 2 X_q) - C;  F_q += 2 k r,  F_p/-= k r,  F_r -= k r."""
    r = (Xm[nb_p1] + Xm[nb_p3] - 2.0 * Xm[nb_p2]) - nb_C
    np.add.at(Fm, nb_p2, 2.0 * nb_k[:, None] * r)
    np.add.at(Fm, nb_p1, -nb_k[:, None] * r)
    np.add.at(Fm, nb_p3, -nb_k[:, None] * r)


def target_forces(Xm, Fm):
    if len(tgt_ids):
        d = tgt_anchor - Xm[tgt_ids]
        d -= np.round(d / np.array([Lx, Ly])) * np.array([Lx, Ly])
        np.add.at(Fm, tgt_ids, tgt_k[:, None] * d)


def spread_forces_at_half():
    Fm = np.zeros((Nb, 2))
    noninv_beam_forces(marker_array(X_h), Fm)
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


# sanity check of the vectorised force against a plain per-beam loop
rng = np.random.default_rng(0)
Xp = X_ib2d + 1e-3 * rng.standard_normal(X_ib2d.shape)
F_vec = np.zeros((Nb, 2))
noninv_beam_forces(Xp, F_vec)
F_loop = np.zeros((Nb, 2))
for i1, i2, i3, kk, Cc in zip(nb_p1, nb_p2, nb_p3, nb_k, nb_C):
    r = Xp[i1] + Xp[i3] - 2.0 * Xp[i2] - Cc
    F_loop[i2] += 2.0 * kk * r
    F_loop[i1] -= kk * r
    F_loop[i3] -= kk * r
err = np.abs(F_vec - F_loop).max() / np.abs(F_loop).max()
if comm.rank == 0:
    print(f"nonInv-beam check (perturbed config): rel. max err = {err:.2e} "
          f"(scale {np.abs(F_loop).max():.3g})")
assert err < 1e-12

# ---------------------------------------------------------------------------
# Output sampling on the IB2d grid (as demo_445/446)
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

tic = time.perf_counter()
for step in range(num_steps):
    ib.evaluate_current_points(X._cpp_object)
    if step == 0:
        X_h.x.array[:] = X.x.array
    else:
        ib.fluid_to_solid(ib_velocity(solver.u_n)._cpp_object, U_s._cpp_object)
        X_h.x.array[:] = X.x.array + 0.5 * dt * U_s.x.array

    spread_forces_at_half()

    solver.solve_one_step()

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
                    n_markers=Nb, mid_index=mid, dx=Lx / Nx, dy=Ly / Ny)
if comm.rank == 0:
    print(f"done in {time.perf_counter()-tic:.1f}s -> "
          f"{out_dir}/afsi_result_g{GRAD_DIV:g}.npz")
