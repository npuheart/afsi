"""IB2d example "128x128 Electromechanical Pumping w/ Calcium Dynamics"
(matIB2d ``Examples/Example_HeartTube/128x128_Electromechanical_Pumping_w_Ca_Dynamics``)
in AFSI.

The high-resolution member of the electromechanical-pumping family: a
310-marker closed tube (two 155-point walls, ``x in [1, 4]``, ``y = 1.25 +- 0.5``)
in a 5 x 2.5 periodic box (``mu = 1``, ``rho = 1``, ``128 x 64``, ``dt = 1e-5``,
``T = 4`` as shipped), with 463 springs (``k = 1e7``; the last 155 tie the two
walls together), 306 invariant beams (``kb = 7.5e7``) and 4 pinned target
points.

**Mechanism (``electro_phys = 1``).**  A-priori, exactly as in demo_588, the
example solves its ``Calcium_Dynamics.m`` and a calcium-driven 1-d
FitzHugh-Nagumo model; its potential history is then written into the
*stiffness* of the 155 cross-tube springs every step
(``springs_info(ePhys_Start:ePhys_End, 3) = (1.6e1 * v)^4``, spring rows
309..463).  The only differences from demo_588 are the higher resolution of the
electrophysiology (``numFHN = 1``, 155 tube points instead of 78) and of the
fluid grid.

pyIB2d does not implement ``electro_phys``; the reference run therefore drives
the identical modulation through its documented ``update_Springs`` hook, using
this demo's ``ib2d_input/update_Springs.py`` (the same ``FitzHugh_Nagumo_1d.py``
this port imports).

Run::

    conda activate afsi-dolfinx
    python run_reference.py                 # pyIB2d reference
    python main.py                          # AFSI
    python compare.py

The shipped setting is ``dt = 1e-5`` over ``T = 4`` s = 400 000 steps (and the
a-priori potential alone is ~0.5 GB); use ``TFINAL`` (and, for the reference,
``--tend``) to truncate the immersed-boundary run.  The calcium stimulus is
periodic in physical time, so truncating does not change its timing.

Output (``OUTPUT_PATH``, default ``./plot``): XDMF fields +
``afsi_result.npz`` (t, X markers, u/p on the IB2d grid, the applied electro
stiffness).
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
sys.path.insert(0, IB2D_EXAMPLE)
from FitzHugh_Nagumo_1d import FitzHugh_Nagumo_1d      # noqa: E402

comm = MPI.COMM_WORLD
assert comm.size == 1, "this demo is serial (IB coupling of afsic gathers on rank 0)"

# the matIB2d example files number the Lagrangian points from 1
ex = ib2d_io.load_example(IB2D_EXAMPLE, index_base=1)
P = ex["params"]
rho, mu = float(P["rho"]), float(P["mu"])
Lx, Ly = float(P["Lx"]), float(P["Ly"])
Nx = int(os.environ.get("NX", P["Nx"]))
Ny = int(os.environ.get("NY", round(Nx * Ly / Lx)))
h = Lx / Nx
dt = float(os.environ.get("DT", P["dt"]))
T = float(os.environ.get("TFINAL", P["Tfinal"]))
print_dump = int(os.environ.get("PRINT_DUMP", P["print_dump"]))
assert Nx % 2 == 0 and Ny % 2 == 0, "Nx, Ny must be even (as in IB2d)"
# the pyIB2d driver advances while (current_time < Tfinal) with
# current_time += dt, so rounding noise can add one last step; replicate it so
# that the two codes end on the same step
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
sp = ex["springs"]
bm = ex["beams"]
tg = ex["targets"]
ds_ib2d = min(Lx / (2 * Nx), Ly / (2 * Ny))
assert "muscles" not in ex and "mass" not in ex and "d_springs" not in ex \
    and "noninv_beams" not in ex and "tracers" not in ex \
    and "concentration" not in ex, "unexpected structure in this example"

# electrophysiology: computed a-priori.  The calcium stimulus is periodic in
# physical time (0.5 s), so shortening the FSI horizon only truncates both
# models and EPHYS_TFINAL simply follows TFINAL.
EPHYS_TFINAL = float(os.environ.get("EPHYS_TFINAL", T))
v_interp, EPHYS_START, EPHYS_END = FitzHugh_Nagumo_1d(dt, EPHYS_TFINAL)
assert v_interp.shape[0] >= num_steps, \
    f"electrophysiology history too short ({v_interp.shape[0]} < {num_steps}); "\
    f"raise EPHYS_TFINAL (currently {EPHYS_TFINAL:g})"
if comm.rank == 0:
    print(f"IB2d example : {IB2D_EXAMPLE}")
    print(f"rho={rho} mu={mu} box={Lx}x{Ly} grid={Nx}x{Ny} dt={dt} T={T} "
          f"steps={num_steps} ds={ds_ib2d}")
    print(f"structure    : {Nb} markers, {len(sp['conn'])} springs "
          f"(k={sp['k'][0]:.2g}), {len(bm['p1'])} beams "
          f"(kb={bm['kb'][0]:.2g}), {len(tg['ids'])} targets")
    print(f"electro      : FitzHugh-Nagumo 1-d over T={EPHYS_TFINAL:g} s -> "
          f"{v_interp.shape[0]} rows; spring rows {EPHYS_START}..{EPHYS_END}; "
          f"v in [{v_interp.min():.4f}, {v_interp.max():.4f}]")
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
# Structure: zero-stiffness FE graph carrying the marker dofs
# ---------------------------------------------------------------------------
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


sp_a, sp_b = sp["conn"][:, 0], sp["conn"][:, 1]
sp_k = sp["k"].copy()          # the electro block overwrites the stiffness
sp_L = sp["L"]
bm_p1, bm_p2, bm_p3 = bm["p1"], bm["p2"], bm["p3"]
bm_k, bm_C = bm["kb"], bm["C"]
tgt_ids, tgt_k = tg["ids"], tg["k"]
tgt_anchor = X_ib2d[tgt_ids].copy()


def spring_forces(Xm, Fm):
    """Linear springs, F = k (L - L0) dhat (IB2d give_Me_Spring_...)."""
    d = Xm[sp_b] - Xm[sp_a]
    d[:, 0] -= np.round(d[:, 0] / Lx) * Lx        # periodic minimum image
    d[:, 1] -= np.round(d[:, 1] / Ly) * Ly
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
    """IB2d target-point force densities k (X_anchor - X)."""
    d = tgt_anchor - Xm[tgt_ids]
    d -= np.round(d / np.array([Lx, Ly])) * np.array([Lx, Ly])
    np.add.at(Fm, tgt_ids, tgt_k[:, None] * d)


def apply_electro(step):
    """IBM_Driver.m's electrophysiology block, verbatim.

    The driver runs it every step with its ``ePhys_Ct`` counter starting at 1
    (row ``step + 1`` here) and writes the value into the spring-stiffness
    column of spring rows ``EPHYS_START:EPHYS_END``.
    """
    sp_k[EPHYS_START - 1:EPHYS_END] = (16.0 * v_interp[step]) ** 4


# sanity check of the vectorised spring force against a plain loop
rng = np.random.default_rng(0)
Xp = X_ib2d + 1e-3 * rng.standard_normal(X_ib2d.shape)
F_check = np.zeros((Nb, 2))
spring_forces(Xp, F_check)
F_loop = np.zeros((Nb, 2))
for a, b, kk, LL in zip(sp_a, sp_b, sp_k, sp_L):
    d = Xp[b] - Xp[a]
    d = d - np.round(d / np.array([Lx, Ly])) * np.array([Lx, Ly])
    L = np.hypot(*d)
    sF = kk * (L - LL) / L
    F_loop[a] += sF * d
    F_loop[b] -= sF * d
err_sp = np.abs(F_check - F_loop).max() / np.abs(F_loop).max()
if comm.rank == 0:
    print(f"force checks (perturbed config): springs rel. err {err_sp:.2e}")
assert err_sp < 1e-12


def spread_forces_at_half():
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
# Q1 (velocity/force) dofs sit on the IB node grid; map them with wrap-around so
# the periodic images all land on [0, Nx) x [0, Ny)
Qc = V_ib.tabulate_dof_coordinates()[:, :2]
gi = np.rint(Qc[:, 0] / (Lx / Nx)).astype(int) % Nx
gj = np.rint(Qc[:, 1] / (Ly / Ny)).astype(int) % Ny


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


def mean_width(Xm):
    """Cross-tube diameter (the 78 electro springs tie marker i to i + 78)."""
    half = Nb // 2
    return np.hypot(Xm[half:, 0] - Xm[:half, 0],
                    Xm[half:, 1] - Xm[:half, 1]).mean()


hist = {"t": [0.0], "X": [marker_array(X).copy()],
        "k_elec": [(16.0 * v_interp[0]) ** 4]}
u0, p0, f0 = grid_fields()
hist["u"], hist["p"], hist["f"] = [u0], [p0], [f0]
write_output(0.0)
if comm.rank == 0:
    print(f"initial      : width={mean_width(marker_array(X)):.5f}  "
          f"max k_elec={hist['k_elec'][0].max():.4g}")

# ---------------------------------------------------------------------------
# Time loop -- Peskin (2002) / IB2d IBM_Driver ordering
# ---------------------------------------------------------------------------
tic = time.perf_counter()
for step in range(num_steps):
    # (1) half-step Lagrangian positions X_h = X + dt/2 U^n(X)
    ib.evaluate_current_points(X._cpp_object)
    if step == 0:
        X_h.x.array[:] = X.x.array
    else:
        ib.fluid_to_solid(ib_velocity(solver.u_n)._cpp_object, U_s._cpp_object)
        X_h.x.array[:] = X.x.array + 0.5 * dt * U_s.x.array

    # (2) electrophysiology: driver's electro block, before the force computation
    apply_electro(step)

    # (3) structure forces at X_h, spread with the 4-point kernel
    spread_forces_at_half()

    # (4) two-stage fluid solve: u_h, u^{n+1}, p^{n+1/2}
    solver.solve_one_step()

    # (5) markers
    ib.fluid_to_solid(ib_velocity(solver.u_h)._cpp_object, U_s._cpp_object)
    X.x.array[:] += dt * U_s.x.array
    X.x.scatter_forward()

    # Dump schedule of the pyIB2d driver (see demo_478): frame k is written at
    # the end of step print_dump, 2 * print_dump, ... (t below is the true time
    # of the state that was written).
    dumped = (step % print_dump == 0 and step >= print_dump) \
        or (step + 1 == num_steps)
    if dumped:
        t = (step + 1) * dt
        Xm = marker_array(X).copy()
        hist["t"].append(t)
        hist["X"].append(Xm)
        hist["k_elec"].append(sp_k[EPHYS_START - 1:EPHYS_END].copy())
        ug, pg, fg = grid_fields()
        hist["u"].append(ug)
        hist["p"].append(pg)
        hist["f"].append(fg)
        write_output(t)
        if comm.rank == 0:
            print(f"step {step+1:6d}  t={t:.5f}  width={mean_width(Xm):.5f}  "
                  f"k_elec max={sp_k[EPHYS_START - 1:EPHYS_END].max():.4g}  "
                  f"umax={np.abs(ug).max():.3f}  "
                  f"wall={time.perf_counter()-tic:.1f}s", flush=True)

elapsed = time.perf_counter() - tic
xf.close()
xs.close()
fname = os.path.join(out_dir, "afsi_result.npz")
np.savez_compressed(fname,
                    t=np.array(hist["t"]), X=np.array(hist["X"]),
                    k_elec=np.array(hist["k_elec"]),
                    u=np.array(hist["u"]), p=np.array(hist["p"]),
                    f=np.array(hist["f"]),
                    dx=Lx / Nx, dy=Ly / Ny, dt=dt, nx=Nx, ny=Ny,
                    half=Nb // 2, n_markers=Nb,
                    ephys_start=EPHYS_START, ephys_end=EPHYS_END)
if comm.rank == 0:
    print(f"done in {elapsed:.1f}s -> {fname}")
