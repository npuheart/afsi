"""IB2d example "Thrips" (pyIB2d ``Examples/Thrips``) in AFSI.

The smallest possible IB2d mass-point model: **one** marker -- a settling
particle -- immersed in a 0.5 x 0.5 periodic box of viscous fluid (mu = 0.1,
rho = 1, 32x32, dt = 1e-4, T = 0.1 s).  The marker carries no structure of its
own; it is coupled by a "mass-spring" (k = 1e3) to a **ghost particle** of mass
M = 1e-3 kg on which gravity (0, -g), g = 9.80665, acts.  The particle
accelerates until the IB fluid drag balances the elastic pull of the falling
ghost -- the 2D point-particle analogue of a settling sphere.

Mass-point semantics (IB2d ``please_Move_Massive_Boundary`` /
``please_Update_Massive_Boundary_Velocity``, identical to demo_450):

* the marker (id 0) moves with the fluid and feels ``F = k (X_ghost - X_marker)``
  (no anchor to its initial position),
* the ghost obeys ``M dV/dt = -F + M g`` -- gravity acts on the ghost only,
* per step: markers take the fluid half-step; the ghost is moved a half-step;
  the tether force is evaluated at the half-step and spread; the fluid is
  solved; markers advance with the fluid velocity; the ghost is restored to its
  step-start position and advanced a full step with the new velocity.

Structure dof carrier: a single-marker FE mesh is built as one **DG0 "point"
cell**, which has exactly one (blocked) dof -- one IB particle.

Run::

    conda activate afsi-dolfinx
    python main.py                      # full example (T = 0.1 s, 32x32)
    TFINAL=0.005 python main.py         # smoke test

Output: ``plot/afsi_result.npz`` (t, X markers, Xmass ghost, u/p on the IB2d
grid) with the same layout as the other ib2d demos.
"""
import os
import time

import numpy as np
from mpi4py import MPI

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
# Parameters
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
num_steps = int(round(T / dt))
out_dir = os.environ.get("OUTPUT_PATH", os.path.join(_here, "plot"))
os.makedirs(out_dir, exist_ok=True)

X_ib2d = ex["X"]
Nb = X_ib2d.shape[0]
mass = ex["mass"]
assert "springs" not in ex and "beams" not in ex and "targets" not in ex, \
    "this demo implements the mass-point + gravity example only"
ds_ib2d = min(Lx / (2 * Nx), Ly / (2 * Ny))
assert float(P["gravity"]) == 1, "this example must switch gravity on"

GRAVITY = 9.80665                      # IBM_Driver/Supp.py: g = 9.80665 m/s^2
g_dir = np.array([float(P["x_gravity_vec_comp"]), float(P["y_gravity_vec_comp"])])
g_vec = GRAVITY * g_dir / np.linalg.norm(g_dir)
if comm.rank == 0:
    print(f"IB2d example : {IB2D_EXAMPLE}")
    print(f"rho={rho} mu={mu} box={Lx}x{Ly} grid={Nx}x{Ny} dt={dt} T={T} "
          f"steps={num_steps} ds={ds_ib2d}")
    print(f"structure    : {Nb} marker(s), {len(mass['ids'])} mass point(s) "
          f"(k={mass['k'][0]:.2g}, M={mass['M'][0]:g} kg), g={g_vec} m/s^2")

# ---------------------------------------------------------------------------
# Fluid
# ---------------------------------------------------------------------------
mesh = dolfinx.mesh.create_rectangle(
    comm, ((0.0, 0.0), (Lx, Ly)), (Nx, Ny),
    cell_type=CellType.quadrilateral, ghost_mode=GhostMode.shared_facet)
solver = PeskinRK2Solver(mesh, (0.0, Lx, 0.0, Ly), dt, rho, mu, grad_div=0.0)
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
    """Eulerian velocity on the IB (vertex) grid."""
    u_ib.interpolate(u)
    return u_ib


# ---------------------------------------------------------------------------
# Structure: one DG0 "point" cell per marker (1 dof = 1 IB particle)
# ---------------------------------------------------------------------------
cells = np.arange(Nb, dtype=np.int64).reshape(-1, 1)
coord_el = basix.ufl.element("Lagrange", "point", 0, shape=(2,))
structure = dolfinx.mesh.create_mesh(comm, cells, coord_el, X_ib2d)
Vs = fem.functionspace(structure, coord_el)
assert Vs.dofmap.index_map.size_local == Nb, \
    "the structure FE mesh must have one dof per IB2d marker"

X = fem.Function(Vs, name="position")
X.interpolate(lambda x: np.array([x[0], x[1]]))
X_h = fem.Function(Vs, name="position_half")
X_h.x.array[:] = X.x.array
U_s = fem.Function(Vs, name="velocity")
F_s = fem.Function(Vs, name="force")

dof_rows = Vs.tabulate_dof_coordinates()[:, :2]
dof_of_ib2d = np.array([int(np.argmin(np.linalg.norm(dof_rows - p, axis=1)))
                        for p in X_ib2d])
assert len(np.unique(dof_of_ib2d)) == Nb, "dof matching is not injective"
assert np.allclose(dof_rows[dof_of_ib2d], X_ib2d, atol=1e-12), \
    "dof coordinates do not recover the IB2d marker list"


def marker_array(fn):
    return fn.x.array.reshape(-1, 2)[dof_of_ib2d]


m_ids, m_k, m_M = mass["ids"], mass["k"], mass["M"]
Xmass = X_ib2d[m_ids].copy()             # ghost-particle positions
Vmass = np.zeros_like(Xmass)             # ghost-particle velocities


def spread_forces_at_half(Xmass_h):
    """Mass-marker tethers at (X_h, Xmass_h), spread with the 4-point kernel.

    Returns the per-mass force ``F = k (X_ghost - X_marker)`` evaluated at the
    half-step (it both acts on the marker and drives the ghost ODE)."""
    Fm = np.zeros((Nb, 2))
    Xm = marker_array(X_h)
    F_mass = m_k[:, None] * (Xmass_h - Xm[m_ids])
    np.add.at(Fm[:, 0], m_ids, F_mass[:, 0])
    np.add.at(Fm[:, 1], m_ids, F_mass[:, 1])
    Fdof = np.empty_like(Fm)
    Fdof[dof_of_ib2d] = Fm
    F_s.x.array[:] = Fdof.ravel()
    F_s.x.array[:] *= ds_ib2d
    ib.evaluate_current_points(X_h._cpp_object)
    ib.solid_to_fluid(f_ib._cpp_object, F_s._cpp_object)
    solver.periodic_fold(f_ib)
    if f_ib is not solver.f:
        solver.f.interpolate(f_ib)
    return F_mass


# ---------------------------------------------------------------------------
# Output sampling on the IB2d grid
# ---------------------------------------------------------------------------
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
xf.write_mesh(mesh)


def write_output(t):
    u_io.interpolate(solver.u_)
    f_io.interpolate(solver.f)
    p_io.interpolate(solver.p_)
    for fn in (u_io, p_io, f_io):
        xf.write_function(fn, t)


hist = {"t": [0.0], "X": [marker_array(X).copy()], "Xmass": [Xmass.copy()]}
u0, p0 = grid_fields()
hist["u"], hist["p"] = [u0], [p0]
write_output(0.0)

# ---------------------------------------------------------------------------
# Time loop -- Peskin (2002) / IB2d IBM_Driver ordering (+ ghost bookkeeping)
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

    # (1b) ghost: half-step move; remember its step-start position
    massOld = Xmass.copy()
    Xmass_h = Xmass + 0.5 * dt * Vmass

    # (2) tether forces at (X_h, Xmass_h), spread with the 4-point kernel
    F_mass = spread_forces_at_half(Xmass_h)

    # (3) two-stage fluid solve
    solver.solve_one_step()

    # (4) marker follows the fluid: X^{n+1} = X^n + dt U_h(X_h)
    ib.fluid_to_solid(ib_velocity(solver.u_h)._cpp_object, U_s._cpp_object)
    X.x.array[:] += dt * U_s.x.array
    X.x.scatter_forward()

    # (4b) ghost: v_half = v - dt/2 (F/M - g); X_mass = X_mass^n + dt v_half;
    #      v^{n+1} = v - dt (F/M - g)   (driver's Move/Update pair)
    acc = F_mass / m_M[:, None] - g_vec
    Vmass_h = Vmass - 0.5 * dt * acc
    Xmass = massOld + dt * Vmass_h
    Vmass = Vmass - dt * acc

    t = (step + 1) * dt
    if (step + 1) % print_dump == 0 or step + 1 == num_steps:
        Xm = marker_array(X).copy()
        hist["t"].append(t)
        hist["X"].append(Xm)
        hist["Xmass"].append(Xmass.copy())
        ug, pg = grid_fields()
        hist["u"].append(ug)
        hist["p"].append(pg)
        write_output(t)
        if comm.rank == 0:
            print(f"step {step+1:6d}  t={t:.4f}  marker=({Xm[0,0]:.5f},"
                  f"{Xm[0,1]:.5f})  ghost_y={Xmass[0,1]:.5f}  "
                  f"vy={Vmass[0,1]:.4f}  |F|max={np.abs(F_s.x.array).max():.3e}"
                  f"  wall={time.perf_counter()-tic:.1f}s", flush=True)

elapsed = time.perf_counter() - tic
xf.close()
fname = os.path.join(out_dir, "afsi_result.npz")
np.savez_compressed(fname, t=np.array(hist["t"]), X=np.array(hist["X"]),
                    Xmass=np.array(hist["Xmass"]), u=np.array(hist["u"]),
                    p=np.array(hist["p"]), dx=Lx / Nx, dy=Ly / Ny,
                    mass_k=float(m_k[0]), mass_M=float(m_M[0]))
if comm.rank == 0:
    print(f"done in {elapsed:.1f}s -> {fname}")
