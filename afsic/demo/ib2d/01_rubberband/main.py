"""IB2d example 1 — "Standard Rubberband" (Rubberband_with_Springs) in AFSI.

An elliptical elastic band of 64 Lagrangian points joined by zero-rest-length
linear springs (k = 2.5e4) oscillates and relaxes in a periodic unit box of
viscous fluid (rho = 1, mu = 0.01).  All parameters and the structure are read
from the original IB2d example files (``input2d``, ``rubberband.vertex``,
``rubberband.spring``):

=========================  =============================================  ==========================================
ingredient                 IB2d (matIB2d)                                  AFSI (this demo)
=========================  =============================================  ==========================================
fluid discretisation       Nx x Ny collocated central FD + FFT             Q2/Q1 Taylor–Hood on Nx x Ny quads
                                                                           (+ grad-div, GRAD_DIV, default 100)
periodicity                FFT                                             periodic reduction P^T A P
time stepping              Peskin 2002 two-stage (BE dt/2 + CN dt)         ``afsic.PeskinRK2Solver`` (identical)
incompressibility          exact FFT projection in each stage              monolithic Stokes solve in each stage
IB grid / kernel           Peskin 4-point on the Nx x Ny grid              ``IBMesh(order=1)``: same grid & kernel
structure                  springs F = k(|dX|-L) dX/|dX|                   1D FE curve, energy k/(2h0)(h0|X_s|-L)^2
Lagrangian weight          F_k * ds,  ds = min(Lx/2Nx, Ly/2Ny)             same
spreading / interpolation  f = sum_k F_k ds delta_h,  U = sum u delta h^2  ``IBInterpolation`` (w = 1)
Lagrangian update          X_h = X + dt/2 U(X);  X += dt U_h(X_h)          same
=========================  =============================================  ==========================================

``FLUID_MESH=half`` instead uses (Nx/2) x (Ny/2) Q2 cells, whose velocity nodes
are the IB2d grid (``IBMesh(order=2)``); it is coarser and leaks more volume.

Run::

    conda activate afsi-dolfinx              # dolfinx 0.10 + MUMPS (no dolfinx_mpc needed)
    python main.py                           # full IB2d run (T = 1.5, dt = 1e-3)
    TFINAL=0.1 python main.py                # short check
    GRAD_DIV=0 python main.py                # plain Taylor–Hood (strong IB leakage)
    IB2D_EXAMPLE=/path/to/case python main.py   # any IB2d spring case (see make_rubberband.py)

Output (``OUTPUT_PATH``, default ``./plot``): XDMF fields + ``afsi_result.npz``
(same layout as ``ib2d_reference.npz``, see ``ib2d_reference.py``).
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
from dolfinx.fem.petsc import assemble_vector, create_vector
from dolfinx.mesh import CellType, GhostMode

from afsic import IBMesh, IBInterpolation
from afsic.euler.PeskinRK2Solver import PeskinRK2Solver

import ib2d_io

_here = os.path.dirname(os.path.abspath(__file__))
_repo = os.path.abspath(os.path.join(_here, *[".."] * 4))
IB2D_EXAMPLE = os.environ.get("IB2D_EXAMPLE", os.path.join(
    _repo, "third_party", "ib2d", "matIB2d", "Examples",
    "Example_Standard_Rubberband", "Rubberband_with_Springs"))

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
Ny = int(os.environ.get("NY", Nx if "NX" in os.environ else P["Ny"]))
dt = float(os.environ.get("DT", P["dt"]))
T = float(os.environ.get("TFINAL", P["Tfinal"]))
print_dump = int(os.environ.get("PRINT_DUMP", P["print_dump"]))
assert Nx % 2 == 0 and Ny % 2 == 0, "Nx, Ny must be even (as in IB2d)"
num_steps = int(round(T / dt))
out_dir = os.environ.get("OUTPUT_PATH", os.path.join(_here, "plot"))
os.makedirs(out_dir, exist_ok=True)

X_ib2d = ex["X"]                       # (Nb, 2) IB2d ordering
conn, k_spring, L_rest, alpha = ex["springs"]
assert np.all(alpha == 1.0), "only linear IB2d springs are ported"
Nb = X_ib2d.shape[0]
# IB2d treats the spring "forces" as Lagrangian force *densities* and multiplies
# them by a fixed Lagrangian spacing before spreading (please_Find_Lagrangian_
# Forces_On_Eulerian_grid.m): F_spread = F_spring * ds, ds = min(Lx/2Nx, Ly/2Ny).
ds_ib2d = min(Lx / (2 * Nx), Ly / (2 * Ny))
if comm.rank == 0:
    print(f"IB2d example : {IB2D_EXAMPLE}")
    print(f"rho={rho} mu={mu} box={Lx}x{Ly} grid={Nx}x{Ny} "
          f"dt={dt} T={T} steps={num_steps} Nb={Nb} springs={len(conn)} ds={ds_ib2d}")
    print(f"fluid: Q2/Q1 FLUID_MESH={os.environ.get('FLUID_MESH', 'full')} GRAD_DIV={os.environ.get('GRAD_DIV', '100')}")

# ---------------------------------------------------------------------------
# Fluid: periodic Taylor–Hood, Peskin two-stage scheme
# ---------------------------------------------------------------------------
# FLUID_MESH=full : Q2/Q1 on  Nx x Ny     quads, mesh vertices = IB2d grid
#                   (IB kernel ops on the vertex grid, Q1 <-> Q2 by interpolation)
# FLUID_MESH=half : Q2/Q1 on (Nx/2)x(Ny/2) quads, Q2 velocity nodes = IB2d grid
FLUID_MESH = os.environ.get("FLUID_MESH", "full").lower()
assert FLUID_MESH in ("full", "half")
ncx, ncy = (Nx // 2, Ny // 2) if FLUID_MESH == "half" else (Nx, Ny)
mesh = dolfinx.mesh.create_rectangle(
    comm, ((0.0, 0.0), (Lx, Ly)), (ncx, ncy),
    cell_type=CellType.quadrilateral, ghost_mode=GhostMode.shared_facet)
GRAD_DIV = float(os.environ.get("GRAD_DIV", "100"))
solver = PeskinRK2Solver(mesh, (0.0, Lx, 0.0, Ly), dt, rho, mu, grad_div=GRAD_DIV)
Vc = solver.V

if FLUID_MESH == "half":
    # IB grid = velocity nodes (order 2) -> spacing Lx/Nx, identical to IB2d's dx
    ibmesh = IBMesh(0.0, Lx, 0.0, Ly, ncx, ncy, 2)
    V_ib = Vc
    u_ib, f_ib = None, solver.f
else:
    ibmesh = IBMesh(0.0, Lx, 0.0, Ly, ncx, ncy, 1)
    V_ib = fem.functionspace(mesh, basix.ufl.element("Lagrange", "quadrilateral", 1, shape=(2,)))
    u_ib, f_ib = fem.Function(V_ib), fem.Function(V_ib)
coords_bg = fem.Function(V_ib)
coords_bg.interpolate(lambda x: np.array([x[0], x[1]]))
ibmesh.build_map(coords_bg._cpp_object)


def ib_velocity(u):
    """Eulerian velocity on the IB grid."""
    if u_ib is None:
        return u
    u_ib.interpolate(u)
    return u_ib


def ib_spread_done():
    """Fold the spread force periodically and hand it to the fluid solver."""
    solver.periodic_fold(f_ib)
    if f_ib is not solver.f:
        solver.f.interpolate(f_ib)


ib = IBInterpolation(ibmesh)

# ---------------------------------------------------------------------------
# Structure: 1D Lagrangian FE curve whose cells are the IB2d springs
# ---------------------------------------------------------------------------
coord_el = basix.ufl.element("Lagrange", "interval", 1, shape=(2,))
structure = dolfinx.mesh.create_mesh(comm, conn, coord_el, X_ib2d)
Vs = fem.functionspace(structure, coord_el)

# per-cell spring data (cells may be reordered by DOLFINx)
DG0 = fem.functionspace(structure, ("DG", 0))
orig = structure.topology.original_cell_index
k_fn, L_fn = fem.Function(DG0), fem.Function(DG0)
k_fn.x.array[:] = k_spring[orig]
L_fn.x.array[:] = L_rest[orig]

X = fem.Function(Vs, name="position")        # current configuration X(s, t)
X.interpolate(lambda x: np.array([x[0], x[1]]))
X_h = fem.Function(Vs, name="position_half")  # force configuration X^{n+1/2}
X_h.x.array[:] = X.x.array
U_s = fem.Function(Vs, name="velocity")
F_s = fem.Function(Vs, name="force")

# map AFSI dof -> IB2d point index (for comparison / output)
Xs0 = X.x.array.reshape(-1, 2)
d2 = ((Xs0[:, None, :] - X_ib2d[None, :, :]) ** 2).sum(-1)
dof_of_ib2d = np.argmin(d2, axis=0)
assert np.allclose(Xs0[dof_of_ib2d], X_ib2d)

# Spring energy, exact for P1 (|X_s| is constant per cell):
#   E = sum_e k/2 (|X_{e,1}-X_{e,0}| - L)^2 = sum_e int_e k/(2 h0) (h0 |X_s| - L)^2 ds
h0 = ufl.CellVolume(structure)
gX = ufl.grad(X_h)                               # = X_s (x) t on the curve
dXs = ufl.TestFunction(Vs)
dx_s = ufl.Measure("dx", domain=structure, metadata={"quadrature_degree": 1})
if np.all(L_rest == 0.0):
    energy = 0.5 * k_fn * h0 * ufl.inner(gX, gX) * dx_s
else:
    stretch = h0 * ufl.sqrt(ufl.inner(gX, gX) + 1e-30)
    energy = 0.5 * k_fn / h0 * (stretch - L_fn) ** 2 * dx_s
force_form = fem.form(-ufl.derivative(energy, X_h, dXs))
F_vec = create_vector(Vs)


def spring_forces():
    """Nodal spring forces F_k at the configuration X_h."""
    with F_vec.localForm() as loc:
        loc.set(0.0)
    assemble_vector(F_vec, force_form)
    F_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    F_s.x.petsc_vec.array[:] = F_vec.array_r
    F_s.x.scatter_forward()


# sanity check against the IB2d spring formula at t = 0
spring_forces()
Fl = np.zeros_like(X_ib2d)
for (a, b), kk, LL in zip(conn, k_spring, L_rest):
    d = X_ib2d[b] - X_ib2d[a]
    nd = np.linalg.norm(d)
    s = kk * (nd - LL) * d / nd
    Fl[a] += s
    Fl[b] -= s
err_F = np.abs(F_s.x.array.reshape(-1, 2)[dof_of_ib2d] - Fl).max() / np.abs(Fl).max()
if comm.rank == 0:
    print(f"spring force vs IB2d formula: rel. max err = {err_F:.2e}")
assert err_F < 1e-10

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
gx = np.arange(Nx) * (Lx / Nx)
gy = np.arange(Ny) * (Ly / Ny)
Vdof = Vc.tabulate_dof_coordinates()[:, :2]
iu = np.rint(Vdof[:, 0] / (Lx / Nx)).astype(int)
ju = np.rint(Vdof[:, 1] / (Ly / Ny)).astype(int)
on_grid = lambda P, i, j: (np.isclose(P[:, 0], i * (Lx / Nx)) & np.isclose(P[:, 1], j * (Ly / Ny)))
keep = (iu < Nx) & (ju < Ny) & on_grid(Vdof, iu, ju)
Q2s = fem.functionspace(mesh, ("Lagrange", 2))     # pressure sampled on the IB2d grid
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


V1 = fem.functionspace(mesh, basix.ufl.element("Lagrange", "quadrilateral", 1, shape=(2,)))
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
    xs.write_function(F_s, t)


hist = {"t": [0.0], "X": [X.x.array.reshape(-1, 2)[dof_of_ib2d].copy()]}
u0, p0 = grid_fields()
hist["u"], hist["p"] = [u0], [p0]
write_output(0.0)

# ---------------------------------------------------------------------------
# Time loop — Peskin (2002) / IB2d IBM_Driver ordering
# ---------------------------------------------------------------------------
tic = time.perf_counter()
for step in range(num_steps):
    # (1) half-step Lagrangian positions X_h = X + dt/2 U^n(X)
    ib.evaluate_current_points(X._cpp_object)
    if step == 0:
        X_h.x.array[:] = X.x.array          # u = 0 initially (as IB2d)
    else:
        ib.fluid_to_solid(ib_velocity(solver.u_n)._cpp_object, U_s._cpp_object)
        X_h.x.array[:] = X.x.array + 0.5 * dt * U_s.x.array

    # (2) spring forces at X_h, spread with the 4-point kernel
    spring_forces()
    F_s.x.array[:] *= ds_ib2d                 # IB2d Lagrangian weight
    ib.evaluate_current_points(X_h._cpp_object)
    ib.solid_to_fluid(f_ib._cpp_object, F_s._cpp_object)
    ib_spread_done()

    # (3) two-stage fluid solve: u_h, u^{n+1}, p^{n+1/2}
    solver.solve_one_step()

    # (4) X^{n+1} = X^n + dt U_h(X_h)
    ib.fluid_to_solid(ib_velocity(solver.u_h)._cpp_object, U_s._cpp_object)
    X.x.array[:] += dt * U_s.x.array
    X.x.scatter_forward()

    t = (step + 1) * dt
    if (step + 1) % print_dump == 0 or step + 1 == num_steps:
        hist["t"].append(t)
        hist["X"].append(X.x.array.reshape(-1, 2)[dof_of_ib2d].copy())
        ug, pg = grid_fields()
        hist["u"].append(ug)
        hist["p"].append(pg)
        write_output(t)
        if comm.rank == 0:
            Xi = hist["X"][-1]
            area = 0.5 * abs(np.sum(Xi[:, 0] * np.roll(Xi[:, 1], -1) - np.roll(Xi[:, 0], -1) * Xi[:, 1]))
            print(f"step {step+1:6d}  t={t:.4f}  max|u|={np.abs(solver.u_.x.array).max():.4e}  "
                  f"area={area:.6f}  X0=({Xi[0,0]:.5f},{Xi[0,1]:.5f})  "
                  f"wall={time.perf_counter()-tic:.1f}s", flush=True)

xf.close()
xs.close()
np.savez_compressed(os.path.join(out_dir, "afsi_result.npz"),
                    t=np.array(hist["t"]), X=np.array(hist["X"]),
                    u=np.array(hist["u"]), p=np.array(hist["p"]),
                    dx=Lx / Nx, dy=Ly / Ny)
if comm.rank == 0:
    print(f"done in {time.perf_counter()-tic:.1f}s -> {out_dir}/afsi_result.npz")
