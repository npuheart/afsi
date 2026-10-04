"""IB2d example "Concentration Diffusion" (pyIB2d
``Examples/Concentration_Diffusion``) in AFSI.

A 496-marker square "chamber" wall (spacing ds = 1/128, every marker a pinned
target point with k = 1e7) sits in a 1 x 1 periodic box of very viscous fluid
(mu = 10) together with a **background concentration** field: a 3 x 3 blob of
value 1 at the box centre (``chamber.concentration``, diffusion coefficient
k = 0.025 on the first line) that diffuses for T = 0.5 s at dt = 1e-4.

The wall is pinned exactly at its anchors, so the IB force -- and with it the
fluid velocity -- is identically zero: this example is a *pure diffusion*
benchmark (IB2d's "background flow item"), and its value is that the passive
scalar has an exact solution.  For a point source in 2D the concentration is
Gaussian, ``C(r,t) = M/(4 pi k t) exp(-r^2/(4 k t))``, so the variance of the
field grows as

    sigma^2(t) = sigma^2(0) + 2 k t          (per coordinate)

which both the pyIB2d reference and this port can be checked against.

AFSI ingredients:

* fluid: periodic Q2/Q1 Taylor-Hood + Peskin (2002) two-stage scheme
  (``PeskinRK2Solver``), as in the other ib2d demos;
* structure: the pinned wall (IB2d target-point formula at X^{n+1/2}), a
  zero-stiffness FE chain carrying the marker dofs;
* scalar: a **Q1 finite-element advection-diffusion solve**
  ``dC/dt + u.grad C = k Lap C`` on the same periodic box, advanced explicitly
  with the lumped mass matrix and the same periodic reduction ``P`` the fluid
  solver uses (``C_full = P C_red``, so the Q1 dofs are the IB2d grid nodes);
  the advecting velocity is the IB (interpolated) fluid velocity, i.e. the
  scalar is one-way coupled to the flow, as in IB2d.

Run::

    conda activate afsi-dolfinx
    python main.py                      # full example (T = 0.5 s, 64x64)
    TFINAL=0.01 python main.py          # smoke test

Output (``OUTPUT_PATH``, default ``./plot``): XDMF fields +
``afsi_result.npz`` (t, X markers, u/p/C on the IB2d grid) with the same layout
as the pyIB2d reference from ``run_reference.py``.
"""
import os
import time

import numpy as np
import scipy.sparse as sp
from mpi4py import MPI

import basix.ufl
import dolfinx
import ufl
from dolfinx import fem
from dolfinx.fem import form
from dolfinx.fem.petsc import assemble_matrix
from dolfinx.mesh import CellType, GhostMode

from afsic import IBMesh, IBInterpolation
from afsic.euler.PeskinRK2Solver import PeskinRK2Solver, periodic_prolongation

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

X_ib2d = ex["X"]                               # (Nb, 2) IB2d ordering
Nb = X_ib2d.shape[0]
tgt = ex["targets"]                            # pinned chamber wall
conc = ex["concentration"]                     # initial field + diffusion coeff
C0_grid, k_diff = conc["C"], conc["k"]
assert C0_grid.shape == (Ny, Nx), \
    f"initial concentration is {C0_grid.shape}, expected {(Ny, Nx)}"
assert "springs" not in ex and "beams" not in ex and "noninv_beams" not in ex, \
    "this demo implements the target-point + concentration example only"
# IB2d's Lagrangian weight (IBM_Driver.py: ds = min(Lx/(2Nx), Ly/(2Ny)))
ds_ib2d = min(Lx / (2 * Nx), Ly / (2 * Ny))
if comm.rank == 0:
    print(f"IB2d example : {IB2D_EXAMPLE}")
    print(f"rho={rho} mu={mu} box={Lx}x{Ly} grid={Nx}x{Ny} dt={dt} T={T} "
          f"steps={num_steps} ds={ds_ib2d}")
    print(f"structure    : {Nb} markers, {len(tgt['ids'])} target points "
          f"(k={tgt['k'][0]:.2g})")
    print(f"scalar       : k_diff={k_diff:g}, initial mass={C0_grid.sum():g} "
          f"in {int((C0_grid > 0).sum())} grid cells")

# ---------------------------------------------------------------------------
# Fluid: periodic Taylor-Hood, Peskin two-stage scheme (as in demo_444/455)
# ---------------------------------------------------------------------------
mesh = dolfinx.mesh.create_rectangle(
    comm, ((0.0, 0.0), (Lx, Ly)), (Nx, Ny),
    cell_type=CellType.quadrilateral, ghost_mode=GhostMode.shared_facet)
solver = PeskinRK2Solver(mesh, (0.0, Lx, 0.0, Ly), dt, rho, mu, grad_div=0.0)
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
# Structure: zero-stiffness FE chain carrying the marker dofs (pinned wall)
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


tgt_ids, tgt_k = tgt["ids"], tgt["k"]
tgt_anchor = X_ib2d[tgt_ids].copy()


def target_forces(Xm, Fm):
    """Target-point forces (tethered to the initial positions)."""
    d = tgt_anchor - Xm[tgt_ids]
    d -= np.round(d / np.array([Lx, Ly])) * np.array([Lx, Ly])   # min. image
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
# Passive scalar: Q1 advection-diffusion on the periodic box
# ---------------------------------------------------------------------------
# The Q1 dofs of this mesh are the IB2d grid nodes (and their periodic images),
# so the periodic reduction of PeskinRK2Solver gives C_full = P C_red directly.
Qc = fem.functionspace(mesh, ("Lagrange", 1))
Wc = fem.functionspace(mesh, basix.ufl.element("Lagrange", "quadrilateral", 1,
                                               shape=(2,)))
u_c = fem.Function(Wc, name="u_on_scalar_nodes")
C = fem.Function(Qc, name="concentration")

Pc, master_c = periodic_prolongation(Qc, (0.0, Lx, 0.0, Ly))
masters_c = np.unique(master_c)
n_red = Pc.shape[1]

# dof -> (grid index i, j) by coordinate rounding (wrapped into the box)
Xc = Qc.tabulate_dof_coordinates()[:, :2]
gi_raw = np.rint(Xc[:, 0] / (Lx / Nx)).astype(np.int64)
gj_raw = np.rint(Xc[:, 1] / (Ly / Ny)).astype(np.int64)
assert gi_raw.min() >= 0 and gi_raw.max() <= Nx, "scalar dofs off the IB2d grid (x)"
assert gj_raw.min() >= 0 and gj_raw.max() <= Ny, "scalar dofs off the IB2d grid (y)"
assert np.allclose(Xc[:, 0], gi_raw * (Lx / Nx), atol=1e-12)
assert np.allclose(Xc[:, 1], gj_raw * (Ly / Ny), atol=1e-12)
gi = gi_raw % Nx                          # periodic images fold onto the box
gj = gj_raw % Ny

_phi = ufl.TestFunction(Qc)
_psi = ufl.TrialFunction(Qc)


def _to_scipy(A):
    ia, ja, va = A.getValuesCSR()
    return sp.csr_matrix((va, ja, ia), shape=A.getSize())


def _assemble(a):
    """Assemble a bilinear UFL form into an assembled PETSc matrix."""
    A = assemble_matrix(form(a))
    A.assemble()
    return A


# lumped mass (Q1 row sums; for a structured quad mesh this is the standard
# nodal quadrature weight) and stiffness, both constant in time
Mc = _assemble(ufl.inner(_phi, _psi) * ufl.dx)
_ia, _ja, _va = Mc.getValuesCSR()
mass_rowsum = np.add.reduceat(_va, _ia[:-1])
mass_red = np.asarray(Pc.T @ mass_rowsum).ravel()
mass_red = np.maximum(mass_red, 1e-30)

S = _to_scipy(_assemble(ufl.dot(ufl.grad(_phi), ufl.grad(_psi)) * ufl.dx))
assert S.shape == (len(Xc), len(Xc))
S_red = sp.csr_matrix(Pc.T @ (S @ Pc))


def advection_matrix():
    """Weak advection matrix ``A_ij = int (u . grad phi_j) phi_i dx``."""
    return _to_scipy(_assemble(ufl.inner(ufl.dot(u_c, ufl.grad(_psi)), _phi)
                               * ufl.dx))


# initial condition: nodal values of the IB2d grid field
C0_full = C0_grid[gj, gi].astype(float)
C.x.array[:] = C0_full
C_red = C0_full[masters_c].copy()

# explicit Euler on the lumped-mass diffusion operator is stable for
# dt * k * lambda_max < 2, where lambda_max solves S_red x = lambda M_red x
lam_max = sp.linalg.eigsh(S_red, k=1, M=sp.diags(mass_red), which="LA",
                          return_eigenvectors=False)[0]
dt_diff_max = 2.0 / (k_diff * lam_max)
if comm.rank == 0:
    print(f"scalar       : {n_red} reduced Q1 dofs, explicit diffusion limit "
          f"dt < {dt_diff_max:.3g} s (dt={dt:g}, "
          f"{'OK' if dt < dt_diff_max else 'VIOLATED'})")
assert dt < dt_diff_max, "explicit scalar step is unstable at this dt"

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

# scalar dofs that sit on the 64x64 grid (all of them, by construction)
keep_c = (gi < Nx) & (gj < Ny)


def grid_fields():
    u = np.zeros((Ny, Nx, 2))
    u[ju[keep], iu[keep]] = solver.u_.x.array.reshape(-1, 2)[keep]
    p_grid_fn.interpolate(solver.p_)
    p = np.zeros((Ny, Nx))
    p[jp[keep_p], ip[keep_p]] = p_grid_fn.x.array[keep_p]
    c = np.zeros((Ny, Nx))
    c[gj[keep_c], gi[keep_c]] = C.x.array[keep_c]
    return u, p, c


V1 = fem.functionspace(mesh, basix.ufl.element("Lagrange", "quadrilateral", 1,
                                               shape=(2,)))
u_io, f_io = fem.Function(V1, name="u"), fem.Function(V1, name="f")
p_io = fem.Function(fem.functionspace(mesh, ("Lagrange", 1)), name="p")
c_io = fem.Function(fem.functionspace(mesh, ("Lagrange", 1)), name="C")
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
    c_io.interpolate(C)
    for fn in (u_io, p_io, f_io, c_io):
        xf.write_function(fn, t)
    disp.x.array[:] = X.x.array - X0
    xs.write_function(disp, t)


def variance(c):
    """Per-coordinate variance of a non-negative grid field (physical units)."""
    tot = c.sum()
    x = np.arange(Nx) * (Lx / Nx)
    y = np.arange(Ny) * (Ly / Ny)
    cx, cy = c.sum(axis=0), c.sum(axis=1)
    mx, my = (cx * x).sum() / tot, (cy * y).sum() / tot
    vx = (cx * (x - mx) ** 2).sum() / tot
    vy = (cy * (y - my) ** 2).sum() / tot
    return mx, my, vx, vy


hist = {"t": [0.0], "X": [marker_array(X).copy()]}
u0, p0, c0 = grid_fields()
hist["u"], hist["p"], hist["C"] = [u0], [p0], [c0]
write_output(0.0)

# ---------------------------------------------------------------------------
# Time loop -- Peskin (2002) / IB2d IBM_Driver ordering + scalar transport
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

    # (2) structure forces at X_h (identifier: the pinned wall -> identically 0)
    spread_forces_at_half()

    # (3) two-stage fluid solve: u_h, u^{n+1}, p^{n+1/2}
    solver.solve_one_step()

    # (4) markers
    ib.fluid_to_solid(ib_velocity(solver.u_h)._cpp_object, U_s._cpp_object)
    X.x.array[:] += dt * U_s.x.array
    X.x.scatter_forward()

    # (5) scalar: explicit lumped-mass Galerkin step of dC/dt + u.grad C = k Lap C
    u_c.interpolate(solver.u_)
    A = advection_matrix()
    r = -(A @ C.x.array) - k_diff * (S @ C.x.array)
    C_red += dt * (Pc.T @ r) / mass_red
    C.x.array[:] = Pc @ C_red

    t = (step + 1) * dt
    if (step + 1) % print_dump == 0 or step + 1 == num_steps:
        hist["t"].append(t)
        hist["X"].append(marker_array(X).copy())
        ug, pg, cg = grid_fields()
        hist["u"].append(ug)
        hist["p"].append(pg)
        hist["C"].append(cg)
        write_output(t)
        if comm.rank == 0:
            mx, my, vx, vy = variance(cg)
            print(f"step {step+1:6d}  t={t:.4f}  mass={cg.sum():.6f}  "
                  f"Cmax={cg.max():.5f}  var=({vx:.5e},{vy:.5e})  "
                  f"2kt={2*k_diff*t:.5e}  wall={time.perf_counter()-tic:.1f}s",
                  flush=True)

elapsed = time.perf_counter() - tic
xf.close()
xs.close()
fname = os.path.join(out_dir, "afsi_result.npz")
np.savez_compressed(fname,
                    t=np.array(hist["t"]), X=np.array(hist["X"]),
                    u=np.array(hist["u"]), p=np.array(hist["p"]),
                    C=np.array(hist["C"]),
                    dx=Lx / Nx, dy=Ly / Ny, k_diff=k_diff)
if comm.rank == 0:
    print(f"done in {elapsed:.1f}s -> {fname}")
