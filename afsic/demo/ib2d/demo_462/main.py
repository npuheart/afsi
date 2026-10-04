"""IB2d example "Non-linear Rubberband" (matIB2d
``Examples/Example_Standard_Rubberband/NL_Rubberband_With_Springs``) in AFSI.

A 64-marker elliptical rubberband (semi-axes 0.4 x 0.2 at the box centre) is
closed by linear springs with **zero resting length** (k = 1e5) and immersed in
a 1 x 1 periodic box of viscous fluid (mu = 0.01, rho = 1, 32 x 32).

What distinguishes this example from demo_444 (pyIB2d
``Rubberband_with_Springs``: identical geometry, k = 2.5e4, alpha = 1) is the
**fifth column of the ``.spring`` file**: IB2d's springs carry a degree of
non-linearity ``alpha`` and use the force law

    F = (alpha + 1)/2 * k * (L - L_rest)^alpha * d / L,

applied as +F to the leader node and -F to the follower.  Here ``alpha = 2``
and ``L_rest = 0``, so the force is ``F = 3/2 k L^2`` along the segment -- a
*hardening* law (three times a linear spring of the same k at unit length) that
this port exercises, together with IB2d's minimum-image treatment of a segment
that wraps around the periodic box.  The ellipse therefore contracts towards a
circle through a non-linear (and much stiffer than demo_444) response.

The example is one of the ``matIB2d`` ones: its files are 1-based, so
``ib2d_io.load_example(..., index_base=1)`` rebases the identifiers and
``run_reference.py`` writes a rebased copy of the ``.spring`` file.

Run::

    conda activate afsi-dolfinx
    python main.py                      # full example (T = 1.5 s, 32x32)
    TFINAL=0.05 python main.py          # smoke test

Output (``OUTPUT_PATH``, default ``./plot``): XDMF fields +
``afsi_result.npz`` (t, X markers, u/p on the IB2d grid) with the same layout
as the pyIB2d reference from ``run_reference.py``.
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

# gamma = 0 leaks area for a zero-rest-length closed ring (the Q2/Q1 pressure
# space cannot represent the transmural jump, see demo_444/demo_447); the
# default here is the value that saturates for those demos
GRAD_DIV = float(os.environ.get("GRAD_DIV", "100"))

X_ib2d = ex["X"]
Nb = X_ib2d.shape[0]
sp = ex["springs"]
sp_a, sp_b = sp["conn"][:, 0], sp["conn"][:, 1]
sp_k, sp_L, sp_alpha = sp["k"], sp["L"], sp["alpha"]
assert "targets" not in ex and "beams" not in ex and "noninv_beams" not in ex \
    and "d_springs" not in ex and "mass" not in ex, \
    "this demo implements the spring-only example only"
assert np.allclose(sp_alpha, sp_alpha[0]), "mixed alpha not supported"
ds_ib2d = min(Lx / (2 * Nx), Ly / (2 * Ny))

if comm.rank == 0:
    print(f"IB2d example : {IB2D_EXAMPLE}")
    print(f"rho={rho} mu={mu} box={Lx}x{Ly} grid={Nx}x{Ny} dt={dt} T={T} "
          f"steps={num_steps} ds={ds_ib2d}")
    print(f"structure    : {Nb} markers, {len(sp_a)} springs "
          f"(k={sp_k[0]:.2g}, L_rest={sp_L[0]:g}, alpha={sp_alpha[0]:g}), "
          f"grad-div gamma={GRAD_DIV:g}")

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
# Structure: zero-stiffness FE ring carrying the marker dofs
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


def _wrap(d, L):
    """IB2d's minimum-image treatment of a component of a segment vector."""
    return np.where(np.abs(d) > L / 2, np.sign(d) * (L - np.sign(d) * d), d)


def spring_forces(Xm, Fm):
    """IB2d's (possibly non-linear) spring force densities.

    ``F = (alpha + 1)/2 k (L - L_rest)^alpha d / L`` on the leader (+F) and
    follower (-F) node, with the minimum-image correction of the .spring
    segment components.
    """
    dx = _wrap(Xm[sp_b, 0] - Xm[sp_a, 0], Lx)
    dy = _wrap(Xm[sp_b, 1] - Xm[sp_a, 1], Ly)
    Lg = np.hypot(dx, dy)
    sF = 0.5 * (sp_alpha + 1.0) * sp_k * (Lg - sp_L) ** sp_alpha / Lg
    np.add.at(Fm[:, 0], sp_a, sF * dx)
    np.add.at(Fm[:, 1], sp_a, sF * dy)
    np.add.at(Fm[:, 0], sp_b, -sF * dx)
    np.add.at(Fm[:, 1], sp_b, -sF * dy)


def spread_forces_at_half():
    Fm = np.zeros((Nb, 2))
    spring_forces(marker_array(X_h), Fm)
    Fdof = np.empty_like(Fm)
    Fdof[dof_of_ib2d] = Fm
    F_s.x.array[:] = Fdof.ravel()
    F_s.x.array[:] *= ds_ib2d
    ib.evaluate_current_points(X_h._cpp_object)
    ib.solid_to_fluid(f_ib._cpp_object, F_s._cpp_object)
    solver.periodic_fold(f_ib)
    if f_ib is not solver.f:
        solver.f.interpolate(f_ib)


# sanity check of the vectorised force against a plain per-spring loop
rng = np.random.default_rng(0)
Xp = X_ib2d + 1e-3 * rng.standard_normal(X_ib2d.shape)
F_vec = np.zeros((Nb, 2))
spring_forces(Xp, F_vec)
F_loop = np.zeros((Nb, 2))
for a, b, kk, LL, aa_ in zip(sp_a, sp_b, sp_k, sp_L, sp_alpha):
    dx = _wrap(np.array([Xp[b, 0] - Xp[a, 0]]), Lx)[0]
    dy = _wrap(np.array([Xp[b, 1] - Xp[a, 1]]), Ly)[0]
    Lg = np.hypot(dx, dy)
    sF = 0.5 * (aa_ + 1.0) * kk * (Lg - LL) ** aa_ / Lg
    F_loop[a] += (sF * dx, sF * dy)
    F_loop[b] -= (sF * dx, sF * dy)
err = np.abs(F_vec - F_loop).max() / np.abs(F_loop).max()
if comm.rank == 0:
    print(f"spring check (perturbed config): rel. max err = {err:.2e} "
          f"(alpha={sp_alpha[0]:g} law, scale {np.abs(F_loop).max():.3g})")
assert err < 1e-12

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


def ring_area(Xm):
    x, y = Xm[:, 0], Xm[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


hist = {"t": [0.0], "X": [marker_array(X).copy()]}
u0, p0 = grid_fields()
hist["u"], hist["p"] = [u0], [p0]
write_output(0.0)
if comm.rank == 0:
    _per = np.sum(np.hypot(X_ib2d[np.roll(np.arange(Nb), -1), 0] - X_ib2d[:, 0],
                           X_ib2d[np.roll(np.arange(Nb), -1), 1] - X_ib2d[:, 1]))
    print(f"initial      : area={ring_area(X_ib2d):.6f}, "
          f"perimeter={_per:.6f}")

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

    # (2) non-linear spring forces at X_h, spread with the 4-point kernel
    spread_forces_at_half()

    # (3) two-stage fluid solve: u_h, u^{n+1}, p^{n+1/2}
    solver.solve_one_step()

    # (4) markers
    ib.fluid_to_solid(ib_velocity(solver.u_h)._cpp_object, U_s._cpp_object)
    X.x.array[:] += dt * U_s.x.array
    X.x.scatter_forward()

    # Dump schedule: the same steps on which the pyIB2d driver writes its VTK
    # frames.  Its loop dumps when its 0-based step counter ``cter`` (checked
    # before being incremented) is a positive multiple of ``print_dump``, i.e.
    # at the end of step ``cter + 1``; mirroring that makes AFSI frame k and
    # reference frame k the same step (the driver's own time stamp for frame k,
    # ``k*print_dump*dt``, is one step behind the state it wrote).
    dumped = (step % print_dump == 0 and step >= print_dump) \
        or (step + 1 == num_steps)
    if dumped:
        t = (step + 1) * dt
        hist["t"].append(t)
        Xm = marker_array(X).copy()
        hist["X"].append(Xm)
        ug, pg = grid_fields()
        hist["u"].append(ug)
        hist["p"].append(pg)
        write_output(t)
        if comm.rank == 0:
            print(f"step {step+1:6d}  t={t:.4f}  area={ring_area(Xm):.6f}  "
                  f"|F|max={np.abs(F_s.x.array).max():.3e}  "
                  f"wall={time.perf_counter()-tic:.1f}s", flush=True)

elapsed = time.perf_counter() - tic
xf.close()
xs.close()
fname = os.path.join(out_dir, "afsi_result.npz")
np.savez_compressed(fname, t=np.array(hist["t"]), X=np.array(hist["X"]),
                    u=np.array(hist["u"]), p=np.array(hist["p"]),
                    n_markers=Nb, dx=Lx / Nx, dy=Ly / Ny)
if comm.rank == 0:
    print(f"done in {elapsed:.1f}s -> {fname}")
