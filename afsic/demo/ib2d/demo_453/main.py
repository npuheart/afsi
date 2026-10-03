"""IB2d example "Single Porous Rubberband" (pyIB2d
``Examples/Single_Porous_Rubberband``) in AFSI.

A 64-point rubberband (linear springs, k = 1e7, rest length 0 -- it wants
to collapse) where *every* marker is a **porous point**: after the usual IB
update the porous points slide along the local normal direction by

    slip = -kappa * F_lag . n_t / |dX/ds| ...

reproduced here verbatim from pyIB2d's ``please_Compute_Porous_Slip_Velocity``
+ the driver's update block:

* the tangent (xL_s, yL_s) at each porous point is a 4th-order one-sided /
  central finite difference of the *porous-list* neighbours, chosen by the
  stencil flag c in {-2,-1,0,1,2} stored in the ``.porous`` file,
* the unit normal is (yL_s, -xL_s)/|(xL_s,yL_s)|,
* the slip velocity is ``Up = -kappa * F_lag * n / |(xL_s,yL_s)|`` and the
  position update is ``xLag_porous -= dt * Up * n``,
* afterwards *all* markers are folded back into the periodic box
  (``xLag %= L``), exactly as the driver does in its porous branch.

The physical effect: the ring leaks through itself as it collapses -- the
fluid can pass through the porous band instead of being squeezed out.

Run::

    conda activate afsi-dolfinx
    python main.py                      # full example (T = 0.1 s, 32x32)
    TFINAL=0.01 python main.py          # smoke test

Output: ``plot/afsi_result.npz`` (t, X markers, u/p on the IB2d grid).
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
sp = ex["springs"]
por = ex["porous"]
assert "mass" not in ex and "beams" not in ex and "noninv_beams" not in ex
assert "targets" not in ex and "muscles" not in ex and "tracers" not in ex
ds_ib2d = min(Lx / (2 * Nx), Ly / (2 * Ny))

if comm.rank == 0:
    print(f"IB2d example : {IB2D_EXAMPLE}")
    print(f"rho={rho} mu={mu} box={Lx}x{Ly} grid={Nx}x{Ny} dt={dt} T={T} "
          f"steps={num_steps} ds={ds_ib2d}")
    print(f"structure    : {Nb} markers, {len(sp['conn'])} springs, "
          f"{len(por['ids'])} porous points "
          f"(kappa={por['kappa'][0]:.2g}, stencil {por['stencil'].min()}.."
          f"{por['stencil'].max()})")

# ---------------------------------------------------------------------------
# Fluid + structure
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


def write_markers(Xm):
    tmp = np.empty_like(Xm)
    tmp[dof_of_ib2d] = Xm
    X.x.array[:] = tmp.ravel()
    X.x.scatter_forward()


# ---------------------------------------------------------------------------
# Forces: springs at X_h
# ---------------------------------------------------------------------------
sp_a, sp_b = sp["conn"][:, 0], sp["conn"][:, 1]
sp_k, sp_L = sp["k"], sp["L"]
p_ids = por["ids"]
# POROUS_SCALE=0 switches the porous slip off (experiment only)
p_kappa = por["kappa"] * float(os.environ.get("POROUS_SCALE", "1"))
p_stencil = por["stencil"]
Np = len(p_ids)


def spring_forces(Xm, Fm):
    d = Xm[sp_b] - Xm[sp_a]
    L = np.hypot(d[:, 0], d[:, 1])
    sF = sp_k * (L - sp_L) / L
    np.add.at(Fm[:, 0], sp_a, sF * d[:, 0])
    np.add.at(Fm[:, 1], sp_a, sF * d[:, 1])
    np.add.at(Fm[:, 0], sp_b, -sF * d[:, 0])
    np.add.at(Fm[:, 1], sp_b, -sF * d[:, 1])


def spread_forces_at_half():
    """Spring forces at X_h, spread with the 4-point kernel.  Returns the
    raw per-marker forces (un-scaled by ds), which the porous block needs
    (driver: ``F_Lag``)."""
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
    return Fm


def porous_slip_update(Xm, Fm, dt):
    """pyIB2d ``please_Compute_Porous_Slip_Velocity`` + driver update.

    The porous points are stored in file order, so the i+k neighbours of the
    finite-difference stencils are the *porous-list* neighbours (the example
    flags every marker, so the list covers the whole ring)."""
    xL, yL = Xm[p_ids, 0], Xm[p_ids, 1]
    xL_s = np.zeros(Np)
    yL_s = np.zeros(Np)
    for i in range(Np):
        c = p_stencil[i]
        if c == -2:
            xL_s[i] = (-25 / 12 * xL[i] + 4 * xL[i + 1] - 3 * xL[i + 2]
                       + 4 / 3 * xL[i + 3] - 0.25 * xL[i + 4]) / ds_ib2d
            yL_s[i] = (-25 / 12 * yL[i] + 4 * yL[i + 1] - 3 * yL[i + 2]
                       + 4 / 3 * yL[i + 3] - 0.25 * yL[i + 4]) / ds_ib2d
        elif c == -1:
            xL_s[i] = (-0.25 * xL[i - 1] - 5 / 6 * xL[i] + 1.5 * xL[i + 1]
                       - 0.5 * xL[i + 2] + 1 / 12 * xL[i + 3]) / ds_ib2d
            yL_s[i] = (-0.25 * yL[i - 1] - 5 / 6 * yL[i] + 1.5 * yL[i + 1]
                       - 0.5 * yL[i + 2] + 1 / 12 * yL[i + 3]) / ds_ib2d
        elif c == 0:
            xL_s[i] = (1 / 12 * xL[i - 2] - 2 / 3 * xL[i - 1] + 2 / 3 * xL[i + 1]
                       - 1 / 12 * xL[i + 2]) / ds_ib2d
            yL_s[i] = (1 / 12 * yL[i - 2] - 2 / 3 * yL[i - 1] + 2 / 3 * yL[i + 1]
                       - 1 / 12 * yL[i + 2]) / ds_ib2d
        elif c == 1:
            xL_s[i] = (-1 / 12 * xL[i - 3] + 0.5 * xL[i - 2] - 1.5 * xL[i - 1]
                       + 5 / 6 * xL[i] + 0.25 * xL[i + 1]) / ds_ib2d
            yL_s[i] = (-1 / 12 * yL[i - 3] + 0.5 * yL[i - 2] - 1.5 * yL[i - 1]
                       + 5 / 6 * yL[i] + 0.25 * yL[i + 1]) / ds_ib2d
        elif c == 2:
            xL_s[i] = (0.25 * xL[i - 4] - 4 / 3 * xL[i - 3] + 3 * xL[i - 2]
                       - 4 * xL[i - 1] + 25 / 12 * xL[i]) / ds_ib2d
            yL_s[i] = (0.25 * yL[i - 4] - 4 / 3 * yL[i - 3] + 3 * yL[i - 2]
                       - 4 * yL[i - 1] + 25 / 12 * yL[i]) / ds_ib2d

    sqrtN = np.hypot(xL_s, yL_s)
    nX = yL_s / sqrtN
    nY = -xL_s / sqrtN
    Up_X = -p_kappa * Fm[p_ids, 0] * nX / sqrtN
    Up_Y = -p_kappa * Fm[p_ids, 1] * nY / sqrtN
    Xm[p_ids, 0] -= dt * Up_X * nX
    Xm[p_ids, 1] -= dt * Up_Y * nY


# sanity check of the vectorised spring force against a plain loop
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
if comm.rank == 0:
    print(f"force checks (perturbed config): springs {err_sp:.2e}")
assert err_sp < 1e-12

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
# Time loop -- Peskin (2002) / IB2d IBM_Driver ordering (+ porous slip)
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

    # (2) forces at X_h, spread with the 4-point kernel
    Fm = spread_forces_at_half()

    # (3) two-stage fluid solve
    solver.solve_one_step()

    # (4) markers follow the fluid: X^{n+1} = X^n + dt U_h(X_h)
    ib.fluid_to_solid(ib_velocity(solver.u_h)._cpp_object, U_s._cpp_object)
    X.x.array[:] += dt * U_s.x.array
    X.x.scatter_forward()

    # (5) porous slip (driver's porous branch, after the marker update):
    #     slide the porous points along the normal, then fold *all* points
    #     back into the periodic box
    Xm = marker_array(X).copy()
    porous_slip_update(Xm, Fm, dt)
    Xm[:, 0] %= Lx
    Xm[:, 1] %= Ly
    write_markers(Xm)

    if (step + 1) % print_dump == 0 or step == num_steps - 1:
        t = (step + 1) * dt
        hist["t"].append(t)
        hist["X"].append(marker_array(X).copy())
        uf, pf = grid_fields()
        hist["u"].append(uf)
        hist["p"].append(pf)
        write_output(t)
        if comm.rank == 0:
            wall = time.perf_counter() - tic
            Xm = hist["X"][-1]
            area = 0.5 * abs(np.dot(Xm[:, 0], np.roll(Xm[:, 1], -1))
                             - np.dot(Xm[:, 1], np.roll(Xm[:, 0], -1)))
            print(f"step {step+1:6d}  t={t:.4f}  area={area:.6f}  "
                  f"wall={wall:.1f}s")

elapsed = time.perf_counter() - tic
xf.close()
xs.close()
fname = os.path.join(out_dir, f"afsi_result_g{GRAD_DIV:g}.npz")
np.savez_compressed(fname,
                    t=np.array(hist["t"]), X=np.array(hist["X"]),
                    u=np.array(hist["u"]), p=np.array(hist["p"]),
                    dx=Lx / Nx, dy=Ly / Ny)
if comm.rank == 0:
    print(f"done in {elapsed:.1f}s -> {fname}")
