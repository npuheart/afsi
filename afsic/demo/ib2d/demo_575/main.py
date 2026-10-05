"""IB2d example "Baby Spider Shower, Flow Based" (matIB2d
``Examples/Example_Baby_Spider_Shower/Flow_Based``) in AFSI.

An 90-marker vertical silk thread (``x = 0.25``, ``y in [0.676, 0.85]``) hangs
in a 1.5 x 1.0 periodic box of *air* (``mu = 1.8e-5``, ``rho = 1.225``); the
thread has 89 springs (``k = 9830``) and 86 invariant beams (``kb = ?``), and
its bottom marker (id 90) carries **one** massive ghost particle
(``k = 1e4``, ``M = 0.0225``) -- the "spider".  Gravity (``9.80665 m/s^2``,
straight down) pulls the spider while the example's
``please_Compute_External_Forcing.m`` blows a wind of ``u = 0.2 tanh(5t) m/s``
through the strip ``x in [0.05, 0.09]``.

This is the only mass-point example with a **single** ghost, i.e. the one that
hits pyIB2d's broken single-mass-point branch -- see ``readme.md`` and
``run_reference.py`` (the reference run patches ``read_Mass_Points`` to take
the general branch; the ported ODE is validated by the replay check).

**Reduced setup (documented in ``readme.md``)**: upstream is ``384 x 256`` with
``T = 5 s`` (100000 steps); this port runs ``NX = 96`` (``NY = 64``) and
``T = 0.2 s`` (4000 steps) at the shipped ``dt = 5e-5``.

Run::

    conda activate afsi-dolfinx
    python run_reference.py             # pyIB2d reference
    python main.py                      # AFSI
    python compare.py

Output (``OUTPUT_PATH``, default ``./plot``): XDMF fields +
``afsi_result.npz`` (t, X markers, Xm/Vm mass points, u/p/f on the IB2d grid).
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
ext_mod = None
if os.path.exists(os.path.join(IB2D_EXAMPLE,
                               "please_Compute_External_Forcing.py")):
    import please_Compute_External_Forcing as ext_mod      # noqa: E402

comm = MPI.COMM_WORLD
assert comm.size == 1, "this demo is serial (IB coupling of afsic gathers on rank 0)"

# the matIB2d example files number the Lagrangian points from 1
ex = ib2d_io.load_example(IB2D_EXAMPLE, index_base=1)
P = ex["params"]
assert (ext_mod is not None) == bool(int(P["arb_ext_force"])), \
    "please_Compute_External_Forcing.py must exist iff arb_ext_force = 1"
rho, mu = float(P["rho"]), float(P["mu"])
Lx, Ly = float(P["Lx"]), float(P["Ly"])
Nx = int(os.environ.get("NX", P["Nx"]))
Ny = int(os.environ.get("NY", round(Nx * Ly / Lx)))
h = Lx / Nx
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
tg = ex.get("targets")                     # optional pinned target points
trs = ex.get("tracers")                    # optional passive tracers
conc = ex.get("concentration")             # optional passive scalar
mass = ex.get("mass")                      # optional massive ghost points
Nt = 0 if trs is None else trs.shape[0]
ds_ib2d = Lx / (2 * Nx)   # IB2d sets ds = Lx/(2Nx)

assert "muscles" not in ex and "d_springs" not in ex \
    and "noninv_beams" not in ex, \
    "this template implements springs, invariant beams, target points, massive "\
    "boundary points, tracers and the arbitrary external force only"

# gravity acts on the massive boundary points only (the Boussinesq coupling of
# the fluid is a separate IB2d flag and is off in every mass-point example)
GRAV = float(P.get("gravity", 0)) != 0
if GRAV:
    _gx, _gy = float(P["x_gravity_vec_comp"]), float(P["y_gravity_vec_comp"])
    _gn = np.hypot(_gx, _gy)
    GRAV_VEC = np.array([_gx / _gn, _gy / _gn])
else:
    GRAV_VEC = None
G_ACC = 9.80665            # m/s^2, IB2d's hard-coded gravity (Supp.py)

if comm.rank == 0:
    desc = []
    if sp is not None:
        desc.append(f"{len(sp['conn'])} springs (k={sp['k'].min():.2g}"
                    f"..{sp['k'].max():.2g})")
    if bm is not None:
        desc.append(f"{len(bm['p1'])} invariant beams (kb={bm['kb'][0]:.2g})")
    if tg is not None:
        desc.append(f"{len(tg['ids'])} target points (k={tg['k'][0]:.2g})")
    if mass is not None:
        desc.append(f"{mass['ids'].size} massive ghost points "
                    f"(k={mass['k'][0]:.2g}, M={np.unique(mass['M'])})")
    if Nt:
        desc.append(f"{Nt} passive tracers")
    if conc is not None:
        desc.append(f"passive scalar (kDiff={conc['k']:g}, "
                    f"C0 in [{conc['C0'].min():g}, {conc['C0'].max():g}])")
    print(f"IB2d example : {IB2D_EXAMPLE}")
    print(f"rho={rho} mu={mu} box={Lx}x{Ly} grid={Nx}x{Ny} dt={dt} T={T} "
          f"steps={num_steps} ds={ds_ib2d}")
    print(f"structure    : {Nb} markers: " + ", ".join(desc))
    print(f"gravity      : {'on, gvec=' + str(GRAV_VEC) if GRAV else 'off'}")
    if ext_mod is not None:
        print(f"forcing      : {os.path.basename(ext_mod.__file__)}: "
              f"x in [{getattr(ext_mod, 'X_MIN', '?')}, "
              f"{getattr(ext_mod, 'X_MAX', '?')}], "
              f"y in [{getattr(ext_mod, 'Y_MIN', '?')}, "
              f"{getattr(ext_mod, 'Y_MAX', '?')}], "
              f"k={getattr(ext_mod, 'K_STIFF', '?')}, "
              f"uMax={getattr(ext_mod, 'U_MAX', '?')}")
    else:
        print("forcing      : none (no please_Compute_External_Forcing.py)")
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


def wrap_periodic(fn, dofs):
    """IB2d wraps marker/tracer coordinates into [0, L) after each update."""
    a = fn.x.array.reshape(-1, 2)
    a[dofs, 0] %= Lx
    a[dofs, 1] %= Ly


# passive tracers: the same interpolation machinery, no force feedback
if Nt:
    tcells = np.column_stack([np.arange(Nt - 1), np.arange(1, Nt)]).astype(np.int64)
    tmesh = dolfinx.mesh.create_mesh(comm, tcells, coord_el, trs)
    Vt = fem.functionspace(tmesh, coord_el)
    Xt, Ut = fem.Function(Vt, name="tracer_position"), \
        fem.Function(Vt, name="tracer_velocity")
    _tdof = Vt.tabulate_dof_coordinates()[:, :2]
    dof_of_tracer = np.array([int(np.argmin(np.linalg.norm(_tdof - p, axis=1)))
                              for p in trs])
    assert np.allclose(_tdof[dof_of_tracer], trs, atol=1e-12)
    _t0 = np.empty_like(trs)
    _t0[dof_of_tracer] = trs
    Xt.x.array[:] = _t0.ravel()


    def tracer_array():
        return Xt.x.array.reshape(-1, 2)[dof_of_tracer]


if tg is not None:
    tgt_ids, tgt_k = tg["ids"], tg["k"]
    tgt_anchor = X_ib2d[tgt_ids].copy()      # pinned: update_target = 0
if sp is not None:
    sp_a, sp_b = sp["conn"][:, 0], sp["conn"][:, 1]
    sp_k = sp["k"]
    sp_L = sp["L"].copy()
if bm is not None:
    bm_p1, bm_p2, bm_p3 = bm["p1"], bm["p2"], bm["p3"]
    bm_k, bm_C = bm["kb"], bm["C"]
if mass is not None:
    # IB2d's ``mass_info``: col 0 the marker id the ghost is tied to, col 1-2
    # the ghost position (initialised at its marker), col 3 the coupling spring
    # stiffness k, col 4 the ghost mass M.  ``mVelocity`` starts at zero.
    mass_ids, mass_k, mass_M = mass["ids"], mass["k"], mass["M"]
    Nm = mass_ids.size
    mass_pos = X_ib2d[mass_ids].copy()
    mass_vel = np.zeros((Nm, 2))
    mass_anchor = mass_pos.copy()


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
    ``give_Me_Target_Lagrangian_Force_Densities``); reproduced verbatim.
    """
    d = tgt_anchor - Xm[tgt_ids]
    for c, L in enumerate((Lx, Ly)):
        over = np.abs(d[:, c]) > L / 2
        sgn = np.sign(d[over, c])
        d[over, c] = sgn * (L - sgn * d[over, c])
    np.add.at(Fm, tgt_ids, tgt_k[:, None] * d)


def mass_forces(Xm, Xg, Fm):
    """IB2d ``give_Me_Mass_Lagrangian_Force_Densities``.

    The mass point at ``Xg`` is tied to its marker (at ``Xm``) by a spring of
    stiffness ``k``; the *marker* feels ``F = k (Xg - Xm)`` (which is spread to
    the fluid), the ghost feels ``-F`` in
    ``please_Update_Massive_Boundary_Velocity``.  ``F`` is returned because the
    ghost's own dynamics need it too.
    """
    d = Xg - Xm[mass_ids]
    F = mass_k[:, None] * d
    np.add.at(Fm, mass_ids, F)
    return F


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


# ---------------------------------------------------------------------------
# Passive scalar (IB2d's ``concentration`` flag): the split explicit
# advection-diffusion scheme of ``Supp.perform_Time_Split_Upwind`` on the
# collocated node grid, with the *new* velocity u^{n+1}
# ---------------------------------------------------------------------------
def node_velocity():
    """u^{n+1} on the IB2d node grid (the grid the scalar lives on)."""
    u_ib.interpolate(solver.u_)
    d = u_ib.x.array.reshape(-1, 2)
    ux, uy = np.zeros((Ny, Nx)), np.zeros((Ny, Nx))
    ux[gj, gi], uy[gj, gi] = d[:, 0], d[:, 1]
    return ux, uy


def _upwind_x(C, U):
    """IB2d ``give_Necessary_Derivative`` in x: periodic upwind difference.

    The upstream interior loop ``range(1, nx-2)`` never touches column
    ``nx-2``, which therefore stays zero; reproduced verbatim (as is the
    sign <= 0 branch, i.e. forward differences for u <= 0).
    """
    G = np.zeros_like(C)
    s = np.sign(U)
    nx = C.shape[1]
    G[:, 0] = np.where(s[:, 0] <= 0, C[:, 1] - C[:, 0],
                       C[:, 0] - C[:, nx - 1]) / h
    G[:, nx - 1] = np.where(s[:, nx - 1] <= 0, C[:, 0] - C[:, nx - 1],
                            C[:, nx - 1] - C[:, nx - 2]) / h
    j = np.arange(1, nx - 2)
    back = s[:, j] <= 0
    G[:, j] = np.where(back, C[:, j + 1] - C[:, j], C[:, j] - C[:, j - 1]) / h
    return G


def _upwind_y(C, V):
    """IB2d ``give_Necessary_Derivative`` in y (see ``_upwind_x``)."""
    G = np.zeros_like(C)
    s = np.sign(V)
    ny = C.shape[0]
    G[0, :] = np.where(s[0, :] <= 0, C[1, :] - C[0, :],
                       C[0, :] - C[ny - 1, :]) / h
    G[ny - 1, :] = np.where(s[ny - 1, :] <= 0, C[0, :] - C[ny - 1, :],
                            C[ny - 1, :] - C[ny - 2, :]) / h
    i = np.arange(1, ny - 2)
    back = s[i, :] <= 0
    G[i, :] = np.where(back, C[i + 1, :] - C[i, :], C[i, :] - C[i - 1, :]) / h
    return G


def _dd(C, axis):
    """IB2d ``DD``: periodic central second difference along ``axis``."""
    G = np.zeros_like(C)
    n = C.shape[axis]
    if axis == 1:
        G[:, 0] = C[:, 1] - 2 * C[:, 0] + C[:, n - 1]
        G[:, -1] = C[:, 0] - 2 * C[:, -1] + C[:, -2]
        G[:, 1:n - 1] = C[:, 2:n] - 2 * C[:, 1:n - 1] + C[:, :n - 2]
    else:
        G[0, :] = C[1, :] - 2 * C[0, :] + C[n - 1, :]
        G[-1, :] = C[0, :] - 2 * C[-1, :] + C[-2, :]
        G[1:n - 1, :] = C[2:n, :] - 2 * C[1:n - 1, :] + C[:n - 2, :]
    return G / h ** 2


def update_concentration(C, U, V, kd):
    """IB2d ``Supp.perform_Time_Split_Upwind``: an x sub-step, then a y
    sub-step on the updated field (the driver's
    ``please_Update_Adv_Diff_Concentration`` calls exactly this)."""
    C = C + dt * (kd * _dd(C, 1) - U * _upwind_x(C, U))
    C = C + dt * (kd * _dd(C, 0) - V * _upwind_y(C, V))
    return C


C_field = None if conc is None else conc["C0"].copy()


def spread_forces_at_half():
    """All Lagrangian forces at the half-step markers, spread to the grid.

    Returns the mass-point spring force ``F_Mass`` (nonzero only at the ghost
    markers' indices) which the massive-boundary update needs.
    """
    Fm = np.zeros((Nb, 2))
    Xm = marker_array(X_h)
    if sp is not None:
        spring_forces(Xm, Fm)
    if bm is not None:
        beam_forces(Xm, Fm)
    if tg is not None:
        target_forces(Xm, Fm)
    F_mass = None
    if mass is not None:
        F_mass = mass_forces(Xm, mass_pos, Fm)
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


hist = {"t": [0.0], "X": [marker_array(X).copy()]}
if mass is not None:
    hist["Xm"], hist["Vm"] = [mass_pos.copy()], [mass_vel.copy()]
if Nt:
    hist["Xt"] = [tracer_array().copy()]
if C_field is not None:
    hist["C"] = [C_field.copy()]
u0, p0, f0 = grid_fields()
hist["u"], hist["p"], hist["f"] = [u0], [p0], [f0]
write_output(0.0)

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

    # (1b) half-step ghost positions  (IBM_Driver step 1:
    #      please_Move_Massive_Boundary(dt/2, mass_info, mVelocity))
    if mass is not None:
        mass_old = mass_pos.copy()
        mass_pos = mass_pos + 0.5 * dt * mass_vel

    # (2) Lagrangian forces at X_h, spread with the 4-point kernel; the mass
    #     springs contribute F_Mass at the ghost markers' indices
    F_mass = spread_forces_at_half()

    # (2b) arbitrary external force, added to the IB force density
    if ext_mod is not None:
        add_external_force(ctime)

    # (2c) massive ghost particles (IBM_Driver steps 2-3): the *same* F_Mass
    #      updates the half-step velocity, moves the ghost by dt from its
    #      position at the top of the step, and then updates the full-step
    #      velocity from the old velocity
    if mass is not None:
        acc = F_mass / mass_M[:, None]
        if GRAV_VEC is not None:
            acc = acc - G_ACC * GRAV_VEC[None, :]
        mass_vel_half = mass_vel - 0.5 * dt * acc
        mass_pos = mass_old + dt * mass_vel_half
        mass_vel = mass_vel - dt * acc

    # (3) two-stage fluid solve: u_h, u^{n+1}, p^{n+1/2}
    solver.solve_one_step()

    # (4) markers
    ib.fluid_to_solid(ib_velocity(solver.u_h)._cpp_object, U_s._cpp_object)
    X.x.array[:] += dt * U_s.x.array
    X.x.scatter_forward()
    wrap_periodic(X, dof_of_ib2d)

    # (4b) passive tracers ride the same interpolation (no force feedback)
    if Nt:
        ib.evaluate_current_points(Xt._cpp_object)
        ib.fluid_to_solid(ib_velocity(solver.u_h)._cpp_object, Ut._cpp_object)
        Xt.x.array[:] += dt * Ut.x.array
        Xt.x.scatter_forward()
        wrap_periodic(Xt, dof_of_tracer)

    # (4c) passive scalar (the driver updates it with u^{n+1} as well)
    if C_field is not None:
        _ux, _uy = node_velocity()
        C_field = update_concentration(C_field, _ux, _uy, conc["k"])

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
        if mass is not None:
            hist["Xm"].append(mass_pos.copy())
            hist["Vm"].append(mass_vel.copy())
        if Nt:
            hist["Xt"].append(tracer_array().copy())
        if C_field is not None:
            hist["C"].append(C_field.copy())
        ug, pg, fg = grid_fields()
        hist["u"].append(ug)
        hist["p"].append(pg)
        hist["f"].append(fg)
        write_output(t)
        if comm.rank == 0:
            Xm = hist["X"][-1]
            extra = ""
            if mass is not None:
                extra += (f"|dXm|max={np.abs(mass_pos - mass_anchor).max():.4f}  ")
            if C_field is not None:
                extra += (f"Cmax={C_field.max():.6e}  sum(C)={C_field.sum():.6e}  ")
            print(f"step {step+1:5d}  t={t:.5f}  {extra}"
                  f"umax={np.abs(ug).max():.3f}  "
                  f"wall={time.perf_counter()-tic:.1f}s", flush=True)

elapsed = time.perf_counter() - tic
xf.close()
xs.close()
fname = os.path.join(out_dir, "afsi_result.npz")
np.savez_compressed(fname,
                    t=np.array(hist["t"]), X=np.array(hist["X"]),
                    u=np.array(hist["u"]), p=np.array(hist["p"]),
                    f=np.array(hist["f"]),
                    dx=Lx / Nx, dy=Ly / Ny, dt=dt, nx=Nx, ny=Ny,
                    n_tracers=Nt, n_mass=0 if mass is None else Nm,
                    mass_ids=np.array([]) if mass is None else mass_ids,
                    mass_k=np.array([]) if mass is None else mass_k,
                    mass_M=np.array([]) if mass is None else mass_M,
                    **({"Xm": np.array(hist["Xm"]), "Vm": np.array(hist["Vm"])}
                       if mass is not None else {}),
                    **({"tracers": np.array(hist["Xt"])} if Nt else {}),
                    **({"C": np.array(hist["C"])} if C_field is not None else {}))
if comm.rank == 0:
    print(f"done in {elapsed:.1f}s -> {fname}")
