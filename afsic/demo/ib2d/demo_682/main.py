"""IB2d example "Rubberband sink" (matIB2d
``Examples/Example_Concentration_Dynamics/Rubberband_Sink_64``) in AFSI.

Third member of the concentration-source family, and the only one that uses the
**reaction/sink** model (``source = 3``): the 249 markers of the contracting
``rubberband`` absorb the scalar at the rate

    fs = k_source * CL,   k_source = -2.0        (source = 3)

so a *uniform* initial field ``C0 = 1`` (with the file's missing ``kDiff``
header, see below) is drained along the band.  Shipped: 1 x 1 periodic box,
``mu = 0.01``, ``rho = 1``, ``64 x 64``, ``dt = 3.125e-5``, ``T = 2``,
``kDiff = 0.01``.

pyIB2d has **no** concentration source at all (upstream gap), so both codes run
the ported MATLAB routines of ``please_Find_Source_For_Concentration.m`` +
``please_Update_Adv_Diff_Concentration_Source.m``: the reference splices them
into the pyIB2d driver (``run_reference.py``), the port implements them with
AFSI's own IB interpolation/spreading (``main.py``); see demo_680's readme for
the full derivation, the verified identities (``sum(Fs) dx dy = sum(fs) ds``,
``CL(1) = 1``) and the ``input2d`` quirk ``c_inf = 0.1`` (unused by model 3).

Run::

    conda activate afsi-dolfinx
    python run_reference.py --tend 0.1
    TFINAL=0.1 python main.py
    python compare.py
"""
import importlib
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

comm = MPI.COMM_WORLD
assert comm.size == 1, "this demo is serial (IB coupling of afsic gathers on rank 0)"


def user_module(name):
    """Import an example-specific module from ``ib2d_input/`` (pyIB2d imports
    the very same modules from its run directory)."""
    if not os.path.exists(os.path.join(IB2D_EXAMPLE, name + ".py")):
        return None
    sys.path.insert(0, IB2D_EXAMPLE)
    try:
        return importlib.import_module(name)
    finally:
        sys.path.remove(IB2D_EXAMPLE)


ext_mod = user_module("please_Compute_External_Forcing")
gen_mod = user_module("give_Me_General_User_Defined_Force_Densities")
act_mod = user_module("give_Muscle_Activation")
upd_springs = user_module("update_Springs")
upd_dsprings = user_module("update_Damped_Springs")
upd_target = user_module("update_Target_Point_Positions")
upd_beams = user_module("update_Beams")
upd_noninv = user_module("update_nonInv_Beams")

# ---------------------------------------------------------------------------
# Parameters: read from the IB2d example, optionally overridden
# ---------------------------------------------------------------------------
ex = ib2d_io.load_example(IB2D_EXAMPLE)
P = ex["params"]


def flag(name, default=0):
    """``input2d`` flag as an int."""
    return int(float(P.get(name, default)))


rho, mu = float(P["rho"]), float(P["mu"])
Lx, Ly = float(P["Lx"]), float(P["Ly"])
Nx = int(os.environ.get("NX", P["Nx"]))
Ny = int(os.environ.get("NY", round(Nx * Ly / Lx)))
dx, dy = Lx / Nx, Ly / Ny
dt = float(os.environ.get("DT", P["dt"]))
T = float(os.environ.get("TFINAL", P["Tfinal"]))
out_dir = os.environ.get("OUTPUT_PATH", os.path.join(_here, "plot"))
os.makedirs(out_dir, exist_ok=True)

GRAD_DIV = float(os.environ.get("GRAD_DIV", "0"))

# the pyIB2d driver advances while (current_time < Tfinal) with
# current_time += dt, so rounding noise can add one last step; replicate it so
# that the dump schedules of the two codes end on the same step
_n, _t = 0, 0.0
while _t < T:
    _t += dt
    _n += 1
num_steps = _n
# a few examples ship no ``print_dump``; dump ~20 frames
print_dump = int(os.environ.get("PRINT_DUMP", P.get("print_dump") or
                                max(1, num_steps // 20)))
ds_ib2d = Lx / (2 * Nx)            # IB2d sets ds = Lx / (2 Nx)

# mechanisms the template does not implement
for _flag_name in ("mass_pts", "3_element_muscle", "electro_phys",
                   "boussinesq"):
    assert not flag(_flag_name), \
        f"this template does not implement {_flag_name} = 1 yet"

X_ib2d = ex["X"].copy()
Nb = X_ib2d.shape[0]
# dolfinx merges coincident mesh vertices, so markers that IB2d keeps as
# separate Lagrangian points but places at the same coordinates (e.g. the
# junction points of a bifurcation) need a distinct vertex of their own; the
# offset is orders of magnitude below the discretisation error and well above
# the mesh-vertex merge tolerance
for _i in range(1, len(X_ib2d)):
    while np.any(np.all(np.isclose(X_ib2d[:_i], X_ib2d[_i], atol=1e-13,
                                   rtol=0.0), axis=1)):
        X_ib2d[_i, 0] += 1e-6 * dx

sp = ex.get("springs")                       # optional springs (lin/non-lin)
dsp = ex.get("d_springs")                    # optional damped springs
bm = ex.get("beams")                         # optional invariant beams
nb = ex.get("noninv_beams")                  # optional non-invariant beams
tg = ex.get("targets")                       # optional target points
mus = ex.get("muscles")                      # optional FV_LT muscle bands
por = ex.get("porous")                       # optional porous points
pe = ex.get("poroelastic")                   # optional poroelastic points
uf = ex.get("user_force")                    # optional generic user force
brink = ex.get("brinkman")                   # optional Brinkman resistance
conc = ex.get("concentration")               # optional passive scalar
trs = ex.get("tracers")                      # optional passive tracers
Nt = 0 if trs is None else trs.shape[0]

# ``*.concentration`` files of this family have no ``kDiff`` header line, so
# ``ib2d_io`` returns ``k = None`` and the coefficient has to come from
# ``input2d`` (an upstream inconsistency, see readme)
if conc is not None and conc["k"] is None:
    conc["k"] = float(P["kDiff"])

# every example-specific module must be switched on in input2d, and vice versa
assert (ext_mod is not None) == bool(flag("arb_ext_force")), \
    "please_Compute_External_Forcing.py must exist iff arb_ext_force = 1"
assert (gen_mod is not None) == bool(flag("user_force_model")), \
    "give_Me_General_User_Defined_Force_Densities.py must exist iff " \
    "user_force_model = 1"
assert (act_mod is not None) == bool(mus is not None), \
    "give_Muscle_Activation.py must exist iff FV_LT_muscle = 1"
for _mod, _flg, _fn in ((upd_springs, "update_springs", "update_Springs.py"),
                        (upd_dsprings, "update_damp_springs",
                         "update_Damped_Springs.py"),
                        (upd_target, "update_target",
                         "update_Target_Point_Positions.py"),
                        (upd_beams, "update_beams", "update_Beams.py"),
                        (upd_noninv, "update_nonInv_beams",
                         "update_nonInv_Beams.py")):
    assert (_mod is not None) == bool(flag(_flg)), \
        f"{_fn} must exist iff {_flg} = 1"

# ---------------------------------------------------------------------------
# Structure bookkeeping: the mutable IB2d "info" arrays the driver hands to the
# update modules (columns as read by pyIB2d's IBM_Driver).
# ---------------------------------------------------------------------------
springs_info = None
if sp is not None:
    springs_info = np.column_stack([sp["conn"][:, 0], sp["conn"][:, 1],
                                    sp["k"], sp["L"], sp["alpha"]]).astype(float)
d_springs_info = None
if dsp is not None:
    d_springs_info = np.column_stack([dsp["conn"][:, 0], dsp["conn"][:, 1],
                                      dsp["k"], dsp["L"], dsp["b"]]).astype(float)
beams_info = None
if bm is not None:
    beams_info = np.column_stack([bm["p1"], bm["p2"], bm["p3"], bm["kb"],
                                  bm["C"]]).astype(float)
noninv_info = None
if nb is not None:
    noninv_info = np.column_stack([nb["p1"], nb["p2"], nb["p3"], nb["kb"],
                                   nb["C"]]).astype(float)
target_info = None
if tg is not None:
    target_info = np.empty((len(tg["ids"]), 4))
    target_info[:, 0] = tg["ids"]
    target_info[:, 1] = X_ib2d[tg["ids"], 0]     # anchors = initial positions
    target_info[:, 2] = X_ib2d[tg["ids"], 1]
    target_info[:, 3] = tg["k"]
mus_info = None
if mus is not None:
    mus_info = np.column_stack([mus["conn"][:, 0], mus["conn"][:, 1],
                                mus["LFO"], mus["SK"], mus["a"], mus["b"],
                                mus["Fmax"]]).astype(float)
pe_ids = None if pe is None else pe["ids"]
pe_c = None if pe is None else pe["c"]
F_Poro_prev = None if pe is None else np.zeros((len(pe_ids), 2))
general_force = None if uf is None else uf

if comm.rank == 0:
    desc = [f"{Nb} markers"]
    if sp is not None:
        desc.append(f"{len(sp['conn'])} springs (k={sp['k'][0]:.3g}, "
                    f"alpha={sp['alpha'].min():g}..{sp['alpha'].max():g})")
    if dsp is not None:
        desc.append(f"{len(dsp['conn'])} damped springs "
                    f"(kappa={dsp['k'][0]:.3g}, b={dsp['b'][0]:.3g})")
    if bm is not None:
        desc.append(f"{len(bm['p1'])} invariant beams "
                    f"(kb={bm['kb'][0]:.3g})")
    if nb is not None:
        desc.append(f"{len(nb['p1'])} non-invariant beams "
                    f"(kb={nb['kb'][0]:.3g})")
    if tg is not None:
        desc.append(f"{len(tg['ids'])} target points (k={tg['k'][0]:.3g})")
    if mus is not None:
        desc.append(f"{len(mus['conn'])} FV_LT muscles "
                    f"(Fmax={mus['Fmax'][0]:.3g}, L_opt={mus['LFO'][0]:g})")
    if por is not None:
        desc.append(f"{len(por['ids'])} porous points "
                    f"(kappa={por['kappa'][0]:.3g})")
    if pe is not None:
        desc.append(f"{len(pe_ids)} poroelastic points (c={pe_c[0]:.3g})")
    if uf is not None:
        desc.append(f"user force (general_force shape {np.shape(uf)})")
    if brink is not None:
        desc.append(f"Brinkman map (k={brink['k']:g}, "
                    f"{int((brink['Brink'] != 0).sum())} blocked nodes)")
    if Nt:
        desc.append(f"{Nt} passive tracers")
    if conc is not None:
        desc.append(f"passive scalar (kDiff={conc['k']:g}, "
                    f"C0 in [{conc['C0'].min():g}, {conc['C0'].max():g}])")
    print(f"IB2d example : {IB2D_EXAMPLE} (index_base={ex['index_base']})")
    print(f"rho={rho} mu={mu} box={Lx}x{Ly} grid={Nx}x{Ny} dt={dt} T={T} "
          f"steps={num_steps} ds={ds_ib2d}")
    print("structure    : " + ", ".join(desc))
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
# Structure: zero-stiffness FE graph carrying the marker dofs
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
assert len(np.unique(dof_of_ib2d)) == Nb, "dof matching is not injective"
assert np.allclose(dof_rows[dof_of_ib2d], X_ib2d, atol=1e-12)


def marker_array(fn):
    """Function on the structure space in IB2d marker order."""
    return fn.x.array.reshape(-1, 2)[dof_of_ib2d]


def write_fun(fn, Xm):
    """Copy IB2d-ordered markers into an FE function of the structure space."""
    tmp = np.empty_like(Xm)
    tmp[dof_of_ib2d] = Xm
    fn.x.array[:] = tmp.ravel()
    fn.x.scatter_forward()


def write_markers(Xm):
    """Copy IB2d-ordered markers back into X."""
    write_fun(X, Xm)


def wrap_periodic(fn, dofs):
    """IB2d folds every marker/tracer into [0, L) after each move."""
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


# ---------------------------------------------------------------------------
# Structure forces (IB2d formulas, evaluated at X^{n+1/2})
# ---------------------------------------------------------------------------
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
    """IB2d ``give_Me_Spring_Lagrangian_Force_Densities`` (linear and
    non-linear): ``sF = (alpha+1)/2 k (L - L0)^alpha dhat`` to the leader,
    ``-sF`` to the follower."""
    a, b = springs_info[:, 0].astype(np.int64), springs_info[:, 1].astype(np.int64)
    k, L0, alpha = springs_info[:, 2], springs_info[:, 3], springs_info[:, 4]
    d = Xm[b] - Xm[a]
    d[:, 0] = _min_image(d[:, 0], Lx)
    d[:, 1] = _min_image(d[:, 1], Ly)
    L = np.hypot(d[:, 0], d[:, 1])
    sF = 0.5 * (alpha + 1.0) * k * np.power(L - L0, alpha) / L
    np.add.at(Fm[:, 0], a, sF * d[:, 0])
    np.add.at(Fm[:, 1], a, sF * d[:, 1])
    np.add.at(Fm[:, 0], b, -sF * d[:, 0])
    np.add.at(Fm[:, 1], b, -sF * d[:, 1])


def damped_spring_forces(Xm, Xm_prev, Fm):
    """IB2d ``give_Me_Damped_Springs_Lagrangian_Force_Densities``:
    ``sF = kappa (L - L_rest) dhat - b dV_leader`` with ``dV`` the leader's
    displacement velocity between the previous half step and this one."""
    a, b = d_springs_info[:, 0].astype(np.int64), \
        d_springs_info[:, 1].astype(np.int64)
    k, L0, beta = d_springs_info[:, 2], d_springs_info[:, 3], \
        d_springs_info[:, 4]
    dx_ = _min_image(Xm[b, 0] - Xm[a, 0], Lx)
    dy_ = _min_image(Xm[b, 1] - Xm[a, 1], Ly)
    L = np.hypot(dx_, dy_)
    Ls = np.where(L > 0, L, 1.0)               # guard the measure-zero L = 0
    dVx = _min_image(Xm[a, 0] - Xm_prev[a, 0], Lx) / dt
    dVy = _min_image(Xm[a, 1] - Xm_prev[a, 1], Ly) / dt
    sFx = k * (L - L0) * dx_ / Ls - beta * dVx
    sFy = k * (L - L0) * dy_ / Ls - beta * dVy
    np.add.at(Fm[:, 0], a, sFx)
    np.add.at(Fm[:, 1], a, sFy)
    np.add.at(Fm[:, 0], b, -sFx)
    np.add.at(Fm[:, 1], b, -sFy)


def beam_forces(Xm, Fm):
    """Invariant (cross-product) beams, IB2d
    ``give_Me_Beam_Lagrangian_Force_Densities``."""
    p1, p2, p3 = (beams_info[:, i].astype(np.int64) for i in range(3))
    kb, C = beams_info[:, 3], beams_info[:, 4]
    Xp, Xq, Xr = Xm[p1].copy(), Xm[p2].copy(), Xm[p3].copy()
    Xp, Xq, Xr = _beam_boundary(Xp, Xq, Xr, Lx)
    for c in (0, 1):
        Xp[:, c], Xq[:, c], Xr[:, c] = _beam_boundary(Xp[:, c], Xq[:, c],
                                                      Xr[:, c], Ly)
    S = ((Xr[:, 0] - Xq[:, 0]) * (Xq[:, 1] - Xp[:, 1])
         - (Xr[:, 1] - Xq[:, 1]) * (Xq[:, 0] - Xp[:, 0]))
    K = kb * (S - C)
    np.add.at(Fm[:, 0], p1, K * (Xr[:, 1] - Xq[:, 1]))
    np.add.at(Fm[:, 1], p1, -K * (Xr[:, 0] - Xq[:, 0]))
    np.add.at(Fm[:, 0], p2, K * ((Xq[:, 1] - Xp[:, 1]) + (Xr[:, 1] - Xq[:, 1])))
    np.add.at(Fm[:, 1], p2, -K * ((Xr[:, 0] - Xq[:, 0]) + (Xq[:, 0] - Xp[:, 0])))
    np.add.at(Fm[:, 0], p3, K * (Xq[:, 1] - Xp[:, 1]))
    np.add.at(Fm[:, 1], p3, -K * (Xq[:, 0] - Xp[:, 0]))


def noninv_beam_forces(Xm, Fm):
    """Non-invariant beams, IB2d
    ``give_Me_nonInv_Beam_Lagrangian_Force_Densities``:
    ``r = (X_p + X_r - 2 X_q) - C``; ``F_q += 2 k r``, ``F_p -= k r``,
    ``F_r -= k r``."""
    p1, p2, p3 = (noninv_info[:, i].astype(np.int64) for i in range(3))
    kb = noninv_info[:, 3]
    cx, cy = noninv_info[:, 4], noninv_info[:, 5]
    r = np.column_stack([Xm[p1, 0] + Xm[p3, 0] - 2.0 * Xm[p2, 0] - cx,
                         Xm[p1, 1] + Xm[p3, 1] - 2.0 * Xm[p2, 1] - cy])
    np.add.at(Fm, p2, 2.0 * kb[:, None] * r)
    np.add.at(Fm, p1, -kb[:, None] * r)
    np.add.at(Fm, p3, -kb[:, None] * r)


def target_forces(Xm, Fm):
    """IB2d ``give_Me_Target_Lagrangian_Force_Densities``: ``k (X_anchor-X)``
    with the |d| > L/2 periodic fold (the anchors come from ``target_info``, so
    an ``update_Target_Point_Positions.py`` module can move them)."""
    ids = target_info[:, 0].astype(np.int64)
    d = np.column_stack([target_info[:, 1] - Xm[ids, 0],
                         target_info[:, 2] - Xm[ids, 1]])
    for c, L in enumerate((Lx, Ly)):
        over = np.abs(d[:, c]) > L / 2
        sgn = np.sign(d[over, c])
        d[over, c] = sgn * (L - sgn * d[over, c])
    np.add.at(Fm, ids, target_info[:, 3][:, None] * d)


def muscle_forces(Xm, Xm_prev, t, Fm):
    """FV_LT muscle bands, IB2d ``give_Muscle_Force_Densities``.

    ``Fm = a_f(t, x_i) Fmax exp(-((Q-1)/SK)^2) (b Fmax - a v)/(Fmax (v+b))``
    with ``Q = LF/L_opt`` and ``v = |LF(X_h) - LF(X_h_prev)|/dt``; the
    activation is the example's own ``give_Muscle_Activation``.
    """
    a, b = mus_info[:, 0].astype(np.int64), mus_info[:, 1].astype(np.int64)
    LFO, SK, ha, hb, Fmax = (mus_info[:, i] for i in range(2, 7))
    dxx = Xm[b, 0] - Xm[a, 0]
    dyy = Xm[b, 1] - Xm[a, 1]
    LF = np.hypot(dxx, dyy)
    LF_P = np.hypot(Xm_prev[b, 0] - Xm_prev[a, 0],
                    Xm_prev[b, 1] - Xm_prev[a, 1])
    v = np.abs(LF - LF_P) / dt
    # NB: the example's activation takes 1-D coordinate arrays (the driver
    # stores xLag/yLag as separate 1-D arrays): xPt = leader x's, xLag = all x's
    Fmag = act_mod.give_Muscle_Activation(v, LF, LFO, SK, ha, hb, Fmax, t,
                                          Xm[a, 0].copy(), Xm[:, 0].copy())
    mFx, mFy = Fmag * dxx / LF, Fmag * dyy / LF
    np.add.at(Fm[:, 0], a, mFx)
    np.add.at(Fm[:, 1], a, mFy)
    np.add.at(Fm[:, 0], b, -mFx)
    np.add.at(Fm[:, 1], b, -mFy)


def user_forces(Xm, Xm_prev, t, Fm):
    """Generic user-defined force interface, IB2d
    ``give_Me_General_User_Defined_Force_Densities`` (called with the raw
    ``.user_force`` array, exactly as pyIB2d does)."""
    fx, fy = gen_mod.give_Me_General_User_Defined_Force_Densities(
        ds_ib2d, Nb, Xm[:, 0].copy(), Xm[:, 1].copy(),
        Xm_prev[:, 0].copy(), Xm_prev[:, 1].copy(), dt, t, general_force)
    Fm[:, 0] += fx
    Fm[:, 1] += fy


def spring_only_forces(Xm):
    """Spring force alone -- IB2d stores ``F_Poro = F_spring`` at the
    poroelastic points."""
    Fp = np.zeros((Nb, 2))
    spring_forces(Xm, Fp)
    return Fp


def spread_forces_at_half(Xh_prev, t):
    """Total Lagrangian force at X_h (all active structure models) x ds.

    Returns the raw (un-scaled) per-marker force, which the porous and
    poroelastic blocks need as IB2d's ``F_Lag``/``F_Poro``.
    """
    Xm = marker_array(X_h)
    Fm = np.zeros((Nb, 2))
    if springs_info is not None:
        spring_forces(Xm, Fm)
    if d_springs_info is not None:
        damped_spring_forces(Xm, Xh_prev, Fm)
    if beams_info is not None:
        beam_forces(Xm, Fm)
    if noninv_info is not None:
        noninv_beam_forces(Xm, Fm)
    if target_info is not None:
        target_forces(Xm, Fm)
    if mus_info is not None:
        muscle_forces(Xm, Xh_prev, t, Fm)
    if gen_mod is not None:
        user_forces(Xm, Xh_prev, t, Fm)
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


# sanity checks of the vectorised forces against plain per-row loops
if springs_info is not None or beams_info is not None \
        or noninv_info is not None:
    rng = np.random.default_rng(0)
    Xp = X_ib2d + 1e-3 * rng.standard_normal(X_ib2d.shape)
    msg = []
    if springs_info is not None:
        Fv = np.zeros((Nb, 2))
        spring_forces(Xp, Fv)
        Fl = np.zeros((Nb, 2))
        for aa, bb, kk, LL, al in springs_info:
            aa, bb = int(aa), int(bb)
            d = Xp[bb] - Xp[aa]
            L = np.hypot(*d)
            sF = 0.5 * (al + 1.0) * kk * (L - LL) ** al / L
            Fl[aa] += sF * d
            Fl[bb] -= sF * d
        e = np.abs(Fv - Fl).max() / np.abs(Fl).max()
        msg.append(f"springs {e:.2e}")
        assert e < 1e-12
    if beams_info is not None:
        Fv = np.zeros((Nb, 2))
        beam_forces(Xp, Fv)
        Fl = np.zeros((Nb, 2))
        for i1, i2, i3, kk, CC in beams_info:
            i1, i2, i3 = int(i1), int(i2), int(i3)
            ax, ay = Xp[i1]
            bx, by = Xp[i2]
            cx, cy = Xp[i3]
            S = (cx - bx) * (by - ay) - (cy - by) * (bx - ax)
            K = kk * (S - CC)
            Fl[i1] += (K * (cy - by), -K * (cx - bx))
            Fl[i2] += (K * ((by - ay) + (cy - by)),
                       -K * ((cx - bx) + (bx - ax)))
            Fl[i3] += (K * (by - ay), -K * (bx - ax))
        e = np.abs(Fv - Fl).max() / np.abs(Fl).max()
        msg.append(f"beams {e:.2e}")
        assert e < 1e-12
    if noninv_info is not None:
        Fv = np.zeros((Nb, 2))
        noninv_beam_forces(Xp, Fv)
        Fl = np.zeros((Nb, 2))
        for i1, i2, i3, kk, cxx, cyy in noninv_info:
            i1, i2, i3 = int(i1), int(i2), int(i3)
            r = Xp[i1] + Xp[i3] - 2.0 * Xp[i2] - np.array([cxx, cyy])
            Fl[i2] += 2.0 * kk * r
            Fl[i1] -= kk * r
            Fl[i3] -= kk * r
        e = np.abs(Fv - Fl).max() / np.abs(Fl).max()
        msg.append(f"non-invariant beams {e:.2e}")
        assert e < 1e-12
    if comm.rank == 0:
        print("force checks (perturbed config): " + ", ".join(msg))


def porous_slip_update(Xm, Fm, dt_):
    """pyIB2d ``please_Compute_Porous_Slip_Velocity`` + the driver's update.

    The porous points are stored in file order, so the i+k neighbours of the
    finite-difference stencils are the *porous-list* neighbours; the tangent is
    a fourth-order one-sided/central difference chosen by the stencil flag,
    the normal is ``(y_s, -x_s)/|(x_s, y_s)|``, and the point slides by
    ``-kappa F_lag . n / |(x_s, y_s)|`` along ``n``.
    """
    ids, kappa, stencil = por["ids"], por["kappa"], por["stencil"]
    Np = len(ids)
    xL, yL = Xm[ids, 0], Xm[ids, 1]
    xL_s, yL_s = np.zeros(Np), np.zeros(Np)
    for i in range(Np):
        c = stencil[i]
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
            xL_s[i] = (1 / 12 * xL[i - 2] - 2 / 3 * xL[i - 1]
                       + 2 / 3 * xL[i + 1] - 1 / 12 * xL[i + 2]) / ds_ib2d
            yL_s[i] = (1 / 12 * yL[i - 2] - 2 / 3 * yL[i - 1]
                       + 2 / 3 * yL[i + 1] - 1 / 12 * yL[i + 2]) / ds_ib2d
        elif c == 1:
            xL_s[i] = (-1 / 12 * xL[i - 3] + 0.5 * xL[i - 2]
                       - 1.5 * xL[i - 1] + 5 / 6 * xL[i]
                       + 0.25 * xL[i + 1]) / ds_ib2d
            yL_s[i] = (-1 / 12 * yL[i - 3] + 0.5 * yL[i - 2]
                       - 1.5 * yL[i - 1] + 5 / 6 * yL[i]
                       + 0.25 * yL[i + 1]) / ds_ib2d
        elif c == 2:
            xL_s[i] = (0.25 * xL[i - 4] - 4 / 3 * xL[i - 3] + 3 * xL[i - 2]
                       - 4 * xL[i - 1] + 25 / 12 * xL[i]) / ds_ib2d
            yL_s[i] = (0.25 * yL[i - 4] - 4 / 3 * yL[i - 3] + 3 * yL[i - 2]
                       - 4 * yL[i - 1] + 25 / 12 * yL[i]) / ds_ib2d
    sqrtN = np.hypot(xL_s, yL_s)
    nX, nY = yL_s / sqrtN, -xL_s / sqrtN
    Up_X = -kappa * Fm[ids, 0] * nX / sqrtN
    Up_Y = -kappa * Fm[ids, 1] * nY / sqrtN
    Xm[ids, 0] -= dt_ * Up_X * nX
    Xm[ids, 1] -= dt_ * Up_Y * nY


def poroelastic_disp(F_Poro):
    """IB2d's poroelastic displacement ``dt * F_spring / (mu c)``."""
    return F_Poro / (mu * pe_c[:, None])


# ---------------------------------------------------------------------------
# Update modules (the driver calls them right after the half-step move, before
# the force computation, with ``current_time`` = start-of-step time; note that
# it hands ``update_Springs`` the *full-step* positions xLag/yLag, IBM_Driver.py
# line 696)
# ---------------------------------------------------------------------------
def update_structure(t, Xfull):
    global springs_info, d_springs_info, target_info, beams_info, noninv_info
    if upd_springs is not None:
        springs_info = upd_springs.update_Springs(
            dt, t, Xfull[:, 0].copy(), Xfull[:, 1].copy(), springs_info)
    if upd_dsprings is not None:
        d_springs_info = upd_dsprings.update_Damped_Springs(
            dt, t, d_springs_info)
    if upd_target is not None:
        target_info = upd_target.update_Target_Point_Positions(
            dt, t, target_info)
    if upd_beams is not None:
        beams_info = upd_beams.update_Beams(dt, t, beams_info)
    if upd_noninv is not None:
        # pyIB2d's driver passes ``beams_info`` here by mistake (IBM_Driver.py
        # line 711); the template passes the non-invariant array, which is what
        # the .m module's signature means, so that these examples can run at all
        noninv_info = upd_noninv.update_nonInv_Beams(dt, t, noninv_info)


# ---------------------------------------------------------------------------
# Arbitrary external force (IB2d's arb_ext_force): the same ported module the
# pyIB2d reference imports, evaluated at the Q1 node dofs of this mesh (which
# coincide with IB2d's collocated fluid grid) and added to solver.f (Q2)
# through its Q2 interpolant
# ---------------------------------------------------------------------------
GRID_INFO = [Nx, Ny, Lx, Ly, dx, dy, int(P["supp"]), Nb, ds_ib2d]
XNODES = np.arange(Nx) * dx
YNODES = np.arange(Ny) * dy
# map every dof of the Q1 velocity space onto the node grid (wrapped images of
# the periodic boundary included, so that the force field is periodic)
Qcoords = V_ib.tabulate_dof_coordinates()[:, :2]
gi = np.rint(Qcoords[:, 0] / dx).astype(int) % Nx
gj = np.rint(Qcoords[:, 1] / dy).astype(int) % Ny
assert np.allclose(Qcoords[:, 0], np.rint(Qcoords[:, 0] / dx) * dx, atol=1e-12)
# the mesh is *not* periodic, so the Q1 space also carries "image" nodes at
# x = Lx / y = Ly whose dofs are empty: reading a **periodic** IB2d field back
# onto the (Ny, Nx) array must skip them (otherwise the boundary column/row of
# the extracted field would be overwritten with zeros -- the source spreading of
# demo_680..684 is the one field that is periodic by construction and not
# smoothed by ``periodic_fold``)
iu_ib = np.rint(Qcoords[:, 0] / dx).astype(int)
ju_ib = np.rint(Qcoords[:, 1] / dy).astype(int)
ib_keep = (iu_ib < Nx) & (ju_ib < Ny)
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


f_br1 = fem.Function(V_ib, name="f_brink_nodal")
f_br2 = fem.Function(solver.V, name="f_brink_q2")


def add_brinkman():
    """IB2d's Brinkman resistance term, F -= mu * Brink(x) * u^n."""
    u_ib.interpolate(solver.u_n)
    u_dof = u_ib.x.array.reshape(-1, 2)
    b = mu * brink["Brink"][gj, gi]
    f_br1.x.array[:] = np.column_stack([-b * u_dof[:, 0],
                                        -b * u_dof[:, 1]]).ravel()
    f_br2.interpolate(f_br1)
    solver.f.x.array[:] += f_br2.x.array[:]


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


def _upwind(C, U, dz, axis):
    """IB2d ``give_Necessary_Derivative``: periodic upwind difference.

    The upstream interior loop ``range(1, n-2)`` never touches the second-to-
    last row/column, which therefore stays zero; reproduced verbatim (as is the
    sign <= 0 branch, i.e. forward differences for u <= 0).
    """
    G = np.zeros_like(C)
    s = np.sign(U)
    n = C.shape[axis]
    if axis == 1:
        G[:, 0] = np.where(s[:, 0] <= 0, C[:, 1] - C[:, 0],
                           C[:, 0] - C[:, n - 1]) / dz
        G[:, n - 1] = np.where(s[:, n - 1] <= 0, C[:, 0] - C[:, n - 1],
                               C[:, n - 1] - C[:, n - 2]) / dz
        j = np.arange(1, n - 2)
        G[:, j] = np.where(s[:, j] <= 0, C[:, j + 1] - C[:, j],
                           C[:, j] - C[:, j - 1]) / dz
    else:
        G[0, :] = np.where(s[0, :] <= 0, C[1, :] - C[0, :],
                           C[0, :] - C[n - 1, :]) / dz
        G[n - 1, :] = np.where(s[n - 1, :] <= 0, C[0, :] - C[n - 1, :],
                               C[n - 1, :] - C[n - 2, :]) / dz
        i = np.arange(1, n - 2)
        G[i, :] = np.where(s[i, :] <= 0, C[i + 1, :] - C[i, :],
                           C[i, :] - C[i - 1, :]) / dz
    return G


def _dd(C, dz, axis):
    """IB2d ``DD``: periodic central second difference along ``axis``."""
    G = np.zeros_like(C)
    n = C.shape[axis]
    if axis == 1:
        G[:, 0] = C[:, 1] - 2 * C[:, 0] + C[:, n - 1]
        G[:, -1] = C[:, 0] - 2 * C[:, n - 1] + C[:, n - 2]
        G[:, 1:n - 1] = C[:, 2:n] - 2 * C[:, 1:n - 1] + C[:, :n - 2]
    else:
        G[0, :] = C[1, :] - 2 * C[0, :] + C[n - 1, :]
        G[-1, :] = C[0, :] - 2 * C[n - 1, :] + C[n - 2, :]
        G[1:n - 1, :] = C[2:n, :] - 2 * C[1:n - 1, :] + C[:n - 2, :]
    return G / dz ** 2


def update_concentration(C, U, V, kd):
    """IB2d ``Supp.perform_Time_Split_Upwind``: an x sub-step, then a y
    sub-step on the updated field."""
    C = C + dt * (kd * _dd(C, dx, 1) - U * _upwind(C, U, dx, 1))
    C = C + dt * (kd * _dd(C, dy, 0) - V * _upwind(C, V, dy, 0))
    return C


# ---------------------------------------------------------------------------
# Concentration source/sink (matIB2d
# ``IBM_Blackbox/please_Find_Source_For_Concentration.m`` +
# ``please_Update_Adv_Diff_Concentration_Source.m``).
#
# **pyIB2d has no concentration source at all** -- its driver never reads the
# ``source`` / ``k_source`` / ``c_inf`` lines of ``input2d`` and always calls
# the *split* upwind stepper, so every example of this family would integrate a
# plain advection-diffusion problem (and with the ``C0 = 0`` shipped by the
# rubberband examples the scalar would stay identically zero).  Both codes here
# therefore run a transcription of the two MATLAB files:
#
#     CL  = dx dy sum_j C(yInd, xInd) phi_X phi_Y      (delta interpolation)
#     fs  = k          (model 1)  /  k (ci - CL) (model 2)  /  k CL (model 3)
#     Fs  = sum_i fs_i ds delta(x - X_i) delta(y - Y_i)   (spread of the source)
#     C^{n+1} = C^n + dt ( k lap C^n - u^{n+1}.grad C^n + Fs )   (un-split!)
#
# The *un-split* form is what matIB2d runs whenever ``source > 0`` (its
# ``perform_Upwind_source``); its upwind helper also fills the second-to-last
# row/column, which pyIB2d's standalone ``give_Necessary_Derivative`` does not
# (see the readme).  The grid transfer is AFSI's own IB machinery (the same
# Peskin spreading/interpolation that carries the forces), the reference uses
# pyIB2d's ``Supp`` delta functions, so the comparison checks the transfer, not
# just the algebra.
# ---------------------------------------------------------------------------
source_model = int(float(P.get("source", 0)))
k_source = float(P.get("k_source", 0.0))
c_inf = float(P.get("c_inf", 1.0))
assert source_model in (0, 1, 2, 3), f"unknown source model {source_model}"
assert not source_model or conc is not None, "source > 0 needs concentration"
assert not source_model or int(float(P.get("advection", 0))) == 0, \
    "the 3rd-order WENO advection (advection = 1) is implemented by neither " \
    "code (see readme)"

# ``*.concentration`` files of this family have no ``kDiff`` header line, so
# ``ib2d_io`` returns ``k = None`` and the coefficient has to come from
# ``input2d`` (an upstream inconsistency, see readme; applied just above)
C_ib = fem.Function(V_ib, name="C_nodal")      # fluid-side carrier of C
C_s = fem.Function(Vs, name="C_at_markers")    # C interpolated to markers
Fs_ib = fem.Function(V_ib, name="Fs_nodal")    # spread source on the grid
F_s_src = fem.Function(Vs, name="source_lag")  # fs * ds at the markers


def _upwind_source(C, U, dz, axis):
    """``give_Necessary_Derivative`` of the *source* stepper (periodic upwind).

    Unlike pyIB2d's standalone helper, the MATLAB loop of
    ``please_Update_Adv_Diff_Concentration_Source.m`` runs ``2:lenX-1`` and
    therefore also fills the second-to-last row/column; reproduced verbatim.
    """
    G = np.zeros_like(C)
    s = np.sign(U)
    n = C.shape[axis]
    if axis == 1:
        G[:, 0] = np.where(s[:, 0] <= 0, C[:, 1] - C[:, 0],
                           C[:, 0] - C[:, n - 1]) / dz
        G[:, n - 1] = np.where(s[:, n - 1] <= 0, C[:, 0] - C[:, n - 1],
                               C[:, n - 1] - C[:, n - 2]) / dz
        j = np.arange(1, n - 1)
        G[:, j] = np.where(s[:, j] <= 0, C[:, j + 1] - C[:, j],
                           C[:, j] - C[:, j - 1]) / dz
    else:
        G[0, :] = np.where(s[0, :] <= 0, C[1, :] - C[0, :],
                           C[0, :] - C[n - 1, :]) / dz
        G[n - 1, :] = np.where(s[n - 1, :] <= 0, C[0, :] - C[n - 1, :],
                               C[n - 1, :] - C[n - 2, :]) / dz
        i = np.arange(1, n - 1)
        G[i, :] = np.where(s[i, :] <= 0, C[i + 1, :] - C[i, :],
                           C[i, :] - C[i - 1, :]) / dz
    return G


def source_field(C):
    """``please_Find_Source_For_Concentration``: the source density Fs on the
    Eulerian grid, evaluated at the *current* marker positions."""
    C_ib.x.array[:] = np.column_stack([C[gj, gi].ravel(),
                                       np.zeros(gj.size)]).ravel()
    ib.evaluate_current_points(X._cpp_object)
    ib.fluid_to_solid(C_ib._cpp_object, C_s._cpp_object)
    CL = C_s.x.array.reshape(-1, 2)[dof_of_ib2d][:, 0]
    if source_model == 1:
        fs = np.full(Nb, k_source)
    elif source_model == 2:
        fs = k_source * (c_inf - CL)
    else:
        fs = k_source * CL
    tmp = np.zeros((Nb, 2))
    tmp[:, 0] = fs * ds_ib2d
    dst = np.empty_like(tmp)
    dst[dof_of_ib2d] = tmp
    F_s_src.x.array[:] = dst.ravel()
    ib.solid_to_fluid(Fs_ib._cpp_object, F_s_src._cpp_object)
    # ``solid_to_fluid`` spreads onto the *non-periodic* nodal grid: a marker
    # whose 4-point stencil wraps around x = Lx / y = Ly deposits its share on
    # the "image" nodes, which must be folded back onto their masters before the
    # field is read onto the (Ny, Nx) IB2d array -- the very same
    # ``periodic_fold`` the force spreading uses
    solver.periodic_fold(Fs_ib)
    Fs = np.zeros((Ny, Nx))
    Fs[ju_ib[ib_keep], iu_ib[ib_keep]] = \
        Fs_ib.x.array.reshape(-1, 2)[ib_keep, 0]
    return Fs, fs


def update_concentration_with_source(C, U, V, kd, f):
    """matIB2d ``perform_Upwind_source``: one un-split advection-diffusion step
    with the source added to the same right-hand side."""
    Cx = _upwind_source(C, U, dx, 1)
    Cxx = _dd(C, dx, 1)
    Cy = _upwind_source(C, V, dy, 0)
    Cyy = _dd(C, dy, 0)
    return C + dt * (kd * (Cxx + Cyy) - (U * Cx + V * Cy) + f)


def source_total(Fs):
    """The discrete line integral the spreading must conserve: sum of the
    source over the grid equals ``sum_i fs_i ds`` up to round-off (checked in
    the time loop)."""
    return Fs.sum() * dx * dy


C_field = None if conc is None else conc["C0"].copy()

# ---------------------------------------------------------------------------
# Output sampling on the IB2d grid
# ---------------------------------------------------------------------------
Vdof = Vc.tabulate_dof_coordinates()[:, :2]
iu = np.rint(Vdof[:, 0] / dx).astype(int)
ju = np.rint(Vdof[:, 1] / dy).astype(int)
on_grid = lambda Q, i, j: (np.isclose(Q[:, 0], i * dx)          # noqa: E731
                           & np.isclose(Q[:, 1], j * dy))
keep = (iu < Nx) & (ju < Ny) & on_grid(Vdof, iu, ju)
Q2s = fem.functionspace(mesh, ("Lagrange", 2))
p_grid_fn = fem.Function(Q2s)
Pdof = Q2s.tabulate_dof_coordinates()[:, :2]
ip = np.rint(Pdof[:, 0] / dx).astype(int)
jp = np.rint(Pdof[:, 1] / dy).astype(int)
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
if target_info is not None:
    hist["Xa"] = [target_info[:, 1:3].copy()]
if Nt:
    hist["Xt"] = [tracer_array().copy()]
if C_field is not None:
    hist["C"] = [C_field.copy()]
u0, p0, f0 = grid_fields()
hist["u"], hist["p"], hist["f"] = [u0], [p0], [f0]
write_output(0.0)
if comm.rank == 0 and target_info is not None:
    ids = target_info[:, 0].astype(np.int64)
    print(f"initial      : max|X_target - anchor|="
          f"{np.abs(marker_array(X)[ids] - target_info[:, 1:3]).max():.2e}")

# ---------------------------------------------------------------------------
# Time loop -- Peskin (2002) / IB2d IBM_Driver ordering
# ---------------------------------------------------------------------------
tic = time.perf_counter()
ctime = 0.0
Xh_prev = X_ib2d.copy()             # driver's xLag_P (previous half step)
for step in range(num_steps):
    # (1) half-step Lagrangian positions X_h = X + dt/2 U^n(X)
    ib.evaluate_current_points(X._cpp_object)
    if step == 0:
        X_h.x.array[:] = X.x.array           # u = 0 initially (as IB2d)
    else:
        ib.fluid_to_solid(ib_velocity(solver.u_n)._cpp_object, U_s._cpp_object)
        X_h.x.array[:] = X.x.array + 0.5 * dt * U_s.x.array
        if pe is not None:
            # the driver's poroelastic branch adds the previous step's
            # (spring-only) force as a displacement
            Xh = marker_array(X_h).copy()
            Xh[pe_ids] += 0.5 * dt * poroelastic_disp(F_Poro_prev)
            write_fun(X_h, Xh)

    # (2) user modules that update the structure parameters, then the forces
    update_structure(ctime, marker_array(X))
    Fm = spread_forces_at_half(Xh_prev, ctime)
    Xh_prev = marker_array(X_h).copy()
    F_Poro = None if pe is None else spring_only_forces(marker_array(X_h))[pe_ids]

    # (3) arbitrary external force, then the Brinkman resistance
    if ext_mod is not None:
        add_external_force(ctime)
    if brink is not None:
        add_brinkman()

    # (4) two-stage fluid solve: u_h, u^{n+1}, p^{n+1/2}
    solver.solve_one_step()

    # (5) markers: X^{n+1} = X^n + dt U_h(X_h) (+ poroelastic displacement)
    ib.fluid_to_solid(ib_velocity(solver.u_h)._cpp_object, U_s._cpp_object)
    Xm = marker_array(X).copy()
    Xm += dt * U_s.x.array.reshape(-1, 2)[dof_of_ib2d]
    if pe is not None:
        Xm[pe_ids] += dt * poroelastic_disp(F_Poro)
        F_Poro_prev = F_Poro
    Xm[:, 0] %= Lx
    Xm[:, 1] %= Ly

    # (5b) porous slip (driver's porous branch, after the marker update)
    if por is not None:
        porous_slip_update(Xm, Fm, dt)
        Xm[:, 0] %= Lx
        Xm[:, 1] %= Ly
    write_markers(Xm)

    # (5c) passive tracers ride the same interpolation (no force feedback)
    if Nt:
        ib.evaluate_current_points(Xt._cpp_object)
        ib.fluid_to_solid(ib_velocity(solver.u_h)._cpp_object, Ut._cpp_object)
        Xt.x.array[:] += dt * Ut.x.array
        Xt.x.scatter_forward()
        wrap_periodic(Xt, dof_of_tracer)

    # (5d) passive scalar (the driver updates it with u^{n+1} as well) plus the
    # matIB2d source term, evaluated at the *new* marker positions and the old
    # field, exactly like the reference splice in run_reference.py
    if C_field is not None:
        _ux, _uy = node_velocity()
        if source_model:
            _Fs, _fs = source_field(C_field)
            if step == 0 or step == num_steps - 1:
                _lhs = _Fs.sum() * dx * dy
                _rhs = _fs.sum() * ds_ib2d
                _rel = abs(_lhs - _rhs) / max(abs(_rhs), 1e-300)
                if comm.rank == 0:
                    print(f"[main] source spreading check: sum(Fs)*dx*dy = "
                          f"{_lhs:.16g} vs sum(fs)*ds = {_rhs:.16g} "
                          f"(rel {_rel:.2e})")
                assert _rel <= 1e-9, \
                    ("source spreading does not conserve the line integral: "
                     f"{_lhs!r} vs {_rhs!r}")
            C_field = update_concentration_with_source(C_field, _ux, _uy,
                                                      conc["k"], _Fs)
        else:
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
        if target_info is not None:
            hist["Xa"].append(target_info[:, 1:3].copy())
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
            Xm_out = hist["X"][-1]
            extra = ""
            if target_info is not None:
                ids = target_info[:, 0].astype(np.int64)
                extra = (f"|X_tgt-Xa|max="
                         f"{np.abs(Xm_out[ids] - target_info[:, 1:3]).max():.2e}"
                         "  ")
            if C_field is not None:
                extra += (f"Cmax={C_field.max():.6e}  "
                          f"sum(C)={C_field.sum():.6e}  ")
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
                    dx=dx, dy=dy, dt=dt, nx=Nx, ny=Ny, ds=ds_ib2d,
                    n_tracers=Nt,
                    **({"Xa": np.array(hist["Xa"])} if "Xa" in hist else {}),
                    **({"tracers": np.array(hist["Xt"])} if Nt else {}),
                    **({"C": np.array(hist["C"])} if C_field is not None else {}))
if comm.rank == 0:
    print(f"done in {elapsed:.1f}s -> {fname}")
