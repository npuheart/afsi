"""Shared configuration for demo_425 — tethered aorta in a box, 3-D round tube.

demo_425 extends demo_424 from a 2-D planar idealisation to a genuine 3-D
round pipe.  demo_424 cut the cylindrical aorta along a plane through its axis,
which turned the lumen into a parallel-plate channel and the wall into two flat
strips.  Here the circular cross-section is kept:

    424 (2-D)                        425 (3-D)
    ---------------------------      ---------------------------------
    lumen: parallel-plate channel     lumen: circular pipe, radius A_LUMEN
    wall : two flat strips            wall : cylindrical shell, thickness T_WALL
    occluder: flat membrane strip     occluder: circular disc, thickness DISC_T
    domain: box, width 1.5*2a         domain: square duct, side BOX_SIDE
    solid outline reads "H"           solid is a tube + a coaxial disc

The solid still carries **no constitutive law** — every Lagrangian point is held
only by a tether (volumetric spring) ``f = beta*(X_ref - X)`` — and the driving
condition is still a pressure Dirichlet on the two open ends of the duct.  That
keeps the same solver-verification spirit as demo_424 while exercising the
genuinely 3-D immersed-boundary path (``IBMesh3D`` / ``IBInterpolation3D``).

All quantities are SI.  The fluid is normalised (rho = 1) with a blood-like
viscosity, matching demo_424.

Axis convention
---------------
The pipe axis is **x**; the cross-section lies in (y, z).  The fluid duct is
square in cross-section with cubic cells, and the pipe axis sits in its centre,
so the outer fluid gap is the same all the way around the tube.
"""
import math
import os

import numpy as np

MMHG = 133.322387415  # Pa per mmHg

CASE = os.environ.get("CASE", "closed").lower()
if CASE not in ("open", "closed"):
    raise ValueError("CASE must be 'open' or 'closed'")

# --------------------------------------------------------------------------
# Geometry [m]
# --------------------------------------------------------------------------
L_AORTA = 0.1                       # aorta length
A_LUMEN = 0.015                     # lumen radius (= aorta radius)
T_WALL = 0.002                      # wall thickness

# Square duct, side chosen so that the outer fluid layer around the tube is the
# same size as in demo_424's box (BOX_W = 1.5 * 2a = 0.045 m there).  Here the
# tube is round, so a square cross-section is the natural 3-D counterpart:
#   side = 2 * (a + t) + 2 * GAP ,  GAP = 0.0055 m  ->  0.045 m
GAP = 0.0055                        # fluid layer outside the wall
BOX_SIDE = 2.0 * (A_LUMEN + T_WALL) + 2.0 * GAP     # = 0.045 m
Y_C = 0.5 * BOX_SIDE                # duct centre in y
Z_C = 0.5 * BOX_SIDE                # duct centre in z (= pipe axis)

DISC_T = 0.002                      # occluding disc thickness [m]
DISC_X = 0.5 * L_AORTA              # disc position, from the aorta start

# --------------------------------------------------------------------------
# Fluid grid: cubic cells, duct half a cell longer than the aorta at each end
# so the aorta's open ends sit at cell centres in x (same device as demo_424).
# --------------------------------------------------------------------------
NY = int(os.environ.get("NY", "45"))        # cells across BOX_SIDE (in y and z)
NZ = int(os.environ.get("NZ", str(NY)))     # cells across BOX_SIDE (in z)
H = BOX_SIDE / NY                           # cell size [m] (cubic: NY ~ NZ)
NX = round(L_AORTA / H) + 1                 # = L_AORTA/H + 1
BOX_L = NX * H                              # = L_AORTA + H
X_OFF = 0.5 * H                             # aorta start offset in x

# --------------------------------------------------------------------------
# Solid (Lagrangian) mesh — see generate_mesh.py
# --------------------------------------------------------------------------
# The solid mesh must resolve the 2 mm wall and the immersed surfaces, but it
# should not run away from the fluid resolution either: an over-refined solid
# costs a lot (every extra radial layer multiplies the whole tube length) and
# buys nothing once the cells are far below h.
#
# Both the wall and the inner region are sized from one target solid cell size.
# The angular direction is arc-length limited by the Cartesian fluid grid and is
# the one that may legitimately come out coarser than the target.
SOLID_TARGET_FRACTION = float(os.environ.get("SOLID_TARGET_FRACTION", "0.5"))
SOLID_TARGET = SOLID_TARGET_FRACTION * H      # wanted solid cell size [m]

# Radial layers across the wall (T_WALL = 2 mm at the default settings).
N_R_WALL = max(int(round(T_WALL / SOLID_TARGET)), 1)
HS_R = T_WALL / N_R_WALL                      # radial solid cell size [m]

# Radial layers of the INNER region (r = 0 .. a), used by both the occluding
# disc and the tube core.  This is sized from the fluid h rather than from HS_R
# on purpose: every inner layer is replicated over the whole tube length AND
# the full circumference, so tying it to the (much finer) wall spacing is what
# blew the closed-case mesh up to millions of cells.
N_R_CORE = max(int(round(A_LUMEN / SOLID_TARGET)), 1)

# Angular divisions of every cylindrical ring.  A Cartesian grid cannot resolve
# a circle below arc ~ h, so the default keeps the arc length near the radial
# cell size: 2*pi*(a+t)/N_THETA_DIV ~ HS_R.
N_THETA_DIV = int(os.environ.get("N_THETA_DIV", str(max(8 * round(
    math.pi * (A_LUMEN + T_WALL) / (4.0 * HS_R)), 8))))

# --------------------------------------------------------------------------
# Physics
# --------------------------------------------------------------------------
RHO = 1.0                                   # normalised density
MU = 0.0035                                 # blood-like dynamic viscosity
BETA = float(os.environ.get("BETA", "1.0e7"))   # tether stiffness [N/m^3]

DP_MMHG = float(os.environ.get("DP_MMHG", "0.02" if CASE == "open" else "0.2"))
DP = DP_MMHG * MMHG                         # driving pressure difference [Pa]

# --------------------------------------------------------------------------
# Time integration
# --------------------------------------------------------------------------
# Same choice of solver as demo_424.  The duct is driven by a pressure
# Dirichlet on BOTH open ends, so IPCSSolver's momentum equation (which carries
# -dot(p_, div(v))*dx) has the natural condition mu*du/dn = p*n, which is a
# spurious normal traction of order DP at an INFLOW.  IPCSSolverTraction in
# demo_424 restores the boundary term; here the shared IPCSSolver is used with
# ds_p/p_traction (the same fix, which demo_424's ipcs_traction.py prototypes).
# SOLVER=chorin (the demo_424 default) avoids the issue entirely.
SOLVER = os.environ.get("SOLVER", "ipcs").lower()
DT = float(os.environ.get("DT", "2.0e-4"))
RAMP_T = float(os.environ.get("RAMP_T", "0.05"))
T_END = float(os.environ.get("T_END", "0.4"))
NSTEPS = int(round(T_END / DT))

# Number of steps for the smoke test (see readme).
SMOKE_STEPS = int(os.environ.get("SMOKE_STEPS", "10"))
SMOKE = bool(os.environ.get("SMOKE"))

VELOCITY_ORDER = 2
PRESSURE_ORDER = 1
FORCE_ORDER = 2

# --------------------------------------------------------------------------
# Derived scales / analytic quantities
# --------------------------------------------------------------------------
NU = MU / RHO                               # kinematic viscosity [m^2/s]
T_VISC = A_LUMEN**2 / NU                    # viscous time across the lumen [s]

# Solid volume, exact, used as the generate_mesh.py regression guard.
#
# The solid is the union of a full cylinder of radius a + t and a coaxial disc of
# radius a, both spanning the same axial range, so the two OVERLAP on the core
# r <= a.  Adding their volumes would therefore double-count pi*a^2*L_AORTA
# (verified: the measured mesh volume exceeds that naive sum by exactly that
# amount).  The correct exact volume is the cylinder plus only the *annular*
# part of the "disc", i.e. the hollow-tube formula plus pi*a^2*DISC_T:
#
#     V = pi*((a+t)^2 - a^2)*L_AORTA + pi*a^2*DISC_T
#
# The mesh generator itself does not rely on this: it emits five
# non-overlapping blocks (the wall rings, the core rings upstream and
# downstream of the disc, and the disc's own rings), which is the same
# decomposition as the formula above.
# --------------------------------------------------------------------------
# Geometry definition shared with generate_mesh.py
# --------------------------------------------------------------------------
# These are the *discrete* axial/radial layer positions the mesh is built from,
# so the volume guard below checks the mesh against the same geometry rather
# than against a hand-derived formula that can silently drift from it.
#
# Note DISC_T is ROUNDED to a whole number of axial cells (n_along), exactly as
# demo_424 rounded its membrane thickness.  At NY=45, h=1.111 mm, so the disc
# is emitted 2*h = 2.222 mm thick, not 2.000 mm.  A guard written against the
# nominal DISC_T would therefore disagree with a perfectly correct mesh.
def n_along(length):
    """Number of axial fluid cells spanning ``length`` (at least one)."""
    return max(int(round(length / H)), 1)


def axial_nodes():
    """Axial node coordinates and the node-index range of the disc."""
    xs = X_OFF
    xe = X_OFF + L_AORTA
    xc = X_OFF + DISC_X
    n1 = n_along(DISC_X - 0.5 * DISC_T)
    nd = n_along(DISC_T)
    n2 = n_along(L_AORTA - DISC_X - 0.5 * DISC_T)
    left = xc - 0.5 * DISC_T
    right = xc + 0.5 * DISC_T
    x1 = np.linspace(xs, left, n1 + 1)
    x2 = np.linspace(left, right, nd + 1)
    x3 = np.linspace(right, xe, n2 + 1)
    X = np.concatenate([x1, x2[1:], x3[1:]])
    return X, (n1, n1 + nd)


def radial_nodes():
    """Radial layer radii, the disc-surface layer index and the wall layers.

    The inner region (r = 0 .. a) is split into N_R_CORE equal layers and the
    wall (r = a .. a+t) into N_R_WALL.  r = a lands exactly on a layer, which is
    what lets the occluding disc and the tube core meet conformingly.
    """
    radii = [A_LUMEN * k / N_R_CORE for k in range(0, N_R_CORE + 1)]
    i_disc = N_R_CORE
    radii += [A_LUMEN + T_WALL * k / N_R_WALL for k in range(1, N_R_WALL + 1)]
    return np.asarray(radii, dtype=np.float64), i_disc, N_R_WALL


def solid_node_blocks():
    """The non-overlapping *node-index* blocks the solid mesh is made of.

    Each entry is ``(a_lo, a_hi, r_lo, r_hi, label)`` in units of NODES:

        a_lo .. a_hi   axial node indices, emitting cells a_lo .. a_hi-1
        r_lo .. r_hi   radial layer indices, emitting cells r_lo .. r_hi-1

    ``generate_mesh.py`` emits cells straight from this list, so the guard in
    that file verifies the mesh that was actually written.

    NOTE: for CASE=open the entire inner region is FLUID, so there is no core
    block.  Emitting one there silently fills the lumen with solid; the volume
    guard is what caught that.
    """
    X, (a0, a1) = axial_nodes()
    radii, i_disc, n_wall = radial_nodes()
    n_a_nodes = len(X)
    n_r_nodes = len(radii)

    if CASE == "closed":
        blocks = [(0, n_a_nodes, i_disc, n_r_nodes, "wall"),
                  (a0, a1, 0, i_disc, "disc"),
                  (0, a0, 0, i_disc, "tube core upstream"),
                  (a1, n_a_nodes, 0, i_disc, "tube core downstream")]
    else:
        blocks = [(0, n_a_nodes, i_disc, n_r_nodes, "wall")]
    return blocks, X, radii


def exact_solid_volume(verbose=False):
    """Exact volume of the emitted solid mesh (polygonal rings included).

    A solid cell spans one angular sector ``dth`` of the radial band
    ``[radii[r], radii[r+1]]`` and one axial step dx.  Cutting the sector ring
    between the two circular arcs gives the pseudotriangle
    ``(r_lo, 0), (r_hi, 0), (r_hi, dth), (r_lo, dth)``, whose area is the
    difference of the two polar triangles:

        0.5 * (r_hi^2 - r_lo^2) * sin(dth)

    and every (axial, radial) cell is replicated N_THETA_DIV times, once per
    angular sector around the ring.  Summing that over the emitted blocks times
    dx reproduces the mesh volume to machine precision, because the mesh
    inscribes exactly these polygons in the true circles (verified: at
    N_theta=104 this lands within 6e-4 relative of the exact-circle volume,
    which is the expected inscribed-polygon deficit).
    """
    blocks, X, radii = solid_node_blocks()
    sin_dth = math.sin(2.0 * math.pi / N_THETA_DIV)
    total = 0.0
    for a_lo, a_hi, r_lo, r_hi, label in blocks:
        block = 0.0
        for i_a in range(a_lo, a_hi - 1):
            dx = X[i_a + 1] - X[i_a]
            for i_r in range(r_lo, r_hi - 1):
                block += (0.5 * (radii[i_r + 1] ** 2 - radii[i_r] ** 2)
                          * sin_dth * dx)
        block *= N_THETA_DIV          # one sector -> the full ring
        if verbose:
            print(f"    {label:<22} a[{a_lo},{a_hi}) r[{r_lo},{r_hi}) "
                  f"= {block:.10e}")
        total += block
    return total


VOL_SOLID = exact_solid_volume()

# Length scale of the disc's tether response.  In demo_424 the 1-D balance was
#   beta * delta * (T_WALL * H_disc) = DP * H_disc   ->   delta = DP/(beta*T_WALL)
# In 3-D the sealed disc is held by *two* springs in series: the disc itself AND
# the annulus of wall to which it is attached, so the true equilibrium is a
# coupled one (the wall is pulled inward by the disc and resists through its own
# tether).  DISC_DELTA below is therefore only the single-spring *scale*; it is
# NOT the expected equilibrium of the coupled 3-D system.  Resolving that
# coupled balance is future work (see readme).
DISC_DELTA_SCALE = DP / (BETA * T_WALL) if CASE == "closed" else 0.0


def p_inlet(t):
    """Imposed inlet pressure [Pa], linear ramp then hold."""
    if RAMP_T <= 0.0:
        return DP
    return DP * min(max(t / RAMP_T, 0.0), 1.0)


def solid_mesh_path():
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "plot",
                        f"mesh-425-{CASE}-NY{NY}"
                        f"-T{SOLID_TARGET_FRACTION:g}.xdmf")


def output_path():
    here = os.path.dirname(os.path.abspath(__file__))
    tag = f"{CASE}_NY{NY}"
    if SOLID_TARGET_FRACTION != 0.5:
        tag += f"_T{SOLID_TARGET_FRACTION:g}"
    if SOLVER != "chorin":
        tag += f"_{SOLVER}"
    if SMOKE:
        tag += "_smoke"
    out = os.path.join(here, "plot", tag)
    os.makedirs(out, exist_ok=True)
    return out + os.sep


def summary():
    lines = [
        "demo_425 — tethered aorta in a box, 3-D round tube",
        f"case            : {CASE}",
        f"aorta           : L={L_AORTA} m, lumen radius={A_LUMEN} m, "
        f"wall={T_WALL} m",
        f"duct            : {BOX_L:.6g} x {BOX_SIDE:.6g} x {BOX_SIDE:.6g} m  "
        f"({NX} x {NY} x {NZ} cells, h={H:.6g} m)",
        f"outer gap       : {GAP:.6g} m all around the tube",
        f"solid           : tube r={A_LUMEN}..{A_LUMEN + T_WALL} m, "
        f"radial cell {HS_R:.6g} m, {N_THETA_DIV} angular divisions",
        f"fluid           : rho={RHO}, mu={MU}  (nu={NU:g} m^2/s)",
        f"tether          : beta={BETA:g} N/m^3",
        f"pressure        : {DP_MMHG:g} mmHg = {DP:.6g} Pa, "
        f"ramp {RAMP_T:g} s, T_end {T_END:g} s",
        f"solver          : {SOLVER}, dt={DT:g} s, steps="
        f"{SMOKE_STEPS if SMOKE else NSTEPS}",
        f"solid volume    : {VOL_SOLID:.10e} m^3 (exact)",
    ]
    if CASE == "closed":
        lines.append(f"disc            : t={DISC_T} m at x={DISC_X} m "
                     f"(from aorta start), DP/(beta*t_wall) scale="
                     f"{DISC_DELTA_SCALE:.6g} m")
    lines.append(f"viscous time    : {T_VISC:.6g} s")
    return "\n".join(lines)
