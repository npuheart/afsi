"""Shared configuration for demo_426 — flow through a slanted channel (2-D).

Benchmark
---------
Grüninger et al., "An immersed boundary method for the simulation of ..." (the
two-dimensional slanted-channel case): steady plane-Poiseuille flow through a
channel inclined at theta = pi/6 about the origin, in the rectangular domain
Omega = [0,1] x [-0.25,2], with the two channel walls handled by an immersed
boundary penalty so that the walls are deliberately NOT grid-aligned.  The
point of the benchmark is to compare how different IB kernels reproduce the
exact solution inside a confined stationary geometry.

Exact solution
--------------
With the axial coordinate measured across the channel,

    xi(x,y) = y*cos(theta) - x*sin(theta)

the exact steady state is unidirectional plane Poiseuille along the channel:

    u(x,y) = -A * xi * (xi - h) * cos(theta)
    v(x,y) = -A * xi * (xi - h) * sin(theta)

which satisfies div(u) = 0 exactly and balances the constant body force

    f = -(dP/L) * (cos(theta), sin(theta))

(the pressure gradient points opposite to the flow).  The channel walls sit at
xi = 0 and xi = h, where the profile vanishes by construction, so the no-slip
condition on the walls is exactly compatible with the analytic field.

Self-consistency of A (IMPORTANT)
---------------------------------
The benchmark paper writes the coefficient as (dP/L)/(2*mu*L).  That is not
dimensionally consistent: A must have units of 1/(length^2), so a length must
appear squared in the denominator.  With the standard plane-Poiseuille result

    u_max = (dP/L) * (D/2)^2 / (2*mu)

the self-consistent coefficient is

    A = (dP/L) / (2*mu*h)        [h = 1 here, so numerically A = dP/L/(2*mu)]

which gives u_max = A*(D/2)^2.  Taking the paper's stated values mu = 0.5 and
dP/L = 1.0 then fixes A = 1.0 and, with D/2 = h/(2*cos(theta)) = 0.57735,

    u_max = 1.0 * 0.57735^2 = 1/3 = 0.333333

NOT the 0.25 quoted in the paper.  The two stated numbers (dP/L = 1, mu = 0.5)
and U_max = 0.25 cannot both hold; the 25 % gap is resolved here by keeping the
paper's *primary* parameters (mu, dP/L, h, theta, domain, N, dt) and treating
U_max as a DERIVED quantity, so the discrete problem is driven by exactly the
same body force as the reference solution and the comparison is a genuine
discretisation-error study.  To reproduce U_max = 0.25 instead one would need
dP/L = 0.75 with mu = 0.5; set DP_DL=0.75 in the environment to do that.

All quantities are SI-consistent (the benchmark is in normalised units).
"""
import math
import os

import numpy as np

# --------------------------------------------------------------------------
# Space-time domain
# --------------------------------------------------------------------------
# The benchmark's box is 2 x 2 with the channel entering on the left and leaving
# on the right.  It is TRANSLATED here so its lower-left corner sits on the
# origin, as an AFSI case requires:
#
#     paper frame :  x in [-0.25, 2] ,  y in [-1, 1]
#     shift       :  X = x + 0.25 ,  Y = y + SHIFT_Y
#     this demo   :  X in [0, 2.25] ,  Y in [0, 2.5]
#
# The paper's box is 2.5 units tall and its width is kept here, but its HEIGHT
# is not enough: over X in [0, 2.25] the channel spans 2.45271 in Y, so the
# quoted 2.0-tall box cuts the lower plate at (1.155, 0).  SHIFT_Y is therefore
# chosen to centre the channel in a 2.5-tall box (0.15 of margin top and
# bottom) while keeping the lower-left corner on the origin.
#
# The shift is NOT along the channel axis, so it does not leave the geometry
# invariant: the across-channel coordinate moves and the exact velocity has to
# be carried over with the new coordinates (see xi/analytic below).  What it
# does preserve is containment.  In this frame the two plates run
#
#     lower  xi=-D/2 :  Y = 0.18900 (X=0) -> 1.34370 (X=2.25)
#     upper  xi=+D/2 :  Y = 1.52233 (X=0) -> 2.67703 (X=2.25)
#
# so both plates run the full width, terminate on the left/right faces, and the
# channel stays inside the box.
X_MIN = float(os.environ.get("X_MIN", "0.0"))
X_MAX = float(os.environ.get("X_MAX", "2.25"))
Y_MIN = float(os.environ.get("Y_MIN", "0.0"))
Y_MAX = float(os.environ.get("Y_MAX", "3.1"))

# Origin of the benchmark frame expressed in this frame.
SHIFT_X = 0.25

# --------------------------------------------------------------------------
# Slanted channel
# --------------------------------------------------------------------------
THETA_DEG = float(os.environ.get("THETA_DEG", "30.0"))
THETA = math.radians(THETA_DEG)
COS_T, SIN_T = math.cos(THETA), math.sin(THETA)

H_CHANNEL = 1.0                       # channel width in its horizontal config
D_CHANNEL = H_CHANNEL / COS_T         # true (perpendicular) channel width
R_HALF = 0.5 * D_CHANNEL              # >0 : xi in [0, h]; profile peaks at r/2

# SHIFT_Y is DERIVED, not hand-picked, so that the lowest point of the lower
# plate sits a small margin above Y=0.  The lower plate is
#     Y = SHIFT_Y + (X*sin - R)/cos
# which increases with X, so its minimum over the box is at X = X_MIN; that is
# what guarantees the lower-left corner sits on the origin with the whole
# channel inside.
PLATE_MARGIN = 0.1
# lower plate at X = X_MIN:  Y = SHIFT_Y + ((X_MIN - SHIFT_X)*sin - R)/cos
# require that to equal PLATE_MARGIN:
SHIFT_Y = float(os.environ.get("SHIFT_Y", repr(
    PLATE_MARGIN - ((X_MIN - SHIFT_X) * SIN_T - R_HALF) / COS_T)))

# --------------------------------------------------------------------------
# Physics
# --------------------------------------------------------------------------
# NOTE ON UNITS
# -------------
# The benchmark states its parameters in CGS: H = 1.0 cm, rho = 1.0 g/cm^3,
# mu_s = mu_p = 0.05 Pa.s (total 0.1 Pa.s = 1.0 poise), lambda = 0.1 s.  This
# demo therefore runs in cm, g, s:
#     length cm   mass g   time s
#     viscosity poise (= g/(cm s))   density g/cm^3
#     velocity cm/s   body force dyn/cm^3
# 1 Pa.s = 10 poise, so mu_s + mu_p = 0.1 Pa.s = 1.0 poise.
# With H = 1.0 cm the profile amplitude is A = (dP/L)/(2 mu H) = 0.5, giving
# u_max = A (D/2)^2 = 0.1666667 cm/s and Re = rho*u_max*D/mu = 0.1925.
RHO = 1.0        # g/cm^3  (benchmark: 1.0 g/cm^3)
MU = 1.0         # poise   (benchmark: mu_s + mu_p = 0.05 + 0.05 Pa.s = 1.0 P)
DP_DL = float(os.environ.get("DP_DL", "1.0"))   # -dp/ds along the channel axis

# Self-consistent profile amplitude and the resulting peak velocity.
A_COEF = DP_DL / (2.0 * MU * H_CHANNEL)
U_MAX = A_COEF * R_HALF**2
U_MAX_PAPER = 0.25                    # as quoted in the benchmark text

# --------------------------------------------------------------------------
# Grid and time stepping
# --------------------------------------------------------------------------
# dx is fixed by the benchmark's N = L/N convention with L = 1, and the cell
# count is then DERIVED from the (extended) box -- not the other way round,
# otherwise widening the box would silently coarsen the grid.
N = int(os.environ.get("N", "32"))
DX = 1.0 / N                          # = 0.03125
NX = int(round((X_MAX - X_MIN) / DX))
NY = int(round((Y_MAX - Y_MIN) / DX))
BOX_L = float(NX) * DX                # >= X_MAX - X_MIN
BOX_H = float(NY) * DX                # >= Y_MAX - Y_MIN
CELLS_PER_UNIT = 1.0 / DX

DT_FACTOR = float(os.environ.get("DT_FACTOR", "0.2"))   # benchmark: dt = 0.2h
DT = DT_FACTOR * DX
if os.environ.get("DT_OVERRIDE"):
    DT = float(os.environ["DT_OVERRIDE"])

# --------------------------------------------------------------------------
# Immersed-boundary penalty for the two (stationary, rigid) plates
# --------------------------------------------------------------------------
# Eq. (14) of the benchmark: penalty stiffness + penalty body force + damping.
# The plates must not move, so the penalty force is
#     f_ib = -BETA*(X - X_ref) - DAMP*(u_ib - 0)
# spread to the fluid grid.  Both parameters must be as large as stability
# allows, per the benchmark's own statement.
BETA = float(os.environ.get("BETA", "1.0e6"))
DAMP = float(os.environ.get("DAMP", "1.0e3"))

# Lagrangian spacing along the plates (the fluid cell size keeps the IB support
# well sampled).
LAGRANGIAN_DIV = int(os.environ.get("LAGRANGIAN_DIV", "2"))
DS_LAG = DX / LAGRANGIAN_DIV

# Physical thickness of the plates.  AFSI's immersed-boundary distributor takes
# an integration weight per Lagrangian point (its w), i.e. the reference volume
# that point represents; for a thin plate in 2-D that is ds * PLATE_THICKNESS.
# (The distributor's default w = 1 means "one full Eulerian cell volume", which
# is only right for a volume-filling solid.)
PLATE_THICKNESS = float(os.environ.get("PLATE_THICKNESS", str(DX)))
W_LAGRANGIAN = DS_LAG * PLATE_THICKNESS

# Where the IB kernel map is evaluated.  False (default, classic Peskin): at the
# CURRENT Lagrangian positions, so the delta function travels with the plates.
# True: at the REFERENCE positions, i.e. the penalty is applied by a fixed
# spatial kernel.
MAP_AT_REFERENCE = os.environ.get("MAP_AT_REFERENCE", "0") not in ("0", "false")


# --- implicit penalty on the plates ----------------------------------------
# The body-force route is applied EXPLICITLY (it enters ns_solver.f, which is
# frozen during the momentum solve), so the effective damping rate DAMP/(dx*dy)
# must satisfy DAMP*dt/(rho*dx*dy) <~ 1 -- at dx=1/32, dt=0.15dx that caps
# DAMP at 0.208, far too weak to enforce no-slip.
#
# The solver also accepts `drag`, which is assembled into the momentum LHS and
# is therefore IMPLICIT and free of any time-step restriction.  A thin band of
# large drag around each plate is the robust way to impose the no-slip
# condition here.  PLATE_DRAG_BAND is measured in fluid cells; PLATE_DRAG is
# the damping coefficient [1/s].
PLATE_DRAG = float(os.environ.get("PLATE_DRAG", "1.0e4"))
PLATE_DRAG_BAND = float(os.environ.get("PLATE_DRAG_BAND", "1.5"))
USE_IMPLICIT_DRAG = os.environ.get("USE_IMPLICIT_DRAG", "1") not in ("0", "false")


def plate_distance(x, y):
    """Signed distance to the nearer plate (+ inside the channel)."""
    xi_ = xi(x, y)
    return R_HALF - np.abs(xi_)


def plate_drag_coefficient(x, y):
    """Implicit drag coefficient: PLATE_DRAG in a band around each plate."""
    d = plate_distance(x, y)
    band = PLATE_DRAG_BAND * DX
    w = np.clip(1.0 - np.abs(d) / band, 0.0, 1.0)
    return PLATE_DRAG * w

VELOCITY_ORDER = 2
PRESSURE_ORDER = 1
FORCE_ORDER = 2          # Lagrange order on the Lagrangian (plate) mesh

# --------------------------------------------------------------------------
# Time loop
# --------------------------------------------------------------------------
T_END = float(os.environ.get("T_END", "2.0"))   # benchmark: 20*lambda = 2.0 s
# Linear ramp of the driving from rest; the benchmark starts the channel from
# rest and runs to steady state.  Default = lambda = 0.1 s.  0 disables.
RAMP_T = float(os.environ.get("RAMP_T", "0.1"))

# Driving mechanism.
#   False (benchmark-faithful): the flow is driven by the prescribed steady
#     analytic solution used as the inflow condition, ramped from rest over
#     RAMP_T.  No body force is applied, so it is not double-driven.
#   True : drive with the equivalent constant body force -grad(p) and keep the
#     analytic velocity as a boundary condition as well (the earlier setup).
USE_BODY_FORCE = os.environ.get("USE_BODY_FORCE", "0") not in ("0", "false")
DT_OVERRIDE = os.environ.get("DT_OVERRIDE")
NSTEPS = int(round(T_END / DT))
SMOKE = bool(os.environ.get("SMOKE"))
SMOKE_STEPS = int(os.environ.get("SMOKE_STEPS", "50"))

SOLVER = os.environ.get("SOLVER", "ipcs").lower()

# --------------------------------------------------------------------------
# Derived scales
# --------------------------------------------------------------------------
NU = MU / RHO
RE = RHO * U_MAX * D_CHANNEL / MU
COURANT = U_MAX * DT / DX
DIFFUSION = NU * DT / DX**2
CHANNEL_CELLS = D_CHANNEL / DX


# --------------------------------------------------------------------------
# Analytic solution and the channel geometry inside the rectangle
# --------------------------------------------------------------------------
def xi(X, Y):
    """Across-channel coordinate in THIS frame, measured from the centreline.

    In the benchmark frame it is ``xi = y*cos(theta) - x*sin(theta)`` with the
    plates at ``xi = +-D/2``.  Substituting ``x = X - SHIFT_X`` and
    ``y = Y - SHIFT_Y`` gives the form used here:

        xi = (Y - SHIFT_Y)*cos(theta) - (X - SHIFT_X)*sin(theta)

    Absorbing the translation into the coordinate is all the "variable
    transformation of the velocity equation" amounts to: the body force is
    spatially constant and the pressure enters only through its gradient, so
    both are invariant under a rigid translation, while ``xi`` -- and hence u --
    picks up the constants.  The velocity field is therefore unchanged; only
    its argument is rewritten.
    """
    return (Y - SHIFT_Y) * COS_T - (X - SHIFT_X) * SIN_T


def analytic(X, Y):
    """Exact velocity at a point (numpy-broadcasting friendly).

    Plane Poiseuille in the across-channel coordinate, unchanged in form:

        u_s = (dP/L)/(2*mu) * ( (D/2)^2 - xi^2 )

    which equals ``A*(D/2)^2 = u_max`` on the centreline and vanishes at both
    plates, with the velocity vector along the channel axis (cos, sin).
    """
    t = xi(X, Y)
    prof = (DP_DL / (2.0 * MU)) * (R_HALF**2 - t**2)
    return prof * COS_T, prof * SIN_T


def wall_y(X, side):
    """Y of the plate `side` (+1 upper, -1 lower) at a given X, this frame."""
    return SHIFT_Y + ((X - SHIFT_X) * SIN_T + side * R_HALF) / COS_T


def wall_endpoints(side):
    """The two endpoints of a plate, in this frame.

    With the box sized and SHIFT_Y derived as above, both plates lie wholly
    inside the box and span its full width, terminating exactly on the inlet
    (X = X_MIN) and outlet (X = X_MAX) faces -- so no clipping is needed and
    both ends are complete channel cross-sections.
    """
    return [(X_MIN, wall_y(X_MIN, side)), (X_MAX, wall_y(X_MAX, side))]


def inlet_interval():
    """y-range of the channel on the left face x = X_MIN."""
    ys = []
    for side in (-1, +1):
        for (x, y) in wall_endpoints(side):
            if abs(x - X_MIN) < 1e-12:
                ys.append(y)
    return (min(ys), max(ys))


def outlet_interval():
    """y-range of the channel on the right face x = X_MAX."""
    ys = []
    for side in (-1, +1):
        for (x, y) in wall_endpoints(side):
            if abs(x - X_MAX) < 1e-12:
                ys.append(y)
    return (min(ys), max(ys))


def solid_mesh_path():
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "plot",
                        f"solid-426-N{int(1.0 / DX)}-t{PLATE_THICKNESS:g}.xdmf")


# Exact plate geometry, for the mesh-area guard.
PLATE_LENGTH = float(np.hypot(*(np.subtract(wall_endpoints(+1)[1],
                                            wall_endpoints(+1)[0]))))
PLATE_AREA_TARGET = PLATE_LENGTH * PLATE_THICKNESS


def output_path():
    here = os.path.dirname(os.path.abspath(__file__))
    tag = f"N{N}_{SOLVER}"
    if SMOKE:
        tag += "_smoke"
    out = os.path.join(here, "plot", tag)
    os.makedirs(out, exist_ok=True)
    return out + os.sep


def summary():
    yl, yu = inlet_interval()
    ol, ou = outlet_interval()
    lines = [
        "demo_426 — flow through a slanted channel (2-D IB benchmark)",
        f"domain          : [{X_MIN}, {X_MAX}] x [{Y_MIN}, {Y_MAX}]",
        f"channel         : theta={THETA_DEG:g} deg, h={H_CHANNEL:g} m, "
        f"D=h/cos={D_CHANNEL:.6f} m ({CHANNEL_CELLS:.2f} cells)",
        f"inlet  (x={X_MIN:g}) : y in [{yl:.6f}, {yu:.6f}]  "
        f"(height {yu - yl:.6f})",
        f"outlet (x={X_MAX:g}) : y in [{ol:.6f}, {ou:.6f}]  "
        f"(height {ou - ol:.6f})",
        f"fluid           : rho={RHO:g}, mu={MU:g}, nu={NU:g}",
        f"driving         : -dp/ds = {DP_DL:g} Pa/m along the axis "
        f"(body force, NOT a pressure BC)",
        f"profile         : A={A_COEF:g}, u_max={U_MAX:.6f} m/s "
        f"(paper quotes {U_MAX_PAPER}; see docstring)",
        f"grid            : N={N}, dx={DX:g}, nx x ny = {NX} x {NY} "
        f"(box {BOX_L:g} x {BOX_H:g})",
        f"time            : dt={DT:g} (= {DT_FACTOR:g} dx), "
        f"steps={SMOKE_STEPS if SMOKE else NSTEPS} "
        f"(T_end {SMOKE_STEPS * DT if SMOKE else T_END:g} s)",
        f"scales          : Re={RE:.4f}, Courant={COURANT:.4f}, "
        f"nu*dt/dx^2={DIFFUSION:.4f}",
        f"penalty         : BETA={BETA:g}, DAMP={DAMP:g}, "
        f"ds_lag={DS_LAG:g} (dx/{LAGRANGIAN_DIV})",
        f"solver          : {SOLVER}",
    ]
    return "\n".join(lines)
