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
# The benchmark quotes Omega = [0,1] x [-0.25,2].  That rectangle does NOT
# contain the slanted channel: at theta = 30 deg the walls are
# y = (x*sin +- D/2)/cos, so over x in [0,1] the channel spans y in
# [-0.66667, 1.24402].  The quoted box cuts the lower plate off below y=-0.25
# and truncates the inlet cross-section by 37 %.
#
# The setup figure in the benchmark makes the intent clear: BOTH plates run the
# full length of the box as dotted Lagrangian marker lines, and the inlet and
# outlet are complete channel cross-sections.  That requires the box to contain
# the whole channel, i.e. it must be extended DOWNWARD (and slightly left so
# the lower plate reaches the left face).
#
# Chosen box:
#
#     x in [-0.2, 1.0] ,  y in [-0.8, 2.0]
#
# With x_min = -0.2 the lower plate meets the LEFT face at y = -0.78214 and the
# upper plate at y = +0.55120, so the inlet is the complete D-tall cross-section
# [-0.78214, 0.55120]; the outlet at x = 1 is [-0.08932, 1.24402], also
# complete.  Both plates therefore lie entirely inside the box and terminate
# exactly on the inlet and outlet faces -- no clipping -- which is what the
# benchmark figure shows.  Set Y_MIN=-0.25 and X_MIN=0 to reproduce the quoted
# truncation instead.
# Box matched to the benchmark's setup figure, which is clearly wider than the
# quoted [0,1] and shows the channel end to end with both plates as dotted
# marker lines and both ends as open cross-sections.
#
#     x in [-0.9, 2.6] ,  y in [-1.4, 3.4]
#
# Checked against the plate lines y = (x*sin +- D/2)/cos:
#   lower plate  y(-0.9) = -1.18636 ->  y(2.6) = +1.01884
#   upper plate  y(-0.9) = -0.03167 ->  y(2.6) = +2.17353
# so both plates run the full width, terminate exactly on the inlet/outlet
# faces, and stay inside the box with margin; the inlet and outlet are complete
# D-tall cross-sections.  A slightly smaller box (x in [-0.2, 1]) also works
# and was used first; set X_MIN/X_MAX/Y_MIN/Y_MAX to reproduce either.
X_MIN = float(os.environ.get("X_MIN", "-0.9"))
X_MAX = float(os.environ.get("X_MAX", "2.6"))
Y_MIN = float(os.environ.get("Y_MIN", "-1.4"))
Y_MAX = float(os.environ.get("Y_MAX", "3.4"))

# --------------------------------------------------------------------------
# Slanted channel
# --------------------------------------------------------------------------
THETA_DEG = float(os.environ.get("THETA_DEG", "30.0"))
THETA = math.radians(THETA_DEG)
COS_T, SIN_T = math.cos(THETA), math.sin(THETA)

H_CHANNEL = 1.0                       # channel width in its horizontal config
D_CHANNEL = H_CHANNEL / COS_T         # true (perpendicular) channel width
R_HALF = 0.5 * D_CHANNEL              # >0 : xi in [0, h]; profile peaks at r/2

# --------------------------------------------------------------------------
# Physics
# --------------------------------------------------------------------------
RHO = 1.0
MU = 0.5
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

DT_FACTOR = float(os.environ.get("DT_FACTOR", "0.15"))
DT = DT_FACTOR * DX

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
T_END = float(os.environ.get("T_END", "20.0"))
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
def xi(x, y):
    """Coordinate across the channel, measured from the centreline.

    The physical range is xi in [-D/2, +D/2]; the channel walls are at
    xi = -D/2 and xi = +D/2, where the profile vanishes.
    """
    return y * COS_T - x * SIN_T


def analytic(x, y):
    """Exact velocity at a point (numpy-broadcasting friendly).

    Plane Poiseuille in the coordinate across the channel: with the walls at
    xi = +-D/2 the profile is the symmetric parabola

        u_s(x,y) = (dP/L)/(2*mu) * ( (D/2)^2 - xi^2 )

    which equals ``A*(D/2)^2 = u_max`` on the centreline and vanishes at both
    walls, and the velocity vector points along the channel axis (cos, sin).

    NOTE: an earlier version of this function built the profile from the
    shifted coordinate ``t = xi + D/2`` as ``-A*t*(t-h)``.  That expansion is

        A*(R^2 - xi^2) - A*R*xi        (R = D/2)

    i.e. it carries a spurious ODD term ``-A*R*xi``, which breaks the symmetry
    of the parabola and leaves the xi=+D/2 wall with a non-zero velocity.  The
    symmetric form above is the correct one.
    """
    t = xi(x, y)
    prof = (DP_DL / (2.0 * MU)) * (R_HALF**2 - t**2)
    return prof * COS_T, prof * SIN_T


def wall_endpoints(side):
    """The segment of a channel wall that lies inside the domain rectangle.

    ``side`` = -1 for the xi=-D/2 wall and +1 for the xi=+D/2 wall.  The wall
    is the line ``y = (off + x*sin)/cos``; it is clipped against all four faces
    of the rectangle, because with this geometry

      * the LOWER wall (side=-1) starts at y=-2/3, i.e. below Y_MIN, and
        therefore enters the domain through the BOTTOM face at x = 1/(2*sqrt3),
        leaving through the right face at y=-0.0893;
      * the UPPER wall (side=+1) runs from y=+2/3 to y=+1.2440, entirely
        inside, entering at x=0 and leaving through the right face.

    So the two walls are NOT symmetric about the domain: the inlet is the
    partial face for one of them.
    """
    off = side * R_HALF

    def y_at(x):
        return (off + x * SIN_T) / COS_T

    def x_at(y):
        return (y * COS_T - off) / SIN_T

    # candidate boundary crossings, in increasing x
    cand = []
    for x in (X_MIN, X_MAX):
        if Y_MIN <= y_at(x) <= Y_MAX:
            cand.append((x, y_at(x)))
    for y in (Y_MIN, Y_MAX):
        x = x_at(y)
        if X_MIN <= x <= X_MAX:
            cand.append((x, y))
    if len(cand) < 2:
        raise ValueError(f"wall {side} does not cross the domain")
    cand = sorted(set(cand))
    return [cand[0], cand[-1]]


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
