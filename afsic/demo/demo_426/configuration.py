"""Shared configuration for demo_426 — flow through a slanted channel (2-D).

Benchmark
---------
The two-dimensional slanted-channel benchmark as used by Li et al. (2025),
section "4.3.1 Slanted channel flow" (originating from Gruninger et al.,
2024, section "4.1 Oldroyd-B flow through an inclined channel"): steady
plane-Poiseuille flow through a channel inclined at theta = pi/6, in the
rectangular domain Omega = [0,1] x [-0.25,2], with the two channel walls
handled by an immersed-boundary penalty so that the walls are deliberately
NOT grid-aligned.  The point of the benchmark is to compare how different IB
kernels reproduce the exact solution inside a confined stationary geometry.

Exact solution (benchmark Eq. (48))
-----------------------------------
With the wall coordinate

    xi~(x,y) = y*cos(theta) - x*sin(theta)

the exact steady state is unidirectional plane Poiseuille along the channel:

    u(x,y) = (dP/L)/(2*mu) * xi~ * (h - xi~) * cos(theta)
    v(x,y) = (dP/L)/(2*mu) * xi~ * (h - xi~) * sin(theta)

which satisfies div(u) = 0 exactly and balances the constant body force

    f = -(dP/L) * (cos(theta), sin(theta))

(the pressure gradient points opposite to the flow).  The channel walls sit at
xi~ = 0 and xi~ = h = 1, where the profile vanishes by construction, so the
no-slip condition on the walls is exactly compatible with the analytic field.
Measured from the centreline (xi = xi~ - h/2, walls at xi = +-h/2) this is the
usual parabola, used below.

Consistency of U_max (CHECKED)
------------------------------
With the benchmark's parameters (mu = 0.5, dP/L = 1, h = 1) the peak speed is

    U_max = (dP/L) * h^2 / (8*mu) = 0.25

which matches the U_max = 0.25 quoted in the benchmark text.  (An earlier
version of this demo placed the channel differently: the CENTRELINE passed
through the origin and the perpendicular width was D = h/cos(theta); in that
geometry the quoted U_max appeared to disagree with (dP/L, mu).  The geometry
was corrected on 2026-09-30 to the one above: the perpendicular channel width
is h, and the LOWER wall (xi~ = 0) passes through the origin.)

All quantities are in the benchmark's normalised units.
"""
import math
import os

import numpy as np

# --------------------------------------------------------------------------
# Space-time domain
# --------------------------------------------------------------------------
# The benchmark's box is Omega = [0,1] x [-0.25,2].  It is TRANSLATED here so
# its lower-left corner sits on the origin, as an AFSI case requires:
#
#     paper frame :  x in [0, 1] ,  y in [-0.25, 2]
#     shift       :  X = x ,  Y = y + 0.25
#     this demo   :  X in [0, 1] ,  Y in [0, 2.25]
#
# The shift is NOT along the channel axis, so it does not leave the geometry
# invariant: the across-channel coordinate moves and the exact velocity has to
# be carried over with the new coordinates (see xi/analytic below).  What it
# does preserve is containment: in this frame the two plates run
#
#     lower  (xi~=0) :  Y = 0.25000 (X=0) -> 0.82735 (X=1)
#     upper  (xi~=h) :  Y = 1.40470 (X=0) -> 1.98205 (X=1)
#
# so both plates run the full width, terminate exactly on the left/right faces
# (both ends are complete channel cross-sections), and the channel stays inside
# the box, with 0.25 of margin below the lower plate corner and 0.268 above the
# upper plate at the outlet.
X_MIN = float(os.environ.get("X_MIN", "0.0"))
X_MAX = float(os.environ.get("X_MAX", "1.0"))
Y_MIN = float(os.environ.get("Y_MIN", "0.0"))
Y_MAX = float(os.environ.get("Y_MAX", "2.25"))

# Origin of the benchmark frame expressed in this frame.
SHIFT_X = 0.0

# --------------------------------------------------------------------------
# Slanted channel
# --------------------------------------------------------------------------
THETA_DEG = float(os.environ.get("THETA_DEG", "30.0"))
THETA = math.radians(THETA_DEG)
COS_T, SIN_T = math.cos(THETA), math.sin(THETA)

H_CHANNEL = 1.0                    # perpendicular channel width h (= 1)
D_CHANNEL = H_CHANNEL / COS_T      # "slanted channel width" D = h/cos(theta):
                                   # the VERTICAL distance between the plates
R_HALF = 0.5 * H_CHANNEL           # perpendicular half-width: walls at xi = +-R

# SHIFT_Y places the paper-frame box [0,1] x [-0.25,2] on the origin:
#     Y = y + SHIFT_Y with SHIFT_Y = 0.25
# so the box bottom y = -0.25 becomes Y = 0.  The channel's lower wall
# (xi~ = 0, which passes through the paper-frame origin) accordingly meets the
# inlet face at Y = SHIFT_Y = 0.25.
SHIFT_Y = float(os.environ.get("SHIFT_Y", "0.25"))

# --------------------------------------------------------------------------
# Physics
# --------------------------------------------------------------------------
# The benchmark's normalised parameters: rho = 1, mu = 0.5, dP/L = 1.  The
# solver is unit-agnostic; lengths (and hence DX) are "metres" only by
# convention.
RHO = 1.0        # benchmark: 1.0
MU = float(os.environ.get("MU", "0.5"))         # benchmark: 0.5
DP_DL = float(os.environ.get("DP_DL", "1.0"))   # -dp/ds along the channel axis

# Profile coefficient: with xi measured from the CENTRELINE (walls at +-R_HALF)
#     u_s = (dP/L)/(2*mu) * (R_HALF^2 - xi^2)
# which is exactly Eq. (48) rewritten about the centreline, and peaks at
#     U_max = (dP/L)/(2*mu) * (h/2)^2 = (dP/L) h^2 / (8 mu) = 0.25
# (mu = 0.5, dP/L = 1, h = 1) -- the value quoted by the benchmark.
A_COEF = DP_DL / (2.0 * MU)
U_MAX = A_COEF * R_HALF**2
U_MAX_PAPER = 0.25                    # as quoted in the benchmark text

# --------------------------------------------------------------------------
# Grid and time stepping
# --------------------------------------------------------------------------
# dx is fixed by the benchmark's dx = L/N convention with L = 1; the cell
# counts are derived from the box.  At N = 32 the box [0,1] x [0,2.25] is
# exactly 32 x 72 cells (no padding needed).
N = int(os.environ.get("N", "32"))
DX = 1.0 / N                          # = 0.03125
NX = int(round((X_MAX - X_MIN) / DX))
NY = int(round((Y_MAX - Y_MIN) / DX))
BOX_L = float(NX) * DX                # >= X_MAX - X_MIN
BOX_H = float(NY) * DX                # >= Y_MAX - Y_MIN
CELLS_PER_UNIT = 1.0 / DX

DT_FACTOR = float(os.environ.get("DT_FACTOR", "0.15"))  # benchmark: dt = 0.15 dx
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
T_END = float(os.environ.get("T_END", "2.0"))   # run to steady state (t >> RAMP_T)
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

# Number of fixed-point iterations per step used to couple the time-centered
# spring force with the marker/fluid velocity.  1 = uncoupled (apply the force
# once, lagged), >1 closes the loop within the step.
IB_ITERATIONS = int(os.environ.get("IB_ITERATIONS", "1"))
DIAG_IB = bool(os.environ.get("DIAG_IB"))

# --- immersed-boundary spreading normalisation ------------------------------
# 历史说明（commit 00d7d8c 已回退）：当时测得对偶性比值 <S*u,F>/<u,SF> 按 h^2 变化，
# 据此认为 distributor 多注入了 1/h^2 的动量并加了 h2 修正。**该测量漏掉了流体侧内积
# 的网格面积 dx*dy**：AFSI 的扩散 f_node += F*W*w/(dx*dy)（w=1）与插值 U += u_node*W
# 本来就是精确伴随的，前提是流体侧用体积权内积
#     sum_ij u_ij (S F)_ij * dx * dy   (dx = 1/(2N) = 流体网格步长的一半)
# 实测（N=16/32/64/128）：节点和比值恒等于 dx*dy，体积权比值恒等于 1.000000
# （afsic/tests/test_duality.py 本来就是按体积权写的，一直通过）。
# 因此默认不再缩放；SPREAD_NORM 只是留给标定的旋钮：
#     "none"    -> 按 AFSI 原样（默认，IB 力不做任何缩放）
#     "h2"      -> 旧行为：乘 DX^2，等于把 IB 力削弱 1024 倍（N=32），板子会失去约束
#     a number  -> 显式缩放因子
# 注意：显式（冻结力）tether 路线本身受时间步限制，突变增益 ∝ beta*dt^2/rho。实测在
# dt = 0.2*DX 下 beta >= ~1e4 就会发散（beta=1e6 几步内爆）；静止板请用隐式 drag 带
# （USE_IMPLICIT_DRAG=1，默认）。详见 docs/demo-426-ib-coupling-findings.md。
_sn = os.environ.get("SPREAD_NORM", "none").lower()
if _sn == "none":
    SPREAD_SCALE = 1.0
elif _sn == "h2":
    SPREAD_SCALE = DX * DX
else:
    SPREAD_SCALE = float(_sn)
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

    In the benchmark frame the wall coordinate is

        xi~ = y*cos(theta) - x*sin(theta),    plates at xi~ = 0 and xi~ = h.

    Substituting ``x = X - SHIFT_X``, ``y = Y - SHIFT_Y`` and measuring from
    the CENTRELINE (xi~ = h/2 = R_HALF) gives the form used here:

        xi = (Y - SHIFT_Y)*cos(theta) - (X - SHIFT_X)*sin(theta) - R_HALF

    so the plates sit at ``xi = +-R_HALF`` and the profile peaks at ``xi = 0``.
    Absorbing the translation into the coordinate is all the "variable
    transformation of the velocity equation" amounts to: the body force is
    spatially constant and the pressure enters only through its gradient, so
    both are invariant under a rigid translation, while ``xi`` -- and hence u --
    picks up the constants.  The velocity field is therefore unchanged; only
    its argument is rewritten.
    """
    return ((Y - SHIFT_Y) * COS_T - (X - SHIFT_X) * SIN_T) - R_HALF


def analytic(X, Y):
    """Exact velocity at a point (numpy-broadcasting friendly).

    Plane Poiseuille in the across-channel coordinate (benchmark Eq. (48)):

        u_s = (dP/L)/(2*mu) * ( (h/2)^2 - xi^2 )

    which equals ``U_max`` on the centreline and vanishes at both plates, with
    the velocity vector along the channel axis (cos, sin).
    """
    t = xi(X, Y)
    prof = (DP_DL / (2.0 * MU)) * (R_HALF**2 - t**2)
    return prof * COS_T, prof * SIN_T


def wall_y(X, side):
    """Y of the plate `side` (+1 upper, -1 lower) at a given X, this frame.

    The plates are the lines xi~ = 0 (lower) and xi~ = h (upper).
    """
    offset = 0.0 if side < 0 else H_CHANNEL
    return SHIFT_Y + ((X - SHIFT_X) * SIN_T + offset) / COS_T


def wall_endpoints(side):
    """The two endpoints of a plate, in this frame.

    With SHIFT_Y = 0.25 the box is [0,1] x [0,2.25] and both plates lie wholly
    inside it, spanning its full width and terminating exactly on the inlet
    (X = X_MIN) and outlet (X = X_MAX) faces -- so no clipping is needed and
    both ends are complete channel cross-sections (0.25 of margin below the
    lower plate corner, 0.268 above the upper plate at the outlet).
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
    # 2026-09-30：固体网格直接放在 demo 文件夹下（原为 plot/ 子目录）。
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here,
                        f"solid-426-N{int(1.0 / DX)}-t{PLATE_THICKNESS:g}.xdmf")


# Exact plate geometry, for the mesh-area guard.
PLATE_LENGTH = float(np.hypot(*(np.subtract(wall_endpoints(+1)[1],
                                            wall_endpoints(+1)[0]))))
PLATE_AREA_TARGET = PLATE_LENGTH * PLATE_THICKNESS


def output_path():
    """Output folder for this run: <demo>/<DRIVING>[_solver][_smoke]/ (2026-09-30).

    两种驱动模式（velocity / pressure）输出的文件名完全相同（velocity.xdmf,
    pressure.xdmf, solid_coords.xdmf, solid_displacement.xdmf），因此以驱动模式
    作为子目录名区分，直接位于 demo_426 文件夹下（原为 plot/N{N}_{solver}/）。
    """
    here = os.path.dirname(os.path.abspath(__file__))
    tag = os.environ.get("DRIVING", "velocity").lower()
    if SOLVER != "ipcs":
        tag += f"_{SOLVER}"
    if SMOKE:
        tag += "_smoke"
    out = os.path.join(here, tag)
    os.makedirs(out, exist_ok=True)
    return out + os.sep


def summary():
    yl, yu = inlet_interval()
    ol, ou = outlet_interval()
    lines = [
        "demo_426 — flow through a slanted channel (2-D IB benchmark)",
        f"domain          : [{X_MIN}, {X_MAX}] x [{Y_MIN}, {Y_MAX}]",
        f"channel         : theta={THETA_DEG:g} deg, h={H_CHANNEL:g} "
        f"(perpendicular), D=h/cos={D_CHANNEL:.6f} (vertical, "
        f"{CHANNEL_CELLS:.2f} cells)",
        f"inlet  (x={X_MIN:g}) : y in [{yl:.6f}, {yu:.6f}]  "
        f"(height {yu - yl:.6f})",
        f"outlet (x={X_MAX:g}) : y in [{ol:.6f}, {ou:.6f}]  "
        f"(height {ou - ol:.6f})",
        f"fluid           : rho={RHO:g}, mu={MU:g}, nu={NU:g}",
        f"driving         : -dp/ds = {DP_DL:g} Pa/m along the axis "
        f"(body force, NOT a pressure BC)",
        f"profile         : A={A_COEF:g}, u_max={U_MAX:.6f} m/s "
        f"(benchmark quotes {U_MAX_PAPER})",
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
