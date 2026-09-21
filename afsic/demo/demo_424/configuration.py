"""Shared configuration for demo_424 — tethered aorta in a box (2-D planar).

Two cases share every parameter except the solid geometry:

  CASE=open    patent aorta: two tethered wall strips
  CASE=closed  occluded aorta: the same strips + a full-occlusion membrane
               at mid-length, so the solid outline is an "H"

2-D planar idealisation
-----------------------
The 3-D cylindrical aorta is cut along a plane through its axis.  The lumen
becomes a parallel-plate channel of half-height A_LUMEN and the wall becomes
two flat strips of thickness T_WALL.  The immersed wall is held in place by a
tether (volumetric spring) only -- there is no constitutive law.

All quantities are SI.  The fluid is normalised (rho = 1) with a blood-like
viscosity, so the flow stays in the laminar, weakly inertial regime
(Re ~ 5 in the patent case).
"""
import os

MMHG = 133.322387415  # Pa per mmHg

CASE = os.environ.get("CASE", "open").lower()
if CASE not in ("open", "closed"):
    raise ValueError("CASE must be 'open' or 'closed'")

# --------------------------------------------------------------------------
# Geometry [m]
# --------------------------------------------------------------------------
L_AORTA = 0.1                       # aorta length
A_LUMEN = 0.015                     # lumen half-height (= aorta radius)
T_WALL = 0.002                      # wall thickness

BOX_W = 1.5 * (2.0 * A_LUMEN)       # box 50% wider than the lumen diameter
Y_C = 0.5 * BOX_W                   # aorta centreline

Y_IN_LO, Y_IN_HI = Y_C - A_LUMEN, Y_C + A_LUMEN            # 0.0075, 0.0375
Y_OUT_LO, Y_OUT_HI = Y_IN_LO - T_WALL, Y_IN_HI + T_WALL    # 0.0055, 0.0395
GAP = Y_OUT_LO                      # outer fluid layer on each side [m]

DISC_T = 0.002                      # occluding membrane thickness [m]
DISC_X = 0.5 * L_AORTA              # membrane position, from the aorta start

# --------------------------------------------------------------------------
# Fluid grid: square cells, box half a cell longer than the aorta at each end
# The aorta's open ends therefore sit at cell centres in x -- the pressure
# Dirichlet nodes never coincide with a Lagrangian end node.
# --------------------------------------------------------------------------
NY = int(os.environ.get("NY", "90"))        # cells across BOX_W
H = BOX_W / NY                              # cell size [m]
NX = round(L_AORTA / H) + 1                 # = L_AORTA/H + 1
BOX_L = NX * H                              # = L_AORTA + H
X_OFF = 0.5 * H                             # aorta start offset in x

# --------------------------------------------------------------------------
# Solid (Lagrangian) mesh
# --------------------------------------------------------------------------
SOLID_DIV = int(os.environ.get("SOLID_DIV", "2"))   # solid cell = H / SOLID_DIV
HS = H / SOLID_DIV

# --------------------------------------------------------------------------
# Physics
# --------------------------------------------------------------------------
RHO = 1.0                                   # normalised density
MU = 0.0035                                 # blood-like dynamic viscosity
BETA = float(os.environ.get("BETA", "1.0e7"))   # tether stiffness [N/m^3]

DP_MMHG = float(os.environ.get("DP_MMHG", "0.02" if CASE == "open" else "0.2"))
DP = DP_MMHG * MMHG                         # driving pressure difference [Pa]

DISC_AREA = (Y_IN_HI - Y_IN_LO)             # membrane span per unit depth [m]

# --------------------------------------------------------------------------
# Time integration
# --------------------------------------------------------------------------
# NOTE on the solver choice: the box is driven by a pressure Dirichlet on BOTH
# open ends.  With IPCSSolver's momentum equation, which carries the term
# -dot(p_, div(v))*dx, the natural (do-nothing) boundary condition becomes
# mu*du/dn = p*n.  At an INFLOW that is a spurious normal traction of order DP,
# which for a nearly parallel flow produces a one-cell pressure boundary layer
# (verified on the solid-free control in test_channel.py: p drops from DP to
# ~0 inside the first cell and the developed gradient comes out ~20x too
# small).  ChorinSolver has no pressure term in its momentum predictor, so its
# natural boundary condition is the homogeneous mu*du/dn = 0 and the pressure
# Dirichlet enters only through the projection step -- which reproduces the
# exact linear pressure to 6 significant digits.  Default: chorin.
SOLVER = os.environ.get("SOLVER", "chorin").lower()
DT = float(os.environ.get("DT", "2.0e-4"))
RAMP_T = float(os.environ.get("RAMP_T", "0.05"))
T_END = float(os.environ.get("T_END", "0.4"))
NSTEPS = int(round(T_END / DT))

VELOCITY_ORDER = 2
PRESSURE_ORDER = 1
FORCE_ORDER = 2

# --------------------------------------------------------------------------
# Derived scales / analytic quantities
# --------------------------------------------------------------------------
NU = MU / RHO                               # kinematic viscosity [m^2/s]
T_VISC = A_LUMEN**2 / NU                    # viscous time across the lumen [s]

# Expected membrane displacement from the 1-D tether balance
#   beta * delta * (H_disc * T_WALL) = DP * H_disc   ->   delta = DP / (beta*T_WALL)
DISC_DELTA = DP / (BETA * T_WALL) if CASE == "closed" else 0.0


def p_inlet(t):
    """Imposed inlet pressure [Pa], linear ramp then hold."""
    if RAMP_T <= 0.0:
        return DP
    return DP * min(max(t / RAMP_T, 0.0), 1.0)


def solid_mesh_path():
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "plot",
                        f"mesh-424-{CASE}-NY{NY}-S{SOLID_DIV}.xdmf")


def output_path():
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "plot", f"{CASE}_NY{NY}{'' if SOLID_DIV == 2 else f'_S{SOLID_DIV}'}{'' if SOLVER == 'chorin' else '_' + SOLVER}")
    os.makedirs(out, exist_ok=True)
    return out + os.sep


def summary():
    lines = [
        f"case            : {CASE}",
        f"aorta           : L={L_AORTA} m, lumen half-height={A_LUMEN} m, "
        f"wall={T_WALL} m",
        f"box             : {BOX_L:.6g} x {BOX_W:.6g} m  ({NX} x {NY} cells, "
        f"h={H:.6g} m)",
        f"outer gap       : {GAP:.6g} m per side",
        f"solid cell      : {HS:.6g} m (h/{SOLID_DIV})",
        f"fluid           : rho={RHO}, mu={MU}  (nu={NU:g} m^2/s)",
        f"tether          : beta={BETA:g} N/m^3",
        f"pressure        : {DP_MMHG:g} mmHg = {DP:.6g} Pa, "
        f"ramp {RAMP_T:g} s, T_end {T_END:g} s",
        f"solver          : {SOLVER}, dt={DT:g} s, steps={NSTEPS}",
    ]
    if CASE == "closed":
        lines.append(f"membrane        : t={DISC_T} m at x={DISC_X} m "
                     f"(from aorta start), delta_expected={DISC_DELTA:.6g} m")
    lines.append(f"viscous time    : {T_VISC:.6g} s")
    return "\n".join(lines)
