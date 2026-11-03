"""Shared parameters of the semilunar-valve demo (CGS units: cm, g, s; pressure in dyn/cm^2).

Set-up after Black et al., "An open-source computational framework for immersed fluid-structure
interaction modeling using FEBio and MFEM" (arXiv:2601.08266), Sec. 4.5 "Semilunar heart valve",
with the geometry rebuilt from the LS-DYNA ICFD heart-valve example (see readme.md). The vessel wall
is rigid: an immersed solid held at its reference position by a tether (volumetric spring).

Every value marked (env) can be overridden by the environment variable of the same name.
"""
import os
import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))


def _env(name, default, cast=float):
    v = os.environ.get(name)
    return default if v is None or v == "" else cast(v)


# ---- geometry (cm) ---------------------------------------------------------------------------------
Z_IN, Z_OUT = -4.0, 14.2            # inlet 4 cm upstream of the root, total length 18.2 cm (paper)
HALF_WIDTH = _env("HALF_WIDTH", 2.0)  # fluid box |x|, |y| <= HALF_WIDTH (wall outer radius <= 1.83)
WALL_T = _env("WALL_T", 0.2)          # vessel wall thickness (paper: rigid vessel, no thickness)
LEAFLET_T = 0.015                     # paper
EXT = 0.10                            # leaflet attachment extension into the wall

# ---- fluid grid ------------------------------------------------------------------------------------
H = _env("H", 0.2)                    # (env) fluid cell size; production 0.05 or finer
NX = NY = int(round(2 * HALF_WIDTH / H))
NZ = int(round((Z_OUT - Z_IN) / H))
VELOCITY_ORDER, PRESSURE_ORDER = 2, 1
FORCE_ORDER = _env("FORCE_ORDER", 1, int)   # (env) Lagrange order of the solid force/position fields

# ---- solid meshes (sizes follow the fluid grid: marker spacing about half a fluid cell) -----------
H_SOLID = _env("H_SOLID", H / 2)            # (env) wall surface element size near the root
WALL_LAYERS = _env("WALL_LAYERS", max(2, int(round(WALL_T / H_SOLID))), int)
VALVE_LAYERS = _env("VALVE_LAYERS", 4, int)            # paper: four elements across the thickness
VALVE_REFINE = _env("VALVE_REFINE", 0 if H >= 0.1 else 1, int)

# ---- fluid (paper) -----------------------------------------------------------------------------------
RHO = 1.1                             # g/cm^3
MU = 0.036                            # dyn s/cm^2

# ---- leaflets: compressible neo-Hookean (paper, after Bonet & Wood) ------------------------------
E_LEAFLET = 2.0e6                     # dyn/cm^2
NU_LEAFLET = 0.4

# ---- tether (rigid wall; also anchors the leaflet attachment extension) ---------------------------
BETA = _env("BETA", 2.0e7)            # (env) dyn/cm^4: f = BETA (X_ref - x)
VALVE_TETHER = _env("VALVE_TETHER", 1, int)   # (env) 1: also tether the leaflet extension (part 5)

# ---- time ------------------------------------------------------------------------------------------
DT = _env("DT", 1.0e-4)               # paper
T_END = _env("T_END", 1.0)            # one heartbeat (paper)
NSTEPS = _env("STEPS", int(round(T_END / DT)), int)
OUTPUT_EVERY = _env("OUTPUT_EVERY", max(NSTEPS // 50, 1), int)

# ---- inflow ------------------------------------------------------------------------------------------
# Paper: plug flow v = -(Q/A) n at the inlet, Q(t) from Oks et al. (2022), shown only as a figure.
# Placeholder until the digitised curve is supplied: one systolic half-sine, zero in diastole.
# Replace q_inlet() (or put a two-column t,Q file at INFLOW_CSV) to use the real waveform.
Q_MAX = _env("Q_MAX", 250.0)          # cm^3/s
T_SYS = _env("T_SYS", 0.30)           # s
INFLOW_CSV = os.environ.get("INFLOW_CSV", "")
_q_table = np.loadtxt(INFLOW_CSV, delimiter=",") if INFLOW_CSV else None


def q_inlet(t):
    if _q_table is not None:
        return float(np.interp(t % _q_table[-1, 0], _q_table[:, 0], _q_table[:, 1]))
    return Q_MAX * np.sin(np.pi * t / T_SYS) if t < T_SYS else 0.0


# ---- paths -------------------------------------------------------------------------------------------
LSDYNA_URL = "https://lsdyna.ansys.com/wp-content/uploads/2022/12/ICFD_example_heartvalve.zip"
LSDYNA_DIR = os.environ.get("LSDYNA_DIR", os.path.join(_here, "data", "lsdyna"))
MESH_DIR = os.environ.get("MESH_DIR", os.path.join(_here, "data", f"mesh_h{H:g}"))


def output_path():
    p = os.environ.get("OUTPUT_PATH") or os.path.join(_here, "data", f"run_h{H:g}_dt{DT:g}")
    os.makedirs(p, exist_ok=True)
    return p + os.sep
