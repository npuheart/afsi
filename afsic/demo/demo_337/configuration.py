"""Configuration for demo_337 — idealized left ventricle FSI.

IB-FE fluid-structure interaction of an idealized left ventricle
(LV ellipsoid) driven by a physiological endocardial pressure load
(diastole & systole).

Reference:
    Ma, Cai, Wang & Gao (2025) "AFSI: Automated Fluid-Structure
    Interaction Solver Development for Nonlinear Solid Mechanics."
    arXiv:2509.00014.

Mesh (data/mesh/) and pulse-fenicsx reference data (data/reference/)
are generated with the fenicsx-pulse Docker image — see readme.md
and data/plot/generate_mesh.py.
"""

from mpi4py import MPI
from afsic import get_project_name

# 1 mmHg in Pa (converts the physiological pressures below)
mmHg = 1333.22368421

config = {
    "nssolver": "chorinsolver",
    "project_name": "demo-337",
    "tag": "lv-fsi",

    # --- Discretisation order ---
    "velocity_order": 2,
    "force_order": 2,
    "pressure_order": 1,
    "num_processors": MPI.COMM_WORLD.size,

    # --- Time ---
    "T":  0.1,        # total simulation time [s]
    "dt": 1 / 1000,   # time step [s]

    # --- Fluid (SI units) ---
    "rho": 1.0,       # density [kg/m^3]
    "mu":  0.01,      # dynamic viscosity [Pa·s]

    # --- Fluid box geometry ---
    "Lx": 5.0, "Ly": 5.0, "Lz": 5.0,
    "Nx": 32,  "Ny": 32,  "Nz": 32,
    "Nl": 20,

    # --- Solid (passive Neo-Hookean) ---
    "mu_s": 0.1,      # shear modulus [Pa]
    "beta": 5e6,      # base-ring penalty [Pa]

    # --- Physiological pressure (endocardium) ---
    "diastole_pressure": 8.0 * mmHg,
    "systole_pressure": 110.0 * mmHg,

    # --- Mesh (pulse-fenicsx / Docker generated, relative to demo_337/) ---
    "mesh_dir": "data/mesh/lv_ellipsoid/geometry",
}

config["num_steps"] = int(config["T"] / config["dt"])

# main.py writes all results here (inside data/)
config["output_path"] = "data/results/"

config["experiment_name"] = get_project_name(
    config["project_name"]) if MPI.COMM_WORLD.rank == 0 else None
config["experiment_name"] = MPI.COMM_WORLD.bcast(
    config["experiment_name"], root=0)
