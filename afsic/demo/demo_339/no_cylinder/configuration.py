"""Channel flow without cylinder — baseline reference.

Physics: 2D channel with parabolic inlet, no-slip top/bottom walls,
pressure outlet.  Re = rho * Um * D / mu ≈ 100.

This serves as the reference case: no immersed body, pure channel flow
to validate the fluid solver baseline before adding the cylinder.
"""

from mpi4py import MPI
from afsic import unique_filename, get_project_name

config = {
    "nssolver": "chorinsolver",
    "project_name": "demo-339",
    "tag": "no-cylinder",
    "velocity_order": 2,
    "pressure_order": 1,

    # --- Fluid (SI) ---
    "Um":  1.0,          # mean inlet velocity [m/s]
    "rho": 1000.0,       # fluid density [kg/m^3]
    "mu":  1.0,          # dynamic viscosity [Pa·s] → Re=100  (based on D=0.1)
    "T":   10.0,         # total simulation time [s]
    "dt":  0.001,        # time step [s]

    # --- Channel geometry ---
    "Lx": 2.2,           # channel length [m]
    "Ly": 0.41,          # channel height [m]
    "Nx": 220,           # fluid grid cells in x
    "Ny": 41,            # fluid grid cells in y
}

config["num_steps"] = int(config['T'] / config['dt'])
config["output_path"] = unique_filename(
    config['project_name'], config['tag']) if MPI.COMM_WORLD.rank == 0 else None
config["output_path"] = MPI.COMM_WORLD.bcast(config["output_path"], root=0)
config["experiment_name"] = get_project_name(
    config['project_name']) if MPI.COMM_WORLD.rank == 0 else None
config["experiment_name"] = MPI.COMM_WORLD.bcast(
    config["experiment_name"], root=0)
