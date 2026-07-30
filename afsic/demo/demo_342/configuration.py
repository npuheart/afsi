"""Configuration for IB-FE flow past a rigid cylinder."""

from mpi4py import MPI
from afsic import unique_filename, get_project_name

# The cylinder is modelled as a Neo-Hookean solid with extremely high
# stiffness to approximate a rigid body, plus a penalty term to fix it
# in place.  Same IB-FE coupling pattern as demo_402 (Turek FSI).

config = {
    "nssolver": "chorinsolver",
    "project_name": "demo-342",
    "tag": "cylinder-ibfe",
    "velocity_order": 2,
    "force_order": 2,
    "pressure_order": 1,
    "num_processors": MPI.COMM_WORLD.size,

    # --- Fluid (SI units) ---
    "Um":  1.0,          # mean inlet velocity [m/s]
    "rho": 1000.0,       # fluid density [kg/m^3]
    "mu":  1.0,          # dynamic viscosity [Pa·s] → Re=100
    "T":   10.0,         # total simulation time [s]
    "dt":  0.001,        # time step [s]

    # --- Channel geometry ---
    "Lx": 2.2,           # channel length [m]
    "Ly": 0.41,          # channel height [m]
    "Nx": 220,           # fluid grid cells in x
    "Ny": 41,            # fluid grid cells in y

    # --- Solid: Neo-Hookean disk, high stiffness → rigid ---
    # E = 2e11 Pa (steel-like), nu = 0.3
    # mu_s = E/(2*(1+nu)) ≈ 7.7e10 Pa
    # lambda_s = E*nu/((1+nu)*(1-2*nu)) ≈ 1.15e11 Pa
    "mu_s":     7.7e10,  # shear modulus [Pa] — ~1000× demo_402
    "lambda_s": 1.15e11, # 1st Lamé constant [Pa]
    "beta":     1e12,    # penalty for fixing disk in place [Pa]
}

config["num_steps"] = int(config['T'] / config['dt'])
config["output_path"] = unique_filename(
    config['project_name'], config['tag']) if MPI.COMM_WORLD.rank == 0 else None
config["output_path"] = MPI.COMM_WORLD.bcast(config["output_path"], root=0)
config["experiment_name"] = get_project_name(
    config['project_name']) if MPI.COMM_WORLD.rank == 0 else None
config["experiment_name"] = MPI.COMM_WORLD.bcast(
    config["experiment_name"], root=0)
