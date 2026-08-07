"""Flow past a cylinder — IB-FE immersed boundary method.

Uses a Neo-Hookean solid disk with extremely high stiffness + penalty
fixation to approximate a rigid cylinder.  Immersed boundary feedback
force coupling (same architecture as demo_402 Turek FSI).

Unlike demo_402, this case has NO elastic tail — only the cylinder disk.
The mesh, units (SI), and geometry differ, but the coupling architecture
is identical to allow manual side-by-side comparison.
"""

from mpi4py import MPI
from afsic import unique_filename, get_project_name

# Material properties for rigid cylinder approximation:
# Young's modulus E ≈ 200 GPa, Poisson's ratio nu = 0.3
# Lame constants in SI [Pa]:
#   mu_s = E / (2*(1+nu)) = 2e11 / 2.6 ≈ 7.7e10
#   lambda_s = E*nu / ((1+nu)*(1-2*nu)) = 2e11*0.3 / (1.3*0.4) ≈ 1.15e11

config = {
    "nssolver": "chorinsolver",
    "project_name": "demo-339",
    "tag": "ibfe",
    "velocity_order": 2,
    "force_order": 2,
    "pressure_order": 1,
    "num_processors": MPI.COMM_WORLD.size,

    # --- Fluid (SI) ---
    "Um": 1.0,         # mean inlet velocity [m/s]
    "T": 10.0,         # simulation time [s]
    "dt": 0.001,       # time step [s]
    "rho": 1000.0,     # fluid density [kg/m^3]
    "Lx": 2.2,         # channel length [m]
    "Ly": 0.41,        # channel height [m]
    "Nx": 220,         # fluid mesh cells in x
    "Ny": 41,          # fluid mesh cells in y
    "mu": 1.0,         # dynamic viscosity [Pa·s]

    # --- Solid: Neo-Hookean, high stiffness → rigid cylinder ---
    # E ≈ 200 GPa, nu = 0.3
    "mu_s": 7.7e10,       # Solid shear modulus (2nd Lame const.) [Pa]
    "lambda_s": 1.15e11,  # Solid 1st Lame constant [Pa]
    "nu_s": 0.3,          # Solid Poisson ratio [-]
    "beta": 1e12,         # Penalty for cylinder fixation [Pa]
}

config["num_steps"] = int(config["T"] / config["dt"])
config["output_path"] = (
    unique_filename(config["project_name"], config["tag"])
    if MPI.COMM_WORLD.rank == 0
    else None
)
config["output_path"] = MPI.COMM_WORLD.bcast(config["output_path"], root=0)
config["experiment_name"] = (
    get_project_name(config["project_name"]) if MPI.COMM_WORLD.rank == 0 else None
)
config["experiment_name"] = MPI.COMM_WORLD.bcast(config["experiment_name"], root=0)
