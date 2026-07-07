from mpi4py import MPI
from afsic import unique_filename, get_project_name

# =============================================================================
# Section 4.5: Channel flow over an elastic thick plate (beamInCrossFlow)
# Reference: Tuković et al. (2018), OpenFOAM FV Solver for FSI
#
# Original form (Richter):
#   Channel: 1.5 m x 0.8 m x 0.4 m (half-domain, symmetry at y=0.4)
#   Plate:   x in [0.45, 0.55], y in [0, 0.2], z in [-0.2, 0]
#   Fluid:   rho=1000 kg/m^3, nu=0.001 m^2/s
#   Solid:   rho=1000 kg/m^3, E=1.4 MPa, nu=0.4 (St.Venant-Kirchhoff)
#   Inlet:   parabolic, peak U=0.2 m/s, Re=40 w.r.t. plate height h=0.2 m
#   Ramp:    U(t) = 0.2 * [1 - cos(pi*t/4)] / 2, max at t=4 s
#
# CGS conversion:
#   1 m = 100 cm,  1 Pa = 10 dyne/cm^2,  1 Pa*s = 10 dyne*s/cm^2
# =============================================================================

# Material: E=1.4 MPa, nu=0.4  ->  Lame constants in CGS [dyne/cm^2]
E_s = 1.4e7       # Young's modulus [dyne/cm^2]
nu_s = 0.4        # Poisson's ratio [-]
mu_s_val = E_s / (2.0 * (1.0 + nu_s))              # = 5.0e6
lambda_s_val = E_s * nu_s / ((1.0 + nu_s) * (1.0 - 2.0 * nu_s))  # = 2.0e7

config = {
    "nssolver": "chorinsolver",
    "project_name": "demo-403",
    "tag": "parallel",
    "velocity_order": 2,
    "force_order": 2,
    "pressure_order": 1,
    "num_processors": MPI.COMM_WORLD.size,

    # --- Geometry (CGS: cm) ---
    "Lx": 150.0,        # channel length [cm]
    "Ly": 40.0,         # channel half-height [cm] (symmetry at top)
    "Lz": 40.0,         # channel width [cm] (z in [-20, 20] for full domain)
    "Nx": 150,          # fluid grid points
    "Ny": 40,
    "Nz": 40,

    # --- Fluid (CGS) ---
    "Um": 20.0,         # peak inlet velocity [cm/s]  (0.2 m/s)
    "rho": 1.0,         # density [g/cm^3]  (1000 kg/m^3)
    "mu": 10.0,         # dynamic viscosity [dyne*s/cm^2]  (1 Pa*s)

    # --- Solid (CGS) ---
    "mu_s": mu_s_val,           # shear modulus [dyne/cm^2]
    "lambda_s": lambda_s_val,   # 1st Lame constant [dyne/cm^2]
    "nu_s": nu_s,               # Poisson's ratio [-]
    "beta": 1e8,                # penalty for fixing plate bottom [dyne/cm^3]

    # --- Time ---
    "T": 6.0,           # total simulation time [s]
    "dt": 0.001,        # time step [s]
    "ramp_time": 4.0,   # ramp duration [s]
}

config["num_steps"] = int(config['T'] / config['dt'])
config["output_path"] = unique_filename(config['project_name'], config['tag']) if MPI.COMM_WORLD.rank == 0 else None
config["output_path"] = MPI.COMM_WORLD.bcast(config["output_path"], root=0)
config["experiment_name"] = get_project_name(config['project_name']) if MPI.COMM_WORLD.rank == 0 else None
config["experiment_name"] = MPI.COMM_WORLD.bcast(config["experiment_name"], root=0)
