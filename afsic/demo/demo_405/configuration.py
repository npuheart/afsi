from mpi4py import MPI
from afsic import unique_filename, get_project_name

# Define the configuration for the simulation
config = {"nssolver": "chorinsolver",
          "project_name": "demo-405",
          "tag": "vessel-wall-3d",
          "velocity_order": 2,
          "force_order": 2,
          "pressure_order": 1,
          "num_processors": MPI.COMM_WORLD.size,
          "T": 0.2,
          "dt": 1/10000,
          "rho": 1.0,
          "Lx": 8.0,
          "Ly": 8.0,
          "Lz": 20.0,
          "Nx": 32,
          "Ny": 32,
          "Nz": 80,
          "mu": 0.036,      # dyn·s/cm² (CGS)
          "U_max": 1.0,     # Max inlet velocity (sinusoidal)
          "freq": 1.0,       # Inlet velocity frequency (Hz)
          "R_inner": 1.3,    # Pipe inner radius (for inlet profile)
          "cx": 4.0,         # Pipe center x in fluid domain
          "cy": 4.0,         # Pipe center y in fluid domain
          "beta": 1e6,       # Penalty for fixing solid (volumetric spring, demo_402 pattern)
          }

config["num_steps"] = int(config['T']/config['dt'])
config["output_path"] = unique_filename(config['project_name'], config['tag']) if MPI.COMM_WORLD.rank == 0 else None
config["output_path"] = MPI.COMM_WORLD.bcast(config["output_path"], root=0)
config["experiment_name"] = get_project_name(config['project_name']) if MPI.COMM_WORLD.rank == 0 else None
config["experiment_name"] = MPI.COMM_WORLD.bcast(config["experiment_name"], root=0)
