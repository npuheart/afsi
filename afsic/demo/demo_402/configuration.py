from mpi4py import MPI
from afsic import unique_filename, get_project_name

#         mu_s=5.0E4,                    # Solid shear modulus or 2nd Lame Coef. [Pa]
#         lambda_s=4.5E5,                # Solid 1st Lame Coef. [Pa]
#         nu_s=0.45,                     # Solid Poisson ratio [-]

# Define the configuration for the simulation
config = {"nssolver": "chorinsolver",
          "project_name": "demo-402", 
          "tag": "parallel",
          "velocity_order": 2,
          "force_order": 2,
          "pressure_order": 1,
          "num_processors": MPI.COMM_WORLD.size,
          "Um": 1.0,                  # mean inlet velocity [cm/s]
          "p_amp": 0.0,                    # no follower pressure for Turek FSI
          "p_period": 2.0,                 # s (unused)
          "T": 30.0,                    # s
          "dt": 0.005/10,
          "rho": 1.0,                   # 1 g/cm^3
          "Lx": 220.0,                  # Turek channel length [cm]
          "Ly": 41.0,                   # Turek channel height [cm]
          "Nx": 220,
          "Ny": 41,
          "mu": 0.01,                  # 1 [Pa*s] , 10 [dyne/cm^2*s]
          "mu_s": 1e4,  # Solid elasticity
          "lambda_s": 1e4,  # Solid elasticity
          "nu_s": 0.45,
          "beta": 1e6,  # Penalty for head/tail fixation, 1e4 [dyne/cm^2]
          "waveform": "fast_open",  # "sin" | "fast_open" | "fast_close"
          "fast_ratio": 0.1,        # fraction of period for the fast phase
          }


config["num_steps"] = int(config['T']/config['dt'])
config["output_path"] = unique_filename(config['project_name'], config['tag']) if MPI.COMM_WORLD.rank == 0 else None
config["output_path"] = MPI.COMM_WORLD.bcast(config["output_path"], root=0)
config["experiment_name"] = get_project_name(config['project_name']) if MPI.COMM_WORLD.rank == 0 else None
config["experiment_name"] = MPI.COMM_WORLD.bcast(config["experiment_name"], root=0)
