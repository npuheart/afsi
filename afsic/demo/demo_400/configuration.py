from mpi4py import MPI
from afsic import unique_filename, get_project_name

#         mu_s=5.0E4,                    # Solid shear modulus or 2nd Lame Coef. [Pa]
#         lambda_s=4.5E5,                # Solid 1st Lame Coef. [Pa]
#         nu_s=0.45,                     # Solid Poisson ratio [-]

# Define the configuration for the simulation
config = {"nssolver": "chorinsolver",
          "project_name": "demo-340", 
          "tag": "parallel",
          "velocity_order": 2,
          "force_order": 2,
          "pressure_order": 1,
          "num_processors": MPI.COMM_WORLD.size,
          "Um": 100.0,                  # 100 cm/s       
          "T": 1.0,                    # s
          "dt": 0.005/10,
          "rho": 1.0,                   # 1 g/cm^3
          "Lx": 200.0,
          "Ly": 100.0,
          "Nx": 128,
          "Ny": 64,
          "mu": 10.0,                  # 1 [Pa*s] , 10 [dyne/cm^2*s]
          "mu_s": 5e5,  # Solid elasticity
          "lambda_s": 4.5e6,  # Solid elasticity
          "nu_s": 0.45,
          "beta": 1e6,  # Penalty for head/tail fixation, 1e4 [dyne/cm^2]
          }


config["num_steps"] = int(config['T']/config['dt'])
config["output_path"] = unique_filename(config['project_name'], config['tag']) if MPI.COMM_WORLD.rank == 0 else None
config["output_path"] = MPI.COMM_WORLD.bcast(config["output_path"], root=0)
config["experiment_name"] = get_project_name(config['project_name']) if MPI.COMM_WORLD.rank == 0 else None
config["experiment_name"] = MPI.COMM_WORLD.bcast(config["experiment_name"], root=0)
