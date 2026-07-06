from mpi4py import MPI
from afsic import unique_filename, get_project_name

# Material properties: Young's modulus E = 5.6 MPa, Poisson's ratio nu = 0.4
# Lame constants in CGS [dyne/cm^2]:
#   mu_s = E / (2*(1+nu)) = 5.6e7 / 2.8 = 2.0e7
#   lambda_s = E*nu / ((1+nu)*(1-2*nu)) = 5.6e7*0.4 / (1.4*0.2) = 8.0e7

# Define the configuration for the simulation
config = {"nssolver": "chorinsolver",
          "project_name": "demo-402",
          "tag": "parallel",
          "velocity_order": 2,
          "force_order": 2,
          "pressure_order": 1,
          "num_processors": MPI.COMM_WORLD.size,
          "Um": 200.0,                  # mean inlet velocity [cm/s] (2 m/s)
          "T": 10.0,                    # s
          "dt": 0.005/100,
          "rho": 1.0,                   # 1 g/cm^3
          "Lx": 220.0,                  # Turek channel length [cm]
          "Ly": 41.0,                   # Turek channel height [cm]
          "Nx": 220,
          "Ny": 41,
          "mu": 10.0,                   # 1 [Pa*s] , 10 [dyne/cm^2*s]
          "mu_s": 2.0e7,                # Solid shear modulus (2nd Lame Coef.) [dyne/cm^2]
          "lambda_s": 8.0e7,            # Solid 1st Lame Coef. [dyne/cm^2]
          "nu_s": 0.4,                  # Solid Poisson ratio [-]
          "beta": 1e8,  # Penalty for head/tail fixation, 1e4 [dyne/cm^2]
          }


config["num_steps"] = int(config['T']/config['dt'])
config["output_path"] = unique_filename(config['project_name'], config['tag']) if MPI.COMM_WORLD.rank == 0 else None
config["output_path"] = MPI.COMM_WORLD.bcast(config["output_path"], root=0)
config["experiment_name"] = get_project_name(config['project_name']) if MPI.COMM_WORLD.rank == 0 else None
config["experiment_name"] = MPI.COMM_WORLD.bcast(config["experiment_name"], root=0)
