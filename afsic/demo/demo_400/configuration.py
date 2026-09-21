import os
from mpi4py import MPI
from afsic import unique_filename, get_project_name

#         mu_s=5.0E4,                    # Solid shear modulus or 2nd Lame Coef. [Pa]
#         lambda_s=4.5E5,                # Solid 1st Lame Coef. [Pa]
#         nu_s=0.45,                     # Solid Poisson ratio [-]

# Define the configuration for the simulation
config = {"nssolver": "chorinsolver",
          "project_name": "demo-400", 
          "tag": "parallel",
          "velocity_order": 2,
          "force_order": 2,
          "pressure_order": 1,
          "num_processors": MPI.COMM_WORLD.size,
          "Um": 0.0,                  # 100 cm/s
          "p_amp": 100.0,                  # 1 dyne/cm^2 
          "p_period": 2.0,                  # s
          "T": 30.0,                    # s
          "dt": 0.005/100,
          "rho": 1.0,                   # 1 g/cm^3
          "Lx": 200.0,
          "Ly": 100.0,
          "Nx": 128,
          "Ny": 64,
          "mu": 0.01,                  # 1 [Pa*s] , 10 [dyne/cm^2*s]
          "mu_s": 1e4,  # Solid elasticity
          "lambda_s": 1e4,  # Solid elasticity
          "nu_s": 0.45,
          "beta": 1e6,  # Penalty for head/tail fixation, 1e4 [dyne/cm^2]
          "waveform": "fast_open",  # "sin" | "fast_open" | "fast_close"
          "fast_ratio": 0.1,        # fraction of period for the fast phase
          }


config["num_steps"] = int(config['T']/config['dt'])
# 环境变量 STEPS 覆盖步数（短程运行；默认仍是 30 s 全时长）
if os.environ.get("STEPS"):
    config["num_steps"] = int(os.environ["STEPS"])
    config["T"] = config["num_steps"] * config["dt"]
config["output_path"] = (os.environ.get("OUTPUT_PATH") or unique_filename(config['project_name'], config['tag'])) if MPI.COMM_WORLD.rank == 0 else None
config["output_path"] = MPI.COMM_WORLD.bcast(config["output_path"], root=0)
config["experiment_name"] = os.environ.get("EXPERIMENT_NAME") or (get_project_name(config['project_name']) if MPI.COMM_WORLD.rank == 0 else None)
config["experiment_name"] = MPI.COMM_WORLD.bcast(config["experiment_name"], root=0)
