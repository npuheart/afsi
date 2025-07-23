from afsic import unique_filename
from mpi4py import MPI
from petsc4py import PETSc

import time
import requests
import numpy as np

import dolfinx
from dolfinx.fem import (Function, functionspace,
                         dirichletbc, locate_dofs_topological)
from dolfinx.mesh import CellType, GhostMode, locate_entities, meshtags
from basix.ufl import element

from afsic import IPCSSolver, TimeManager
from afsic import swanlab_init, swanlab_upload

# Define the configuration for the simulation
config = {"nssolver": "ipcssolver",
          "project_name": "demo-2000", 
          "tag": "parallel",
          "T": 10.0,
          "dt": 2e-4,
          "rho": 1.0,
          "Ne": 32,
          "Nl": 20,
          "nv": 0.01,
          }

config["num_steps"] = int(config['T']/config['dt'])
config["output_path"] = unique_filename(config['project_name'], config['tag']) if MPI.COMM_WORLD.rank == 0 else None
config["output_path"] = MPI.COMM_WORLD.bcast(config["output_path"], root=0)
config["experiment_name"] = requests.get(f"http://counter.pengfeima.cn/{config['project_name']}").text if MPI.COMM_WORLD.rank == 0 else None
config["experiment_name"] = MPI.COMM_WORLD.bcast(config["experiment_name"], root=0)
swanlab_init(config['project_name'], config['experiment_name'], config)





time_manager = TimeManager(config['T'], config['num_steps'], fps=20)



# for step in range(  config['num_steps']):
for step in range(10):
    current_time = step * config['dt']
    time_manager.should_output(step)
    data_log = {}
    if MPI.COMM_WORLD.rank == 0:
        time.sleep(1)
        print(f"Step {step+1}/{config['num_steps']}, Time: {current_time:.2f}s")
        swanlab_upload(current_time, data_log, capacity=10, pressure_left=20)



# ib_mesh = IBMesh([Point(0, 0), Point(1, 1)], [Ne, Ne], order_velocity)








