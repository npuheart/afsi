"""Flow past a cylinder — body-fitted mesh (cylinder is a hole).

The no-slip condition on the cylinder surface is enforced directly
via a Dirichlet BC (u=0 on the cylinder facet).  No IBM needed.
"""

from mpi4py import MPI
from afsic import unique_filename, get_project_name

config = {
    "nssolver": "chorinsolver",
    "project_name": "demo-339",
    "tag": "body-fitted",
    "velocity_order": 2,
    "pressure_order": 1,

    "Um":  1.0, "rho": 1000.0, "mu": 1.0,
    "T": 10.0, "dt": 0.001,
    "Lx": 2.2, "Ly": 0.41,
    "Nx": 220, "Ny": 41,  # ignored — mesh comes from .geo
}

config["num_steps"] = int(config['T'] / config['dt'])
config["output_path"] = unique_filename(config['project_name'], config['tag']) if MPI.COMM_WORLD.rank == 0 else None
config["output_path"] = MPI.COMM_WORLD.bcast(config["output_path"], root=0)
config["experiment_name"] = get_project_name(config['project_name']) if MPI.COMM_WORLD.rank == 0 else None
config["experiment_name"] = MPI.COMM_WORLD.bcast(config["experiment_name"], root=0)
