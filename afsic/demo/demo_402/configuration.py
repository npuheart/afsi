import os

from mpi4py import MPI
from afsic import unique_filename, get_project_name

# Material properties: Young's modulus E = 5.6 MPa, Poisson's ratio nu = 0.4
# Lame constants in CGS [dyne/cm^2]:
#   mu_s = E / (2*(1+nu)) = 5.6e7 / 2.8 = 2.0e7
#   lambda_s = E*nu / ((1+nu)*(1-2*nu)) = 5.6e7*0.4 / (1.4*0.2) = 8.0e7

# 运行期可覆盖（便于同一算例做短程/分辨率对比；默认值即 Turek FSI2 原始配置）：
#   T / DT / NX / NY / UM / LX / LY / MU_S / LAMBDA_S / RHO / MU
T_OVERRIDE = float(os.environ.get("T", "10.0"))            # 物理时长 [s]
DT_OVERRIDE = float(os.environ.get("DT", str(0.005 / 100)))  # 时间步 [s]
NX_OVERRIDE = int(os.environ.get("NX", "220"))
NY_OVERRIDE = int(os.environ.get("NY", "41"))
UM_OVERRIDE = float(os.environ.get("UM", "200.0"))          # 入口平均速度 [cm/s]
LX_OVERRIDE = float(os.environ.get("LX", "220.0"))          # 通道长 [cm]
LY_OVERRIDE = float(os.environ.get("LY", "41.0"))           # 通道高 [cm]
MU_S_OVERRIDE = float(os.environ.get("MU_S", "2.0e7"))      # 固体剪切模量 [dyne/cm^2]
LAMBDA_S_OVERRIDE = float(os.environ.get("LAMBDA_S", "8.0e7"))
RHO_OVERRIDE = float(os.environ.get("RHO", "1.0"))          # 流体密度 [g/cm^3]
MU_OVERRIDE = float(os.environ.get("MU", "10.0"))           # 流体粘度 [dyne*s/cm^2]

# Define the configuration for the simulation
config = {
    "nssolver": "chorinsolver",
    "project_name": "demo-402",
    "tag": "parallel",
    "velocity_order": 2,
    "force_order": 2,
    "pressure_order": 1,
    "num_processors": MPI.COMM_WORLD.size,
    "Um": UM_OVERRIDE,  # mean inlet velocity [cm/s] (2 m/s)
    "T": T_OVERRIDE,  # s
    "dt": DT_OVERRIDE,
    "rho": RHO_OVERRIDE,  # 1 g/cm^3
    "Lx": LX_OVERRIDE,  # Turek channel length [cm]
    "Ly": LY_OVERRIDE,  # Turek channel height [cm]
    "Nx": NX_OVERRIDE,
    "Ny": NY_OVERRIDE,
    "mu": MU_OVERRIDE,  # 1 [Pa*s] , 10 [dyne/cm^2*s]
    "mu_s": MU_S_OVERRIDE,  # Solid shear modulus (2nd Lame Coef.) [dyne/cm^2]
    "lambda_s": LAMBDA_S_OVERRIDE,  # Solid 1st Lame Coef. [dyne/cm^2]
    "nu_s": 0.4,  # Solid Poisson ratio [-]
    "beta": 1e6,  # Penalty for head/tail fixation, 1e4 [dyne/cm^2]
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
