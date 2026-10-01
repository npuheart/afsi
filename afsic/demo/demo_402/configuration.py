import os

from mpi4py import MPI
from afsic import unique_filename, get_project_name

# ===================== 论文参数的 CGS 换算（m,kg,s -> g,cm,s）=====================
# 论文：改版 Turek–Hron FSI 基准（li2025local §4.3.2），SI 单位。
#   L      2.46 m       -> 246 cm          H   0.41 m        -> 41 cm
#   d      0.1  m       -> 10  cm          圆心 (0.2,0.2) m  -> (20,20) cm
#   梁     l=0.35 m     -> 35  cm          h = 0.02 m       -> 2 cm
#   监测点 A (0.6,0.2) m -> (60,20) cm
#   U      2 m/s        -> 200 cm/s
#   ρ      1000 kg/m³   -> 1 g/cm³         μ 1 Pa·s         -> 10 dyne·s/cm²
#   μ_s    1e6 Pa       -> 1e7 dyne/cm²    λ_s 8e6 Pa       -> 8e7 dyne/cm²
#         （1 Pa = 10 dyne/cm²；1 Pa·s = 10 dyne·s/cm²）
#   Re = ρ U d/μ = 1*200*10/10 = 200  ✓
#   网格 N=128（最长边）-> Δx = L/N = 246/128 = 1.9219 cm；近似方格取 Ny=21
#         （dy = 41/21 = 1.9524 cm，与 Δx 差 1.6%）
#   Δt 论文 = 0.00164Δx = 3.152e-5 s；本 demo 取 5e-5 s（已确认允许不一致）
#   κ_s 论文 = 5.0e4·Δx/Δt²，即无量纲组 κ̂ = κ_sΔt²/(ρΔx) = 5e4 -> CGS:
#       κ_s = κ̂·ρ·Δx/Δt² [dyne/cm⁴]（见 main.py 的 PENALTY_MODE=paper；
#       显式耦合稳定预算内实跑取 κ̂=1.0）
# =============================================================================
T_OVERRIDE = float(os.environ.get("T", "10.0"))            # 物理时长 [s]
DT_OVERRIDE = float(os.environ.get("DT", str(5e-5)))       # 时间步 [s]（论文 3.152e-5）
NX_OVERRIDE = int(os.environ.get("NX", "128"))
NY_OVERRIDE = int(os.environ.get("NY", "21"))
UM_OVERRIDE = float(os.environ.get("UM", "200.0"))          # 入口平均速度 [cm/s] (2 m/s)
LX_OVERRIDE = float(os.environ.get("LX", "246.0"))          # 通道长 [cm] (2.46 m)
LY_OVERRIDE = float(os.environ.get("LY", "41.0"))           # 通道高 [cm] (0.41 m)
MU_S_OVERRIDE = float(os.environ.get("MU_S", "1.0e7"))      # 固体剪切模量 [dyne/cm²] (1e6 Pa)
LAMBDA_S_OVERRIDE = float(os.environ.get("LAMBDA_S", "8.0e7"))  # 第一 Lamé 系数 (8e6 Pa)
RHO_OVERRIDE = float(os.environ.get("RHO", "1.0"))          # 流体密度 [g/cm³] (1000 kg/m³)
MU_OVERRIDE = float(os.environ.get("MU", "10.0"))           # 流体粘度 [dyne·s/cm²] (1 Pa·s)

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
    "nu_s": 0.444,  # Solid Poisson ratio [-]，由论文 μ_s/λ_s 推得 (8/(2*9)=0.444)
    "beta": 1e6,  # 旧固定罚（dyne/cm^3）；PENALTY_MODE=paper 时不使用
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
