"""Direct Forcing 圆柱绕流 — 配置。

经典的 DFG 2D-3 基准：Re=100 的圆柱绕流。
Direct forcing: 直接在流体网格上标记固体区域，强制速度为零。
"""

from mpi4py import MPI
from afsic import unique_filename, get_project_name

config = {
    "project_name": "demo-339",
    "tag": "direct-forcing",
    "velocity_order": 2,
    "pressure_order": 1,
    "num_processors": MPI.COMM_WORLD.size,

    # --- 流体 (CGS) ---
    "Um": 100.0,       # 平均入口速度 [cm/s]
    "T": 2.0,          # 模拟时长 [s]
    "dt": 0.00005,     # 时间步长 [s] (CFL≈0.5: 100*5e-5/0.01=0.5)
    "rho": 1.0,        # 密度 [g/cm^3]
    "Lx": 2.2,         # 通道长度 [cm]
    "Ly": 0.41,        # 通道高度 [cm]
    "Nx": 220,         # 流体网格 x (加密)
    "Ny": 41,          # 流体网格 y (加密)
    "mu": 0.1,         # 动力粘度 [g/(cm·s)] (Re=100)

    # --- 圆柱 (刚性) ---
    "cylinder_cx": 0.2,       # 圆心 x [cm]
    "cylinder_cy": 0.2,       # 圆心 y [cm]
    "cylinder_r": 0.05,       # 半径 [cm]

    # --- 曳力/升力 (CGS: cm·g·s) ---
    "D": 0.1,          # 圆柱直径 [cm] — 注意: 此处应为几何直径
    # Re = rho*Um*D/mu = 1*100*0.1/1 = 10
    # 如需 Re=100，设置 mu=0.1 或 Um=1000
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
