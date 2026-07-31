"""Direct Forcing 圆柱绕流 — 配置。

经典的 DFG 2D-3 基准：Re=100 的圆柱绕流。
Direct forcing: 直接在流体网格上标记固体区域，强制速度为零。

与 demo_339 其余三个实现（no_cylinder / ibfe / body_fitted）统一为 SI 单位，
保证四者在相同物理参数下可对比。
"""

from mpi4py import MPI
from afsic import unique_filename, get_project_name

config = {
    "project_name": "demo-339",
    "tag": "direct-forcing",
    "velocity_order": 2,
    "pressure_order": 1,
    "num_processors": MPI.COMM_WORLD.size,

    # --- 流体 (SI) — 与其余实现一致 ---
    "Um": 1.0,         # 平均入口速度 [m/s]
    "T": 10.0,         # 模拟时长 [s]
    "dt": 0.001,       # 时间步长 [s]
    "rho": 1000.0,     # 密度 [kg/m^3]
    "Lx": 2.2,         # 通道长度 [m]
    "Ly": 0.41,        # 通道高度 [m]
    "Nx": 220,         # 流体网格 x (加密)
    "Ny": 41,          # 流体网格 y (加密)
    "mu": 1.0,         # 动力粘度 [Pa·s] (Re=100)

    # --- 圆柱 (刚性) ---
    "cylinder_cx": 0.2,       # 圆心 x [m]
    "cylinder_cy": 0.2,       # 圆心 y [m]
    "cylinder_r": 0.05,       # 半径 [m]

    # --- 曳力/升力 (SI) ---
    "D": 0.1,          # 圆柱直径 [m]
    # Re = rho*Um*D/mu = 1000*1*0.1/1 = 100
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
