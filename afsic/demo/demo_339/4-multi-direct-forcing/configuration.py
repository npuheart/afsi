"""Multi-direct forcing（迭代直接力法）圆柱绕流 — 配置。

DFIBMFoam (Mi et al. 2025, https://github.com/MsureCFD/DFIBMFoam) 算法的
FEniCSx/AFSI 移植：
  - AB2 对流 + 3/2-1/2 半隐式扩散的分步投影
  - Peskin 四点 δ 核插值/扩散（复用 afsic 的 IBMesh/IBInterpolation）
  - multi-direct forcing：每步 10 次迭代累加体积力，插值用 tU=U*-1.5dt·∇p^n+IBMf·dt
  - 压力泊松 ∇²p = (2/(3dt))∇·U；速度修正 U -= 1.5dt·∇p

几何/物理参数与 demo_339 其余实现统一（SI, DFG 2D-3, Re=100, 固定圆柱）。
输出写到本目录下的 output/ 文件夹（h5+xdmf），便于查找。
"""

import os
from mpi4py import MPI

_demo_dir = os.path.dirname(os.path.abspath(__file__))

config = {
    "project_name": "demo-339",
    "tag": "multi-direct-forcing",
    "velocity_order": 2,
    "pressure_order": 1,
    "num_processors": MPI.COMM_WORLD.size,

    # --- 流体 (SI) — 与其余实现一致 ---
    "Um": 1.0,          # 平均入口速度 [m/s]
    "T": 10.0,          # 模拟时长 [s]
    "dt": 0.001,        # 时间步长 [s]
    "rho": 1000.0,      # 密度 [kg/m^3]
    "Lx": 2.2,          # 通道长度 [m]
    "Ly": 0.41,         # 通道高度 [m]
    "Nx": 220,          # 流体网格 x
    "Ny": 41,           # 流体网格 y
    "mu": 1.0,          # 动力粘度 [Pa·s] (Re=100)

    # --- 圆柱 (固定) ---
    "cylinder_cx": 0.2,
    "cylinder_cy": 0.2,
    "cylinder_r": 0.05,
    "D": 0.1,

    # --- IBM (multi-direct forcing) ---
    "marker_mode": "boundary",  # "boundary"=仅边界环(DFIBMFoam 原版) | "disk"=填充圆盘
    "marker_spacing_h": 0.5,    # 内部标记间距 = 0.5*h（disk 模式）
    "n_markers": 128,           # 边界环标记点数（boundary 模式，Δs≈2.45mm < h）
    "n_iter": 10,               # 每步直接力迭代次数（DFIBMFoam 默认 10）
    "mask_interior": True,      # 每步把圆柱内部速度硬置零（保证实体固体；修复卡门涡街）
}

config["num_steps"] = int(config["T"] / config["dt"])

# 环境变量覆盖步数（用于短程验证）
if os.environ.get("STEPS"):
    config["num_steps"] = int(os.environ["STEPS"])
    config["T"] = config["num_steps"] * config["dt"]

# 输出到本 demo 目录的 output/ 下（velocity/pressure 的 xdmf+h5）
config["output_path"] = os.path.join(_demo_dir, "output") + os.sep
os.makedirs(config["output_path"], exist_ok=True)
config["experiment_name"] = "multi-direct-forcing-demo"
