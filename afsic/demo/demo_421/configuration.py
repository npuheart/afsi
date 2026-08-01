"""demo_421 鱼游动（DFIBMFoam CircularFishSwimming 移植）— 配置。

算法（与 demo_339/multi_direct_forcing 相同，DFIBMFoam multi-direct forcing）：
  - AB2 对流 + 3/2-1/2 半隐式扩散的分步投影
  - Peskin 四点 δ 核插值/扩散（复用 afsic 的 IBMesh/IBInterpolation）
  - multi-direct forcing：每步 n_iter 次迭代累加体积力
  - 移动鱼体：每个边界标记点的期望速度 U^d = dX/dt（由鱼体摆动运动学给出），
    而非固定圆柱的 U^d=0

鱼体几何/运动学忠实移植 DFIBMFoam 的 CircularFishSwimming 案例
（/tmp/DFIBMFoam/CircularFishSwimming/code/IBM.C）：
  - NACA 4 位厚度分布鱼身 + 行波摆动 h(x,t)
  - 鱼整体绕圆心作圆周游动（半径 rad、周期 cycleT），鱼体纵轴切于圆周轨迹
输出写到本目录下的 output/（velocity/pressure xdmf+h5 + 鱼体标记轨迹 csv）。
"""

import os
from mpi4py import MPI

_demo_dir = os.path.dirname(os.path.abspath(__file__))

config = {
    "project_name": "demo-421",
    "tag": "fish-swimming-ibm",
    "velocity_order": 2,
    "pressure_order": 1,
    "num_processors": MPI.COMM_WORLD.size,

    # --- 流体：闭合水槽（圆周游动，回流） ---
    # 注意：afsic IBM 内核的 base_node 用 X/dh 且假设网格从 (0,0) 出发
    #（build_map/get_index 才用 (x-x0)/dx），故域必须从原点出发，否则 IBM
    # 施力/插值会整体偏移 x0 个格点（见 readme「已知局限」）。故 x0=y0=0，
    # 鱼绕圈中心放在 (0.7,0.7)（=域中心）。
    "x0": 0.0,           # 域原点 x [m]（必须 0，见上）
    "y0": 0.0,           # 域原点 y [m]（必须 0）
    "Lx": 1.4,           # 域宽 [m]
    "Ly": 1.4,           # 域高 [m]
    "Nx": 280,           # 流体网格 x（h≈0.005，鱼身厚 0.0125m ≈ 2.5 格）
    "Ny": 280,           # 流体网格 y
    "rho": 1000.0,       # 密度 [kg/m^3]
    "mu": 0.01,          # 动力粘度 [Pa·s]（Re≈10^3 量级，见 readme）
    "T": 1.0,            # 模拟时长 [s]（默认 2 个摆动周期；STEPS 可覆盖）
    "dt": 0.001,         # 时间步长 [s]

    # --- 鱼体（DFIBMFoam CircularFishSwimming 参数） ---
    "fish_length": 0.1,  # 鱼体长/弦长 L [m]
    "n_sections": 120,   # 鱼体截面数（Δs=L/nS=0.83mm < h；240 标记兼顾速度与分辨率）
    "wavelength": 0.1,   # 行波波长 λ [m]
    "wave_period": 0.5,  # 行波周期 T_wave [s]
    "orbit_radius": 0.3, # 圆周游动半径 rad [m]
    "cycle_period": 37.7,# 绕圈周期 cycleT [s]
    "orbit_center": [0.7, 0.7],  # 绕圈中心（域中心；原版 DFIBMFoam 为原点）
    "n_fish": 1,         # 鱼的数量
    "fish_index": 0,     # 当前鱼编号（多鱼时相位差 = 2π·i/n_fish）

    # --- IBM (multi-direct forcing) ---
    "marker_mode": "boundary",  # 仅鱼体表面标记（DFIBMFoam 原版，细长体适用）
    "n_iter": 5,                # 每步直接力迭代次数（速度/精度折衷）
    "mask_interior": False,     # 细长鱼身无需内部掩码（与原版一致）
    "out_interval": 20,         # 每 N 步输出一次 xdmf
}

config["num_steps"] = int(config["T"] / config["dt"])

# 环境变量覆盖步数/分辨率（用于短程验证与加密运行）
if os.environ.get("STEPS"):
    config["num_steps"] = int(os.environ["STEPS"])
    config["T"] = config["num_steps"] * config["dt"]
if os.environ.get("NX"):
    config["Nx"] = int(os.environ["NX"])
if os.environ.get("NY"):
    config["Ny"] = int(os.environ["NY"])

# 输出到本 demo 目录的 output/ 下
config["output_path"] = os.path.join(_demo_dir, "output") + os.sep
os.makedirs(config["output_path"], exist_ok=True)
config["experiment_name"] = "fish-swimming-demo"
