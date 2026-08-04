"""demo_336 方腔驱动圆盘 — multi-direct forcing（迭代直接力法）配置。

求解器结构与 afsic/demo/demo_421/main.py 完全一致（DFIBMFoam multi-direct
forcing 的 FEniCSx 移植），把"游动鱼体"换回 demo_336 原始算例的**圆盘**：

  - 流体域：1×1 方腔，顶盖以 U=(1,0) 匀速滑动（lid-driven cavity），其余三壁无滑移；
    压力在角落固定一个 DOF（闭合腔消去零模态）
  - 圆盘：圆心 (0.6, 0.5)、半径 r=0.2，与 demo_336 原始算例
    （~/afsi-data/336-lid-driven-disk/mesh/circle_20）一致
  - 圆盘运动：默认 "free" —— 圆盘作为**刚性体随方腔流场被驱动**（平动 + 旋转），
    每步由标记处的流体速度确定刚体速度 (V_c, ω)，与 demo_336"圆盘随流运动"语义一致；
    也可设 "fixed"（U^d=0，固定圆盘，demo_339 圆柱风格）

注意：afsic IBM 内核要求流体域从原点 (0,0) 出发（base_node 用 X/dh），
故 x0=y0=0（方腔正好满足）。
"""

import os
from mpi4py import MPI

_demo_dir = os.path.dirname(os.path.abspath(__file__))

config = {
    "project_name": "demo-336",
    "tag": "multi-direct-forcing",
    "velocity_order": 2,
    "pressure_order": 1,
    "num_processors": MPI.COMM_WORLD.size,

    # --- 流体：1×1 方腔驱动 (lid-driven cavity) ---
    "x0": 0.0,             # 域原点 x [m]（必须 0，afsic IBM 内核要求）
    "y0": 0.0,             # 域原点 y [m]（必须 0）
    "Lx": 1.0,             # 方腔边长 x [m]
    "Ly": 1.0,             # 方腔边长 y [m]
    "Nx": 128,             # 流体网格 x（NX 可覆盖）
    "Ny": 128,             # 流体网格 y（NY 可覆盖）
    "U_lid": 1.0,          # 顶盖滑动速度 [m/s]（沿 +x，恒定）
    "rho": 1.0,            # 密度 [kg/m^3]（与 demo_336 原始一致）
    "mu": 0.01,            # 动力粘度 [Pa·s] → Re = ρ·U_lid·L/μ = 100
    "T": 10.0,             # 模拟时长 [s]（与原始 demo_336 一致；STEPS 可覆盖）
    "dt": 0.0025,          # 时间步长 [s]（128×128 下 CFL≈0.32）

    # --- 圆盘（与 demo_336 原始算例一致） ---
    "cx": 0.6,             # 初始圆心 x
    "cy": 0.5,             # 初始圆心 y
    "r": 0.2,              # 半径 [m]
    "D": 0.4,              # 直径 [m]

    # --- 圆盘运动模式 ---
    "disk_motion": "free",   # "free"=刚体随流驱动（平动+旋转，默认） | "fixed"=固定 U^d=0

    # --- IBM (multi-direct forcing) ---
    "marker_mode": "disk",     # "disk"=填充圆盘（含内部标记，保证实体） | "boundary"=仅边界环
    "marker_ds_h": 0.5,        # 边界环标记间距 = 0.5*h（Δs≈h/2，保证插值分辨率）
    "marker_spacing_h": 0.5,   # 内部标记间距 = 0.5*h（disk 模式）
    "n_iter": 10,              # 每步直接力迭代次数（DFIBMFoam 默认 10）
    "mask_interior": True,     # 仅 "fixed" 模式有效：每步把圆盘内部速度硬置零（保证实体）
    "write_solid": True,       # 输出固体（圆盘）网格 disk.xdmf（几何 + 刚体速度场）
    "solid_rings": 4,          # 固体可视化圆盘网格的环数
    "solid_circ": 48,          # 固体可视化圆盘网格的周向分段数
    "out_interval": 40,        # 每 N 步输出一次 xdmf
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
if os.environ.get("DISK_MOTION"):
    config["disk_motion"] = os.environ["DISK_MOTION"]
if os.environ.get("MARKER_MODE"):
    config["marker_mode"] = os.environ["MARKER_MODE"]

# 输出到本 demo 目录的 output/ 下（velocity/pressure 的 xdmf+h5 + forces.csv）
config["output_path"] = os.path.join(_demo_dir, "output") + os.sep
os.makedirs(config["output_path"], exist_ok=True)
config["experiment_name"] = "lid-driven-cavity-disk-mdf"
