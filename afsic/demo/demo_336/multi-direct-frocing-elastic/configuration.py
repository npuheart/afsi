"""demo_336 方腔驱动圆盘 — multi-direct-frocing-elastic：弹性固体版配置。

在 multi-direct-frocing（mdf 版，afsic/demo/demo_336/multi-direct-frocing）基础上
**增加固体的推进方程**：圆盘不再是刚体，而是一块**弹性固体**（可变形、有本构），
每个时间步：
  1) 流体速度插值到固体节点（fluid_to_solid）
  2) 固体推进方程：X_s += V_s·dt（运动学平流）
  3) 由变形梯度 F=∇X_s 按本构（可压缩 neo-Hookean 型）组装弹性力
  4) 弹性力扩散回流体（solid_to_fluid，替换式）→ 进入流体动量方程

流体求解器与 mdf 版一致（AB2 预测 + 压力泊松 + L2 投影）。

耦合为**分区显式（partitioned/staggered）**：每步依次推进流体、推进固体、
交换界面力一次，不构成 monolithic 联立求解 —— 流体与固体"相对解耦"。
"""

import os
from mpi4py import MPI

_demo_dir = os.path.dirname(os.path.abspath(__file__))

config = {
    "project_name": "demo-336",
    "tag": "multi-direct-forcing-elastic",
    "velocity_order": 2,
    "force_order": 2,
    "pressure_order": 1,
    "num_processors": MPI.COMM_WORLD.size,

    # --- 流体：1×1 方腔驱动（同 mdf 版） ---
    "x0": 0.0,             # 域原点（必须 0，afsic IBM 内核要求）
    "y0": 0.0,
    "Lx": 1.0,
    "Ly": 1.0,
    "Nx": 128,             # NX 可覆盖
    "Ny": 128,             # NY 可覆盖
    "U_lid": 1.0,          # 顶盖滑动速度（沿 +x）
    "rho": 1.0,            # 流体密度
    "mu": 0.01,            # 流体动力粘度 → Re=100
    "T": 1.0,              # 模拟时长 [s]（STEPS 可覆盖；默认在稳定包络内）
    "dt": 0.0025,          # 时间步长 [s]

    # --- 弹性圆盘（初始位姿与原始算例一致） ---
    "cx": 0.6,
    "cy": 0.5,
    "r": 0.2,
    "D": 0.4,
    "solid_rings": 8,      # 固体可视化/求解圆盘网格环数
    "solid_circ": 96,      # 周向分段数（共 1+8×96=769 节点，1408 三角）

    # --- 弹性本构（可压缩 neo-Hookean，direct-forcing 带惯性版） ---
    # 本版为真正的 direct-forcing：固体带惯性（rho_s），直接力 F_IBM=(V_s-U_l)/dt
    # 强制流体在标记处匹配固体速度，反作用力（added-mass 项）喂回固体动量方程，
    # 固液耦合在固体速度更新中隐式处理 → 稳定、不再像无质量平流版那样短时间翻转。
    "rho_s": 1.0,          # 固体密度（≥ ρ_f 更稳且更能"挡流"；added-mass 项 = ρ_f）。
    "mu_s": 0.05,          # 剪切模量（Lame μ）
    "lambda_s": 0.5,       # 第一 Lame 参数 λ（近不可压缩会体积锁定，可调小）
    "solid_active": True,  # False = 纯方腔无固体（参照，用于对比固体对流体的影响）
    "clamp_solid": True,   # 质心钳位：防止圆盘被主涡带出域（纯平动修正，不损变形）

    "out_interval": 40,    # 每 N 步输出一次
    "write_solid": True,   # 输出固体（参考网格 + 位移场 / 力场）
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
if os.environ.get("RHO_S"):
    config["rho_s"] = float(os.environ["RHO_S"])
if os.environ.get("MU_S"):
    config["mu_s"] = float(os.environ["MU_S"])
if os.environ.get("LAMBDA_S"):
    config["lambda_s"] = float(os.environ["LAMBDA_S"])
if os.environ.get("SOLID_ACTIVE"):
    config["solid_active"] = (os.environ["SOLID_ACTIVE"].lower() in ("1", "true", "yes", "on"))

# 输出到本 demo 目录的 output/ 下
config["output_path"] = os.path.join(_demo_dir, "output") + os.sep
os.makedirs(config["output_path"], exist_ok=True)
config["experiment_name"] = "lid-driven-cavity-elastic-disk"
