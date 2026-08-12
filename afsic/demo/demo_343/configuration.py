"""demo_343 配置：圆盘随流体通过二维理想瓣膜（IB-FE FSI）。

物理参数（与 main.py 分离）。输出路径本地化到 plot/，支持 STEPS 覆盖步数。
"""
import os
from mpi4py import MPI

_demo_dir = os.path.dirname(os.path.abspath(__file__))

config = {"nssolver": "chorinsolver",
          "project_name": "demo-343",
          "tag": "parallel",
          "velocity_order": 2,
          "force_order": 2,
          "pressure_order": 1,
          "num_processors": MPI.COMM_WORLD.size,
          "T": 3.0,
          "dt": 1/64000,    # 细网格（320×64）+ 大入口速度，需更小 dt 防下瓣膜根部瞬态翻转
          "rho": 1.0,        # 与 demo_340 同物理；勿用 339 的 rho=1000（配 343 大入口速度会发散）
          "Lx": 8.0,
          "Ly": 1.61,
          "Nx": 64*5,
          "Ny": 64,
          "Nl": 20,
          "mu": 0.1,         # 与 demo_340 同物理
          "mu_s": 5.6e5,     # 上瓣膜参考剪切模量（FRH 实际用 C0/C1）
          "mu_s_down_factor": 10.0,  # 下瓣膜刚度 = 10×mu_s（更硬）
          "nv_s": 0.4,
          "C0": 2e5,          # FRH 剪切模量（与 demo_340 一致）
          "C1": 1e6,          # FRH 纤维增强（与 demo_340 一致）
          "kappa": 4e5,       # FRH 体积模量（对齐 demo_340；原 4e6 偏大→FSI 力大）
          "beta": 5e7,       # 瓣膜根部固定惩罚（原 5e7）
          "deviatoric": False,
          "fps": 100,
          }

config["num_steps"] = int(config['T']/config['dt'])
# 环境变量 STEPS 覆盖步数（短程验证）
if os.environ.get("STEPS"):
    config["num_steps"] = int(os.environ["STEPS"])
    config["T"] = config["num_steps"] * config["dt"]
# 输出到本地 plot/
config["output_path"] = os.path.join(_demo_dir, "plot") + os.sep
os.makedirs(config["output_path"], exist_ok=True)
config["experiment_name"] = "demo-343"
