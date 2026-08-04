#!/usr/bin/env python
"""对比：原始 IB 方法(fsi_paralell.py) 在 mu_s=0.2 下的圆盘行为（离线）。

目的：回答"IB 方法能算, direct-forcing 不能算吗?"——用与 DF 相同的 mu_s=0.2、
rho 不变(原始无 rho_s, 固体为运动学无质量)跑原始 IB, 看它的圆盘是否也被主涡
带到壁面/出域。若是, 则触壁是"轻软体沿流线公转、轨道贴着壁面"的共性问题,
与求解方法无关。

实现：读取 fsi_paralell.py 源码, 注入配置覆盖(mu_s/T/输出路径)后 exec,
并屏蔽 swanlab 网络调用。T 用环境变量 T 覆盖(默认 5.0), 输出到 /tmp/ib_compare/。

用法：
    conda activate afsi-dolfinx
    cd afsic/demo/demo_336
    python run_ib_compare.py
    T=5 python run_ib_compare.py
"""
import os

import numpy as np
from mpi4py import MPI

import afsic

# 离线补丁：屏蔽 swanlab / get_project_name / unique_filename（网络 + 固定输出路径）
afsic.get_project_name = lambda *a, **k: "ib-compare"
afsic.swanlab_init = lambda *a, **k: None
afsic.swanlab_upload = lambda *a, **k: None
afsic.unique_filename = lambda *a, **k: "/tmp/ib_compare/"

rank = MPI.COMM_WORLD.rank
T = float(os.environ.get("T", "5.0"))

base = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(base, "fsi_paralell.py"), "r", encoding="utf-8") as f:
    src = f.read()

# 注入配置覆盖（在源码的 config 字典里替换值）
src = src.replace('"T": 10.0', f'"T": {T:.1f}')
src = src.replace('"mu_s": 0.1', '"mu_s": 0.2')      # 用户指定 mu_s=0.2
# dolfinx 0.10 兼容：create_vector(form) 已废弃，需函数空间（仓库已知坑）
src = src.replace("b1 = create_vector(L_hat)", "b1 = create_vector(Vs)")

ns = {"__name__": "__main__"}
exec(compile(src, "fsi_paralell.py", "exec"), ns)

if rank == 0:
    print("原始 IB(fsi_paralell) 运行结束，输出见 /tmp/ib_compare/")
