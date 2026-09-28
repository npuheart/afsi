#!/usr/bin/env python
"""demo_336 四组合对照运行器：SOLVER={chorin,ipcs} × IB 载荷={旧, 新(IB_DIRECT_LOAD)}。

补丁：屏蔽 swanlab/网络；输出目录由 OUT 环境变量指定（默认 /tmp/demo336_matrix/）。
T 由环境变量覆盖（默认用 fsi_paralell.py 的 10.0）。

用法（在 demo_336 下）：
    SOLVER=ipcs IB_DIRECT_LOAD=1 OUT=/tmp/demo336/ipcs_new T=10 \
        caffeinate -i mpirun -n 4 python -B -u run_matrix.py
"""
import os

from mpi4py import MPI

import afsic

afsic.get_project_name = lambda *a, **k: "demo336-matrix"
afsic.swanlab_init = lambda *a, **k: None
afsic.swanlab_upload = lambda *a, **k: None

_out = os.path.join(os.environ.get("OUT", "/tmp/demo336_matrix"))
if MPI.COMM_WORLD.rank == 0:
    os.makedirs(_out, exist_ok=True)
MPI.COMM_WORLD.barrier()
afsic.unique_filename = lambda *a, **k: _out + "/"

base = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(base, "fsi_paralell.py"), "r", encoding="utf-8") as f:
    src = f.read()

T = os.environ.get("T")
if T:
    src = src.replace('"T": 10.0', f'"T": {float(T)}')

ns = {"__name__": "__main__"}
exec(compile(src, "fsi_paralell.py", "exec"), ns)

if MPI.COMM_WORLD.rank == 0:
    print(f"[run_matrix] done: SOLVER={os.environ.get('SOLVER', 'chorin')} "
          f"IB_DIRECT_LOAD={os.environ.get('IB_DIRECT_LOAD', '0')} -> {_out}/")
