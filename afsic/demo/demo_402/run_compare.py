#!/usr/bin/env python
"""demo_402（Turek FSI2：固定圆柱 + 弹性尾巴）离线对比运行器。

同一套物理/网格，只换流体求解器，用来对比 **Chorin 投影法** 与 **IPCS**
（incremental pressure correction）在浸没边界 FSI 上的表现。

用法（在仓库根的 afsi 环境里跑）：

    cd afsic/demo/demo_402
    RUN=../../../.tools/afsi-run.sh
    SOLVER=chorin T=1.0 $RUN python -B -u run_compare.py
    SOLVER=ipcs   T=1.0 $RUN python -B -u run_compare.py

环境变量
    SOLVER   chorin（默认）| ipcs        流体求解器
    T        物理时长 [s]（默认 1.0；demo 原配置是 10 s = 200000 步）
    DT       时间步 [s]（默认沿用 configuration.py 的 5e-5）
    NX/NY    流体网格（默认 220×41）
    OUT      输出目录（默认 <demo>/plot/compare_<solver>/）

本脚本做的三件事（都不改动 demo 源码语义）：
  1. 离线补丁：屏蔽 swanlab 的网络登录/上传，并把 `get_project_name`（会请求
     api.pengfeima.cn）与 `unique_filename`（时间戳输出目录）换成确定实现，
     使同一目录可重复对比；
  2. 用源码注入覆盖 `T` / `dt` / 网格密度（同 demo_336/run_ib_compare.py 的做法）；
  3. 把 `swanlab_upload` 换成本地记录器：写出 history.csv，含 u_L2、p_L2、
     固体力范数、体积、**尾尖位移**与累计耗时。

尾尖位移的定义与 Turek 基准一致：弹性尾巴末端（参考构型 x 最大的那批节点）
相对参考位置的位移 (dx, dy)，单位 cm。
"""
import os
import sys
import time

import numpy as np
from mpi4py import MPI

import afsic

HERE = os.path.dirname(os.path.abspath(__file__))

SOLVER = os.environ.get("SOLVER", "chorin").lower()
T = float(os.environ.get("T", "1.0"))
DT = os.environ.get("DT", "")
NX = os.environ.get("NX", "")
NY = os.environ.get("NY", "")
OUT = os.environ.get("OUT", os.path.join(HERE, "plot", f"compare_{SOLVER}"))
os.makedirs(OUT, exist_ok=True)
RANK = MPI.COMM_WORLD.rank

# ---------------------------------------------------------------- 离线补丁 --
afsic.get_project_name = lambda *a, **k: f"demo-402-{SOLVER}"
afsic.swanlab_init = lambda *a, **k: None
afsic.unique_filename = lambda *a, **k: OUT.rstrip("/") + "/"

# ------------------------------------------------------------ 记录器 ---------
ns = {"__name__": "__main__", "__file__": os.path.join(HERE, "main.py")}
csv_path = os.path.join(OUT, "history.csv")
csv_file = open(csv_path, "w") if RANK == 0 else None
if csv_file:
    csv_file.write("t,u_L2,p_L2,solid_force_L2,volume,inlet_scale,"
                   "tip_dx,tip_dy,cyl_dx,cyl_dy,max_u,umax_x,umax_y,elapsed_s\n")

_t0 = time.time()
_ref_coords = None
_comm = MPI.COMM_WORLD
_rank = _comm.rank


def _sum_n(a):
    """跨 rank 求分量和与计数（a 可以是 (n,) 或 (n,2)）。"""
    a = np.asarray(a, dtype=float)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    s = a.sum(axis=0).copy()
    n = np.array([float(a.shape[0])])
    _comm.Allreduce(MPI.IN_PLACE, s, op=MPI.SUM)
    _comm.Allreduce(MPI.IN_PLACE, n, op=MPI.SUM)
    return s, int(n[0])


def _global_mean(sel_mask, disp):
    """掩码选出的局部 dof 位移的全局分量平均（MPI 安全）。"""
    if sel_mask.sum():
        return _sum_n(disp[sel_mask])
    return np.zeros(disp.shape[1]), 0


def _tip(coords_ref, cur):
    """尾尖（参考构型 x 最大处）相对参考位置的位移；MPI 下按全局 x 最大处取。"""
    xmax = np.array([coords_ref[:, 0].max() if len(coords_ref) else -np.inf])
    _comm.Allreduce(MPI.IN_PLACE, xmax, op=MPI.MAX)
    sel = coords_ref[:, 0] >= xmax[0] - 1e-9
    s, n = _global_mean(sel, cur - coords_ref)
    d = s / n if n else np.array([np.nan, np.nan])
    return float(d[0]), float(d[1])


def _cyl(coords_ref, cur):
    """圆柱区（参考构型 x <= 圆心+R）的平均刚体漂移，用来检查惩罚约束是否守住。"""
    sel = coords_ref[:, 0] <= 25.0 + 1e-9      # cx + R = 20 + 5 [cm]
    s, n = _global_mean(sel, cur - coords_ref)
    d = s / n if n else np.array([np.nan, np.nan])
    return float(d[0]), float(d[1])


def _upload(current_time, data_log, **params):
    """替代 swanlab.log：本地写 CSV（所有归约在每个 rank 上都要做，避免死锁）。"""
    global _ref_coords
    sc, Vs = ns.get("solid_coords"), ns.get("Vs")
    tip_dx = tip_dy = cyl_dx = cyl_dy = float("nan")
    if sc is not None and Vs is not None:
        if _ref_coords is None:
            _ref_coords = Vs.tabulate_dof_coordinates()[:, :2].copy()
        cur = sc.x.array.reshape(-1, 2)
        tip_dx, tip_dy = _tip(_ref_coords, cur)
        cyl_dx, cyl_dy = _cyl(_ref_coords, cur)
    u_arr = ns["ns_solver"].u_.x.array.reshape(-1, 2)
    mag = np.linalg.norm(u_arr, axis=1)
    i_max = int(np.argmax(mag)) if len(mag) else 0
    coords_u = ns["V"].tabulate_dof_coordinates()[:, :2]
    gmax = np.array([mag[i_max] if len(mag) else -np.inf])
    _comm.Allreduce(MPI.IN_PLACE, gmax, op=MPI.MAX)
    # 谁持有全局最大点，就由谁报告它的坐标
    owner = np.array([0], dtype=np.int32)
    if len(mag) and abs(mag[i_max] - gmax[0]) < 1e-12:
        owner[0] = _rank
    _comm.Allreduce(MPI.IN_PLACE, owner, op=MPI.MAX)
    umax_xy = np.zeros(2)
    if owner[0] == _rank and len(mag):
        umax_xy = coords_u[i_max]
    _comm.Bcast(umax_xy, root=int(owner[0]))
    if _rank != 0:
        return
    row = [f"{current_time:.8e}",
           f"{data_log.get('u_norm', float('nan')):.10e}",
           f"{data_log.get('p_norm', float('nan')):.10e}",
           f"{data_log.get('solid_force_norm', float('nan')):.10e}",
           f"{data_log.get('volume', float('nan')):.10e}",
           f"{data_log.get('inlet_velocity', float('nan')):.10e}",
           f"{tip_dx:.10e}", f"{tip_dy:.10e}",
           f"{cyl_dx:.10e}", f"{cyl_dy:.10e}",
           f"{gmax[0]:.10e}", f"{umax_xy[0]:.4f}",
           f"{umax_xy[1]:.4f}", f"{time.time() - _t0:.3f}"]
    csv_file.write(",".join(row) + "\n")
    csv_file.flush()


afsic.swanlab_upload = _upload

# ------------------------------------------------------------ 运行 main.py ---
# T / DT / NX / NY / UM 由 configuration.py 直接读环境变量，这里无需注入。
with open(os.path.join(HERE, "main.py"), "r", encoding="utf-8") as fh:
    src = fh.read()

# main.py 原本只在 rank 0 调用 swanlab_upload；我们的记录器内部有集合通信，
# 必须让所有 rank 都调用（打印仍只在 rank 0）。这里做纯文本替换，不改磁盘上的 main.py。
_old = '''        if MPI.COMM_WORLD.rank == 0:
            data_log["u_norm"] = u_L2
            data_log["p_norm"] = p_L2
            data_log["solid_force_norm"] = F_L2
            data_log["volume"] = volume
            data_log["inlet_velocity"] = inlet_velocity.scale
            print(f"Step {step+1}/{config['num_steps']}, Time: {current_time:.2f}s")
            swanlab_upload(current_time, data_log)'''
_new = '''        data_log["u_norm"] = u_L2
        data_log["p_norm"] = p_L2
        data_log["solid_force_norm"] = F_L2
        data_log["volume"] = volume
        data_log["inlet_velocity"] = inlet_velocity.scale
        swanlab_upload(current_time, data_log)   # 所有 rank 参与（内部有集合通信）
        if MPI.COMM_WORLD.rank == 0:
            print(f"Step {step+1}/{config['num_steps']}, Time: {current_time:.2f}s")'''
if _old in src:
    src = src.replace(_old, _new)
elif MPI.COMM_WORLD.size > 1:
    raise RuntimeError("main.py 的 upload 代码块不匹配，MPI 下会死锁；请检查 main.py")

# 输出频率：TimeManager(..., fps=100) -> 每个 (fps*T) 一次；诊断时可用 FPS 加密
FPS = os.environ.get("FPS")
if FPS:
    assert "fps=100" in src, "找不到 TimeManager 的 fps 参数"
    src = src.replace("fps=100", f"fps={int(FPS)}")

if RANK == 0:
    print(f"[run_compare] solver={SOLVER} T={T} "
          f"dt={DT or 'from configuration.py'} out={OUT}")

exec(compile(src, os.path.join(HERE, "main.py"), "exec"), ns)

if csv_file:
    csv_file.close()
    print(f"[run_compare] history -> {csv_path}")
