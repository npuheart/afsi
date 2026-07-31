#!/usr/bin/env python
"""Short verification runner for demo_339.

在统一参数下，对四个圆柱绕流实现各运行少量步数，验证程序能否正确运行，
并采集关键量（u_L2 / p_L2 范数；direct_forcing 额外输出 Cd/Cl）用于对比。

离线安全：屏蔽 swanlab 与 get_project_name 的网络调用。
步数可通过环境变量 SHORT_STEPS 覆盖（默认 100）。
用法：
    conda activate afsi-dolfinx
    python run_short.py
    SHORT_STEPS=50 python run_short.py
"""
import os
import sys
import types
import importlib.util

import numpy as np
from mpi4py import MPI

import afsic

BASE = os.path.dirname(os.path.abspath(__file__))
SUB_DIRS = ["no_cylinder", "body_fitted", "ibfe", "direct_forcing"]
NUM_STEPS = int(os.environ.get("SHORT_STEPS", "100"))
SCRATCH = os.path.join(BASE, "_short_run")

rank = MPI.COMM_WORLD.size and MPI.COMM_WORLD.rank
summary = []  # (sub, u_L2, p_L2, extra)

# ---------------------------------------------------------------------------
# 离线补丁：把 afsic 中需要网络的函数替换为无害实现。
# 注意：main.py 里是 `from afsic import swanlab_init, ...`，会取到补丁后的引用。
# ---------------------------------------------------------------------------
afsic.get_project_name = lambda *a, **k: "short-test"
afsic.swanlab_init = lambda *a, **k: None
afsic.swanlab_upload = lambda *a, **k: None


def load_config(sub):
    """加载某个子目录的 configuration.py 并返回其 config 字典。"""
    spec = importlib.util.spec_from_file_location(
        "cfg_" + sub.replace("-", "_"),
        os.path.join(BASE, sub, "configuration.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.config


def register_config(cfg):
    """把修改后的 config 注册为可被 `from configuration import config` 找到。"""
    m = types.ModuleType("configuration")
    m.config = cfg
    sys.modules["configuration"] = m


def run_case(sub):
    path = os.path.join(BASE, sub)
    main_path = os.path.join(path, "main.py")

    cfg = load_config(sub)
    # —— 统一参数 + 短程步数 ——
    cfg["num_steps"] = NUM_STEPS
    cfg["T"] = NUM_STEPS * cfg["dt"]
    out_dir = os.path.join(SCRATCH, sub)
    os.makedirs(out_dir, exist_ok=True)
    cfg["output_path"] = out_dir + "/"
    cfg["experiment_name"] = "short-" + sub

    register_config(cfg)

    if rank == 0:
        print(f"\n{'=' * 72}\n=== {sub}  |  Nx={cfg.get('Nx')} Ny={cfg.get('Ny')} "
              f"dt={cfg['dt']} steps={NUM_STEPS}  t_end={cfg['T']:.3f}\n{'=' * 72}",
              flush=True)

    g = {"__name__": "__main__", "__file__": main_path, "__builtins__": __builtins__}
    with open(main_path) as fh:
        code = fh.read()
    exec(compile(code, main_path, "exec"), g)

    # ---- 汇总：统一计算最终 u_L2 / p_L2 范数 ----
    extra = ""
    u_L2 = p_L2 = float("nan")
    try:
        from ufl import dot, dx
        from dolfinx.fem import form, assemble_scalar
        u_ = g["ns_solver"].u_
        p_ = g["ns_solver"].p_
        mesh = g["mesh"]
        u_L2 = mesh.comm.allreduce(
            assemble_scalar(form(dot(u_, u_) * dx)), op=MPI.SUM)
        p_L2 = mesh.comm.allreduce(
            assemble_scalar(form(dot(p_, p_) * dx)), op=MPI.SUM)
        if sub == "direct_forcing":
            dh = g.get("drag_history", [])
            if dh:
                arr = np.array(dh)
                D, rho, Um = cfg["D"], cfg["rho"], cfg["Um"]
                cd = 2.0 * arr[-1, 1] / (rho * Um**2 * D)  # 无量纲化 Cd
                extra = f"  Cd(final)={cd:.4f}"
        elif sub == "ibfe":
            F = g.get("solid_force", None)
            if F is not None:
                sF = F.x.array
                extra = f"  |F_s|max={np.abs(sF).max():.3e}"
    except Exception as e:
        if rank == 0:
            print(f"[warn] post-run summary failed for {sub}: {e}", flush=True)

    if rank == 0:
        print(f"[RESULT] {sub}: u_L2={u_L2:.6f}  p_L2={p_L2:.6f}{extra}", flush=True)
    summary.append((sub, u_L2, p_L2, extra.strip()))


if __name__ == "__main__":
    for sub in SUB_DIRS:
        try:
            run_case(sub)
        except Exception as e:
            if rank == 0:
                import traceback
                print(f"\n[FAIL] {sub}: {e}", flush=True)
                traceback.print_exc()
            sys.exit(1)
    if rank == 0:
        print("\n" + "=" * 72)
        print("汇总 (统一参数, 短程):")
        print(f"{'case':<16}{'u_L2':>12}{'p_L2':>12}  extra")
        for sub, u, p, extra in summary:
            print(f"{sub:<16}{u:>12.6f}{p:>12.6f}  {extra}")
        print("=" * 72)
