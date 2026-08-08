#!/usr/bin/env python
"""demo_339 四个算例 —— 单核离线短程运行 + 结果对比。

对四个算例（1-no-cylinder / 2-body-fitted / 3-ibfe / 4-multi-direct-forcing）
分别运行其 main.py（单进程、MPI 世界大小=1），统一短步数，输出写到本地相对路径
_demo_339/_short_run/<case>/，最后汇总各算例的最终范数对比。

离线安全：把 afsic 的 swanlab_init / swanlab_upload / get_project_name /
unique_filename 替换为无害实现，避免联网卡死并让输出落在本地。

用法：
    STEPS=300 python _short_run/run_compare.py all          # 依次跑四个并汇总
    STEPS=300 python _short_run/run_compare.py 3-ibfe       # 只跑单个算例
"""
import os
import sys
import types
import importlib.util
import subprocess

import numpy as np
from mpi4py import MPI

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # demo_339/
SCRATCH = os.path.join(BASE, "_short_run")
SUB_DIRS = [
    "1-no-cylinder",
    "2-body-fitted",
    "3-ibfe",
    "4-multi-direct-forcing",
]
NUM_STEPS = int(os.environ.get("STEPS", "300"))


def patch_afsic_offline():
    """把 afsic 中需要网络的函数替换为无害实现；输出路径改到本地 _short_run。"""
    import afsic

    afsic.get_project_name = lambda *a, **k: "local-run"
    afsic.swanlab_init = lambda *a, **k: None
    afsic.swanlab_upload = lambda *a, **k: None

    def _local_unique_filename(project_name, tag="normal"):
        d = os.path.join(SCRATCH, tag)
        os.makedirs(d, exist_ok=True)
        return d + os.sep

    afsic.unique_filename = _local_unique_filename


def load_config(sub):
    spec = importlib.util.spec_from_file_location(
        "cfg_" + sub.replace("-", "_"),
        os.path.join(BASE, sub, "configuration.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.config


def register_config(cfg):
    m = types.ModuleType("configuration")
    m.config = cfg
    sys.modules["configuration"] = m


def run_one(sub):
    """在单进程内执行某个算例的 main.py，输出到 _short_run/<sub>/。"""
    patch_afsic_offline()
    main_path = os.path.join(BASE, sub, "main.py")

    cfg = load_config(sub)
    cfg["num_steps"] = NUM_STEPS
    cfg["T"] = NUM_STEPS * cfg["dt"]
    out_dir = os.path.join(SCRATCH, sub)
    os.makedirs(out_dir, exist_ok=True)
    cfg["output_path"] = out_dir + os.sep
    cfg["experiment_name"] = "run-" + sub
    register_config(cfg)

    rank = MPI.COMM_WORLD.rank
    if rank == 0:
        print(f"\n{'=' * 72}\n=== {sub} | dt={cfg['dt']} steps={NUM_STEPS} "
              f"t_end={cfg['T']:.3f} | out={out_dir}\n{'=' * 72}", flush=True)

    g = {"__name__": "__main__", "__file__": main_path, "__builtins__": __builtins__}
    with open(main_path) as fh:
        code = fh.read()
    exec(compile(code, main_path, "exec"), g)

    # ---- 最终范数（各 main.py 时间循环末尾的模块级变量） ----
    u_L2 = float(g.get("u_L2", np.nan))
    p_L2 = float(g.get("p_L2", np.nan))
    extra = ""
    if sub == "3-ibfe":
        F = g.get("solid_force", None)
        if F is not None:
            extra = f" |F_s|max={np.abs(F.x.array).max():.3e}"
    elif sub == "4-multi-direct-forcing":
        extra = (f" Cd={float(g.get('Cd', np.nan)):.4f} "
                 f"Cl={float(g.get('Cl', np.nan)):+.4f}")

    if rank == 0:
        print(f"[RESULT] {sub}: u_L2={u_L2:.6f} p_L2={p_L2:.6f}{extra}", flush=True)


def run_all():
    rows = []
    for sub in SUB_DIRS:
        r = subprocess.run(
            [sys.executable, os.path.abspath(__file__), sub],
            cwd=BASE,
            capture_output=True,
            text=True,
        )
        print(r.stdout, end="", flush=True)
        if r.returncode != 0:
            print(f"[{sub}] FAILED (rc={r.returncode})", file=sys.stderr, flush=True)
            print(r.stderr[-3000:], file=sys.stderr, flush=True)
            rows.append(f"[RESULT] {sub}: FAILED")
            continue
        for ln in r.stdout.splitlines():
            if ln.startswith("[RESULT]"):
                rows.append(ln)

    print("\n" + "=" * 72)
    print("对比汇总（单核, %d 步, t_end=%.3fs）" % (NUM_STEPS, NUM_STEPS * 0.001))
    print("=" * 72)
    print("\n".join(rows))
    summary_path = os.path.join(SCRATCH, "summary.txt")
    with open(summary_path, "w") as fh:
        fh.write("\n".join(rows) + "\n")
    print(f"\n汇总已存: {summary_path}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] != "all":
        run_one(sys.argv[1])
    else:
        run_all()
