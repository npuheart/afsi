#!/usr/bin/env python
"""运行标准 body_fitted/main.py 到 10s（离线安全 + 本地 output/）。

补丁 afsic 的 swanlab/get_project_name 避免网络卡死，输出改到
body_fitted/output/（velocity/pressure 的 xdmf+h5）。
"""
import os
import sys
import types
import importlib.util

import afsic

BASE = os.path.dirname(os.path.abspath(__file__))
BF_DIR = os.path.join(BASE, "body_fitted")

# ---- 离线补丁 ----
afsic.swanlab_init = lambda *a, **k: None
afsic.swanlab_upload = lambda *a, **k: None
afsic.get_project_name = lambda *a, **k: "bf-demo"

# ---- 加载 body_fitted 配置，输出改到本地 ----
spec = importlib.util.spec_from_file_location(
    "cfg_bodyfitted", os.path.join(BF_DIR, "configuration.py"))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
cfg = mod.config
cfg["output_path"] = os.path.join(BF_DIR, "output") + os.sep
os.makedirs(cfg["output_path"], exist_ok=True)
if os.environ.get("STEPS"):
    cfg["num_steps"] = int(os.environ["STEPS"])
    cfg["T"] = cfg["num_steps"] * cfg["dt"]
m = types.ModuleType("configuration")
m.config = cfg
sys.modules["configuration"] = m

# ---- 运行 body_fitted/main.py ----
main_path = os.path.join(BF_DIR, "main.py")
g = {"__name__": "__main__", "__file__": main_path, "__builtins__": __builtins__}
exec(compile(open(main_path).read(), main_path, "exec"), g)
