# -----------------------------------------------------------------------------
# 版权所有 (c) 2025 保留所有权利。
#
# 本文件隶属于 Poromechanics Solver 项目，主要开发者为：
#   - 马鹏飞：mapengfei@mail.nwpu.edu.cn
#   - 王璇：wangxuan2022@mail.nwpu.edu.cn
#
# 本软件仅供内部使用和学术研究之用。未经明确许可，严禁重新分发、修改或用于商业用途。
# 详细授权条款请参阅：https://www.pengfeima.cn/license-strict/
# -----------------------------------------------------------------------------

# Vessel luminal pressure - linear ramp-up
import numpy as np

mmHg = 1333.22368421

def calculate_pressure_linear(t, t_load=0.2, t_cycle=0.8,
                              diastole_pressure=8.0*mmHg, systole_pressure=110.0*mmHg):
    """Linearly ramps from 0 to systole_pressure over first t_load seconds, then holds."""
    if t < t_load:
        value = systole_pressure * t / t_load
    else:
        value = systole_pressure
    return value
