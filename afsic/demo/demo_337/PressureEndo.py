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

# 心室内壁压力、主动收缩张力
# 线形增长、周期变化
import numpy as np

mmHg = 1333.22368421
def calculate_pressure(t, t_load=0.2, t_end_diastole=0.5, t_cycle=0.8,
                      diastole_pressure=8.0*mmHg, systole_pressure=110.0*mmHg, start_pressure=0.0*mmHg):
    local_t = t % t_cycle
    value = 0
    
    if local_t < t_load:
        value = diastole_pressure * local_t / t_load + start_pressure * (t_load - local_t) / t_load
    elif t_load <= local_t < t_end_diastole:
        value = diastole_pressure
    elif t_end_diastole <= local_t < (0.15 + t_end_diastole):    
        a = -(local_t - t_end_diastole)**2 / 0.004
        value = diastole_pressure + (systole_pressure-diastole_pressure) * (1.0 - np.exp(a))
    elif (0.15 + t_end_diastole) <= local_t < t_end_diastole+0.3:
        a = -(local_t - t_end_diastole - 0.3)**2 / 0.004
        value = diastole_pressure + (systole_pressure-diastole_pressure) * (1.0 - np.exp(a))
    elif t_end_diastole+0.3 <= local_t < t_cycle:
        value = diastole_pressure
    
    return value

def calculate_pressure_linear(t,t_load = 0.2,t_end_diastole = 0.5,t_cycle = 0.8,diastole_pressure = 8.0*mmHg,systole_pressure = 110.0*mmHg):
    local_t = t % t_cycle
    if t < t_cycle:
        value = systole_pressure * local_t / t_cycle  # 第一个周期的舒张期内壁压力线性增长
    else:
        value = systole_pressure
    return value

def calculate_tension(t, t_load = 0.2, t_end_diastole = 0.5, t_cycle = 0.8, max_tension = 600.0*mmHg):
    local_t = t % t_cycle
    if local_t < t_end_diastole:
        value = 0.0
    elif local_t >= t_end_diastole and local_t < (t_cycle + t_end_diastole) / 2.0:
        a = -(local_t - t_end_diastole) * (local_t - t_end_diastole) / 0.005
        value = max_tension * (1.0 - np.exp(a))
    elif local_t >= (t_cycle + t_end_diastole) / 2.0 and local_t < t_cycle:
        a = -(local_t - t_cycle) * (local_t - t_cycle) / 0.005
        value = max_tension * (1.0 - np.exp(a))
    return value


def calculate_tension_linear(t, t_load = 0.2, t_end_diastole = 0.5, t_cycle = 0.8, max_tension = 600.0*mmHg):
    local_t = t % t_cycle
    if t < t_cycle:
        value = max_tension * local_t / t_cycle  # 第一个周期的舒张期内壁压力线性增长
    else:
        value = max_tension
    return value
