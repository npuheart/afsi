"""DFIBMFoam CircularFishSwimming 鱼体几何与运动学（FEniCSx 移植）。

忠实移植原版 OpenFOAM 代码（/tmp/DFIBMFoam/CircularFishSwimming/code/IBM.C）：
  - `updateIbpCoordinate(time)`：NACA 4 位厚度鱼身 + 行波摆动 + 圆周游动
  - `updateIbpVelocity(dt)`：期望速度 U^d = (X(t) - X(t-dt)) / dt

鱼体模型（L = 鱼长，λ = 波长，T_w = 摆动周期）：
  摆动中线（行波）:
    h(x, t) = L·(0.351·sin(x/L - 1.796) + 0.359) · sin(2π(x/λ - t/T_w))
  半厚度（NACA 4 位厚度分布）:
    d(x) = 0.125·L/0.2 · (0.2969√(x/L) - 0.1260(x/L) - 0.3516(x/L)²
                          + 0.2843(x/L)³ - 0.1015(x/L)⁴)
  上表面 y = h + d，下表面 y = h - d。

圆周游动：鱼整体绕原点以半径 rad 圆周运动，鱼体纵轴切于圆周轨迹：
  angle(t) = 2π·t/cycleT + 2π·i/n_fish
  rotation = angle - π/2

返回的标记点按 DFIBMFoam 顺序交错排列：idx=2i 为上表面第 i 截面、
idx=2i+1 为下表面第 i 截面（每点取该微段的中心）。
"""

import numpy as np


def naca_half_thickness(x, L):
    """NACA 4 位厚度分布的半厚度（t/c=0.125/0.2 缩放，与 DFIBMFoam 一致）。

    Parameters
    ----------
    x : array_like — 距鱼头距离 [0, L]
    L : float — 弦长
    """
    xi = np.asarray(x, dtype=float) / L
    return (0.125 * L / 0.2
            * (0.2969 * np.sqrt(xi)
               - 0.1260 * xi
               - 0.3516 * xi ** 2
               + 0.2843 * xi ** 3
               - 0.1015 * xi ** 4))


def fish_surface(cfg, t):
    """返回时刻 t 的鱼体表面 IB 标记点。

    Parameters
    ----------
    cfg : dict — configuration.py 的 config
    t : float — 时刻 [s]

    Returns
    -------
    mx, my : (2·n_sections,) float array — 标记点坐标（交错：上/下表面）
    nx, ny : (2·n_sections,) float array — 外法线
    ds     : (2·n_sections,) float array — 每标记对应微段弧长 Δs
    """
    L = cfg["fish_length"]
    nS = cfg["n_sections"]
    lam = cfg["wavelength"]
    waveT = cfg["wave_period"]
    rad = cfg["orbit_radius"]
    cycleT = cfg["cycle_period"]
    idx = cfg["fish_index"]
    objNum = cfg["n_fish"]

    dL = L / nS
    i = np.arange(nS)
    x1 = i * dL
    x2 = (i + 1) * dL
    xi1 = x1 / L
    xi2 = x2 / L

    # 行波摆动中线
    h1 = L * (0.351 * np.sin(xi1 - 1.796) + 0.359) * np.sin(2 * np.pi * (x1 / lam - t / waveT))
    h2 = L * (0.351 * np.sin(xi2 - 1.796) + 0.359) * np.sin(2 * np.pi * (x2 / lam - t / waveT))

    # 半厚度
    d1 = naca_half_thickness(xi1 * L, L)
    d2 = naca_half_thickness(xi2 * L, L)

    # 圆周游动：圆心位置 + 鱼体朝向（纵轴切于轨迹）
    # orbit_center：绕圈中心（本 demo 取域中心 (0.7,0.7)，因 IBM 内核要求域从原点出发）
    ocx, ocy = cfg.get("orbit_center", [0.0, 0.0])
    init_angle = idx * 2.0 * np.pi / objNum
    angle = init_angle + 2.0 * np.pi * t / cycleT
    cx = ocx + rad * np.cos(angle)
    cy = ocy + rad * np.sin(angle)
    rot = angle - np.pi / 2.0
    cr, sr = np.cos(rot), np.sin(rot)

    # 局部坐标 → 全局（旋转 + 平移）
    def rot_tr(x, y):
        return cx + cr * x - sr * y, cy + sr * x + cr * y

    u1x, u1y = rot_tr(x1, h1 + d1)   # 上表面段首
    u2x, u2y = rot_tr(x2, h2 + d2)   # 上表面段末
    l1x, l1y = rot_tr(x1, h1 - d1)   # 下表面段首
    l2x, l2y = rot_tr(x2, h2 - d2)   # 下表面段末

    # 标记 = 微段中心，按 DFIBMFoam 顺序交错（2i=上, 2i+1=下）
    n = 2 * nS
    mx = np.empty(n)
    my = np.empty(n)
    mx[0::2] = 0.5 * (u1x + u2x)
    my[0::2] = 0.5 * (u1y + u2y)
    mx[1::2] = 0.5 * (l1x + l2x)
    my[1::2] = 0.5 * (l1y + l2y)

    # 微段弧长
    dsu = np.hypot(u2x - u1x, u2y - u1y)
    dsl = np.hypot(l2x - l1x, l2y - l1y)
    ds = np.empty(n)
    ds[0::2] = dsu
    ds[1::2] = dsl

    # 外法线（上: (-ty, tx)，下: (ty, -tx)）
    nx = np.empty(n)
    ny = np.empty(n)
    nx[0::2] = -(u2y - u1y) / dsu
    ny[0::2] = (u2x - u1x) / dsu
    nx[1::2] = (l2y - l1y) / dsl
    ny[1::2] = -(l2x - l1x) / dsl

    return mx, my, nx, ny, ds


def fish_desired_velocity(cfg, t, dt):
    """期望速度 U^d = (X(t) - X(t-dt)) / dt（DFIBMFoam updateIbpVelocity）。

    返回交错数组（2·n_sections 个 2 分量 = 4·n_sections 元素）。
    """
    mx, my, _, _, _ = fish_surface(cfg, t)
    mxl, myl, _, _, _ = fish_surface(cfg, t - dt)
    Ud = np.empty(2 * mx.size)
    Ud[0::2] = (mx - mxl) / dt
    Ud[1::2] = (my - myl) / dt
    return Ud
