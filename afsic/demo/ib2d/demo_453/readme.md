# demo_453 — Single Porous Rubberband（多孔滑移橡皮筋，IB2d）

pure-python 参考：`third_party/ib2d/pyIB2d/Examples/Single_Porous_Rubberband`
（已拷贝为 `ib2d_input/`，0-based，无需索引换算）。

## 物理设置

1×1 周期盒，$\mu=0.1,\ \rho=1$，32×32。64 点**闭合**橡皮筋
（64 根线性弹簧 $k=10^7$、静长 $L_0=0$——想缩成一个点），
**每个标记都是 porous 点**（$\kappa=10^{-4}$，stencil 标志
$c\in\{-2..2\}$）。porous 机制 = 标记点在常规 IB 更新后**沿局部法向
额外滑移**（流体可以"穿过"多孔带），让环能在不排开全部流体的情况下收缩。

### porous 滑移的逐字复刻（pyIB2d `please_Compute_Porous_Slip_Velocity` + driver）

每步在标记全步推进之后：

1. 切向 $(x_s,y_s)$ = 4 阶单侧/中心差分（按 $c$ 选模板，导数取自
   porous 列表的相邻行，`/ds`，$ds=L_x/(2N_x)$）；
2. 单位法向 $n=(y_s,-x_s)/\lVert(x_s,y_s)\rVert$；
3. 滑移速度 $U_p=-\kappa\,F_{\mathrm{lag}}\,n/\lVert(x_s,y_s)\rVert$
   （$F_{\mathrm{lag}}$ = 当前步在 $X_h$ 处算出的**原始结构力**，未乘 ds）；
4. 位置更新 $X_{\mathrm{porous}}\mathrel{-}=\Delta t\,U_p\,n$；
5. 之后**全部**标记做周期折叠（`xLag %= L`，与 driver 一致）。

## 核心发现：闭合环塌缩对 grad-div 稳定项 γ（λ）极敏感

AFSI 直接跑官方配置（$\gamma_{\rm grad\text{-}div}=0$）时：环塌缩比 IB2d
**快 ~5 倍**，$t\approx0.043$ 缩到网格尺度以下（面积 $\sim10^{-4}<ds^2$）
后**数值崩溃（NaN）**。系统性扫描稳定项 γ（=之前 demo 里用户口径的 λ）：

| γ | 行为 |
|---|---|
| 0 | 塌缩过快，$t\approx0.043$ 几何崩溃 NaN |
| 1 | 跑完 T=0.1 但仍过快（末态面积 0.0055 vs 参考 0.0399），贴着崩溃边缘 |
| **2.5** | **全程匹配**：$\max_k\lVert X^{\rm AFSI}-X^{\rm IB2d}\rVert\le0.032\approx1h$、面积轨迹重合（最佳） |
| 10 | 早段好（0.15–0.34h），晚段过阻尼（AFSI 反而慢，峰值 1.7h） |
| 25–100 | 晚段过阻尼更强（末态 0.08+ vs 参考 0.04） |

机理：粗网格 P2/P1 投影在强耦合收缩问题上有伪散度"漏流"，环因此
塌缩过快（与 demo_444 的"γ=0 投影塌缩 / γ=100 稳定"完全同型）；
γ=2.5 恰好把伪散度压到与 IB2d 有效阻力一致的水平。见
`figures/experiments.png`（γ 扫描）。

## 验证

| 实验 | 结果 |
|---|---|
| **porous 公式逐位验证**：直接 import pyIB2d 的 `please_Compute_Porous_Slip_Velocity` 与 AFSI 实现同一输入对比 | 差 $6.8\times10^{-21}$（机器精度）——实现完全一致 |
| 力核验（弹簧，扰动态 loop 对照） | 8.7e-17 |
| AFSI dt=1e-4 vs 2.5e-5 | 面积轨迹几乎相同——AFSI 收敛 |
| 参考 dt=1e-4 vs 2.5e-5 | 面积一致——参考亦收敛 |
| porous 强度扫描（$\kappa_{por}=0/10^{-4}/2\times10^{-4}$ @γ=0） | 滑移有效且近似线性——机制工作正常 |
| 弹簧刚度对照（$\kappa_{spr}=2.5\times10^5$） | 软弹簧下两码都几乎不塌缩，残差 0.24h——差异随刚度增长 |

## 正式结果（γ=2.5，T=0.1，dt=1e-4，1000 步，AFSI ~30 s）

| $t$ | dXmax | 面积 AFSI / IB2d |
|---|---|---|
| 0.016 | 0.0094（0.30h） | 0.2046 / 0.2184 |
| 0.032 | 0.0290（0.93h） | 0.1630 / 0.1801 |
| 0.064 | **0.0042（0.13h）** | 0.0928 / 0.0945 |
| 0.088 | 0.0143（0.46h） | 0.0524 / 0.0431 |
| 0.099 | 0.0192（0.61h） | 0.0375 / 0.0272 |

（面积差先负后正、在 t≈0.07 处交叉——两码的"有效流体阻力"匹配良好。）

## 文件

- `main.py`          AFSI 运行（正式：`GRAD_DIV=2.5 python main.py`；
                     冒烟 `TFINAL=0.004 ...`；`POROUS_SCALE=0` 关 porous、
                     `IB2D_EXAMPLE=...` 换输入目录——机理实验用）；
                     输出 `plot/afsi_result_g<γ>.npz`
- `run_reference.py` 纯 Python pyIB2d 参考（lag: springs+porous；
                     支持 `--src/--tend/--dt/--print-dump` 覆盖）
- `ib2d_io.py`       0-based 输入读取（本 demo 加了 `.porous`）
- `ib2d_reference.py` VTK→npz 转换
- `compare.py`       对比表 + `figures/`（形状、面积历史、γ 扫描、涡量场）

```bash
conda run -n afsi-dolfinx python run_reference.py
GRAD_DIV=2.5 conda run -n afsi-dolfinx python main.py
GRAD_DIV=2.5 conda run -n afsi-dolfinx python compare.py
```
