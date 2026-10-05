# demo_588 — IB2d `Example_HeartTube/Electromechanical_Pumping_w_Ca_Dynamics`（钙动力学驱动的电生理心管泵）

IB2d 官方算例（`matIB2d/Examples/Example_HeartTube/Electromechanical_Pumping_w_Ca_Dynamics`）在 AFSI 中的复现。
与 demo_587 同一个管子、同一个流体、同一个耦合方式（膜电位 → 跨管弹簧刚度），
**唯一区别是膜电位的驱动电流来自钙动力学模型**（`Calcium_Dynamics.m`）而不是预设脉冲。

## 1. 算例内容

| 项目 | 值（官方 `input2d`） |
|---|---|
| 周期盒 / 网格 | `5 x 5`，`64 x 64`，`mu = 1`，`rho = 1` |
| 时间 | `dt = 1e-4`，`T = 4`（40000 步） |
| 结构 | 156 个标记点（上下各 78），232 根弹簧（`k = 1e7`；第 155–232 行是跨管弹簧）、152 根不变梁、4 个目标点 |
| 开关 | `electro_phys = 1`，`update_springs = 0`，`FV_LT_muscle = 0` |

## 2. 移植的机制

官方驱动在 IBM 循环之前一次性求解两个模型：

1. `Calcium_Dynamics.m`（Williams et al. 2008 质量作用定律，无量纲）：

```
dCa /dt = (k4 Caf - k3 Ca)(1 - Caf) + k1 (C - Ca - Caf)          [刺激 ON]
dCa /dt = (k4 Caf - k3 Ca)(1 - Caf) + k2 Ca (C - S - Ca - Caf)   [刺激 OFF]
dCaf/dt = -(k4 Caf - k3 Ca)(1 - Caf)
```

`k1..k4 = 9, 50, 40, 19.4`，`C = S = 200`，前向 Euler，
刺激窗口 `rem(dt*i, 0.5) < 0.005`（**物理时间周期 0.5 s**，与 `T_final` 无关），
`dt = t_Final/Nsteps`（`Nsteps` 为 IBM 步数，故 `dt_Ca ≈ dt_IBM/numFHN`）。

2. 一维 FitzHugh–Nagumo（与 demo_587 同一套离散），但电流改成钙信号：

```
I(t) = 0.5 * Caf(t)      在注入窗口 j = floor(0.25N) ... floor(0.3N) 内
```

（`Iapp_from_Calcium_Dynamics`：该版本的 `IIapp` **每步重建为 0**，而不是像预设脉冲版那样保持历史。）

然后每一步、力计算之前把膜电位写进跨管弹簧刚度：

```matlab
springs_info(155:232, 3) = ( 1.6e1 v )^4      % 与 demo_587 相同
```

本 demo 里 `v` 峰值约 **1.26**，因此 `(16v)^4` 峰值约 **1.6e5**（比 demo_587 的 5.6e4 强得多），
管径从 1.0 收缩到 **0.32**，且收缩集中在 `t ≈ 0.005–0.09 s`（钙刺激在 `t ∈ [0, 0.005]` 打开）。

## 3. 移植要点与上游疑点

1. demo_587 里记录的三条上游疑点同样适用于本算例（列 3 = 刚度而非静长；
   `vStore` 的 `NT_IBM + 1` 自动扩行；`wStore`/`IIappStore` 写了但从不读）。
2. **pyIB2d 没有 `electro_phys` 实现**（`IBM_Driver.py:213` 读了 flag 后不再引用），
   所以参考运行通过其文档化的 `update_springs` 钩子注入**同一份** `FitzHugh_Nagumo_1d.py`
   给出的 `(16v)^4`。参考与 AFSI 唯一的差别是 FSI 离散本身。
3. **`T_final` 可以安全截断**：本例的刺激是**物理时间周期 0.5 s** 的钙波，
   与 `T_final` 无关（`dt_Ca = t_Final/Nsteps` 在数值上仍是 `dt_IBM/numFHN`），
   所以把 `TFINAL` 从 4 s 缩短只截断两个模型，不改变时序。
   经数值检验：`T = 0.1` 与 `T = 0.4` 前 1001 行电位最大差 `5.8e-4`（相对 1.26 为 0.05%，
   来自 `dt_Ca` 的微小差异），刺激窗口位置完全一致。
4. `Calcium_Dynamics.py` 与 1-based MATLAB 逐行转写**逐位一致**（`max|ΔCa| = max|ΔCaf| = 0`）；
   钙驱动的 FHN 与逐行转写最大差 `1.1e-16`。

## 4. 结果（`compare.py`）

参考运行 **295 s**（5000 步 @ 1000 步/秒的波动主要来自本机负载；CFL 意义上的纯计算约 10 s user），
AFSI 运行 **514 s**（1000 步 `dt = 1e-4`，`NX = 64`，`GRAD_DIV = 0`）。
两边各 50 帧（`print_dump = 20`，`t = 0 … 0.098 s`）。

| 指标 | 结果 |
|---|---|
| **电生理耦合**：78 根跨管弹簧的 `(16v)^4` | AFSI 与参考**逐位相同**（`max\|Δ\| = 0`），峰值 1.646e5 |
| 管径 | AFSI 0.3198–1.0000 / IB2d 0.3404–1.0000；逐帧 `max\|Δw\| = 2.4e-2` |
| 收缩相位 | 两边同时在 `t ≈ 0.005–0.01` 起缩、`t ≈ 0.09` 附近到极小 |
| 标记点偏差（156 点） | 中位 3.9e-2 m = **0.50 h**，最大 9.3e-2 m = 1.19 h |
| 末帧速度场 | `max\|u\|` AFSI 33.2 / IB2d 35.8，`max\|Δu\| = 26.5` |

**能验证**：① 钙动力学 + 钙驱动 FHN 求解器（与 MATLAB 逐行转写逐位/1e-16 一致）；
② 电生理耦合（写进弹簧刚度的数值完全一致）；③ 收缩深度与相位（管径 0.32/0.34，`max|Δw| = 2.4e-2`）。

**不能验证（如实记录）**：
* **瞬时速度场**：`max|Δu| = 26.5`，与峰值同量级。本例收缩极剧烈（`max|u| ≈ 190 m/s`，
  `u dt/h ≈ 0.24`，且钙刺激开启处速度峰值达 193 m/s），两种离散（IB2d 的 FFT 投影 + 共置罚力
  vs AFSI 的 Taylor–Hood + 插值力密度）在这种设置下不能逐帧对上。
* 参考侧的"电生理"是由本 demo 的钩子注入的（原因见 §3.2），因此该 demo 验证的是
  **钙/FHN 求解器 + FSI**，而不是"pyIB2d 自带的电生理实现"（后者不存在）。

## 5. 运行

```bash
conda activate afsi-dolfinx
python run_reference.py --tend 0.1 --print-dump 20      # pyIB2d 参考
TFINAL=0.1 PRINT_DUMP=20 python main.py                # AFSI
python compare.py
```

环境变量：`TFINAL`、`DT`、`NX`/`NY`、`PRINT_DUMP`、`OUTPUT_PATH`、`GRAD_DIV`（默认 0）、
`EPHYS_TFINAL`（默认跟随 `TFINAL`，见 §3.3）。

> 运行时间是在本机同时有多个 dolfinx 作业时测得的，绝对时间仅供参考（`run_reference.py`
> 的 wall time 里有大量 CPU 争用导致的等待）。
