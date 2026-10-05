# demo_478 — IB2d `Example_Concentration_Gradients/Impedance_Pumping_Tube`

IB2d 官方算例（`matIB2d/Examples/Example_Channel_Flow/Example_Concentration_Gradients/Impedance_Pumping_Tube`）。
它是 `Example_HeartTube` 家族的 `concentration` 成员：**新的泵送律**（阻抗泵，`update_Springs`）
+ 已经移植过的弹簧/梁/目标点/被动标量机制。

## 算例内容

* `5 x 5` 周期盒、`mu = 0.5`、`rho = 1`、`128 x 128`、`dt = 1e-3`；
  官方 `T = 0.5`（500 步）→ 本 demo **用完整设置**（约 30 s/边）。
  `print_dump` 官方为 5（100 帧）；本 demo 用 **10**（50 帧），只为把场输出减半，
  否则每个 `afsi_result.npz`/`ib2d_reference.npz` 会接近 100 MB。
* **结构**：412 个拉格朗日点构成一条直径 1 的弹性管（`x in [0.5, 4.5]`、`y in [2, 3]`）：
  **616 根弹簧**（`k` 在 1e4–1e6）、**408 根不变梁**（`kb = 100`、`C = 0`）、
  **4 个钉死目标点**、以及 `kDiff = 5e-5` 的被动标量（管内一个 336 节点、峰值 0.998 的方块）。
* **泵送律**（`update_Springs.m`，已逐行移植为 `ib2d_input/update_Springs.py`，
  两个代码共用同一份）：把 **16 根弹簧**（MATLAB 行 `(N-2)+60 : (N-2)+75`）的静长按

  ```
  RL_i = d - |0.9 d sin(2 pi freq t)|,   d = 1, freq = 10 Hz
  ```

  周期性捏紧——因为取绝对值，**每个周期被捏两次**。这是"阻抗泵"：管壁被快速挤压，
  体积变化驱动流体。（对比 demo_464 的 `Update_Springs_Peristaltic_Pumping` 是**行波**静长，
  几何与律都不同。）捏紧处管壁速度约 `0.9*pi*10/2 ≈ 14 m/s`，流场峰值速度可达 30 m/s 以上。

## 移植要点

1. **静长是可变的**：模板里 `sp_L = sp["L"].copy()`，每步在力计算**之前**
   （用全程位置，与驱动一致）调用 `update_rest_lengths(t)` 更新这 16 根。
2. **标量算子**与 demo_476/477 完全相同（`perform_Time_Split_Upwind` 分裂格式、
   周期 upwind 对流 + 中心扩散、用 `u^{n+1}`、共置节点网格）；该算子已在 demo_476
   被验证到 **1e-11**（浮点舍入）。
3. **无需外力模块**（`arb_ext_force = 0`）：模板自动跳过；
   `compare.py` 增加了**泵机制专用指标**（16 根被捏弹簧的当前长度、管直径 = 标记点 `y` 跨度）。
4. 官方 `dt = 1e-3` 下 `u_max dt / h ≈ 0.9`，**两个代码都在时间分辨率的边缘**
   （这不是移植问题，而是官方设置本身）。

## 结果

参考运行约 28 s、AFSI 运行约 27 s，各 50 帧（`t = 0 .. 0.491`）。

| 指标（末帧 `t = 0.491`） | 结果 |
|---|---|
| 管直径（`y` 跨度）AFSI / IB2d | 1.06753 / 1.05548（**1.14%**） |
| 16 根被捏弹簧平均长度 AFSI / IB2d | 0.50135 / 0.51499（2.7%，`max\|ΔL\| = 2.6e-2`） |
| 标记点偏差（412 点） | 2.13e-2 m = 0.55 h |
| 速度峰值 AFSI / IB2d | 20.24 / 19.39（4.3%），`max\|Δu\| = 4.87` |
| 浓度总量 AFSI / IB2d | 0.54881 / 0.54721（**0.29%**，守恒性好） |
| 浓度峰值 AFSI / IB2d | 0.79546 / 0.81804（2.8%） |
| 浓度相对 $L_2$ 误差 | 1.79e-1 |

**能验证**：泵送运动学——管直径 1.1%、被捏弹簧长度 2.7%、标量总量 0.3%，
即"捏多深、捏多快"两边一致。

**不能验证（已如实记录）**：瞬时速度场/标量分布（`max|Δu|` 达 0.25 `max|u|`、浓度
相对 $L_2$ 1.8e-1）。为此做了两个**收敛性判别**：

* **dt 减半（5e-4，100 帧）不改善**：浓度相对 $L_2$ 0.179 → 0.190、
  `max|Δu|` 0.25 → 0.34 `max|u|`、标记点 0.55 h → 0.88 h ⇒ **不是时间分辨率问题**；
* **`GRAD_DIV = 1.0` 不改善**（0.179 → 0.187、标记点 0.55 → 0.64 h）⇒ 与 demo_474/477
  不同，本例不需要、也帮不上 grad-div 稳定化。

因此差异来源与分叉动脉一族相同：**剧烈运动的高速管壁在两种离散下的耦合方式不同**
（IB2d 为 FFT 投影 + 共置罚力，AFSI 为 Taylor–Hood + 插值力密度），
在 CFL≈0.9、`max|u| ≈ 30 m/s` 的边缘设置下被放大。两边的**运动学**（结构本身）
是真被验证了的。

## 运行

```bash
conda activate afsi-dolfinx
python run_reference.py                                   # ~28 s
python main.py                                            # ~27 s
python compare.py
# 收敛性交叉验证（dt 减半），compare.py 会自动配对 *_dt5e-4
python run_reference.py --dt 5e-4 --out ib2d_reference_dt5e-4.npz --run /tmp/ib2d_478_dt2   # ~54 s
DT=5e-4 OUTPUT_PATH=plot_dt5e-4 python main.py                                               # ~50 s
python compare.py
```
