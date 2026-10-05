# demo_574 — IB2d `Example_Mass_Points/Gravity_vs_Flow_256x256`（质量点 / 细网格版）

## 上游路径

* 数据：`third_party/ib2d/matIB2d/Examples/Example_Mass_Points/Gravity_vs_Flow_256x256/`
  （含 `please_Compute_External_Forcing.m`，已逐行移植为
  `ib2d_input/please_Compute_External_Forcing.py`；与 demo_570 的唯一区别是
  射流带 `y ∈ [0.1, 0.13]`（更窄）而网格是 `256 x 256`）。
* 算法源码：`third_party/ib2d/pyIB2d/IBM_Blackbox/`。

质量点（`mass_pts`）算法的完整说明、上游 bug 清单、复算检验方法见 **demo_571 的 readme**；
官方 `dt` 与稳定性边缘的关系见 **demo_570 的 readme**。

## 算例内容（官方设置）

| 项 | 值 |
|---|---|
| 流体 | `1 x 1` 周期盒、`mu = 0.01`、`rho = 1`、**`256 x 256`**（本 demo 用 `128 x 128`，见下） |
| 时间 | `dt = 5e-5`、官方 `Tfinal = 0.35` |
| 结构 | 3 个 **226 点**圆环（`x ≈ 0.2 / 0.5 / 0.8`、`y ≈ 0.55`）+ 2766 个钉死目标点构成的内部"槽道" |
| 弹簧 / 梁 | 1017 根（`k = 1e5` / `1e7`）、678 根不变梁（`kb = 1e9`、`C ≈ -2.4e-8`） |
| 质量点 | 678 个：**`k = 1e4`（软）**、`M = 6.5e-4 / 0.0875 / 2.5` |
| 重力 | 开（`gvec = (0,-1)`） |
| 外力 | 开：三股竖直射流，带 `y ∈ [0.1, 0.13]`，`uY_tar = -uMax*0.5|sin(20πt)|*抛物线(x)`，`kStiff = 1e4`、`uMax = 250` |
| 标量 / 示踪 | 关 |

与 demo_570 的关键差别：**ghost 弹簧软了 100 倍（`k = 1e4`）**，
`ω·dt = 0.20`（`M = 6.5e-4`）远在稳定域内 —— 因此本 demo **不需要减半 `dt`**。

## 缩减设置

* 网格：**`NX = 128`**（`NY = 128`）代替官方 `256 x 256`
  （4 倍网格 + 7001 步在本机的负载下不可行；`main.py` 与 `run_reference.py --nx 128`
  用同一个值，两边一致）；
* 时间：**`T = 0.04`（401 步）** 代替官方 `Tfinal = 0.35`，
  `dt = 5e-5` 保持官方值，`print_dump = 40`（20 帧）。

## 移植要点：`mass_pts` 算法（摘要）

完整推导、上游 bug/quirks 清单与"复算检验"见 **demo_571 的 readme**；与本算例相关的要点：

* 状态：`mass_info[i] = [标记点 id, x_ghost, y_ghost, k, M]`，`mVelocity = 0`，
  ghost 从它标记点的初始位置出发（`IBM_Driver.py` 373-417）；
* 每步顺序（`IBM_Driver.py` 689-691, 731-742）：

  1. 步首半步移动 ghost：`x_ghost += (dt/2) * mVelocity`（在标记点半步移动**之后**、
     力计算**之前**）；
  2. 力例程在弹簧/梁/目标点力之后追加 `F = k (X_ghost - X_marker)`（只写进该标记点的槽位），
     这个 `F` 同时是"撒向流体"的力和 ghost 速度更新用的 `F_Mass_Bnd`；
  3. `mVelocity_h = mVelocity - (dt/2) * (F/M - g*gvec)`；
  4. `X_ghost = X_ghost(步首) + dt * mVelocity_h`（先退回步初位置，再走整步）；
  5. `mVelocity = mVelocity(步首) - dt * (F/M - g*gvec)`（用同一个 `F`）。
* AFSI 侧：ghost 是普通 numpy 数组（自由粒子**不需要** FE 空间），`F` 用 `np.add.at`
  加进与弹簧/梁/目标点**同一个** Lagrangian 力数组，再用既有 4 点核一起撒到网格；
  重力只进入 ghost 的速度更新（`Supp.py` 330-410，`g = 9.80665`）；
* ghost 是"自由"的：驱动会对标记点做周期回绕，但**从不回绕 ghost**（上游行为，本移植照抄；
  `compare.py` 里对 `ghost - marker` 偏移取最小镜像）。
* 上游 bug（本移植在 `run_reference.py` 里绕开，详见 demo_571）：
  `.mass` 只有**一列**索引（旧模板误写成两列）；pyIB2d 的"单质量点"分支在
  标记点数 > 1 时会 `ValueError`；pyIB2d 不输出 ghost，因此参考解的 ghost 历史
  由 wrapper 记录。

## 结果

运行时间（实测）：pyIB2d 参考 **95.1 s**、AFSI **580.7 s**；各 20 帧（含 `t = 0`），
`t = 0 .. 0.03805`。

| 指标（末帧 `t = 0.03805`） | 结果 |
|---|---|
| **质量点（ghost）轨迹 max\|dXm\|**（headline） | 4.03e-3 m = **0.515 h**（rms 0.304 h） |
| 三组各自：`M = 6.5e-4 / 0.0875 / 2.5` | 0.515 h / 0.164 h / 0.0118 h |
| 组质心偏差 | 0.510 h / 0.128 h / 0.0058 h |
| **ghost−marker 偏移之差**（纯弹簧耦合） | 6.16e-5 m = **0.0079 h** ← 机制本身 |
| 标记点偏差（3444 点） | 4.028e-3 m = 0.516 h |
| ghost 速度偏差 `max\|dVm\|` | 0.0862 m/s |
| 参考解末帧 ghost 位移 | 0.0216 m = 2.8 h |
| 速度峰值 AFSI / IB2d | 1.850 / 2.005（**7.7%**），`max\|du\|` = 0.307 |
| 复算检验（同一标记点轨迹下的 ghost ODE） | `max\|dXm\| = 0.000e+00 m` |

**能验证**：

* 质量点算法逐位一致（复算检验 = 0）；
* 本 demo 是 **mass_pts 一族里定量结果最好的一个**：ghost 轨迹 0.515 h、
  纯弹簧耦合 0.0079 h、速度峰值差 7.7%、`max|du|` 只有 0.31 m/s；
* **质量比敏感性得到确认**（与 demo_571/572 的结论一致）：
  `M = 2.5` 的一组偏差只有 0.012 h（重物几乎只由重力和弹簧决定），
  而最轻的 `M = 6.5e-4` 一组是 0.515 h（附加质量/阻力主导）；且因为这里的
  `k = 1e4`（比 demo_570 软 100 倍），同一组轻 ghost 的偏差从 1.84 h 降到 0.515 h。

**不能验证（如实记录）**：

* ghost/标记点的绝对位置（末帧 0.515 h；相对参考解 0.0216 m 的 ghost 位移约 24%），
  来源同样是两种离散在动边界耦合上的差别；
* 官方 `256 x 256` 网格与 `Tfinal = 0.35`：本 demo 用 `128 x 128` / `T = 0.04`，
  因此**不能**用本 demo 的数据去讨论 256 网格上的收敛性或长时间行为；
* 支撑带内的流体速度（`max|du| = 0.31 m/s`，与 `max|u|` 同量级）。

## 运行

```bash
conda activate afsi-dolfinx
python run_reference.py --tend 0.04 --print-dump 40 --nx 128   # 参考解（~1.6 min）
NX=128 TFINAL=0.04 PRINT_DUMP=40 python main.py               # AFSI（~9.7 min）
python compare.py
```

其他环境变量：`NX`（默认官方 256）、`NY`、`DT`、`TFINAL`、`PRINT_DUMP`、
`OUTPUT_PATH`（默认 `./plot`）、`GRAD_DIV`。

产物：`plot/afsi_result.npz`、`ib2d_reference.npz`、
`figures/compare_trajectories.png`（headline）、`compare_history.png`、
`compare_fields.png`、`compare_table.csv`。
