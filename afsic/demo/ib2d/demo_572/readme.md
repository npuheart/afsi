# demo_572 — IB2d `Example_Mass_Points/Channel_Flow_Cellular_Race`（质量点 / 槽道流里的细胞赛跑）

## 上游路径

* 数据：`third_party/ib2d/matIB2d/Examples/Example_Mass_Points/Channel_Flow_Cellular_Race/`
  （含 `please_Compute_External_Forcing.m`，本 demo 逐行移植为
  `ib2d_input/please_Compute_External_Forcing.py`，AFSI 与 pyIB2d 参考跑同一份）。
* 算法源码：`third_party/ib2d/pyIB2d/IBM_Blackbox/`。

质量点算法本身见 **demo_571 的 readme**（那里有完整推导、上游 bug 清单与"复算检验"），
本文件只记录本算例特有的设置与结果。

## 算例内容（官方设置）

| 项 | 值 |
|---|---|
| 流体 | `1 x 1` 周期盒、`mu = 0.01`、`rho = 1`、`64 x 64` |
| 时间 | `dt = 5e-5`、官方 `Tfinal = 0.35`（7001 步） |
| 结构 | 3 个 54 点圆环（`y = 0.5`，`x = 0.25 / 0.5 / 0.75`）+ 696 个钉死目标点（`k = 1e7`，ids 163-858）构成的内部"槽道" |
| 弹簧 / 梁 | 243 根（`k = 5e7` / `5e5`）、162 根不变梁（`kb = 1e7`） |
| 质量点 | 162 个：`k = 1e6`、`M = 0.01 / 0.5 / 1.0` |
| 重力 | **关** |
| 外力 | 开，**稳态**三股水平射流（见下） |
| 标量 / 示踪 | 关 |

外力（`please_Compute_External_Forcing.m`）：在薄带 `x in [0.1, 0.16]` 内、
`y` 中心分别 `0.11 / 0.41 / 0.71` 的三个窗口中施加速度目标罚力

```
Fx = -kStiff (uX - uX_tar),  Fy = -kStiff uY,
uX_tar = -uMax * 5*tanh(t) * ((Lx/2 + w/2 + yShift) - y) ((Lx/2 - w/2 + yShift) - y)
```

`kStiff = 1e4`、`w = 0.2`、`uMax = 250`、`yShift = 0 / ±0.3`，即三股**抛物线型**
（在 `y` 方向）水平射流，随时间以 `5 tanh(t)` 平滑开启后保持（与 demo_570 的
`0.5|sin(20πt)|` 振荡射流不同）。上游代码里抛物线用的是 `Lx/2` 而自变量是 `y`，
本例 `Lx = Ly = 1`，数值上无差别，移植保持原式。

**本 demo 的缩减**：官方 `Tfinal = 0.35`，本 demo 用 **`T = 0.1`（2001 步）**。

## 移植要点

与 demo_571 完全相同（ghost 用 numpy 数组实现、力加进同一个 Lagrangian 力数组、
严格按驱动的 6 步顺序；`.mass` 单列 id 的修正；ghost 历史记录；复算检验）；
本算例额外用到模板的 `arb_ext_force`：AFSI 侧把 `solver.u_n` 插值到 Q1 节点网格
（与 IB2d 的共置网格同一套格点）→ 调用同一个 `please_Compute_External_Forcing.py`
→ 结果插值到 Q2 加到 `solver.f`（与 demo_467-478 的机制一致）。

**额外发现**：本算例的 `input2d` 里 `target_pts = 1`，但**上游 pyIB2d 会把
`channel.target` 里的 696 个目标点钉在它们的初始位置**（`update_target = 0`），
而 3 个圆环的标记点是"自由"的 1-162 号——两者在同一份 `.vertex` 文件里。

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

运行时间（macOS M4，10 核；负载波动很大）：pyIB2d 参考 **74.0 s**（计时复跑）、
AFSI **539.9 s**（2001 步，来自 main.py 的计时）；各 41 帧（含 `t = 0`），
`t = 0 .. 0.10005`。

| 指标（末帧 `t = 0.10005`） | 结果 |
|---|---|
| **质量点（ghost）轨迹 max\|dXm\|**（headline） | 2.249e-2 m = **1.44 h**（rms 0.633 h） |
| 三个环各自：`M = 0.01 / 0.5 / 1` | 1.44 h / 0.503 h / 0.369 h |
| 环质心偏差 | 0.560 h / 0.292 h / 0.249 h |
| **ghost−marker 偏移之差**（纯弹簧耦合） | 2.66e-4 m = **0.017 h** ← 机制本身 |
| 标记点偏差（858 点） | 2.248e-2 m = 1.44 h |
| ghost 速度偏差 `max\|dVm\|` | 0.700 m/s |
| 参考解末帧 ghost 位移 | 0.0611 m = 3.9 h |
| 速度峰值 AFSI / IB2d | 1.739 / 2.277（23.6%），`max\|du\|` = 2.120 |

**能验证**：质量点算法逐位一致（复算检验 = 0）；ghost 相对自己标记点的偏移对到 0.017 h；
三环随射流的下游漂移**顺序与量级**一致（轻环被吹得更远：`M = 0.01` 走 0.061 m，
`M = 1` 走得明显更少），质心偏差 0.25–0.56 h。

**不能验证（如实记录）**：

* ghost/标记点的绝对位置（末帧 1.44 h，rms 0.63 h）：这是动边界耦合的固有差异
  （AFSI Taylor–Hood + 插值力密度 vs IB2d 共置 FFT 投影 + 罚力）；
* 流体瞬时速度场：`max|u|` 差 23.6%，`max|du|` = 2.1 m/s。`max|du|` 出现在圆环标记点的
  4 点 delta 支撑带内（网格尺度尖峰），这也是绝对位置偏差的主要来源；
* 目标点"槽道壁"虽然钉死（`k = 1e7`），但钉死的实现方式在两代码里不同
  （IB2d：罚力，AFSI：同一套罚力——本 demo 用的就是同一个目标点力公式，
  但流体求解器不同），因此槽道壁附近的速度仍有差别。

## 运行

```bash
conda activate afsi-dolfinx
python run_reference.py --tend 0.1 --print-dump 50    # pyIB2d 参考解（74 s .. 3 min）
TFINAL=0.1 PRINT_DUMP=50 python main.py               # AFSI（540 s）
python compare.py
```

环境变量：`NX`（默认官方 64）、`NY`、`DT`、`TFINAL`、`PRINT_DUMP`、
`OUTPUT_PATH`（默认 `./plot`）、`GRAD_DIV`。

产物：`plot/afsi_result.npz`、`ib2d_reference.npz`、
`figures/compare_trajectories.png`（headline）、`compare_history.png`、
`compare_fields.png`、`compare_table.csv`。
