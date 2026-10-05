# demo_573 — IB2d `Example_Mass_Points/Massive_Cell_In_Channel`（质量点 / 槽道里的单细胞）

## 上游路径

* 数据：`third_party/ib2d/matIB2d/Examples/Example_Mass_Points/Massive_Cell_In_Channel/`
  （含 `please_Compute_External_Forcing.m`，已逐行移植为
  `ib2d_input/please_Compute_External_Forcing.py`，两个代码共用同一份）。
* 算法源码：`third_party/ib2d/pyIB2d/IBM_Blackbox/`。

质量点算法（`mass_pts`）的完整说明、上游 bug 清单、以及"用 pyIB2d 自己的标记点轨迹
复算移植算法"的检验方法，见 **demo_571 的 readme**。

## 算例内容（官方设置）

| 项 | 值 |
|---|---|
| 流体 | `1 x 1` 周期盒、`mu = 0.01`、`rho = 1`、`64 x 64` |
| 时间 | `dt = 5e-5`、官方 `Tfinal = 0.35`（7001 步） |
| 结构 | 1 个 54 点圆环（`x = 0.5`、`y = 0.5`、半径 ≈ 1/30）+ 232 个钉死目标点（`k = 1e7`，上下两壁 `y = 0.4 / 0.6`） |
| 弹簧 / 梁 | 54 根弹簧（`k = 5e7`）；**没有梁**（`beams = 0`，文件中无 `.beam`） |
| 质量点 | 54 个（每个标记点一个）：`k = 1e6`、`M = 0.02` |
| 重力 | 关 |
| 外力 | 开：**单股**抛物线槽道射流（`x in [0.1, 0.16]`、`y in [0.41, 0.59]`、`uX_tar = -uMax * 5 tanh(t) * ((Lx/2+w/2) - y)((Lx/2-w/2) - y)`，`kStiff = 1e4`、`w = 0.2`、`uMax = 250`） |
| 标量 / 示踪 | 关 |

这是 demo_572 的单细胞版本：没有梁、没有周期赛跑，细胞被一股平滑开启的槽道射流
"推着走"，同时因每个标记点都挂着质量为 `M = 0.02` 的 ghost 而具有惯性。

**本 demo 的缩减**：官方 `Tfinal = 0.35`，本 demo 用 **`T = 0.1`（2001 步）**。

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

运行时间：pyIB2d 参考 **37.6 s**（计时复跑）、AFSI **613.7 s**（2001 步，来自
main.py 的计时）；两边各 41 帧（含 `t = 0`），`t = 0 .. 0.10005`。

| 指标（末帧 `t = 0.10005`） | 结果 |
|---|---|
| **质量点（ghost）轨迹 max\|dXm\|**（headline） | 1.228e-2 m = **0.786 h**（rms 0.546 h） |
| 环质心偏差 | 0.465 h |
| **ghost−marker 偏移之差**（纯弹簧耦合） | 4.44e-6 m = **2.8e-4 h** ← 机制本身（最好的一次） |
| 标记点偏差（286 点） | 1.228e-2 m = 0.786 h |
| ghost 速度偏差 `max\|dVm\|` | 0.255 m/s |
| 参考解末帧 ghost 位移 | 0.0563 m = 3.6 h |
| 速度峰值 AFSI / IB2d | 1.231 / 1.226（**0.4%**），`max\|du\|` = 0.674 |

**能验证**：

* 质量点算法逐位一致（复算检验 = 0）；
* **纯弹簧耦合**（ghost 相对标记点的偏移）对到 `2.8e-4 h` —— 这是所有 mass_pts 算例里
  最好的一次，因为只有一个细胞、没有周期回绕、ghost 轨迹平滑；
* 流场的**峰值速度**对到 0.4%（`1.231` vs `1.226` m/s），说明射流本身被两套求解器
  算得一致；
* 细胞被射流带走的位移量级一致（末帧两边 ghost 位移都在 0.056 m 附近）。

**不能验证（如实记录）**：

* 标记点/ghost 的绝对位置（末帧 0.79 h，rms 0.55 h）。注意这里的偏差**不是**
  流场速度的偏差（`max|u|` 只差 0.4%），而是细胞在射流中被"推"了 0.056 m 后，
  两种离散给出的**相位/位移**逐渐分开——即细胞位置对速度场在 4 点 delta 支撑带内
  的微小差别（`max|du|` = 0.67 m/s，出现在标记点带内）非常敏感；
* 支撑带内（marker 邻域）的流体速度：`max|du|` = 0.674 m/s，与 `max|u|` 同量级，
  这是 AFSI/IB2d 力密度撒布与投影方式差异的已知表现（参见 demo_467 的"窄带尖峰"记录）。

## 运行

```bash
conda activate afsi-dolfinx
python run_reference.py --tend 0.1 --print-dump 50    # pyIB2d 参考解（38 s .. 2 min）
TFINAL=0.1 PRINT_DUMP=50 python main.py               # AFSI（614 s）
python compare.py
```

环境变量：`NX`、`NY`、`DT`、`TFINAL`、`PRINT_DUMP`、`OUTPUT_PATH`（默认 `./plot`）、`GRAD_DIV`。

产物：`plot/afsi_result.npz`、`ib2d_reference.npz`、
`figures/compare_trajectories.png`（headline）、`compare_history.png`、
`compare_fields.png`、`compare_table.csv`。
