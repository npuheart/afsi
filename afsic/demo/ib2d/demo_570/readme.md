# demo_570 — IB2d `Example_Mass_Points/Gravity_vs_Flow`（质量点 / 重力对抗射流）

## 上游路径

* 数据：`third_party/ib2d/matIB2d/Examples/Example_Mass_Points/Gravity_vs_Flow/`
  （`channel.vertex/spring/beam/target/mass`、`input2d`、
  `please_Compute_External_Forcing.m` → 已逐行移植为
  `ib2d_input/please_Compute_External_Forcing.py`，AFSI 与 pyIB2d 参考跑同一份）。
* 算法源码：`third_party/ib2d/pyIB2d/IBM_Blackbox/{IBM_Driver.py, Supp.py,
  please_Find_Lagrangian_Forces_On_Eulerian_grid.py}`。

这是 AFSI 里**质量点（`mass_pts`，massive "ghost" boundary points）**机制的旗舰算例；
demo_571–576 共用同一套 `main.py` / `run_reference.py` / `compare.py` 模板。

## 算例内容（官方设置）

| 项 | 值 |
|---|---|
| 流体 | `1 x 1` 周期盒、`mu = 0.01`、`rho = 1`、`64 x 64` |
| 时间 | `dt = 5e-5`、官方 `Tfinal = 0.35`（7001 步） |
| 结构 | 3 个 54 点圆环（`x = 0.2 / 0.5 / 0.8`、`y = 0.55`、半径 ≈ 1/30）+ 696 个钉死目标点（`k = 1e7`，ids 163-858）组成的内部"槽道" |
| 弹簧 / 梁 | 243 根（`k = 5e7` / `5e5`）、162 根不变梁（`kb = 1e7`、`C = 0`） |
| 质量点 | 162 个（每个标记点一个）：`k = 1e6`、`M = 2.75 / 0.5 / 6.5e-4` |
| 重力 | 开（`gvec = (0,-1)`，`g = 9.80665`） |
| 外力 | 开：三股**竖直**射流（见下） |
| 标量 / 示踪 | 关 |

外力（`please_Compute_External_Forcing.m`）：在薄带 `y in [0.1, 0.16]` 内、
`x` 中心分别 `0.2 / 0.5 / 0.8` 的三个窗口中施加速度目标罚力

```
Fy = -kStiff (uY - uY_tar),  Fx = -kStiff uX,
uY_tar = -uMax * 0.5|sin(20 pi t)| * ((Lx/2+w/2+yShift) - x) ((Lx/2-w/2+yShift) - x)
```

`kStiff = 1e4`、`w = 0.2`、`uMax = 250`、`yShift = 0 / ±0.3`，即三股抛物线型
（在 `x` 方向）竖直射流，`0.5|sin(20πt)|` 以 20 Hz 开关（`T = 0.35` 内 7 个来回）。
这就是"gravity vs flow"：重力把三个质量不同的环往下拉，射流把它们往上顶。

## 移植要点：`mass_pts` 算法

IB2d 的质量点是一个**自由**"幽灵"粒子：它不随流体运动，而是被一根刚度为 `k`
的弹簧系在某个拉格朗日标记点上；弹簧力一方面驱动 ghost，另一方面**反作用在标记点**
（因而撒到流体）上。上游实现（`IBM_Driver.py` 373-417）：

```
mass_info[i] = [ lag id, x_ghost, y_ghost, k, M ],   mVelocity = 0
x_ghost(0) = 该标记点的初始位置
```

每步顺序（`IBM_Driver.py` 689-691 与 731-742）：

1. 步首半步移动 ghost：`mass_info, massLagsOld = please_Move_Massive_Boundary(dt/2,
   mass_info, mVelocity)`（**在标记点半步移动之后、力计算之前**）；
2. 标记点照常移到半步位置 `xLag_h`，力例程在算完弹簧/梁/目标点力后追加质量点力
   （`please_Find_Lagrangian_Forces_On_Eulerian_grid.py:
   give_Me_Mass_Lagrangian_Force_Densities`）：

   ```
   F[i] = k_i ( X_ghost_i - X_marker[i] )     % 加到 Lagrangian 力数组的 marker[i] 槽位
   ```

   这个 `F` 既是标记点/流体受到的力（与其它结构力一起用同一个 4 点 delta 核撒到网格），
   也是 ghost 速度更新用的 `F_Mass_Bnd`；
3. `mVelocity_h = please_Update_Massive_Boundary_Velocity(dt/2, mass_info, mVelocity, F,
   gravity_Info)`：

   ```
   v -= (dt/2) * ( F[i]/M_i - g * gvec )     % 有重力时；Supp.py 330-410，g = 9.80665
   ```

4. `mass_info[:,[1,2]] = massLagsOld`，再 `Move(dt, mass_info, mVelocity_h)`：
   `X_ghost = X_ghost(步首) + dt * mVelocity_h`（**不是**从半步位置再走半步）；
5. `mVelocity = please_Update_Massive_Boundary_Velocity(dt, mass_info, mVelocity, F, gravity_Info)`
   —— 用的仍是步首的 `mVelocity` 和同一个 `F`。

AFSI 侧（`main.py`）：ghost 就是普通 numpy 数组（`mass_pos`/`mass_vel`，自由粒子
**不需要** FE 空间）；`mass_forces()` 返回 `F = k (Xg - Xm)` 并用 `np.add.at` 加进与
弹簧/梁/目标点**同一个** `Fm`，由 `spread_forces_at_half()` 用既有 4 点核一次撒完
（与 IB2d 把 `fx_mass` 加进 `fx` 再撒完全等价）；时间循环严格按上面 1-5 的顺序执行；
重力只进入 ghost 的速度更新（Boussinesq 是另一个开关，本例关闭）。

### 发现的上游 bug / quirk

1. **`.mass` 只有一列索引**（`id k M`）：上一批 demo 的 `run_reference.py` 里
   `id_cols` 表把 `mass` 写成 2 列（会把刚度 `k` 也减 1）。本 demo 修正为 `"mass": 1`。
2. **单质量点分支是坏的**（`IBM_Driver.py` 387-400）：`.mass` 只有 1 行时走该分支，
   里面写的是 `mass_info[0,1] = xLag[mass_info[0].astype('int')]`（把整行 5 列当索引）
   → `ValueError`。触发条件是"标记点数 > 1 且质量点数 == 1"（demo_575 命中；
   demo_458 的 Thrips 是"1 标记点 + 1 质量点"，`xLag` 是 `np.float64` 标量，
   走安全分支，实测躲过）。
3. **pyIB2d 从不输出 ghost**：VTK 只写标记点，所以参考解的 ghost 历史
   （`Xm/Vm/mass_M/mass_step`）由 `run_reference.py` 用 wrapper 记录
   `please_Update_Massive_Boundary_Velocity` 得到。
4. **ghost 不随周期回绕**：驱动把标记点 `% Lx/% Ly`，但 `mass_info` 从不回绕
   （没有 minimum-image）→ 真跑久了 ghost 越过盒子边界时会与回绕后的标记点差一个 `L`，
   弹簧力出现尖峰。本 demo 未触发；`compare.py` 对偏移取最小镜像。
5. **dump 标号偏移**：驱动在 `cter % print_dump == 0 且 cter >= print_dump` 时写盘，
   但写的是**该步末**的状态而时间戳是 `cter*dt`（少一步）。`main.py` 在与驱动完全相同的
   步上 dump 并盖真实时间；`compare.py` 按 `round(t/dt)` 对齐。

### ghost 算法的"复算"检验（关键验证）

`run_reference.py` 会额外记录每一步驱动实际使用的半步标记点位置，并用它复算本移植的
ghost ODE（`main.py` 的 mass 块）：**8001 步 `max|dXm| = 0.000e+00 m`**，
即给定相同的标记点轨迹，移植的质量点算法与 pyIB2d **逐位一致**。

## 缩减设置（重要）与实测的数值稳定性

官方 `Tfinal = 0.35`、`dt = 5e-5`。本 demo 的最终版本用
**`dt = 2.5e-5`（dt 减半）、`T = 0.2`（8001 步）**，网格与所有物理参数保持官方值。

为什么必须把 `dt` 减半：最轻的 ghost 组（`M = 6.5e-4`、`k = 1e6`）的固有频率
`ω = sqrt(k/M) = 3.92e4 rad/s`，在官方 `dt = 5e-5` 下 **`ω·dt ≈ 1.96`**，正好贴在
IB2d 自己那套显式 ghost 积分（同步 leapfrog，稳定域 `ω·dt < 2`）的边界上：

* pyIB2d 参考解在 `dt = 5e-5` 下**恰好稳定**（ghost 速度全程在 1–2 m/s 之间振荡）；
* AFSI 在 `dt = 5e-5` 下**发散**：`max|Vm|` 从 t≈0.01 起每 ~0.02 s 翻一倍
  （`M = 2.75/0.5` 两组完全正常，只有最轻的一组失控），`max|u|` 到 t≈0.22 变成 NaN；
* `dt = 2.5e-5`（`ω·dt ≈ 0.98`）后 AFSI 的 ghost 速度**有界振荡**
  （实测 `max|Vm| ∈ [0.5, 1.6]`，不再增长），并且**物理量不变**：
  在 `dt = 5e-5` 与 `dt = 2.5e-5` 两个 AFSI 解里，`t = 0.04` 的 `max|dXm|` 分别是
  0.0310 与 0.0309 m。

因此 `run_reference.py` 与 `main.py` **两边都用 `dt = 2.5e-5`**，属于
"官方算例把时间步放在稳定性边缘"这一上游性质的直接后果，不是移植公式的问题
（`demo_571` 的复算检验证明质量点算法本身逐位一致）。

## 结果

运行时间（macOS M4，10 核，系统负载 90+，均为本 demo 实测）：
pyIB2d 参考 **811 s**（8001 步，含其 VTK 汇总）、AFSI **1066.9 s**。
两边各 21 帧（含 `t = 0`），`t = 0 .. 0.20003`。

| 指标（末帧 `t = 0.20003`） | 结果 |
|---|---|
| **质量点（ghost）轨迹 max\|dXm\|**（headline） | 2.869e-2 m = **1.84 h**（rms 1.02 h） |
| 三组各自：`M = 6.5e-4 / 0.5 / 2.75` | 1.84 h / 1.17 h / 0.92 h（质心 1.00 / 0.93 / 0.79 h） |
| **ghost−marker 偏移之差**（纯弹簧耦合） | 4.60e-4 m = **0.029 h** ← 机制本身 |
| 标记点偏差（858 点） | 2.869e-2 m = 1.84 h |
| ghost 速度偏差 `max\|dVm\|` | 0.514 m/s |
| 参考解末帧 ghost 位移 | 0.1444 m = 9.2 h |
| 速度峰值 AFSI / IB2d | 3.07 / 4.87（36.8%），`max\|du\|` = 3.30 |
| 复算检验（同一标记点轨迹下的 ghost ODE） | 8001 步 `max\|dXm\| = 0.000e+00 m` |

**能验证**：

* 质量点**算法**：用 pyIB2d 自己的每步标记点轨迹复算移植的 ghost ODE，
  8001 步逐位一致（0.0）——符号、`dt/2` 半步、重力、"先 Move 再 Update"的顺序全对；
* **质量点弹簧耦合**：两边的"ghost 相对自己标记点的偏移"只差 0.029 h
  （即 ghost 位置偏差几乎完全等于其标记点的偏差，是"被标记点带偏的"）；
* **赛跑的物理**：三组质量不同的 ghost 上升/下落的行为一致（重的掉、轻的被射流顶起），
  组内质心偏差 0.79–1.00 h，趋势与相位一致；
* 与 demo_574（同族、把 `k` 从 `1e6` 软到 `1e4`）对比可以看到：把 ghost 弹簧软下来后
  ghost 轨迹偏差从 1.84 h 掉到 0.52 h，说明偏差主要来自"刚性 ghost 把标记点带得
  更敏感"，而不是机制写错。

**不能验证（如实记录）**：

* ghost/标记点的绝对位置（末帧 1.84 h，rms 1.02 h；相对参考解 0.14 m 的 ghost 位移
  约 20%）。这是 AFSI（Taylor–Hood + 插值力密度）与 IB2d（共置 FFT 投影 + 罚力）
  在动边界耦合上的固有差异，demo_478/571/572 都有同类记录；
* 流体瞬时速度场：`max|u|` 差 36.8%，`max|du|` = 3.3 m/s。两者都出现在圆环标记点的
  4 点 delta 支撑带内（IB2d 的共置投影在该带内会长出网格尺度尖峰，见 demo_467 的记录），
  而射流带本身（`y ∈ [0.1,0.16]`，远离所有标记点）的速度剖面两边是对齐的；
* **官方 `dt = 5e-5` 下的 AFSI 解不存在**（见上：发散），因此本 demo 的所有数字都是
  `dt = 2.5e-5` 的；官方 `Tfinal = 0.35` 也没有跑（用 `T = 0.2`，仍覆盖
  `|sin(20 pi t)|` 的 4 个完整开关周期）。

## 运行

```bash
conda activate afsi-dolfinx
python run_reference.py --tend 0.2 --dt 2.5e-5 --print-dump 400   # 参考解（~13.5 min）
DT=2.5e-5 TFINAL=0.2 PRINT_DUMP=400 python main.py                # AFSI（~18 min）
python compare.py
```

其他环境变量：`NX`（默认官方 64）、`NY`、`OUTPUT_PATH`（默认 `./plot`）、
`GRAD_DIV`（Taylor–Hood grad-div 稳定化，默认 0）。

产物：`plot/afsi_result.npz`、`ib2d_reference.npz`（含 ghost 历史 `Xm/Vm/mass_M`）、
`figures/compare_trajectories.png`（headline：三组 ghost 质心轨迹 + 误差）、
`compare_history.png`、`compare_fields.png`、`compare_table.csv`。
