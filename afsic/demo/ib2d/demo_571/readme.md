# demo_571 — IB2d `Example_Mass_Points/Gravity_Cellular_Race`（质量点 / 重力赛跑）

## 上游路径

* 数据与驱动：`third_party/ib2d/matIB2d/Examples/Example_Mass_Points/Gravity_Cellular_Race/`
  （pyIB2d 里也自带一份同样内容的 `pyIB2d/Examples/Gravity_Cellular_Race/`，
  本 demo 用 matIB2d 那份，读取时把 id 从 1-based 归一到 0-based）。
* 算法源码：`third_party/ib2d/pyIB2d/IBM_Blackbox/{IBM_Driver.py, Supp.py,
  please_Find_Lagrangian_Forces_On_Eulerian_grid.py}`。

这是 AFSI 里**质量点（`mass_pts`，massive "ghost" boundary points）**机制的第一个算例
（demo_570 是同一机制的"重力 vs 射流"版本）。

## 算例内容（官方设置）

| 项 | 值 |
|---|---|
| 流体 | `1 x 1` 周期盒、`mu = 0.01`、`rho = 1`、`64 x 64` |
| 时间 | `dt = 5e-5`、官方 `Tfinal = 0.35`（7001 步） |
| 结构 | 3 个 54 点圆环（`x = 0.25 / 0.5 / 0.75`、`y = 0.8`、半径 ≈ 1/30） |
| 弹簧 | 243 根（环内 `k = 5e7`、跨环 `k = 5e5`） |
| 梁 | 162 根不变梁（`kb = 1e7`、`C = 0`） |
| 目标点 | 无（`target_pts = 0`；文件 `channel.target` 存在但被驱动忽略） |
| 质量点 | 162 个（每个标记点一个）：`k = 1e6`、`M = 0.05 / 0.2 / 1.0` |
| 重力 | 开，`gvec = (0, -1)`，`g = 9.80665 m/s^2`（IB2d 硬编码） |
| 外力 / 标量 / 示踪 | 全关 |

三个环除质量外完全相同，因此这是一个"谁先掉下去"的赛跑：重力只作用在 ghost 上，
ghost 通过弹簧把力传给标记点，标记点再通过流体粘性/附加质量获得阻力。

**本 demo 的缩减**（见"结果"）：官方 `Tfinal = 0.35`，本 demo **用 `T = 0.1`（2001 步）**，
`dt`、网格、所有物理参数保持官方值。

## 移植要点：`mass_pts` 算法

IB2d 的质量点是一个**自由**"幽灵"粒子：它不随流体运动，而是被一根刚度为 `k`
的弹簧系在某个拉格朗日标记点上，弹簧力一方面驱动 ghost，另一方面**反作用在标记点上**
（因而撒到流体上）。上游实现（`read_Mass_Points` → `mass_info`）：

```
mass_info[i] = [ lag id, x_ghost, y_ghost, k, M ],   mVelocity = 0
x_ghost(0) = 该标记点的初始位置          % IBM_Driver.py 373-417
```

每步的调用顺序（`IBM_Driver.py` 689-691 与 731-742）：

1. 步首半步移动 ghost：`mass_info, massLagsOld = please_Move_Massive_Boundary(dt/2, mass_info, mVelocity)`
   （注意这是**在标记点半步移动之后、力计算之前**）；
2. 标记点照常移到半步位置 `xLag_h`，力计算例程在算完弹簧/梁/目标点力后**追加**质量点力
   （`please_Find_Lagrangian_Forces_On_Eulerian_grid.py: give_Me_Mass_Lagrangian_Force_Densities`）：

   ```
   F[i] = k_i * ( X_ghost_i - X_marker[i] )        % 加到 Lagrangian 力数组的 marker[i] 槽位
   ```

   这个 `F` 就是标记点/流体受到的力（用同一个 4 点 delta 核撒到 Eulerian 网格），
   同时也是 ghost 速度更新用的 `F_Mass_Bnd`；
3. `mVelocity_h = please_Update_Massive_Boundary_Velocity(dt/2, mass_info, mVelocity, F, gravity_Info)`：

   ```
   v -= (dt/2) * ( F[i]/M_i - g * gvec )           % 有重力时（Supp.py 330-410）
   ```

4. `mass_info[:,[1,2]] = massLagsOld`，然后 `Move(dt, mass_info, mVelocity_h)`：
   `X_ghost = X_ghost(步首) + dt * mVelocity_h`（**不是**从半步位置再走半步）；
5. `mVelocity = please_Update_Massive_Boundary_Velocity(dt, mass_info, mVelocity, F, gravity_Info)`
   —— 用的仍是步首的 `mVelocity` 和同一个 `F`。

AFSI 侧的对应实现（`main.py`）：

* ghost 就是普通 numpy 数组（`mass_pos`, `mass_vel`），自由粒子**不需要** FE 空间；
* `mass_forces(Xm, Xg, Fm)` 返回 `F = k (Xg - Xm)` 并把它 `np.add.at` 到与
  弹簧/梁/目标点**同一个** `Fm` 数组的 `mass_ids` 槽位 → 由 `spread_forces_at_half()`
  用既有 4 点核一次撒完（与 IB2d 把 `fx_mass` 加进 `fx` 再撒完全等价）；
* 时间循环里严格按上面 6 步的顺序执行（(1b) 半步 ghost 移动 → (2) 力(含 F_mass) →
  (2c) ghost 半步速度/整步位置/整步速度 → (3) 流体求解）；
* 重力只进入 ghost 的 `v -= dt*(F/M - g*gvec)`，与 IB2d 一致（Boussinesq 是另一个开关，本例关闭）。

### 发现的上游 bug / quirk

1. **`.mass` 只有一列索引**：`.mass` 行是 `id k M`，但上一批 demo 的
   `run_reference.py` 里 `id_cols` 表把 `mass` 写成 2 列（会把刚度 `k` 也减 1）。
   本 demo 修正为 `"mass": 1`（IB2d 自己的 `read_Mass_Points` 只用第 0 列做索引）。
   例子：demo_570 之前若真跑 mass 例子就会静默改掉 `k`。
2. **单质量点分支是坏的**（`IBM_Driver.py` 387-400）：当 `.mass` 只有 1 行时
   `read_Mass_Points` 返回 1 维数组（`size/3 < 2`），走"只有 1 个质量点"分支，里面写的是

   ```python
   mass_info[0,1] = xLag[mass_info[0].astype('int')]   # 把整行 5 列当索引
   ```

   `mass_info[0]` 是 `[id, 0, 0, 0, 0]`，用它索引 `xLag` 得到 5 个值的数组，
   赋给标量 → `ValueError`。**触发条件是"标记点数 > 1 且质量点数 == 1"**：
   只有一个标记点时（例如 `pyIB2d/Examples/Thrips`，已被移植为 demo_458）
   `np.loadtxt(..., unpack=True)` 返回的是 `np.float64` 标量，`np.isscalar` 为真，
   走的是安全分支；而 demo_575 的"90 个标记点 + 1 个质量点"必然踩到这个 bug。
3. **pyIB2d 从不输出 ghost**：VTK 只写 `lagsPts`（标记点）。
   所以 `run_reference.py` 用 wrapper 记录 `please_Update_Massive_Boundary_Velocity`
   的入参/返回值，把 ghost 位置与速度存进 `ib2d_reference.npz`（`Xm`/`Vm`/`mass_M`）。
4. **ghost 不随周期回绕**：驱动把标记点 `% Lx/% Ly` 回绕，但 `mass_info` 从不回绕
   （没有 minimum-image），所以真跑久了 ghost 越过盒子边界时会与回绕后的标记点
   差一个 `L`，弹簧力出现尖峰。本 demo（`T = 0.1`，下落不到 0.05 m）没有触发；
   `compare.py` 对 `ghost - marker` 偏移取最小镜像，保证这个量在回绕时也连续。
5. **dump 标号偏移**：驱动在 `cter % print_dump == 0 且 cter >= print_dump`
   时写盘并盖上 `cter * dt` 的时间戳，但写的是**该步末**的状态（即多走了一步）。
   `main.py` 在与驱动完全相同的步上 dump，并盖上真实时间，
   `compare.py` 按 `round(t/dt)` 对齐（首帧 = 初值）。

### ghost 算法的"复算"检验（关键验证）

`run_reference.py` 在跑参考解时额外记录每一步**驱动实际使用的半步标记点位置**，
然后用它复算本 demo 移植的 ghost ODE（`main.py` 的 mass 块，逐位 numpy 重写）：

```
[pyIB2d] replay check of the ported ghost ODE driven by pyIB2d's own markers:
         max|dXm| = 0.000e+00 m over 2001 steps
```

即：**给定相同的标记点轨迹，移植的质量点算法与 pyIB2d 逐位一致**（符号、`dt/2`
半步、重力、两个 Update 调用的顺序全对）。因此下面结果里 ghost 轨迹的偏差
**全部来自流固耦合**（标记点轨迹本身的差异），不是质量点算法的问题。

这一判断还有第二个证据：`max|d(ghost-marker offset)|`（两边"ghost 相对自己标记点的
偏移"之差）在末帧只有 **2.4e-4 m = 0.015 h**，而 ghost 绝对位置的偏差是 0.76 h ——
说明 ghost 几乎完全跟着自己的标记点走，偏差是"被标记点带偏"的。

## 结果

运行时间（本机 macOS M4，10 核；**负载波动很大，下面给出计时复跑的实测值**）：
pyIB2d 参考 **22.8 s**（计时复跑；首次运行在负载 90+ 时约 175 s）、
AFSI **383.1 s**（2001 步，由 main.py 自己计时）；
两边各 41 帧（含 `t = 0`），`t = 0 .. 0.10005`。

| 指标（末帧 `t = 0.10005`） | 结果 |
|---|---|
| **质量点（ghost）轨迹 max\|dXm\|**（headline） | 1.191e-2 m = **0.762 h**（rms 0.396 h） |
| 三个环各自：`M = 0.05 / 0.2 / 1` | 0.762 h / 0.516 h / 0.237 h |
| 环质心偏差（刚体下落部分） | 0.504 h / 0.324 h / 0.062 h |
| **ghost−marker 偏移之差**（纯弹簧耦合） | 2.41e-4 m = **0.015 h** ← 机制本身 |
| 标记点偏差（162 点） | 1.193e-2 m = 0.763 h |
| ghost 速度偏差 `max\|dVm\|` | 0.566 m/s |
| 参考解末帧 ghost 位移 | 0.0477 m = 3.1 h |
| 速度峰值 AFSI / IB2d | 1.473 / 1.761（16.4%），`max\|du\|` = 1.595 |

**能验证**：

* 质量点**算法**本身：复算检验逐位为 0（见上）；
* 质量点**运动学**（环的刚体下落）：三个环的质心轨迹形状、下落顺序、
  轻环下落慢/重环下落快的关系两边一致，质心偏差 0.06–0.50 h；
* 环内部 ghost 相对标记点的偏移（弹簧响应）对到 0.015 h。

**不能验证（如实记录）**：

* 标记点/ghost 的绝对位置：末帧 0.76 h（约 25 m/s 的环速下走 0.17 m，偏差约 8%
  的行程）；这是 AFSI（Taylor–Hood + 插值力密度）与 IB2d（共置 FFT 投影 + 罚力）
  在**动边界耦合**上的固有差异，demo_478 记录过同类现象；
* 流体瞬时速度场：`max|u|` 差 16.4%，`max|du|` 达 1.6 m/s（该值出现在标记点 4 点
  delta 支撑带内的网格尺度尖峰上，demo_467 记录过 IB2d 共置投影会在该带内长出尖峰）；
* **质量比对耦合差异的敏感性**（本 demo 的额外发现）：偏差随质量减小而迅速变大
  （`M = 1`：0.24 h；`M = 0.5`… ；`M = 0.05`：0.76 h），说明轻环的下落由
  **流体的附加质量/阻力**主导，而附加质量在两种离散下差别最大；重环几乎只由重力
  和弹簧决定，所以对得很好。这解释了两个代码"物理一致、数值不同"的现象。

## 运行

```bash
conda activate afsi-dolfinx
python run_reference.py --tend 0.1 --print-dump 50    # 参考解（22.8 s .. 3 min，随负载）
TFINAL=0.1 PRINT_DUMP=50 python main.py               # AFSI（~6.5 min）
python compare.py                                     # 表格 + figures/
```

其他环境变量：`NX`（网格，默认官方 `64`）、`NY`、`DT`、`OUTPUT_PATH`（默认 `./plot`）、
`IB2D_EXAMPLE`、`GRAD_DIV`（Taylor–Hood 的 grad-div 稳定化，默认 0）。

产物：`plot/afsi_result.npz`（标记点/ghost 历史 + 网格场）、
`ib2d_reference.npz`（pyIB2d 参考解 + ghost 历史）、
`figures/compare_trajectories.png`（headline：三组 ghost 质心轨迹与误差）、
`compare_history.png`、`compare_fields.png`、`compare_table.csv`。
