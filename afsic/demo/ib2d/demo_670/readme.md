# demo_670 — IB2d `Examples_Education/Pendulum/PointMass-256x256`（质量点摆）

## 上游路径

* 数据：`third_party/ib2d/matIB2d/Examples/Examples_Education/Pendulum/PointMass-256x256/`
  （`pendulum.vertex`、`pendulum.spring`、`pendulum.target`、`pendulum.mass`、
  `input2d`，以及生成几何的 `Give_Pendulum_Geometry_PointMass.m`）。
* 算法源码：`third_party/ib2d/pyIB2d/IBM_Blackbox/{IBM_Driver.py, Supp.py,
  please_Find_Lagrangian_Forces_On_Eulerian_grid.py}`。

这是 IB2d 教学算例里的**质量点（`mass_pts`，massive "ghost" boundary
point）**最小模型：一个只有 2 个标记点的单摆。本目录从 demo_575 派生（它含有
"只有 1 个质量点"的上游 bug 补丁），与 `demo_570` 共用同一套
`main.py` / `run_reference.py` / `compare.py` 基础设施。

`PointMass-256x256` / `-512x512` / `-1024x1024` 三个上游目录的
`pendulum.{vertex,spring,target,mass}` **逐字节相同**，只有 `input2d` 里的
`Nx = Ny` 不同（256 / 512 / 1024）——是一组网格加密研究。本仓库把它们分别
移植为 demo_670 / 671 / 672，并把缩减后的 `NX` 也连成一串（64 / 96 / 128）。

`Give_Pendulum_Geometry_PointMass.m` 只是"生成 `.vertex/.spring/...` 文件"的
MATLAB 脚本（`main2d.m` 在文件不存在时调用它），pyIB2d 驱动**从不调用**它，
因此没有移植；四个数据文件已随上游原样附带。

## 算例内容（官方设置）

| 项 | 值 |
|---|---|
| 流体 | `1 x 1` 周期盒、`mu = 0.01`、`rho = 1000`、`256 x 256` |
| 时间 | `dt = 2.5e-5`、官方 `Tfinal = 5.0`（200000 步） |
| 结构 | 2 个标记点：支点 marker 2 `(0.5, 2/3)`（被 target 钉死），摆锤 marker 1 `(0.33820, 0.54911)` |
| 弹簧 | 1 根线性弹簧（`k = 1e8`、`L0 = 0.2 m`），即摆杆 |
| 目标点 | 1 个（marker 2，`k = 5e7`） |
| 质量点 | 1 个（挂在 marker 1，`k = 2.5e6`、`M = 1000 kg`），即摆锤质量 |
| 重力 | 开（`gvec = (0, -1)`，`g = 9.80665`） |
| 梁 / 外力 / 标量 / 示踪 | 全关 |

初始角约偏离竖直 54°，故重力会驱动摆锤；小角度摆周期
`2π√(L/g) ≈ 0.90 s`（初始摆长恰好等于 `L0 = 0.2 m`，弹簧处于自然长度）。

## 移植要点：`mass_pts` 算法

IB2d 的质量点是一个**自由**"幽灵"粒子：它不随流体运动，而是被一根刚度为 `k`
的弹簧系在某个拉格朗日标记点上；弹簧力一方面驱动 ghost，另一方面**反作用在标记点**
（因而撒到流体）上。上游实现（`IBM_Driver.py` 373-417、689-742）：

```
mass_info[i] = [ lag id, x_ghost, y_ghost, k, M ],   mVelocity = 0
x_ghost(0) = 该标记点的初始位置
```

每步顺序：

1. 步首半步移动 ghost：`please_Move_Massive_Boundary(dt/2, ...)`（**在标记点半步移动之后、力计算之前**）；
2. 标记点照常移到半步位置，力例程在弹簧/目标点力之后追加质量点力
   `F[i] = k_i (X_ghost_i - X_marker[i])`（`give_Me_Mass_Lagrangian_Force_Densities`），
   这一份 `F` 同时被 4 点 delta 核撒到网格、并用于 ghost 速度更新；
3. `v -= (dt/2) (F[i]/M_i - g·gvec)`（`Supp.py` 330-410，`g = 9.80665`）；
4. `mass_info[:,[1,2]] = massLagsOld`，再 `Move(dt, ...)`：
   `X_ghost = X_ghost(步首) + dt·v_h`（**不是**从半步位置再走半步）；
5. `v = please_Update_Massive_Boundary_Velocity(dt, ...)`，用步首的 `v` 与同一个 `F`。

AFSI 侧（`main.py`）：ghost 就是普通 numpy 数组（**不需要** FE 空间）；
`mass_forces()` 返回 `F = k (Xg - Xm)` 并用 `np.add.at` 加进与弹簧/目标点**同一个**
`Fm`，由 `spread_forces_at_half()` 一次撒完；时间循环严格按上面 1-5 的顺序执行；
重力只进入 ghost 的速度更新。

### 发现的上游 bug / quirk（本算例全部命中）

1. **单行 `.spring` 直接崩溃**：`read_Spring_Points` 里 `n, m = springs.shape`，
   文件只有 1 根弹簧时 `np.genfromtxt` 返回 1-D 数组 → `ValueError: not enough
   values to unpack (expected 2, got 1)`。`run_reference.py` 用逐行等价、
   仅把形状修正为 `(n, m)` 的实现替换之。
2. **单行 `.target`/`.beam`/... 崩溃**：`read_Target_Points` 等用
   `np.loadtxt(..., usecols=...)`，单行时返回 1-D，随后 `aux[:, 0]` 抛
   `IndexError`。`run_reference.py` 统一把这些 reader 包成返回 `(n, m)`。
3. **`.mass` 只有一列索引**：`.mass` 行是 `id k M`；demo_575 的 `id_cols`
   表已把 `mass` 修正为 1 列（旧 demo 写成 2 列会把刚度 `k` 也减 1）。
4. **"只有 1 个质量点"分支是坏的**（`IBM_Driver.py` 385-401）：`.mass` 只有
   1 行时走该分支，里面是 `mass_info[0,1] = xLag[mass_info[0].astype('int')]`
   （把整行 5 列当索引）→ `ValueError`。本算例正好"2 个标记点 + 1 个质量点"，
   命中！`run_reference.py` 把 `read_Mass_Points` 打补丁追加一行 `k = 0` 的
   哑 ghost（挂到**另一个**标记点，`k = 0` 不产生任何力），从而走正确的通用分支。
5. **ghost 不随周期回绕**：驱动把标记点 `% Lx/% Ly`，但 `mass_info` 从不回绕
   → 长时间后 ghost 越过盒子边界时会与回绕后的标记点差一个 `L`。本例 0.2 s 内
   未触发；`compare.py` 对偏移取最小镜像。
6. **pyIB2d 从不输出 ghost**：VTK 只写标记点，参考解的 ghost 历史
   （`Xm/Vm/mass_M/mass_step`）由 `run_reference.py` 包装
   `please_Update_Massive_Boundary_Velocity` 记录得到。
7. **dump 标号偏移**：驱动在 `cter % print_dump == 0 且 cter >= print_dump`
   时写盘，时间戳是该步**步首**的时间；`main.py` 在与驱动完全相同的步上 dump，
   `compare.py` 按 `round(t/dt)` 对齐。

### ghost 算法的"复算"检验（关键验证）

`run_reference.py` 会额外记录每一步驱动实际使用的半步标记点位置，并用它复算本移植的
ghost ODE（`main.py` 的 mass 块）：**4001 步 `max|dXm| = 0.000e+00 m`**，
即给定相同的标记点轨迹，移植的质量点算法与 pyIB2d **逐位一致**（符号、`dt/2` 半步、
重力、"先 Move 再 Update"的顺序全对）。因此剩下的偏差只能来自流体耦合。

## 缩减设置（重要）

官方 `256 x 256`、`dt = 2.5e-5`、`Tfinal = 5.0`（200000 步）。本 demo 用
**`NX = 64`（官方 1/4）、`dt = 5e-5`（官方 2 倍）、`T = 0.2 s`（4001 步）**。

为什么 `dt` 只能放大到 2 倍：摆杆刚度 `k = 1e8` 极硬，显式 IB 耦合的稳定域很窄。
实测（`NX = 64`）：

| `dt` | 结果 |
|---|---|
| `2.5e-5`（官方） | 稳定，AFSI 与参考解 `max\|dXm\| = 1.6e-3 m`（0.2 s 内） |
| `5e-5`（采用） | 稳定；与 `2.5e-5` 的 AFSI 解相差 `max\|dXm\| = 1.7e-5 m` |
| `1e-4` | 前 0.1 s 正常，`t ≈ 0.3` 起 `max\|u\|` 失控（36 m/s），末帧 NaN |

pyIB2d 参考解在 `dt = 1e-4` 下仍是收敛的（`2.5e-5` 与 `1e-4` 相差
`max\|dXm\| = 4.7e-5 m`），所以 `1e-4` 的失稳是 AFSI 侧显式耦合的稳定限制，
不是物理。故两边都用 `dt = 5e-5`。

## 结果

运行时间（macOS，本 demo 实测）：pyIB2d 参考 **7.0 s**（4001 步驱动，总进程 11.8 s）、
AFSI **61.3 s**（4001 步）。两边各 9 帧（含 `t = 0`），`t = 0 .. 0.20005`。

| 指标（末帧 `t = 0.20005`） | 结果 |
|---|---|
| **ghost（摆锤质量）轨迹 max\|dXm\|**（headline） | 1.854e-3 m = **0.119 h** |
| ghost 速度偏差 `max\|dVm\|` | 3.52e-2 m/s |
| **ghost−marker 偏移之差**（纯弹簧耦合，机制本身） | 1.585e-3 m = **0.101 h** |
| 标记点偏差（2 点） | 1.274e-3 m = 8.15e-2 h |
| 参考解末帧 ghost 位移 | 0.0646 m（相对误差 ≈ 2.9%） |
| 流体 `max\|u\|` AFSI / IB2d | 1.764 / 1.186（相对差 48.8%），`max\|du\|` = 2.21 |
| 复算检验（同一标记点轨迹下的 ghost ODE） | 4001 步 `max\|dXm\| = 0.000e+00 m` |
| AFSI 的 `dt` 收敛性（`2.5e-5` vs `5e-5`） | `max\|dXm\| = 1.7e-5 m` |

**能验证**：

* **质量点算法**：用 pyIB2d 自己的每步标记点轨迹复算移植的 ghost ODE，
  4001 步逐位一致（`0.0`）——`dt/2` 半步、重力、"先 Move 再 Update"的顺序全对；
* **总体运动**：两边摆锤都从初始 54° 位置被重力驱动向下向右摆动，
  末帧 ghost 位移 `0.0646 m`，AFSI 偏差 `1.85e-3 m`（≈ 参考位移的 2.9%）；
* **`dt` 收敛**：AFSI 在 `2.5e-5` 与 `5e-5` 的解相差 1.7e-5 m，说明采用的
  `dt = 5e-5` 已收敛。

**不能验证（如实记录）**：

* 流体瞬时速度场：`max|u|` 差 48.8%、`max|du|` = 2.21 m/s。IB2d 的共置
  FFT 投影会在标记点 4 点 delta 支撑带内长出网格尺度尖峰（demo_467/478/570
  都有同类记录），而 AFSI 用 Taylor–Hood + 插值力密度，两者在该带内固有差异；
* 长时间的绝对轨迹：本 demo 只跑到 `T = 0.2 s`（官方 5.0 s 的 1/25，
  约摆周期的 1/4），仿真内没有完成一个完整摆动；
* 官方 `dt = 2.5e-5`/`Tfinal = 5` 的"完整"算例未复现（200000 步，超出本机预算）；
* 网格收敛性：pyIB2d 参考解自身在 `NX = 64/96/128` 之间变化约 1e-3 m
  （见 demo_671/672 的 readme），因此本 demo 的 `NX = 64` 解与官方 `256²`
  不是同一个离散解。

## 运行

```bash
conda activate afsi-dolfinx
python run_reference.py --tend 0.2 --dt 5e-5 --print-dump 500 --nx 64   # 参考解（~7 s）
NX=64 DT=5e-5 TFINAL=0.2 PRINT_DUMP=500 python main.py                 # AFSI（~61 s）
python compare.py
```

其他环境变量：`NX`（默认官方 256）、`NY`、`OUTPUT_PATH`（默认 `./plot`）、
`GRAD_DIV`（Taylor–Hood grad-div 稳定化，默认 0）。

产物：`plot/afsi_result.npz`、`ib2d_reference.npz`（含 ghost 历史 `Xm/Vm/mass_M`）、
`figures/compare_trajectories.png`（headline：ghost 轨迹 + 误差）、
`figures/compare_history.png`、`figures/compare_fields.png`、`figures/compare_table.csv`。
