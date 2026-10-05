# demo_576 — IB2d `Example_Concentration_Dynamics/Leaf_64`（质量点 + 浓度标量 / 叶片）

## 上游路径

* 数据：`third_party/ib2d/matIB2d/Examples/Example_Concentration_Dynamics/Leaf_64/`
  （`leaf.vertex/spring/beam/target/mass/concentration`、`input2d`、
  `please_Compute_External_Forcing.m` → 已逐行移植为
  `ib2d_input/please_Compute_External_Forcing.py`；`give_me_source_model.m` 没有被驱动引用，
  见下）。
* 算法源码：`third_party/ib2d/pyIB2d/IBM_Blackbox/`。

质量点（`mass_pts`）算法说明、上游 bug 清单、复算检验方法见 **demo_571 的 readme**；
被动标量算子（`Supp.perform_Time_Split_Upwind`）已在 demo_476/478 验证过。

## 算例内容（官方设置）

| 项 | 值 |
|---|---|
| 流体 | `30 x 5` 周期盒、`mu = 1.895e-4`、`rho = 1.145e-3`、`384 x 64` |
| 时间 | `dt = 5e-5`、官方 `Tfinal = 3.25`（65000 步） |
| 结构 | 33 点"叶片"（`x ∈ [2.94, 3.98]`、`y ≈ 2.5`）；**前 10 个点是钉死目标点**（`k = 25000`，叶柄），其余可动 |
| 弹簧 / 梁 | 36 根（`k = 25000`）、25 根不变梁（`kb = 2e8`） |
| 质量点 | **26 个**（挂在 id 8–33 的标记点上）：`k = 1`（极软）、`M = 5e-4` |
| 重力 | 关 |
| 外力 | 开：入口薄条 `x ∈ [0, 0.16]`、`y ∈ [0.005, 5]` 的水平风，`uX_tar = 5 tanh(t) m/s`（`kStiff = 0.1`，很软） |
| 浓度标量 | `concentration = 1`、`kDiff = 0.01`（见下：上游初始场恒为 0） |

**本 demo 的缩减**：官方 `384 x 64`、`Tfinal = 3.25 s`（65000 步）；
本 demo 用 **`NX = 192`（`NY = 32`）**、**`T = 0.1 s`（2001 步）**，
`dt = 5e-5` 与全部物理参数保持官方值。

* 网格减半后，随算例发布的 `leaf.concentration`（`64 x 384`）**按 2x2 分块平均**
  降到 `32 x 192`：`main.py` 里做（`_coarsen_scalar`），`run_reference.py` 在写运行目录时
  对同一文件做同样的分块平均，保证两个代码的初始条件**逐位相同**；
* `T = 0.1` 只覆盖风的开启段（`5 tanh(0.1) = 0.50 m/s`）——入口在 `x ≤ 0.16`、
  叶片在 `x ≈ 3`，**风还没吹到叶片**。因此本 demo 验证的是"实现是否一致"
  （标记点 + 质量点 + 标量算子 + 分块平均的初始场），不是长时间的叶片摆动。

## 发现的上游 bug / quirk（本算例特有）

1. **`leaf.concentration` 没有 `kDiff` 头行，而且整场恒为 0**：
   文件是 64 行 x 384 列，第一行就是数据（不像 `HeartTube.concentration` 那样第一行是扩散系数），
   而且所有 64x384 个值都是 `0`。后果：
   * `IBM_Driver.read_In_Concentration_Info` 用 `kDiff = float(f.readline())` 读第一行 →
     pyIB2d **开箱即崩**（`ValueError: could not convert string to float: '0.0 0.0 ...'`）；
   * MATLAB 版会把第一行数据当成 `kDiff = 0`，另外少读一行；
   * 无论哪种解释，**初始浓度恒为 0，所以浓度场永远恒为 0**（本 demo 的 compare
     结果里 `max|dC| = 0`、相对 L2 = 0，是"两边都恒 0"，不是"算子被验证"）。

   本 demo 的处理（都写在 readme 里、不修改上游文件）：
   `ib2d_io.read_concentration` 能识别"有/无头行"两种格式并把 `k` 返回 `None`；
   `main.py` 此时回退到 `input2d` 的 `kDiff = 0.01`；
   `run_reference.py` 在运行目录里补上 `kDiff` 头行（并顺带做分块平均），
   使 pyIB2d 能跑起来。因为 `C0 ≡ 0`，`kDiff` 取哪个值对结果都没有影响。
2. `give_me_source_model.m`（"简单热源" `fs = 2`）**没有被任何驱动引用**
   （matIB2d/pyIB2d 都不读它），因此没有移植；
3. `.mass` 的 id 列数是 1、pyIB2d 不输出 ghost：见 demo_571 的 readme。

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

运行时间（实测）：pyIB2d 参考 **112.4 s**、AFSI **273.0 s**（各 2001 步）、
两边各 41 帧（含 `t = 0`），`t = 0 .. 0.10005`。

| 指标（末帧 `t = 0.10005`） | 结果 |
|---|---|
| **质量点（ghost）轨迹 max\|dXm\|**（headline） | 1.95e-5 m = **1.25e-4 h**（rms 4.1e-5 h） |
| 26 个 ghost 的质心偏差 | 3.20e-6 m = 2.0e-5 h |
| **ghost−marker 偏移之差**（纯弹簧耦合） | 1.47e-5 m = 9.4e-5 h |
| 标记点偏差（33 点，前 10 个是钉死的叶柄） | 3.42e-5 m = 2.2e-4 h |
| ghost 速度偏差 `max\|dVm\|` | 9.44e-4 m/s |
| 参考解末帧 ghost 位移 | 2e-4 m ≈ 0.02 h |
| 速度峰值 AFSI / IB2d | 0.413 / 0.479（13.7%），`max\|du\|` = 0.109 |
| 浓度 `max\|dC\|` / 相对 L2 | **0 / 0**（两边初始场都恒为 0，见上） |
| 复算检验（同一标记点轨迹下的 ghost ODE） | `max\|dXm\| = 0.000e+00 m` |

**能验证**：

* 质量点算法逐位一致（复算检验 = 0）；
* 在本 demo 的（很温和的）减速流场里，**所有量都对得很好**：标记点 2.2e-4 h、
  ghost 1.25e-4 h、ghost−marker 偏移 9.4e-5 h、速度峰值差 13.7% 但绝对差只有 0.11 m/s；
* 模板的四件套（弹簧 + 梁 + 目标点 + 质量点）与被动标量能在**非方形网格**
  （`192 x 32`）上同时工作，且分块平均后的初始浓度场两个代码一致。

**不能验证（如实记录）**：

* **浓度标量本身没有被验证**：上游初始场恒为 0，所以两边都是 0；
  本 demo 只能说明"读到了同一份（恒 0）初始场、算子没炸"，不能说明标量输运算得对；
* 叶片在风中的**摆动/扑动**没有出现：`T = 0.1` 太短，风还没到达叶片
  （官方 `Tfinal = 3.25` 对应 65000 步，本机负载下不可行）；
* 官方 `384 x 64` 网格：本 demo 用 `192 x 32`（叶片的 33 个标记点本来就密于网格：
  点距 `0.0094–0.032` vs 官方 `h = 0.078`，减半网格后更是"一条线糊在几个网格里"）；
* 支撑带内的流体速度（`max|du| = 0.109 m/s`）。

## 运行

```bash
conda activate afsi-dolfinx
python run_reference.py --tend 0.1 --print-dump 50 --nx 192   # 参考解（~1.9 min）
NX=192 TFINAL=0.1 PRINT_DUMP=50 python main.py                # AFSI（~4.6 min）
python compare.py
```

其他环境变量：`NX`（默认官方 384）、`NY`、`DT`、`TFINAL`、`PRINT_DUMP`、
`OUTPUT_PATH`（默认 `./plot`）、`GRAD_DIV`。

产物：`plot/afsi_result.npz`（含 `Xm/Vm/C`）、`ib2d_reference.npz`、
`figures/compare_trajectories.png`（headline）、`compare_history.png`
（含浓度项）、`compare_fields.png`（浓度场）、`compare_table.csv`。
