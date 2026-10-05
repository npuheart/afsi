# demo_575 — IB2d `Example_Baby_Spider_Shower/Flow_Based`（质量点 / 单质量点的"蜘蛛"）

## 上游路径

* 数据：`third_party/ib2d/matIB2d/Examples/Example_Baby_Spider_Shower/Flow_Based/`
  （`spider.vertex/spring/beam/mass`、`please_Compute_External_Forcing.m`（已逐行移植为
  `ib2d_input/please_Compute_External_Forcing.py`）、`update_Springs.m`（**未被启用**：
  `input2d` 里 `update_springs = 0`）。
* 算法源码：`third_party/ib2d/pyIB2d/IBM_Blackbox/`。

质量点算法的完整说明见 **demo_571 的 readme**。本算例的特殊之处：它是**唯一一个
只有一个质量点**的算例，因此正好踩到 pyIB2d 的单质量点分支 bug（见下）。

## 算例内容（官方设置）

| 项 | 值 |
|---|---|
| 流体 | `1.5 x 1.0` 周期盒、`mu = 18e-6`（空气）、`rho = 1.225`、`384 x 256` |
| 时间 | `dt = 5e-5`、官方 `Tfinal = 5.0`（100000 步） |
| 结构 | 一条沿 `x = 0.25` 竖直悬挂的 90 点"蛛丝"（`y in [0.676, 0.85]`，点距 `1.95e-3`） |
| 弹簧 / 梁 | 89 根（`k = 9830.4`，`L = 1.953e-3`）、86 根不变梁（`kb = 99614.72`、`C = 0`） |
| 目标点 | 无（`target_pts = 0`，`spider.target` 里那 252 行 id 到 297 的目标点被驱动完全忽略） |
| 质量点 | **1 个**：挂在最下面的标记点（id 90）上，`k = 1e4`、`M = 0.0225` |
| 重力 | 开（`gvec = (0,-1)`，`g = 9.80665`）——蜘蛛往下掉 |
| 外力 | 开：`x in [0.05, 0.09]` 整条竖带内的水平风 `uX_tar = 0.2 tanh(5t) m/s`，`kStiff = 1e4` |

**本 demo 的缩减**：官方 `384 x 256`、`Tfinal = 5 s`（100000 步）；
本 demo 用 **`NX = 96`（`NY = 64`）**、**`T = 0.1 s`（2001 步）**，`dt = 5e-5` 与全部
物理参数保持官方值。

### 为什么必须缩小网格与时间

* 100000 步 × （384×256 网格 + 已记录的高系统负载）在本机完全不可行；
* 上游 `Tfinal = 5 s` 是"蜘蛛下落 + 被风横吹"的长时间行为；`T = 0.1 s` 覆盖了
  蛛丝的初始振荡、ghost 的自由下落（`0.5 g t^2 ≈ 0.049 m`）与风的平滑开启
  （`tanh(5t)` 在 `t = 0.1` 已达 0.46 的极限风）；
* **注意**：`dt = 5e-4`（10 倍）在本算例下**两个代码都会数值发散**
  （蛛丝的点距只有网格步长的 1/8，弹簧耦合的显式反馈在放大 `dt` 后失稳），
  所以这里保持官方 `dt`。这是本 demo 的一个实测结论。

## 发现的上游 bug（本算例直接命中）

1. **单质量点分支坏掉**（`IBM_Driver.py` 387-400）：

   ```python
   if ( mass_aux.size/3.0 < 2.0 ):        # 只有 1 个质量点
       mass_info = np.empty((1,5))
       mass_info[0,0] = mass_aux[0]
       ...
       mass_info[0,1] = xLag[mass_info[0].astype('int')]   # ← 用整行 5 列做索引
   ```

   `mass_info[0]` 是 `[id, 0, 0, 0, 0]`，用它索引 `xLag` 得到 5 个值，赋给标量 →
   `ValueError: setting an array element with a sequence`。
   本算例 `spider.mass` 只有一行、而 `spider.vertex` 有 90 行，
   `np.isscalar(xLag)` 为假 → 走上面这条坏分支 → pyIB2d **开箱即崩**
   （注意 demo_458 的 Thrips 是"1 个标记点 + 1 个质量点"，`xLag` 是 `np.float64`
   标量，`np.isscalar` 为真，所以恰好躲过了这个 bug —— 实测确认）。

   `run_reference.py` 的绕法：monkeypatch `read_Mass_Points`，在只有一行时追加一个
   **`k = 0` 的哑元 ghost**，让驱动走（正确的）通用分支。哑元必须挂在**不同的**
   标记点上：`give_Me_Mass_Lagrangian_Force_Densities` 写力用的是
   `fx[IDs] = fx[IDs] + k*(xPts - xLag[IDs])`（`=` 而非 `+=`），
   numpy 对重复索引是 **last-write-wins**，如果哑元用同一个 id，
   它那一行 0 力会把真实弹簧力覆盖掉（这个坑是实测出来的：复算检验从 0 变成 2.3e-8）。
   用不同 id 后，哑元的力恒为 0，对流体和真实 ghost 都无影响（我们只取第 0 行做对比）。
2. `.mass` 的 id 列数是 1（见 demo_571 的 readme）；
3. pyIB2d 不输出 ghost（见 demo_571 的 readme）。

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

运行时间（macOS M4，10 核，系统负载 90+）：
pyIB2d 参考 **81.8 s**、AFSI **617.4 s**（各 2001 步、41 帧，`t = 0 .. 0.10005`）。

| 指标（末帧 `t = 0.10005`） | 结果 |
|---|---|
| **蜘蛛（唯一 ghost）轨迹偏差**（headline） | 3.830e-3 m = **0.245 h** |
| ghost−marker 偏移之差（纯弹簧耦合） | 7.02e-6 m = **4.5e-4 h** |
| 蛛丝标记点（90 点）偏差 | 4.013e-3 m = 2.57e-1 h |
| ghost 速度偏差 `max\|dVm\|` | 7.88e-2 m/s |
| 参考解末帧 ghost 位移 | 4.1e-3 m = 0.26 h |
| 速度峰值 AFSI / IB2d | 9.217e-2 / 9.710e-2（**5.1%**），`max\|du\|` = 0.168 |
| 复算检验（同一标记点轨迹下的 ghost ODE） | `max\|dXm\| = 0.000e+00 m`（500 步与 2001 步都测过） |

**能验证**：质量点算法逐位一致（复算检验 = 0）；ghost−marker 偏移（纯弹簧耦合）
对到 **4.5e-4 h**；流体速度峰值对到 5.1%；蛛丝标记点的绝对偏差 0.26 h 是整族里较小的。

**不能验证（如实记录）**：

* 蜘蛛在 `T = 0.1` 内只下落约 0.004 m（参考解），因此"ghost 轨迹相对自身位移的误差"
  达到 ~90% —— 这不是机制算错（纯弹簧耦合 4.5e-4 h、复算检验 0），而是
  **位移太小、分母太小**；要真正验证"蜘蛛下落 + 风把蛛丝吹歪"的行为需要官方
  的 `Tfinal = 5 s`；
* 蛛丝附近的速度场（90 个点挤在约 11 个网格里，`max|du| = 0.168 m/s`）；
* 上游 `update_Springs.m`（变刚度/"断裂"模型）与 `spider.target` 都没启用，
  因此没有移植，也不能用本 demo 讨论它们。

### 其他需要说明的

* 官方 `Tfinal = 5 s`、`384 x 256` 的完整算例没有复现（见上"缩减"）；
* 蛛丝点距（`1.95e-3`）远小于网格步长（本 demo `h = 0.0156`，官方 `h = 3.9e-3`），
  90 个标记点挤在约 11 个网格里，力密度是"糊成一条线"的；两个代码都这样，
  但这意味着**蛛丝附近的速度场本身**分辨率很低，不适合逐点比较；
* 上游的 `update_Springs.m`（"蛛丝断裂"/变刚度模型）没有被启用（`update_springs = 0`），
  因此**没有**移植；同理 `spider.target`（252 个 id 越界的目标点）在本例被驱动忽略，
  也**没有**被使用。

## 运行

```bash
conda activate afsi-dolfinx
python run_reference.py --tend 0.1 --print-dump 50 --nx 96   # pyIB2d 参考解
NX=96 TFINAL=0.1 PRINT_DUMP=50 python main.py                # AFSI
python compare.py
```

环境变量：`NX`（默认官方 384）、`NY`、`DT`、`TFINAL`、`PRINT_DUMP`、
`OUTPUT_PATH`（默认 `./plot`）、`GRAD_DIV`。

产物：`plot/afsi_result.npz`、`ib2d_reference.npz`（含 ghost 历史与
`mass_ids/mass_M/mass_step`）、`figures/compare_trajectories.png`（headline）、
`compare_history.png`、`compare_fields.png`、`compare_table.csv`。
