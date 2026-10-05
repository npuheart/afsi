# demo_577 — IB2d `Rayleigh_Taylor_Instability`（Boussinesq 浮力耦合，主验证）

IB2d 官方算例（pyIB2d `Examples/Rayleigh_Taylor_Instability`，等价于 matIB2d
`Examples/Example_Boussinesq/Rayleigh_Taylor_Instability`）。
**本 demo 是这次任务的核心：把 IB2d 的 Boussinesq 近似移植进 AFSI**，
即浓度场通过浮力项进入动量方程（浮力 ↔ 对流-扩散双向耦合）。

## 1. Boussinesq 公式与实现位置（逐条对照 upstream）

### upstream（`pyIB2d/IBM_Blackbox/IBM_Driver.py`）

* 循环前（约 570–600 行，`if boussinesq_Yes:`）：

  ```python
  if exp_Coeff == 0:              # ← 上游怪癖
      exp_Coeff = 1.0             #   0 被悄悄改成 1.0（而不是关闭浮力）
  fBouss_X, fBouss_Y = please_Form_Boussinesq_Forcing_Terms(exp_Coeff, Nx, Ny, gravity_Info)
  ```

  其中（`please_Form_Boussinesq_Forcing_Terms.py`）：

  ```
  g     = 0.981                       # ← pyIB2d 的硬编码值（matIB2d 用 9.81，见 §6）
  mat[yInd, xInd] = 1                 # boussinesq.mesh 列出的 Eulerian 节点
  fBouss_X = exp_Coeff * g * gravity_Info[1] * mat
  fBouss_Y = exp_Coeff * g * gravity_Info[2] * mat
  ```

  `gravity_Info = [gravity_Yes, xG/|G|, yG/|G|]`（`IBM_Driver.py` 约 510 行，
  **归一化**后的重力向量）。

* 时间循环第 3 步（流体求解**之前**，约 765 行）：

  ```python
  if boussinesq_Yes:
      Fxh = Fxh + rho*fBouss_X*C      # C 是"上一步"的浓度：
      Fyh = Fyh + rho*fBouss_Y*C      # driver 在本步流体求解之后(第 822 行)才更新 C
  ```

  公式即

  ```
  F_x += rho · exp_Coeff · g · g_x · mat · C
  F_y += rho · exp_Coeff · g · g_y · mat · C
  ```

### AFSI（`main.py`）

* 循环前：读 `boussinesq` / `expansion_coeff` / `gravity` / `x_gravity_vec_comp` /
  `y_gravity_vec_comp`，构造 `GRAVITY_INFO`（含归一化与 `exp_Coeff==0 → 1.0` 的上游怪癖，
  代码中有注释），`chdir` 到 `ib2d_input`（模块从 cwd 读 `boussinesq.mesh`，与 driver 的行为一致）
  后调用**同一个**模块 `please_Form_Boussinesq_Forcing_Terms.py` 得到 `F_BOUSS`。
* 循环内第 **(3c)** 步——在 `add_external_force` 之后、`solver.solve_one_step()` 之前，
  与 upstream 第 3 步位置一致——调用 `add_boussinesq(C_field)`：

  ```python
  fx = rho * F_BOUSS[0] * C      # (Ny, Nx) 节点数组，C 为"上一步"浓度
  fy = rho * F_BOUSS[1] * C
  f_b1.x.array[:] = np.column_stack([fx[gj, gi], fy[gj, gi]]).ravel()   # Q1 节点场
  f_b2.interpolate(f_b1)                                               # → Q2
  solver.f.x.array[:] += f_b2.x.array[:]
  ```

* 本例数值：`expansion_coeff = 10`、`x_gravity_vec_comp = 0`、`y_gravity_vec_comp = -1`、
  `g = 0.981`（pyIB2d）→

  ```
  fBouss_X = 0         fBouss_Y = -9.81·mat          (mat = 1 在 23793 个节点上)
  F_y += rho · (-9.81) · C                           （掩膜外无浮力）
  ```

* 掩膜来自 `boussinesq.mesh`（0 起编号，pyIB2d 布局）：
  `x ∈ [13, 115]`、`y ∈ [13, 243]`（103 × 231 = 23793 个节点）。

### 独立的力级核对（`compare.py`）

末帧在**外力为零**的 21825 个掩膜节点上：

```
median( f_y / C ) = -9.810000        （应与 rho·exp_Coeff·g·g_y = -9.81 完全相等）
|f_y − (rho·fBouss_y·C + F_ext,y)|max = 7.38e-3   = 0.0013% of |f|max = 5.79e2
```

即：浮力项被**逐节点**复现（残差来自用 `u^n` 而非 `u^{n−1}` 评估罚力，`O(dt)`）。

## 2. 算例内容

* `0.5 x 1.0` 周期盒、`mu = 0.01`、`rho = 1`、`128 x 256`、`dt = 1.25e-5`、`supp = 4`；
  官方 `T = 2.0`（160000 步、`print_dump = 800`），本 demo 用 **`T = 0.012`（960 步）**，见 §5。
* **结构**：2660 个 `k = 1e7` 的**钉死目标点**围出"水槽"边界（`x ∈ [0.05, 0.45]`、
  `y ∈ [0.05, 0.95]`，矩形一圈）；
* **外力**（算例自带 `please_Compute_External_Forcing.py`，两个代码共用）：
  窗口 `x ∈ [0.2, 0.3]`、`y ∈ [0.48, 0.52]`，`kStiff = 1e4`、`uMax = 10`，
  目标速度 `uX^tar = 0, uY^tar = -10 tanh(20t)`，**只在 `t < 0.01` 施加**——这是给
  界面一个向下的初始扰动；
* **浓度**：`boussinesq.concentration` 首行 `kDiff = 1e-6`，其余为 `(256, 128)` 场
  （`C ∈ [0, 1]`，`∑C = 20871.5`）：底部/顶部是重流体（`C = 1`），中间一段轻流体
  （`C = 0`），在 `y ≈ 0.38–0.63` 用双曲正切过渡连接（重上轻下 → RTI），
  两侧边界列 `x = 0, 127` 也填 `C = 1`（轴向/展向边界效应）。

## 3. 结果（`compare.py`）

| 指标（末帧 `t = 0.01126`；参考解同帧 `t = 0.01125`） | 结果 |
|---|---|
| 标记点（2660 个钉死点）最大周期偏差 | 7.46e-7 m = **1.9e-4 h**（`h = 1/256`） |
| 速度场相对 $L_2$ 误差（IB 带外） | **2.61e-2** |
| 窄带外 `max\|du\|` | 1.30e-1 |
| **竖向速度** `max\|u_y\|` AFSI / IB2d | 1.8957 / 1.9102（**−0.76%**） |
| 竖向速度 $L_2$ 范数 AFSI / IB2d | 50.797 / 50.925（**−0.25%**） |
| 浓度相对 $L_2$ 误差 | **2.44e-3** |
| 浓度 `max\|ΔC\|` | 4.78e-2（`max C = 1`） |
| 峰值 `max C` AFSI / IB2d | 1.0 / 1.0 |
| 总量 `∑C` AFSI / IB2d（初值 20871.5） | 20878.87 / 20871.82（**+0.034%**） |
| **浮力项逐节点核对**（外力=0 的 21825 节点） | `median(f_y/C) = -9.810000`（= `rho·exp_Coeff·g·g_y`）；残差 **0.0013%** of `\|f\|max` |

逐帧（`figures/compare_table.csv`，10 帧）：`relL2(u)` 从 2.6e-1（第 1 帧，射流刚启动的瞬态）
单调降到 **2.6e-2**；`max|u_y|` 的相对误差始终 **≤ 1.5%**；`relL2(C)` ≤ 2.4e-3。

## 4. 关于"浮力驱动"的量级（重要、如实说明）

本例的浮力加速度是 `a_y = -9.81·C`（`C ∈ [0,1]`），**量级只有 ~10 m/s²**。
而 `t < 0.01` 的向下罚力射流把目标速度直接拉到 `-10 tanh(20t)`（t=0.01 时约 −1.9 m/s），
实际流场里 `max|u_y|` 在 t=0.0113 时是 1.9 m/s。浮力在 0.011 s 内能贡献的
速度尺度只有 `9.81 × 0.0113 ≈ 0.11 m/s`（约为 `max|u_y|` 的 **6%**）。

因此：

* **浮力项的正确性**由 §1 的**力级核对**（`median(f_y/C) = -9.81` 精确相等、
  残差 0.0013%）直接证明——这是最强的证据；
* **浮力的动力学后果**（把浮力差异放大到速度场上）在本窗口内被射流淹没，
  在 **demo_578**（`F_y = −4905·C`，浮力驱动的 `u_y ≈ −0.77 m/s`）里得到
  **1.3% 以内**的速度场验证。本 demo 的速度/浓度指标更多验证
  "浓度 ↔ 浮力 → 力 → 速度"整条链路的**数值一致性**，而不是"浮力导致 RTI 增长"。

## 5. 不可比 / 已知限制

* **时间只跑了官方值的 0.6%**（`T = 0.012` vs `T = 2.0`）。官方 160000 步在本机
  （负载 ~100）需要数十小时；本 demo 覆盖的是**浮力加速 + 射流扰动**的早期线性段。
  RTI 自身的指数增长（`σ ≈ sqrt(A g k)`，本设置下量级 10–35 s⁻¹）在 0.011 s 内
  只有 `e^{0.1~0.4}` 的增长，**没有**发展到卷起（mushroom/roll-up）。
  要看真正的 RTI 演化需 `T ≳ 0.1`（8000 步），命令：
  `python run_reference.py --tend 0.1 --print-dump 1000` + `TFINAL=0.1 PRINT_DUMP=1000 python main.py`
  （空载约 15 min + 8 min，本机高负载下约 10 倍）。
* **网格用官方值 128 × 256、`dt` 用官方值 1.25e-5**（未减半、未加粗）。
* **`g` 的上游不一致**（见 §6）使本例的浮力比 MATLAB 版本**弱 10 倍**；
  本 demo 与 pyIB2d 参考一致，但**不能**与 MATLAB 版做定量比较。
* `x = 0` 与 `x = 127` 两列 `C = 1` 但在掩膜之外（掩膜从 13 开始），
  所以这两列没有浮力——这是 upstream 的几何设定，不是移植差异。
* 掩膜边界的浮力是**台阶式**的（`mat` 0/1 突变），AFSI 的 Q1→Q2 插值与 IB2d 的
  节点式相加在掩膜边缘会有 `O(h)` 差异；`relL2(u) = 2.6e-2` 主要来自这里和罚力/投影差异。

## 6. 发现的上游 bug

**`g` 常数不一致（10 倍）**：

| 文件 | 值 |
|---|---|
| `matIB2d/Examples/Example_Boussinesq/Rayleigh_Taylor_Instability/please_Form_Boussinesq_Forcing_Terms.m` | `g = 9.81` |
| `pyIB2d/Examples/Rayleigh_Taylor_Instability/please_Form_Boussinesq_Forcing_Terms.py` | `g = 0.981` |

pyIB2d 的移植把 9.81 打成了 0.981，浮力弱了 10 倍（本 demo 因为要与 pyIB2d 参考对比，
必须复现 0.981；`main.py`/`ib2d_input` 里都保留了 pyIB2d 的原文件）。
demo_578/579 用的是 MATLAB 原版模块（`g = 9.81`）。

另外（已复现/注释，不算 bug 但很隐蔽）：

* `exp_Coeff == 0` 被 silently 改成 1.0（`IBM_Driver.py`）；
* driver 里"必须有 gravity + concentration"的检查被**注释掉**了，
  所以 `boussinesq = 1` 而不开两者时不会报错（AFSI 端改成了显式 `assert`）。

## 7. 运行（实测耗时；当时机器负载 ~100，远高于空载）

```bash
conda activate afsi-dolfinx
python run_reference.py --tend 0.012 --print-dump 100     # 参考解，216 s
TFINAL=0.012 PRINT_DUMP=100 python main.py                # AFSI，1367 s
python compare.py                                         # 出表 + figures/
```

输出：`plot/fluid.xdmf`、`plot/structure.xdmf`、`plot/afsi_result.npz`、
`ib2d_reference.npz`、`figures/compare_table.csv` 与 4 张对比图。
（`compare.py` 支持 `AFSI_NPZ=<path>` 覆盖输入、`BOUSSINESQ=0/1` 覆盖浮力开关以便做对照实验。）
