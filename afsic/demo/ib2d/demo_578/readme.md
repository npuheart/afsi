# demo_578 — IB2d `Example_Boussinesq/Kelvin_Helmholtz_Instability/Kelvin_Helmholtz_Looks_Like_A_Cat`（Boussinesq 浮力 + 浓度）

IB2d 官方算例（`matIB2d/Examples/Example_Boussinesq/Kelvin_Helmholtz_Instability/Kelvin_Helmholtz_Looks_Like_A_Cat_For_Fun`）。
**本 demo 验证的新机制是 Boussinesq 近似**：浓度场通过浮力项与动量方程双向耦合。
这是 demo_577（Rayleigh–Taylor）之后的第二个 Boussinesq 算例。

## 1. Boussinesq 公式与实现位置（本包的移植核心）

pyIB2d 的调用点在 `IBM_Blackbox/IBM_Driver.py` 第 3 步（流体求解前）：

```python
if boussinesq_Yes:
    Fxh = Fxh + rho*fBouss_X*C      # 膨胀系数 * 重力 * 掩膜 * 浓度
    Fyh = Fyh + rho*fBouss_Y*C
```

其中 `fBouss` 在时间循环前只算一次（`please_Form_Boussinesq_Forcing_Terms`）：

```
fBouss_X = exp_Coeff * g * gravity_Info[1] * mat
fBouss_Y = exp_Coeff * g * gravity_Info[2] * mat
gravity_Info = [gravity_Yes, xG/|G|, yG/|G|]
```

* `mat` = `boussinesq.mesh` 里列出的 Eulerian 节点上的 0/1 掩膜。**本例的掩膜是整个网格**
  （生成脚本写出的 32768 = 256×128 个节点），所以 `mat ≡ 1`。
* 本例 `expansion_coeff = 500`、`y_gravity_vec_comp = -1`、`g = 9.81`
  （取自算例自带的 MATLAB 模块），因此

  ```
  F_y += rho * (-500 * 9.81) * C = -4905 * C
  ```

* `C` 用的是**上一步**的浓度（driver 在本步流体求解**之后**才更新 `C`，见 `IBM_Driver.py`
  第 822 行）；AFSI 端也是在第 3c 步用还未更新的 `C_field`。
* **上游怪癖（已在代码中复现并注释）**：`exp_Coeff == 0` 时 driver 会把它
  **悄悄改成 1.0**（而不是关闭该项）；本例为 500，不触发。

AFSI 端实现位置：`main.py`
* 循环前：读取 `boussinesq`/`expansion_coeff`/`gravity`/`x_gravity_vec_comp`/
  `y_gravity_vec_comp`，构造 `GRAVITY_INFO`，`chdir` 到 `ib2d_input` 后调用
  **同一个**移植模块 `please_Form_Boussinesq_Forcing_Terms.py`（模块自己读
  `boussinesq.mesh`），得到 `F_BOUSS`；
* 循环内第 (3c) 步（外力之后、`solver.solve_one_step()` 之前）调用 `add_boussinesq(C_field)`：
  把 `rho*fBouss_X*C`、`rho*fBouss_Y*C` 写到 Q1 节点场再插值到 Q2，加进 `solver.f`。

**验证方式**：两种代码调用的是同一个 Python 模块，因此比较只反映求解器/投影/插值差异。
`compare.py` 另外做了一次**独立核对**：末帧 `|f_y − (rho·fBouss_y·C + F_ext,y)|max = 3.79e1`，
只有 `|f|max = 2.03e4` 的 **0.19%**（残差来自用 `u^n` 而非 `u^{n−1}` 评估罚力，`O(dt)`）。

> ⚠️ **上游 bug（g 常数不一致）**：matIB2d 的
> `please_Form_Boussinesq_Forcing_Terms.m` 用 `g = 9.81`，而 pyIB2d 自带算例
> `pyIB2d/Examples/Rayleigh_Taylor_Instability/please_Form_Boussinesq_Forcing_Terms.py`
> 写的是 `g = 0.981`（差 10 倍）。demo_577 用 pyIB2d 的 0.981（那是该算例真正加载的模块），
> 本 demo 用 MATLAB 原版的 9.81；两者都忠实于各自的 upstream 文件，差别记录在案。

## 2. 算例内容

* `1.0 x 0.5` 周期盒、`mu = 0.01`、`rho = 1`、`256 x 128`、`dt = 2.5e-5`、`supp = 4`；
  官方 `T = 1.0`（40000 步）；本 demo 用 **`T = 0.005`（200 步）**，见 §5。
* **结构**：`target_pts = 0`、`springs = 0`、`beams = 0`——算例虽然带了
  `boussinesq.target`（205 个点、`k = 1e6`），但该文件**不会被读取**。
  205 个 `boussinesq.vertex` 标记点是**无力载荷的惰性示踪点**（只被流体平流），
  本 demo 用它们的走位来核对平流。
* **外力**：`please_Compute_External_Forcing.m` → 已逐行移植为 `.py`，两个代码共用。
  对**全部** Eulerian 节点施加罚力（`kStiff = 1e4`、`uMax = 1`）：

  ```
  Fx -= kStiff (uX - uX^tar),  uX^tar = ±0.25 uMax tanh(20 t)     （四条水平带）
  Fy -= 2.5 kStiff (uY - 2.5 uMax tanh(20 t))    当 y<0.025 或 y>0.475
  Fy -= kStiff uY                                 其余
  ```

  MATLAB 原版的 `%if t<0.01` 时间窗是**注释掉的**，所以外力全程施加（移植保留）。
* **浓度**：`boussinesq.concentration` 是 `(128, 256)` 的 4 条水平带（`C = 0,1,2,3`），
  `kDiff = 1e-4`（写在 `input2d` 里，不是文件首行）；标量推进沿用 demo_476/477 验证过的
  分裂 upwind 格式（`Supp.perform_Time_Split_Upwind`），用 `u^{n+1}`。
  （`input2d` 的 `advection`/`source` 开关在 pyIB2d 中**根本没有被读取**，两个代码都只跑
  一阶 upwind 平流 + 中心扩散，这一点是一致的。）

## 3. 移植要点

1. **文件格式差异**：matIB2d 的 `.concentration` **首行不是 kDiff**（kDiff 在 `input2d` 里），
   而 pyIB2d 的读取器要求首行是 kDiff。`run_reference.py::write_pyib2d_concentration`
   在写 run 目录副本时补上首行；`ib2d_io.read_concentration` 自动识别两种布局。
2. **索引基准**：matIB2d 文件按 1 起编号，`main.py` 用 `index_base=1`，
   `run_reference.py` 的 `shift_ids` 同样减 1。
3. **`boussinesq.mesh` 是 1 起编号**（MATLAB），移植模块里做了 `-1`（pyIB2d 版本对 0 起的
   数据不偏移）；两个代码都用这一个模块，因此自洽。
4. 本例 `target_pts = 0`，模板里 target 相关的分支做了空集保护（`tg is None`）。
5. 移植模板在 demo_477 基础上增加了：Boussinesq 模块加载/加力、`tg` 可缺省、
   `compare.py` 的浮力残差核对与周期性标记点偏差。

## 4. 结果（`compare.py`，末帧 `t = 0.00502`）

| 指标 | 结果 |
|---|---|
| 标记点（205 个惰性点）最大周期偏差 | 6.23e-4 m = **1.59e-1 h**（`h = 1/256`） |
| 速度场相对 $L_2$ 误差（IB 带外） | **5.10e-2** |
| 窄带外 `max\|du\|` | 1.60e-1 |
| **浮力驱动竖向速度** `max\|u_y\|` AFSI / IB2d | 0.7649 / 0.7550（**+1.31%**） |
| 竖向速度 $L_2$ 范数 AFSI / IB2d | 101.667 / 101.060（**+0.60%**） |
| 浓度相对 $L_2$ 误差 | **3.22e-3** |
| 浓度 `max\|ΔC\|` | 6.39e-2（`max C = 3`） |
| 峰值 `max C` AFSI / IB2d | 3.0 / 3.0 |
| 总量 `∑C` AFSI / IB2d（初值 46723.0） | 46828.37 / 46847.62（**−0.041%**） |
| 浮力项独立核对（末帧） | `\|f_y−(rho·fBouss_y·C+F_ext,y)\|max = 3.79e1 =` **0.19%** of `\|f\|max = 2.03e4` |

逐帧（`figures/compare_table.csv`）：`relL2(u)` 4.0e-2 → 5.1e-2，`relL2(C)` 3.5e-3 → 3.2e-3，
`max|u_y|` 的相对误差 1.5% → 1.3%，随时间稳定，没有发散趋势。

**结论**：Boussinesq 浮力耦合（幅值 −4905·C、掩膜=全网格、用上一步浓度）被正确复现；
浮力驱动的竖向速度与参考在 **1.3%** 以内，浓度误差 **0.3%**，总量守恒差 0.04%。

## 5. 不可比 / 已知限制

* **只跑了 200 步（`T = 0.005`，官方 `T = 1.0`）**。`tanh(20t)` 的罚力目标此时只升到
  `tanh(0.1) ≈ 0.0997`（约 10%），所以看到的是**线性增长段**，不是充分发展的 KH 卷起；
  本 demo 验证的是"浮力项 + 罚力 + 标量"三者的一致性，不验证长时间非线性演化。
  加大 `--tend`/`TFINAL` 即可延长（成本线性）。
* **`mu` 用官方值 0.01**（`input2d`），`dt` 也用官方 2.5e-5，未做减半/稳定性标定。
* 205 个标记点是**惰性**的（`target_pts = 0`），它们与流体之间没有反馈力；
  本 demo 的"标记点偏差"只是平流走位核对，不是结构力学的验证。
* 上游 `boussinesq.target` 文件存在但被 `target_pts = 0` 忽略——本 demo 不做任何 pinning，
  这与 upstream 行为一致（不是移植缺陷）。
* MATLAB 的 `g = 9.81` 与 pyIB2d 的 `g = 0.981` 不一致（见 §1 警示）；若用 pyIB2d 的 0.981，
  浮力会弱 10 倍，结果不可与之直接比较。

## 6. 运行（实测耗时，机器负载 ~100，比空载慢数倍）

```bash
conda activate afsi-dolfinx
python run_reference.py --tend 0.005 --print-dump 50     # 参考解，~125 s
TFINAL=0.005 PRINT_DUMP=50 python main.py                # AFSI，~389 s
python compare.py                                        # 出表 + figures/
```

输出：`plot/fluid.xdmf`、`plot/structure.xdmf`、`plot/afsi_result.npz`、
`ib2d_reference.npz`、`figures/compare_table.csv` 与 4 张对比图。
