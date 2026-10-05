# demo_701 — IB2d `Example_Boussinesq/Kelvin_Helmholtz_Instability/Kelvin_Helmholtz_Qualitative_2`

IB2d 官方算例（`matIB2d/Examples/Example_Boussinesq/Kelvin_Helmholtz_Instability/Kelvin_Helmholtz_Qualitative_2`）。
这是 demo_579（`…_Qualitative`）的兄弟目录：**同一套"剪切层 + 被动标量"物理，但外力只加在
5 条很窄的竖直条带上**（见"移植要点"），其余设置一致。文件名里的 `Boussinesq` 只是家族名——
上游自己的 README 明确写着"本算例**不使用** Boussinesq 近似，只是用来说明 KH 不稳定外观"
（`input2d` 里 `boussinesq = 0`）。

## 算例内容

* `1 x 0.5` 周期盒、`mu = 2e-3`、`rho = 1`、官方 `1024 x 512`、`dt = 2.5e-5`、
  官方 `Tfinal = 0.5`（2 万步、`print_dump = 400`）；本 demo 用 **`256 x 128`、`T = 0.005`
  （200 步、`print_dump = 50`、5 帧）**。
* **没有结构**：`springs = target_pts = beams = mass_pts = 0`，因此没有任何拉格朗日力；
  唯一的"标记点"是生成脚本写出的 513 个被动点（`y = 0.25` 的一条水平线），只随流体平流。
* **被动标量**：`concentration = 1`、`kDiff = 1e-5`，初始场是**四条水平浓度带**
  （`y < 0.125 → 0`、`0.125–0.25 → 1`、`0.25–0.375 → 0`、其余 `→ 1`），
  正是剪切层里被卷起的那几层。
* **外力**：`arb_ext_force = 1`，`gravity = 1`（`gvec = (0,-1)`，但无质量点，故不产生力）。

## 移植要点

1. **上游不附带任何数据文件**：目录里只有 `input2d`、`README.txt`、`main2d.m`、
   `Make_Tub_Geometry_and_Initial_Concentration.m`（生成器）、
   `please_Compute_External_Forcing.m`、`please_Form_Boussinesq_Forcing_Terms.m`、
   `test_Concentration.m`。上游 README 要求**先跑 MATLAB 生成脚本**才能得到
   `.vertex`/`.concentration`。本 demo 用 `make_input.py` **逐条复刻**该脚本的输出
   （水平线标记点 + 四条浓度带），因此两个代码读到的输入是同一份。
2. 生成脚本里有个上游怪癖：它先调用
   `give_Me_Immsersed_Boundary_Geometry_and_Concentration`（构造一个**矩形**轮廓），
   随后立刻用 `xLag = 0:2*ds:Lx; yLag = 0.25*ones(...)` **覆盖**成一条直线，
   矩形被丢弃。`make_input.py` 按最终结果（直线）生成，并在注释里写明。
3. **与 demo_579 的唯一区别是外力窗口**：兄弟算例对**每个**欧拉节点施力
   （`xMin1 = 0, xMax1 = Lx`），本算例只在
   `x ∈ [0, 0.02] / [0.23, 0.27] / [0.48, 0.52] / [0.73, 0.77] / [0.98, 1.0]` 五条竖直条带内施力
   （`kStiff = 1e4`、`w = 0.3`、`uMax = 2.5`、`uX^tar = -uMax·5·tanh(t)·(…)(…)`，
   `uY^tar = 0.075·uMax·sin(2πx/0.2)·tanh(15t)`）。已逐行移植为
   `ib2d_input/please_Compute_External_Forcing.py`，参考与 AFSI **共用同一份**。
4. `boussinesq = 0`，所以 `please_Form_Boussinesq_Forcing_Terms` 在本算例里**不会被调用**
   （上游死文件）；本 demo 也不生成 `.mesh`/`.target`（上游同样没有）。
5. 生成的数据（`.vertex`/`.concentration`）形状随网格变化：`ib2d_input/` 里提交的是
   **`NX = 256`** 的版本；换网格运行会先看到带修复命令的断言提示。

## 结果

`NX = 256`（`128 x 256` 网格）、`dt = 2.5e-5`、`T = 0.005`（200 步、5 帧）；参考约 30 s、AFSI 约 34 s。

| 指标（末帧 `t = 0.00502`） | compare.py 原文 |
|---|---|
| 标记点偏差（513 个被动点） | 8.435e-6 m |
| 速度场相对 $L_2$ | **5.00e-2** |
| `max\|u_y\|` AFSI / IB2d | 0.0599 / 0.0630（−4.9%） |
| `L2\|u_y\|` AFSI / IB2d | 2.7412 / 2.7803（**−1.41%**） |
| 浓度相对 $L_2$ | **2.14e-4** |
| 浓度 `max\|ΔC\|` | 6.67e-3（峰值 `C_max = 1`） |
| 浓度总量 `∑C` AFSI / IB2d | 16384.0148 / 16384.0150（**−0.000%**） |

**能验证**：外力条带的加载（速度场 $L_2$ 5%、`L2|u_y|` 1.4%、被动点 8e-6 m）与被动标量的
输运/守恒（相对 $L_2$ 2.1e-4、总量差 <1e-4%），即"剪切层把浓度带卷起来"的核心过程两边一致。

**不能验证**：瞬时的局部峰值（`max|u_y|` 差 4.9%）——KH 剪切层的局部极值对分辨率与
相位极敏感（官方网格是 `1024 x 512`，本 demo 用 `256 x 128`）；要复现峰值需接近官方网格。

> 家族对照：`demo_579`（`…_Qualitative`）用同样的机制但在全域施力，其 readme 里还做了
> "把 `boussinesq` 打开"的对照实验（非 upstream 配置），可一并参考。

## 运行

```bash
conda activate afsi-dolfinx
python make_input.py --nx 256                                     # 生成数据（上游不附带）
python run_reference.py --nx 256 --tend 0.005 --print-dump 50      # ~30 s
NX=256 TFINAL=0.005 PRINT_DUMP=50 python main.py                   # ~34 s
python compare.py
```
