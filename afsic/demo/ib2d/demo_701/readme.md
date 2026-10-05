# demo_579 — IB2d `Example_Boussinesq/Kelvin_Helmholtz_Instability/Kelvin_Helmholtz_Qualitative`（KH 剪切层 + 浓度）

IB2d 官方算例（`matIB2d/Examples/Example_Boussinesq/Kelvin_Helmholtz_Instability/Kelvin_Helmholtz_Qualitative`）。
本 demo 是 demo_577/578 之后的 Boussinesq 家族第三个成员，**同时**验证：
`concentration`（被动标量）+ `arb_ext_force`（任意外力）+ `gravity` 三个开关的组合，
且没有 `mass_pts`。

## 0. 为什么选这个算例（选型说明）

`Example_Boussinesq` 下**没有**第三个"`boussinesq = 1` 且无 `mass_pts`"的算例：
该目录里 `boussinesq = 1` 的只有 `Falling_Sphere`（`mass_pts = 1`，由另一个 agent 处理）、
`Rayleigh_Taylor_Instability`（→ demo_577）和
`Kelvin_Helmholtz_Looks_Like_A_Cat_For_Fun`（→ demo_578）。
按任务规定退回次选：满足 **`concentration = 1` + `arb_ext_force = 1` + `gravity = 1`
+ `mass_pts = 0`** 的候选是 `Kelvin_Helmholtz_Qualitative` 与 `Kelvin_Helmholtz_Qualitative_2`
（都在 `Example_Boussinesq` 下）。`Example_Concentration_Dynamics` 下的 7 个目录 **没有**
一个同时满足该组合（`Leaf_64` 满足 concentration+arb_ext 但 `mass_pts = 1`，其余
`gravity = 0` 且 `arb_ext_force = 0`），故不选。这里选 `Kelvin_Helmholtz_Qualitative`
（`1024 x 512 / dt = 1e-4`，比 `_2` 的 `dt = 2.5e-5` 便宜）。

> **注意**：本算例自己的 `README.txt` 明说 "*This example doesn't use the Boussinesq
> approximation - just for illustrative purposes*"，`input2d` 里 `boussinesq = 0`。
> 因此主运行**按 upstream 原样**（不启用浮力）验证；浮力耦合本身由 demo_577/578 验证。

## 1. 数据生成（本算例不附带数据文件）

upstream 只给了 MATLAB 脚本 `Make_Tub_Geometry_and_Initial_Concentration.m`，其开头写着
"Before running main2d.m, please run ... to create the input files"。本 demo 把它逐行移植为
`make_geometry.py`，生成（1 起编号，matIB2d 布局）：

| 文件 | 内容 |
|---|---|
| `boussinesq.vertex` | y = 0.15 / 0.25 / 0.35 三条水平线，`x = 0 : 2ds : Lx`（`ds = 0.25 dx`），256x128 时 1539 点 |
| `boussinesq.target` | 1 起 id + `k = 1e6`（**`target_pts = 0`，文件写了但不会被读**） |
| `boussinesq.concentration` | `(Ny, Nx)` 四带浓度 `C ∈ {0,1,2,3}`，界面为正弦扰动（`Amp = 0.025`、`λ = 0.2`、`OffY = 0.025`），**首行不是 kDiff** |
| `boussinesq.mesh` | Boussinesq 掩膜（此处 = 全网格 32768 点） |

**上游 bug（已复现并注释）**：`print_Boussinesq_Mesh` 的守卫写成
`if ( ( yInd < 0.425 ) || ( yInd > 0.025 ) )`，其中 `yInd` 是**整数索引**，条件恒为真，
于是整张网格都被写出；原本想写的浮点带状条件没有生效。移植按实际行为复现。

## 2. 算例内容（主运行，`boussinesq = 0`）

* `1.0 x 0.5` 周期盒、`mu = 0.002`、`rho = 1`、官方 `1024 x 512`（本 demo 降到 **`256 x 128`**）、
  `dt = 1e-4`、`supp = 4`；官方 `T = 1.0`（10000 步）→ 本 demo 用 **`T = 0.02`（200 步）**。
* **外力**（`please_Compute_External_Forcing.m` → 已逐行移植）作用于**全部**节点，
  `kStiff = 1e4`、`uMax = 2.5`：

  ```
  uX^tar = 0.25 uMax tanh(50 (·)) tanh(20 t)     （五条剪切带，界面在 y=0.05,0.15,0.25,0.35,0.45）
  uY^tar = 0.075 uMax sin(2πx/0.2) tanh(15 t)
  Fx -= kStiff (uX - uX^tar);  Fy -= kStiff (uY - uY^tar)
  ```

  MATLAB 的 `%if t<0.01` 是被注释掉的，外力全程施加（移植保留）。
* **浓度**：`kDiff = 1e-5`（在 `input2d` 里），用 demo_476/477 已 1e-11 验证过的分裂 upwind
  格式推进（`u^{n+1}`）。`input2d` 的 `advection = 1`（注释说是 3 阶 WENO）与 `source` 开关
  **pyIB2d 根本不读**，两个代码都只跑一阶 upwind + 中心扩散，这一点两边一致。
* **结构**：`target_pts = springs = beams = 0`，1539 个标记点是**惰性示踪点**（无反馈力），
  用来核对平流（它们在周期盒里会绕回，比较用最小镜像）。

## 3. 移植要点

1. **没有 upstream 数据** → `make_geometry.py` 必须先生成；分辨率必须与运行一致
   （`--nx/--ny` 对 `NX/NY`）。
2. 1 起编号 → `main.py` 用 `index_base=1`；`.concentration` 无 kDiff 首行 →
   `run_reference.py::write_pyib2d_concentration` 补首行，`ib2d_io.read_concentration` 自动识别。
3. `boussinesq.mesh` 是 1 起编号，移植模块 `-1`。
4. 模板把 target 分支改成可缺省（本算例 `tg is None`）。
5. `main.py` / `compare.py` / `run_reference.py` 都支持 `BOUSSINESQ` / `--boussinesq` 覆盖，
   便于跑"启用浮力"的对照实验（见 §5）。

## 4. 结果（`compare.py`，末帧 `t = 0.0201`）

| 指标 | 结果 |
|---|---|
| 标记点（1539 个惰性点）最大周期偏差 | 1.70e-5 m = **4.3e-3 h**（`h = 1/256`） |
| 速度场相对 $L_2$ 误差（IB 带外） | **4.71e-3** |
| 窄带外 `max\|du\|` | 6.54e-3 |
| **剪切驱动竖向速度** `max\|u_y\|` AFSI / IB2d | 0.0546 / 0.0546（差 **< 0.05%**） |
| 竖向速度 $L_2$ 范数 AFSI / IB2d | 6.9901 / 6.9901（**−0.001%**） |
| 速度峰值 `max\|u\|` AFSI / IB2d | 0.236 / 0.234（+0.9%） |
| 浓度相对 $L_2$ 误差 | **8.28e-6** |
| 浓度 `max\|ΔC\|` | 3.86e-4（`max C = 3`） |
| 峰值 `max C` AFSI / IB2d | 3.0 / 3.0 |
| 总量 `∑C` AFSI / IB2d（初值 49152.0） | 49153.9114 / 49153.9108（**+0.000%**） |

逐帧（`figures/compare_table.csv`）：`relL2(u)` = 4.85e-3 / 4.75e-3 / 4.72e-3 / 4.71e-3
（随时间基本不变），`relL2(C)` 1.9e-7 → 8.3e-6，`max|u_y|` 逐帧完全吻合到 4 位有效数字。

**结论**：这套算例里两种代码的一致性极高（速度 0.5%、浓度 1e-5、总量差 0）——
因为它**没有**罚力钉死结构、没有 `mass_pts`，流动是光滑的周期剪切，
所以它成了"任意外力 + 被动标量 + 周期性 IB 插值"链路的干净基准。

## 5. 对照实验：把 `boussinesq` 打开（非 upstream 配置）

上游虽置 `boussinesq = 0`，但算例自带 `expansion_coeff = 250` 和
`please_Form_Boussinesq_Forcing_Terms.m`，`boussinesq.mesh` 又是全网格，
因此可以（**明确标注为偏离 upstream 的对照**）用
`--boussinesq 1` / `BOUSSINESQ=1` 跑同一套设置，看看浮力耦合是否同样吻合：

```bash
python run_reference.py --nx 256 --tend 0.02 --print-dump 50 --boussinesq 1 \
    --out ib2d_reference_boussinesq.npz --run ib2d_run_boussinesq
BOUSSINESQ=1 NX=256 TFINAL=0.02 PRINT_DUMP=50 OUTPUT_PATH=./plot_boussinesq python main.py
BOUSSINESQ=1 AFSI_NPZ=plot_boussinesq/afsi_result.npz python compare.py \
    ib2d_reference_boussinesq.npz figures_boussinesq
```

此时 `F_y += rho · exp_Coeff · g · g_y · mat · C = -2452.5 C`（`g = 9.81`，MATLAB 原版）。
实测（末帧 `t = 0.0201`，`figures_boussinesq/compare_table.csv`）：

| 指标 | 主运行（`boussinesq = 0`） | 对照（`boussinesq = 1`） |
|---|---|---|
| `max\|u_y\|` IB2d | 0.0546 | **0.5272**（浮力把竖向速度放大 ~10 倍） |
| `max\|u_y\|` AFSI / IB2d | 0.0546 / 0.0546（<0.05%） | 0.6034 / 0.5272（**+14.4%**） |
| `L2\|u_y\|` AFSI / IB2d | −0.001% | 68.863 / 68.793（**+0.102%**） |
| 速度场 `relL2(u)`（IB 带外） | 4.71e-3 | 4.93e-2 |
| 浓度 `relL2(C)` / `max\|ΔC\|` | 8.3e-6 / 3.9e-4 | 3.7e-3 / 6.3e-2 |
| 总量 `∑C` AFSI / IB2d | +0.000% | 49878.15 / 49901.55（**−0.047%**） |
| 标记点最大周期偏差 | 1.70e-5 m | 1.79e-3 m（`= 0.46 h`） |
| 浮力项残差 | —（未开） | **0.148%** of `\|f\|max = 4.15e3` |

结论：**打开浮力后**两种代码的 `L2` 级量（`L2|u_y|`、`∑C`、`relL2(u)`≈5%）仍然吻合，
浮力项本身被逐节点复现（残差 0.15%）；但 **`max|u_y|` 的峰值差 14%**——
因为 `-2452.5 C` 这个强体力与 `kStiff = 1e4` 的 y 向罚力在局部形成刚性平衡，
两种代码对"强体力 + 罚力"的响应在峰值点上不同（`L2` 上只有 0.1%）。
这与 demo_578 的结论一致：**量级都是 1%–10%，`L2` 指标远好于 `max` 指标**。

## 6. 不可比 / 已知限制

* **分辨率降低 16 倍**（`1024 x 512 → 256 x 128`）：官方分辨率需要
  `ds = 0.25 dx`、`dt = 1e-4`、10000 步，单次运行时间不可接受；降低分辨率不改变
  无量纲设置（剪切带界面、扰动波长都用绝对长度给出），但不是官方分辨率下的定量结果。
* **时间只跑了 2%**（`T = 0.02` vs 官方 1.0）：`tanh(20t)` 的目标速度此时只到 ~0.33，
  `tanh(15t)` 到 ~0.26，即仍在线性上升段；看到的是"罚力建立剪切"的过程，
  不是充分发展的 KH 卷起。
* **标记点是惰性的**（`target_pts = 0`），不构成对结构力学的验证。
* `advection = 1`（WENO）与 `source` 被 pyIB2d 忽略，两个代码都跑一阶 upwind；
  因此本 demo 的"浓度一致性"只覆盖**一阶 upwind** 格式，不代表 WENO 分支。
* 主运行 `boussinesq = 0`，所以 §4 的数字**不能**用来验证浮力项；浮力项的验证在
  demo_577/578（以及 §5 的对照实验）。

## 7. 运行（实测耗时，机器负载 ~100，比空载慢数倍）

```bash
conda activate afsi-dolfinx
python make_geometry.py --nx 256 --ny 128                    # 上游数据生成器，~1.3 s
python run_reference.py --nx 256 --tend 0.02 --print-dump 50  # 参考解，~3 min
NX=256 TFINAL=0.02 PRINT_DUMP=50 python main.py               # AFSI，~171 s
python compare.py                                             # 出表 + figures/
```

输出：`plot/fluid.xdmf`、`plot/structure.xdmf`、`plot/afsi_result.npz`、
`ib2d_reference.npz`、`figures/compare_table.csv` 与 4 张对比图。
