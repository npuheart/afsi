# demo_675 — IB2d `Example_Celestial_Mechanics/Planets`（**上游本身不可运行**）

## 上游路径

* 数据/脚本：`third_party/ib2d/matIB2d/Examples/Example_Celestial_Mechanics/Planets/`
  （只有 `input2d`、`main2d.m`、**`Planets.m`**，**没有任何 `plantes.*` 数据文件**）。
* 算法源码：`third_party/ib2d/pyIB2d/IBM_Blackbox/{IBM_Driver.py, Supp.py,
  please_Find_Lagrangian_Forces_On_Eulerian_grid.py}`。
* 代码基：`afsic/demo/ib2d/_template/`。

## 结论：该上游算例损坏，无法复现（保留忠实移植 + 失败证据）

`Planets.m` 的**唯一职能**是在 MATLAB 端生成 `BeamCurve.vertex/.beam/.target`
（和 `Example_Wobbly_Beam` 用的是同一个"生成器"函数），因此没有 MATLAB 就无法
得到数据；本 demo 用 `ib2d_input/Planets.py` **逐行移植**该生成器，产出与
`Planets.m` 完全相同的 `plantes.*` 文件。用这些文件跑两个求解器，**两边都在
第 1 步发散成 NaN**。

### 证据 1：上游数据文件缺失 + 名称不一致

* 目录里只有 `Planets.m`，没有 `.vertex/.beam/.target`；
* 生成器内部硬编码 `struct_name = 'BeamCurve'`，而 `input2d` 写的是
  `string_name = plantes`（**没有引号**的笔误）——两者不一致，按原样跑
  MATLAB 会找不到 `plantes.*`。本 demo 的 `Planets.py` 默认按 `input2d` 的
  `string_name` 输出（即 `plantes.*`），使 `input2d` 保持逐字不变。

### 证据 2：生成的几何跨越周期缝、且端点恰好落在边界 y = 0

`Planets.m` 的几何是 `y = −2(x² − (a/2)²)`，`x ∈ [−a/2, a/2] = [−0.25, 0.25]`
（腾挪到盒子中央的那行 `xLag(1) = Lx/2 - a/2` 被**注释掉**了）：

* `x ∈ [−0.25, 0]` 的 32 个标记点在周期盒里回绕到 `x + 1`，于是梁链上出现
  一条跨度为 `0.75` 的"梁"，`kb·S` 立刻爆掉；
* 抛物线两端的 `y` 恰好等于 `0`（生成器显式写 `yLag(1) = 0`，另一端同值），
  正好压在周期边界上。

### 证据 3：两个求解器都在第 1 步 NaN（本机实测）

以官方设置（`32 x 32`、`dt = 5e-5`、`Tfinal = 0.05`）跑：

| 求解器 | 结果 |
|---|---|
| pyIB2d（`run_reference.py`） | 第 1 帧起 `X`、`u` 全为 `NaN`（`t = 0.0005`） |
| AFSI（`main.py`） | step 11（`t = 0.00055`）标记点位移 `1.00`（跨盒回绕）、`max\|u\| = 1158.6`；step 21 起 NaN |
| `compare.py` | 表格 `dXmax/h`、`max\|u\|`、`max\|du\|`、`relL2`、`df` **全为 NaN** |

### 证据 4：不是 dt 太小 / 太大，也不是"移植写错"（对照实验，均为 pyIB2d 参考解）

用同一台机器、`NX = 32` 做对照（把**可运行的** `demo_673` 的 `BeamCurve` 数据
逐项替换）：

| 配置 | 结果 |
|---|---|
| demo_673 原数据（拱在 `y ∈ [0.5, 0.625]`，`kb = 7.5e9`） | 正常 |
| 上述数据整体 `y −= 0.49`（`y_min = 0.01`） | 正常 |
| 上述数据整体 `y −= 0.50`（`y_min = 0`，即端点压在边界上） | **NaN** |
| 上述数据整体 `x −= 0.50`（跨越周期缝） | **NaN** |
| Planets 生成几何（`x ∈ [−0.25, 0.25]`，`kb = 7.5e10`） | **NaN** |
| 同一几何"搬进盒内"（`x += 0.5`）后 | 仍 **NaN**（`kb` 太硬） |
| 同一几何 + `kb = 7.5e9` + `y += 0.01` | 前 3 帧 `max\|u\| = 0 → 10.3 → 15.6`，随后 **NaN** |
| 同一几何但关掉梁（`beams = 0`，只留 2 个目标点） | 正常（`u ≡ 0`） |

另外把 `dt` 从官方 `5e-5` 一路减到 `1e-6`（步数 ×50）仍是第 1 步 NaN，
说明失稳来自**几何跨界 + `kb = 7.5e10` 过硬**，不是时间步选错。

### 判定

按任务约定"若某目录已经移植过或不可移植，则报告并跳过"：**demo_675 判定为
上游损坏、不可复现**。目录保留为忠实记录（生成器移植 + 上述证据），
`compare.py` 仍可运行但输出全 NaN。

## 缩减设置

无（官方 `32 x 32`、`dt = 5e-5`、`Tfinal = 0.05` 已足够小；失败与网格/步长无关）。

## 结果

| 指标 | 结果 |
|---|---|
| 结构 | 64 标记点 + 62 不变梁（`kb = 7.5e10`，`C = 0`）+ 2 目标点（`k = 2e8`） |
| AFSI 首次失稳 | step 11，`max\|u\| = 1158.6`，`t = 0.00055` |
| AFSI 起 NaN | step 21，`t = 0.00105` |
| pyIB2d 起 NaN | 第 1 帧，`t = 0.0005` |
| `compare.py` | 全部指标 `NaN`（`dXmax/h`、`max\|u\|`、`max\|du\|`、`relL2`、`df`） |

**能验证**：

* `Planets.py` 忠实复刻 `Planets.m` 的几何与输出格式（64 点抛物线、
  `kb = 7.5e10`、`k_target = 2e8`、`x ∈ [−0.25, 0.25]`、`y ∈ [0, 0.125]`）；
* 该数据下 AFSI 与 pyIB2d **都**立即发散（两个独立实现的一致"失败"）；
* 对照实验表明失稳由数据（跨界 + 过硬）引起，而非移植错误。

**不能验证（如实记录）**：

* 一切物理量（`max|dX|`、`max|u|`、`max|du|`、`relL2`、`df`）：全 NaN；
* 网格收敛性、`dt` 收敛性：无法讨论（解不存在）；
* 上游原作者期望的"天体力学"构型：`Planets.m` 与随附数据/名称均不一致，
  无法从仓库推断其本意。

## 运行

```bash
conda activate afsi-dolfinx
python ib2d_input/Planets.py                                            # 生成 plantes.*
python run_reference.py --tend 0.05 --dt 5e-5 --print-dump 10 --nx 32   # 参考解（第 1 帧 NaN）
NX=32 DT=5e-5 TFINAL=0.05 PRINT_DUMP=10 python main.py                  # AFSI（step 21 起 NaN）
python compare.py                                                       # 全 NaN 表格
```

产物：`plot/afsi_result.npz`、`ib2d_reference.npz`（均含 NaN）、
`figures/compare_*.png`、`figures/compare_table.csv`。
