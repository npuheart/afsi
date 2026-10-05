# demo_689 — IB2d `Example_Meshmerize_Me/Insect_Wing_Veins`（昆虫翅脉网格 + 薄缝射流）

IB2d 官方算例（`matIB2d/Examples/Example_Meshmerize_Me/Insect_Wing_Veins/`）。
Meshmerize 家族成员；**家族共享的移植要点（外力模块 3 个函数、半分辨率规则、
`T`/`print_dump` 缩减、速度场对比的限制）见 demo_685 的 readme**。

## 上游路径

* 数据：`third_party/ib2d/matIB2d/Examples/Example_Meshmerize_Me/Insect_Wing_Veins/`
  （`wing_vein.vertex`（32823 点）、`wing_vein.target`、`input2d`；
  `please_Compute_External_Forcing.m` → 已逐行移植为同名 `.py`；
  `main2d.m`、`Wingitize_Me.m`（SVG→网格生成脚本）、`cropped_wing*.svg`、
  未使用的 `wing_flow.vertex/target` 未随 demo 提供
  ——`input2d` 的 `string_name` 是 `wing_vein`）。
* 算法源码：`third_party/ib2d/pyIB2d/IBM_Blackbox/`。

## 算例内容

* `0.5 x 0.25` 周期盒、**`mu = 0.1`**（demo_688 是 0.01）、`rho = 1000`、
  官方 `1024 x 512`、`dt = 5e-5`、`supp = 4`；官方 **`Tfinal = 1.0`**
  （20000 步，demo_688 是 10）→ 本 demo 用 **`256 x 128`** 与 **`T = 0.02`**。
* **结构**：与 demo_688 **同一份** `wing_vein.vertex/target`（32823 点，
  全部由 `k = 1e5` 的目标点钉死）。两个算例的区别是
  `cropped_wing.svg`（网格来源）与 `mu`/`Tfinal`；上游把同一份网格数据放在两个目录里。
* **驱动**：与 demo_688 **完全相同**的薄缝 x 射流
  （`uMax = 0.1`、`w = 0.0051`、窗口 `x ∈ [0.015, 0.017]`、`y ∈ [0.1927, 0.1978]`）。

## 移植要点

* 家族共享部分见 demo_685；本例特有：`mu = 0.1`（`nu = 1e-4`，
  比 688 大 10 倍）、`Tfinal = 1.0`（官方，本 demo 取 `T = 0.02`）、
  以及它与 688 共用同一份移植后的外力模块（常数完全相同，因此
  两个 demo 的 `ib2d_input/please_Compute_External_Forcing.py` 只有 docstring 不同）。

## 结果

参考运行 ~61 s、AFSI 运行 ~40 s（各 400 步、41 帧，`t = 0 .. 0.01955`）。

| 指标（末帧 `t = 0.01955`） | 结果 |
|---|---|
| 标记点偏差（32823 点，全部钉死） | 2.57e-8 m = **1.3e-5 h**（逐位级） |
| 外力场（窄带外） | `max\|df\| = **4.2e-19**`（窗口内两侧外力逐位相同） |
| 流场 `max\|u\|` AFSI / IB2d | 8.056e-5 / 8.905e-5（**9.5e-2** 相对，量级 1e-4） |
| 速度场相对 $L_2$（窄带外） | 7.86e-1 |

**能验证**：与 demo_688 相同（32823 点结构逐位级吻合、外力模块逐位一致）；
本例 `mu` 大 10 倍，弱驱动段的速度幅值差从 54% 降到 9.5%。

**不能验证 / 已知限制**：与 demo_688 相同（流场处于 1e-4 数值噪声量级、
`T` 只跑 1/50、网格减到 1/4、无 MATLAB 第三方核对），
另外 `nu` 大 10 倍使粘性更主导，`T = 0.02` 覆盖的物理更"短"。

## 运行

```bash
conda activate afsi-dolfinx
python run_reference.py --tend 0.02 --nx 256 --print-dump 10   # pyIB2d，~61 s
NX=256 TFINAL=0.02 PRINT_DUMP=10 python main.py                # AFSI，~40 s
python compare.py                                              # figures/ + compare_table.csv
```
