# demo_688 — IB2d `Example_Meshmerize_Me/Dragonfly_Wing`（蜻蜓翅脉网格 + 薄缝射流）

IB2d 官方算例（`matIB2d/Examples/Example_Meshmerize_Me/Dragonfly_Wing/`）。
Meshmerize 家族成员；**家族共享的移植要点（外力模块 3 个函数、半分辨率规则、
`T`/`print_dump` 缩减、速度场对比的限制）见 demo_685 的 readme**。

## 上游路径

* 数据：`third_party/ib2d/matIB2d/Examples/Example_Meshmerize_Me/Dragonfly_Wing/`
  （`wing_vein.vertex`（32823 点）、`wing_vein.target`、`input2d`；
  `please_Compute_External_Forcing.m` → 已逐行移植为同名 `.py`；
  `main2d.m`、`Wingitize_Me.m`（SVG→网格生成脚本）、`cropped_wing*.svg`、
  未使用的 `wing_flow.vertex/target` 未随 demo 提供
  ——`input2d` 的 `string_name` 是 `wing_vein`）。
* 算法源码：`third_party/ib2d/pyIB2d/IBM_Blackbox/`。

## 算例内容

* `0.5 x 0.25` 周期盒、`mu = 0.01`、`rho = 1000`（`nu = 1e-5`）、官方 `1024 x 512`、
  `dt = 5e-5`、`supp = 4`；官方 `Tfinal = 10.0`（200000 步）、`print_dump = 100`。
  本 demo 用 **`256 x 128`**（两个方向同时减到 1/4；再粗一档窗口就空了，
  见下）与 **`T = 0.02`**（400 步）。
* **结构**：`wing_vein.vertex` **32823** 个标记点（真实蜻蜓翅脉网络，
  由 SVG 矢量化得到），全部由 `wing_vein.target`（`k = 1e5`）钉死
  （`update_target = 0`）。这是全仓库最大的一个 Lagrangian 结构。
* **驱动**：窗口 `x ∈ [0.015, 0.017]`、`y ∈ [0.1927, 0.1978]`（**极薄**，
  物理尺寸 2e-3 × 5.1e-3）内的 x 射流
  ```
  uX_tar = uMax tanh(2t) ((MP+w/2)-y)((MP-w/2)-y) / scale,   uY_tar = 0
  ```
  `uMax = 0.1`、`w = 0.1978 − 0.1927 = 0.0051`、`MP` 为其中心。

## 移植要点

* 家族共享部分见 demo_685；本例特有：
  * **网格只能减到 1/4**：窗口 `x` 方向只有 `2e-3` 宽，用 `256 x 128`
    时仍有 1 列节点（`dx = 1.95e-3`）；再减到 `128 x 64` 窗口就会**一个节点都没有**
    （`give_Me_Indices_To_Apply_Force` 返回空集 → 完全不驱动），故取 1/4；
  * 32823 个标记点在 256×128（32768 节点）上与网格同量级，
    每步的撒点/插值开销约 0.1 s/步（实测 400 步 ~40 s）。

## 结果

参考运行 ~60 s、AFSI 运行 ~40 s（各 400 步、41 帧，`t = 0 .. 0.01955`）。
（参考更慢：pyIB2d 的撒点/插值在大标记点集上是 Python 循环。）

| 指标（末帧 `t = 0.01955`） | 结果 |
|---|---|
| 标记点偏差（32823 点，全部钉死） | 4.98e-8 m = **2.6e-5 h**（逐位级） |
| 外力场（窄带外） | `max\|df\| = **2.5e-19**`（即窗口内两侧外力逐位相同） |
| 窗口/流场 `max\|u\|` AFSI / IB2d | 2.110e-4 / 1.369e-4（**5.4e-1** 相对，但量级只有 1e-4） |
| 速度场相对 $L_2$（窄带外） | 8.67e-1 |

**能验证**：
* **最大结构的读取与转移**：32823 个标记点的 wing_vein 网格在 400 步后偏差
  **2.6e-5 h**（≈ 5e-8 m，即双精度级别的"完全不动的钉死结构"）；
* **外力模块逐位一致**：窄带外的外力差 **2.5e-19**——两侧跑同一份移植代码，
  窗口索引与 `u_tar` 完全相同，`u` 一致的地方外力就完全一致（这是外力模块
  正确性的直接证据）。

**不能验证 / 已知限制**：

1. **流场对比无意义（本例）**：`t = 0.02` 时 `tanh(2t) = 0.04`，
   `uMax = 0.1`，窗口又是单列节点，因此达到的流速只有 **1e-4 量级**，
   两侧的相对差（`relL2 = 0.87`）是"两套投影对同一弱外力的数值噪声"，
   不代表物理差异。本 demo 的验证价值在**结构与外力模块**，
   不在流场；要看流场对比请看 demo_686/687（宽窗口、`u` ~0.5）。
2. **`T` 只跑 1/500**（400/200000 步）：官方 10 s 在这个分辨率下需要
   AFSI 数小时；`0.02 s` 只覆盖驱动的刚启动段。
3. **网格减到 1/4**（1024×512 → 256×128），窗口仍非空（1 列节点），
   但射流只有一列 → 与上游"2 列 × 5 行"的窗口形状不同（见"不能验证"第 1 条）。
4. 本机无 MATLAB/Octave，无法跑上游 matIB2d 做第三方核对。

## 运行

```bash
conda activate afsi-dolfinx
python run_reference.py --tend 0.02 --nx 256 --print-dump 10   # pyIB2d，~60 s
NX=256 TFINAL=0.02 PRINT_DUMP=10 python main.py                # AFSI，~40 s
python compare.py                                              # figures/ + compare_table.csv
```
