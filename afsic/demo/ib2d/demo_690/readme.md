# demo_690 — IB2d `Example_Meshmerize_Me/old`（旧版叶脉网格 + 流槽射流）

IB2d 官方算例（`matIB2d/Examples/Example_Meshmerize_Me/old/`）。
Meshmerize 家族成员；**家族共享的移植要点（外力模块的 3 个函数、半分辨率规则、
`T`/`print_dump` 缩减、速度场对比的限制）见 demo_685 的 readme**，
本文只写本例特有的设置与结果。

## 上游路径

* 数据：`third_party/ib2d/matIB2d/Examples/Example_Meshmerize_Me/old/`
  （`leaf_vein.vertex/target`、`input2d`；`please_Compute_External_Forcing.m` → 已逐行移植为同名 `.py`；
  `main2d.m`、`Leafitize_Me.m`（SVG→网格生成脚本）、`optimal_leaf.svg` 未随 demo 提供）。
* 算法源码：`third_party/ib2d/pyIB2d/IBM_Blackbox/`。

## 算例内容

* `1 x 1` 周期盒、`mu = 1e-3`、`rho = 1`、官方 `512 x 512`（方格 `h = 1.95e-3`）、
  `dt = 1e-4`、`supp = 4`；官方 `Tfinal = 0.5`（5000 步）、`print_dump = 100`。
  本 demo 用 **`256 x 256`**（两个方向同时减半）与 **`T = 0.02`**（200 步）。
* **结构**：`leaf_vein.vertex` **2335** 点（比 demo_685 的叶脉网更大、更密）
  全部由目标点 `k = 1e5` 钉死（`update_target = 0`）。
* **驱动**：窗口 `x ∈ [0.41, 0.545]`、`y ∈ [0.115, 0.12]` 内的 y 射流
  ```
  uY_tar = -uMax · 5 tanh(t) · ((MP+w/2)-x)((MP-w/2)-x),   uX_tar = 0
  ```
  `uMax = 250`、`w = 0.5515 − 0.4029 = 0.1486`、`MP` 为其中心。
  与 685 的三处区别（上游原样保留）：**没有 `scale` 归一化**、
  时间因子是 `5 tanh(t)`（不是 `tanh(2t)`）、窗口与几何都不同。

## 移植要点

* 家族共享部分见 demo_685；本例特有：无 `scale`、`5 tanh(t)`、
  `uMax = 250`（是 685 的 100 倍）→ 驱动更"硬"。
* **注意 `kStiff·dt = 1e4 × 1e-4 = 1` 恰好是显式罚力的临界稳定点**，
  这也解释了本例速度场的大偏差（见下）。

## 结果

参考运行 ~12 s、AFSI 运行 ~47 s（各 200 步、21 帧，`t = 0 .. 0.0201`）。

| 指标（末帧 `t = 0.02010`） | 结果 |
|---|---|
| 标记点偏差（2335 点，钉死叶脉网） | 2.99e-5 m = **7.7e-3 h** |
| 窗口内 `max\|u\|` AFSI / IB2d | 1.2106e-1 / 3.3449e-1（**6.4e-1** 相对） |
| 速度场相对 $L_2$（窄带外） | 6.24e-1 |
| 窄带外 `max\|du\|` | 7.82e-2 = 7.7e-1 `max\|u\|` |
| 外力场（窄带外） | `max\|df\| = 2.1e2`（`\|f\|max = 3.8e2`） |
| 早期帧 `t = 0.00110` | `max\|u\|` 5.44e-3/3.44e-3（AFSI **更**强，58%）、标记点 5.4e-5 h |

**能验证**：`leaf_vein`（2335 点）的几何/目标点读取与 IB 转移（偏差 7.7e-3 h，
与 demo_685 的 8.0e-3 h 一致）；两侧共用同一份外力模块（窗口与型线逐位一致）。

**不能验证 / 已知限制**：

1. **速度场不可比**（末帧 `max|u|` 差 **64%**，`relL2 = 0.62`）。
   原因是驱动窗口只有 1 个格点高、`uMax = 250`（比 685 大 100 倍）且
   **`kStiff·dt = 1` 正好落在显式罚力的临界稳定点**：
   `u^{n+1} = u^n + dt(−k(u^n − u_tar)) = u_tar` —— 离散罚力一步即达目标、
   完全无阻尼。pyIB2d 的 FFT 投影时间推进把这种"单格点、零阻尼"罚力
   逐步放大（表中比值从 0.2 倍一路涨到 2.8 倍），而 AFSI 的
   Taylor-Hood/Peskin 两段式保持接近目标值。这是**时间推进器的差异**，
   不是移植错误（结构、外力模块都逐位一致）。同类现象见 demo_685；
   demo_686/687（窗口宽、`kStiff·dt = 0.5`）就只有 5–6% 的幅值差。
2. **`T` 只跑 1/25**（200/5000 步）；官方 0.5 s 在 256×256 上需要 AFSI ~20 分钟。
3. 网格减半（512×512 → 256×256）。
4. 本机无 MATLAB/Octave，无法跑上游 matIB2d 做第三方核对。

## 运行

```bash
conda activate afsi-dolfinx
python run_reference.py --tend 0.02 --nx 256 --print-dump 10   # pyIB2d，~12 s
NX=256 TFINAL=0.02 PRINT_DUMP=10 python main.py                # AFSI，~47 s
python compare.py                                              # figures/ + compare_table.csv
```
