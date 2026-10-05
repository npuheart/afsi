# demo_687 — IB2d `Example_Meshmerize_Me/Flow_Around_Starfish`（海星网格 + 脉冲射流）

IB2d 官方算例（`matIB2d/Examples/Example_Meshmerize_Me/Flow_Around_Starfish/`）。
Meshmerize 家族成员；**家族共享的移植要点（外力模块的 3 个函数、半分辨率规则、
`T`/`print_dump` 缩减、速度场对比的限制）见 demo_685 的 readme**，
本文只写本例特有的设置与结果。

## 上游路径

* 数据：`third_party/ib2d/matIB2d/Examples/Example_Meshmerize_Me/Flow_Around_Starfish/`
  （`starfish_flowtank.vertex/target`、`input2d`；`please_Compute_External_Forcing.m` → 已逐行移植为同名 `.py`；
  `main2d.m`、`VisIt_Session.session`/`.gui`、SVG 与 SVG→网格生成脚本 未随 demo 提供）。
* 算法源码：`third_party/ib2d/pyIB2d/IBM_Blackbox/`。

## 算例内容

* `1 x 0.25` 周期盒、`mu = 0.1`、`rho = 1000`（`nu = 1e-4`）、官方 `1024 x 256`、
  `dt = 5e-5`、`supp = 4`；官方 `Tfinal = 1.0`（20000 步）、`print_dump = 100`。
  本 demo 用 **`512 x 128`**（两个方向同时减半）与 **`T = 0.02`**（400 步）。
* **结构**：与 demo_686 相同的 3010 点海星轮廓，但**没有弹簧**，只有目标点
  `k = 1e5`（`update_target = 0`，形状固定）。
* **驱动**：窗口 `x ∈ [0.04, 0.06]`、`y ∈ [0.02, 0.22]` 内的**脉冲** x 射流
  ```
  freq_P = 5; period = 1/freq_P; rest_time = period; tot_period = 2*period
  t_mod = mod(t, tot_period)
  uX_tar = -uMax sin(2π·freq_P·t) ((Ly/2+w/2)-y)((Ly/2-w/2)-y)   若 t_mod <= period
  uX_tar = 0                                                      否则（静止期）
  ```
  `uMax = 2.5e3`、`w = 0.25`。上游怪癖：外层 `if t_mod <= tot_period` 恒为真
  （`t_mod < tot_period`），移植按实际行为照抄。

## 移植要点

* 家族共享部分见 demo_685；本例特有：脉冲逻辑（`mod`、`freq_P`、静止期）、
  `uMax = 2.5e3`、窗口占满左边界、无弹簧（对比 demo_686）。
* `T = 0.02` 覆盖脉冲周期的 20%（`period = 0.2 s`）——驱动刚启动的强加速段
  （`|f|max` 达到 2.4e5，见下）。

## 结果

参考运行 ~22 s、AFSI 运行 ~80 s（各 400 步、41 帧，`t = 0 .. 0.01955`）。

| 指标（末帧 `t = 0.01955`） | 结果 |
|---|---|
| 标记点偏差（3010 点，仅目标点约束） | 3.25e-5 m = **1.7e-2 h** |
| 窗口内 `max\|u\|` AFSI / IB2d | 6.3527e-1 / 6.6931e-1（**5.1e-2** 相对） |
| 速度场相对 $L_2$（窄带外） | 1.74e-1 |
| 窄带外 `max\|du\|` | 1.48e-1 = 7.8e-1 `max\|u\|` |
| 外力场（窄带外） | `max\|df\| = 1.2e3`（`\|f\|max = 2.4e5`） |
| 早期帧 `t = 0.00505` | 标记点 1.2e-3 h、`max\|u\|` 4.28e-2/4.45e-2（3.9%）、`relL2(u)` 3.5e-1 |

**能验证**：脉冲驱动的时间结构（`sin(2π·5t)` 的上升段两侧同步）、
固定海星轮廓的网格结构（偏差 1.7e-2 h）、以及两侧共用同一份外力模块；
速度幅值差 5.1%（比 685/690 好，因为窗口宽、`kStiff·dt = 0.5` 未到临界）。

**不能验证 / 已知限制**：

1. **`T` 只跑 1/50**（400/20000 步）：官方 1 s 在 512×128 上需要 AFSI ~1 小时。
   脉冲的完整 on/off 周期（0.4 s）没有跑完，只覆盖了第一个脉冲的前 20%。
2. **速度场只做半定量比较**：窄带外 `relL2 = 0.17`、`max|du| = 0.78 max|u|`；
   本例的射流很强（`uMax = 2.5e3`）且窗口占满左边界，初始加速段对
   两套投影/时间推进最敏感（`|f|max` 2.4e5，`u` 到 0.67）。
3. 网格减半（1024×256 → 512×128）+ `print_dump` 100 → 10。
4. 本机无 MATLAB/Octave，无法跑上游 matIB2d 做第三方核对。

## 运行

```bash
conda activate afsi-dolfinx
python run_reference.py --tend 0.02 --nx 512 --print-dump 10   # pyIB2d，~22 s
NX=512 TFINAL=0.02 PRINT_DUMP=10 python main.py                # AFSI，~80 s
python compare.py                                              # figures/ + compare_table.csv
```
