# demo_678 — IB2d `Example_Scallop/Scallop_3Elem_Muscle_Almost_Swims`（扇贝 / "差点会游泳"）

> 本 demo 是任务书里的**可选**项（"如有余力再做"），已按与 670–677 相同的标准完成。

## 上游路径

* 数据：`third_party/ib2d/matIB2d/Examples/Example_Scallop/Scallop_3Elem_Muscle_Almost_Swims/`
  （`scallop.vertex/.spring/.beam/.3_muscle`、`input2d`、`main2d.m`、
  `Make_Scallop_Geo_and_Input_Files.m`、`test_Activation_Wave.m`、
  **`give_3_Element_Muscle_Activation.m`**，外加一个上游备份 `...m~`）。
* 算法源码：`third_party/ib2d/pyIB2d/IBM_Blackbox/{IBM_Driver.py, Supp.py,
  please_Find_Lagrangian_Forces_On_Eulerian_grid.py}`。
* 代码基：`afsic/demo/ib2d/_template/`（弹簧 + 不变梁 + 三元件肌肉分支），
  与 demo_677 共用同一套改动。

## 算例内容

| 项 | 值 |
|---|---|
| 流体 | `1 x 1` 周期盒、`mu = 2.5e-4`、`rho = 1`、`256 x 256`（官方） |
| 时间 | `dt = 1e-4`、`Tfinal = 1.5`（官方，15000 步） |
| 结构 | 103 个标记点（与 demo_676/677 同一外壳轮廓，`x ∈ [0.75, 0.848]`、`y ∈ [0.152, 0.25]`） |
| 弹簧 | 102 根线性弹簧（`k = 2.5e6`、`L0 = 1/512`） |
| 不变梁 | 101 根（`kb = 2.5e10`，比 demo_677 软 40 倍） |
| 三元件肌肉 | **1** 条（marker 30↔74，`Fmax = 200`、`LFO = 0.049292`、`SK = 0.3`、Hill `a = 0.25`、`b = 4`） |
| 阻尼弹簧 / 目标点 / 质量点 / 外力 | 全关 |

激活是**平滑的**功率冲程：`t mod (1/freq) < 1/(3 freq)` 时
`af = sin(π·freq·t)`（`freq = 3` Hz，脉冲持续 `1/9 ≈ 0.111 s`），其余时间 `af = 0`。
外壳每个周期闭合一次再打开，游泳的净位移很小——"Almost Swims"。

## 移植要点

* 三元件力律与 demo_677 完全一致（pyIB2d 里与 FV_LT 共用代码），
  但本算例的激活模块里 `Fm = af_Val·Fmax·F1·F2`（**没有 `abs`**）、
  且是 **10 参数**签名（无 `i`），正好与 pyIB2d 的调用匹配；
* 激活模块已逐行移植为 `ib2d_input/give_3_Element_Muscle_Activation.py`，
  AFSI 与 pyIB2d 参考共用同一份；
* 弹簧/不变梁沿用模板。

### 发现的上游 bug / quirk

1. **pyIB2d 漏 import**（与 demo_677 同）：`please_Find_Lagrangian_Forces_On_Eulerian_grid.py`
   的三元件分支调用 `give_3_Element_Muscle_Activation` 却从未 import →
   `NameError`；`run_reference.py` 注入移植模块。
2. **单行 `.muscle` 崩溃**：本算例只有 1 条肌肉带，`read_Hill_3Muscle_Points`
   等 `usecols` reader 返回 **1-D** 数组，`muscles[:, 0]` 抛 `IndexError`；
   `run_reference.py` 把同类 reader 统一包成返回 `(n, m)`，并单独重写
   `read_Spring_Points`（它在 `n, m = springs.shape` 处崩）。
3. **扩展名不一致**：pyIB2d 读 `<name>.muscle`，算例发的是 `<name>.3_muscle`
   （9 列）；`ib2d_input/scallop.muscle` 是前 7 列的逐字拷贝。
4. 上游还随包了一个编辑备份文件 `give_3_Element_Muscle_Activation.m~`，未纳入
   `ib2d_input/`（与运行无关）。

## 缩减设置

官方 `256 x 256`、`dt = 1e-4`、`Tfinal = 1.5`。本 demo 用
**`NX = 96`、`dt = 5e-5`、`T = 0.12 s`（2401 步）= 一个完整激活周期**
（`1/freq = 1/3 s` 中的前 0.12 s 覆盖完整的 `sin` 冲程与随后 0.009 s 松弛；
`T` 取 0.12 也正好让 dump 落在整数步上：`2401 = 60 × 40 + 1`）。

`256²` 需约 4.6 GB 内存（被用户内存约束排除），故降到 `NX = 96`
（两次运行 11.7 s / 71.3 s）。本算例比 676/677 温和得多（`kb` 软 40 倍、
激活平滑），在 `NX = 96` 上**没有出现失稳**，参考解一路干净到 `T = 0.2`。

## 结果

运行时间（macOS，本 demo 实测）：pyIB2d 参考 **11.7 s**、AFSI **71.3 s**
（2401 步，峰值内存 740 MB）。两边各 61 个对齐帧，`t = 0 .. 0.12005`。

| 指标 | 结果 |
|---|---|
| 标记点偏差（103 点，末帧）`max\|dX\|` | 3.533e-2 m = **3.39 h** |
| `t ≤ 0.01`（6 帧）`max\|dX\|` | 5.69e-5 m = **0.005 h** |
| `t ≤ 0.05`（26 帧）`max\|dX\|` | 6.00e-3 m = 0.576 h |
| 流体 `max\|u\|` 峰值 AFSI / IB2d | 1.269 / 1.386（**差 8.5%**） |
| 流体 `max\|du\|`（全程） | 0.729 m/s |
| 相对 L2（速度场，全程） | 6.69e-01 |
| 弹簧平均长度 AFSI / IB2d | 0.001957 / 0.001960（差 **0.17%**） |
| 肌肉长度（marker 30↔74）AFSI / IB2d | `0.04929→0.02180`（最小 0.02162）/ `0.04929→0.03230`（最小 0.03051） |
| 外壳质心净位移 AFSI / IB2d | `(-0.0475, +0.0475)` / `(-0.0239, +0.0239)` |

**能验证**：

* 三元件肌肉 + 平滑激活：`t ≤ 0.01` 内标记点偏差仅 `0.005 h`，
  全程弹簧平均长度差 0.17%，`max|u|` 峰值差 8.5%——本族里**最干净的一个**；
* "游泳"的定性结论：两边外壳质心都沿 **`(-x, +y)`** 方向净移动
  （AFSI `(-0.0475, +0.0475)`、IB2d `(-0.0239, +0.0239)`），
  即同一方向、AFSI 位移约为参考的 2 倍；
* 肌肉收缩行为一致：两边 LF 都从 `LFO = 0.049292` 缩短
  （AFSI→0.0216、IB2d→0.0305），AFSI 收缩更多。

**不能验证（如实记录）**：

* 长时间绝对轨迹：末帧 `max|dX| = 3.39 h`。对一个"游动"的结构，标记点的绝对位置
  由累积的净位移决定，而 AFSI 与 IB2d 的净位移本身相差 2 倍（见上），
  因此**绝对位置不可比**；可比的是同一时刻的**形状**（前 0.01 s 0.005 h）与
  流场量级（8.5%）；
* 完整游泳序列：官方 `Tfinal = 1.5 s`（4.5 个激活周期）未复现（本 demo 跑 1 个周期）；
* 网格/`dt` 收敛性：本机预算内无法完成（`256²` 约需 4.6 GB 内存）。

## 运行

```bash
conda activate afsi-dolfinx
python run_reference.py --tend 0.12 --dt 5e-5 --print-dump 40 --nx 96   # 参考解（~12 s）
NX=96 DT=5e-5 TFINAL=0.12 PRINT_DUMP=40 python main.py                 # AFSI（~71 s）
python compare.py
```

环境变量：`NX`（默认官方 256）、`NY`、`DT`、`TFINAL`、`PRINT_DUMP`、
`OUTPUT_PATH`（默认 `./plot`）、`GRAD_DIV`（默认 0）。

产物：`plot/afsi_result.npz`、`ib2d_reference.npz`、
`figures/compare_profiles.png`、`figures/compare_history.png`、
`figures/compare_fields.png`、`figures/compare_table.csv`。
