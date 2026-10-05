# demo_676 — IB2d `Example_Scallop/Scallop_FV_LT_Muscle`（扇贝 / FV-LT 肌肉）

## 上游路径

* 数据：`third_party/ib2d/matIB2d/Examples/Example_Scallop/Scallop_FV_LT_Muscle/`
  （`scallop.vertex/.spring/.beam/.muscle`、`input2d`、`main2d.m`、
  `Make_Scallop_Geo_and_Input_Files.m`、`test_Activation_Wave.m`、
  **`give_Muscle_Activation.m`**）。
* 算法源码：`third_party/ib2d/pyIB2d/IBM_Blackbox/{IBM_Driver.py, Supp.py,
  please_Find_Lagrangian_Forces_On_Eulerian_grid.py}`。
* 代码基：`afsic/demo/ib2d/_template/`（已支持弹簧 + 不变梁 + FV_LT 肌肉）。

`give_Muscle_Activation.m` 已逐行移植为 `ib2d_input/give_Muscle_Activation.py`
（AFSI 与 pyIB2d 参考跑同一份）。`Make_Scallop_Geo_and_Input_Files.m`（几何生成器）
与 `test_Activation_Wave.m`（画图测试脚本）驱动都不调用，未移植（.m 原样保留）。

## 算例内容

| 项 | 值 |
|---|---|
| 流体 | `1 x 1` 周期盒、`mu = 1e-4`、`rho = 1`、`256 x 256`（官方） |
| 时间 | `dt = 1e-4`、`Tfinal = 1.5`（官方，15000 步） |
| 结构 | 103 个标记点：扇贝外壳轮廓，`x ∈ [0.75, 0.848]`、`y ∈ [0.152, 0.25]` |
| 弹簧 | 102 根线性弹簧（`k = 2e6`、`L0 = 1.953e-3` = 1/512） |
| 不变梁 | 101 根（`kb = 1e11`，参考叉积 `C` 由生成器写入磁盘） |
| FV_LT 肌肉 | **1** 条（marker 30 ↔ 74，`Fmax = 100`、`LFO = 0.049292`、`SK = 0.3`、Hill `a = 0.25`、`b = 4`） |
| 目标点 / 质量点 / 外力 | 全关 |

肌肉激活是算例自带的 `give_Muscle_Activation.m`：**只随时间**变化的方波
（`freq = 3` Hz，`t mod 1/3 < 1/6` 时 `+1`，否则 `-2`），与 HeartTube 系列的
"行波"不同，它不依赖 `xPt`。

## 移植要点

* FV_LT 力律（`give_Muscle_Force_Densities`）逐条复刻：
  `v = |LF(X_h) − LF(X_h^prev)|/dt`、`Q = LF/LFO`、`F1 = exp(−((Q−1)/SK)²)`、
  `F2 = (1/Fmax)(b·Fmax − a·v)/(v+b)`、`F = af·Fmax·F1·F2`，
  再 `F_leader += F·d̂`、`F_follower −= F·d̂`；
* 激活模块用**算例自带的** `give_Muscle_Activation`（9 参数签名，pyIB2d 以向量调用），
  两边共用同一份 `ib2d_input/give_Muscle_Activation.py`；
* 不变梁、线性弹簧沿用模板（`kb`、`C` 从 `.beam` 逐字读取）。

### 发现的上游 bug / quirk

1. **单行 `.muscle` 让 pyIB2d 直接崩**：本算例只有 1 条肌肉带，
   `read_Muscle_Points` 用 `np.loadtxt(..., usecols=...)` 返回 **1-D** 数组，
   随后 `please_Find_Lagrangian_Forces_On_Eulerian_grid.py:351` 的
   `muscles[:,0]` 抛 `IndexError: too many indices for array`。
   `run_reference.py` 把同一类的 reader（spring/beam/target/muscle/mass/…）
   统一包成返回 `(n, m)`，并单独重写了 `read_Spring_Points`
   （它在函数内部就 `n, m = springs.shape` 崩掉）。
2. **结构分辨率与流体网格解耦**：`.spring` 的 `L0 = 1.953125e-3 = 1/512`
   说明这套标记点是按 512 点离散的，而官方流体网格是 `256²`
   （标记点/网格 = 1/2）。把流体网格降到 96/64 后，同样 `kb = 1e11` 的梁
   相对网格变得**硬得多**，两个求解器都会在 `t ≈ 0.08–0.14 s` 失稳
   （实测见下）。这是本 demo 只能跑很短的物理时间的原因。

## 缩减设置（重要）

官方 `256 x 256`、`dt = 1e-4`、`Tfinal = 1.5`。本 demo 用
**`NX = 96`、`dt = 5e-5`（官方 2 倍）、`T = 0.06 s`（1200 步）**。

为什么停得这么早（本机实测，`NX=96`、`dt=5e-5`）：

| 工况 | 结果 |
|---|---|
| AFSI，`NX=96`、`dt=5e-5` | `t ≈ 0.083` 起 `max\|u\|` 从 8.5 跳到 27.7，随后 NaN |
| AFSI，`NX=64`、`dt=1e-4` | `t ≈ 0.092` 起 NaN |
| pyIB2d，`NX=96`、`dt=5e-5` | `T=0.15` 内干净（`max\|u\|` 峰值 31） |
| pyIB2d，`NX=64`、`dt=1e-4` | `t ≈ 0.165` 起 NaN |
| pyIB2d，`NX=128`、`dt=5e-5` | `t ≈ 0.143` 起 NaN |

因此取 `T = 0.06 s`（两边都干净），覆盖激活方波 `+1` 相位的**前半段**
（`+1` 相位持续到 `t = 1/6 ≈ 0.167 s`）。

## 结果

运行时间（macOS，本 demo 实测）：pyIB2d 参考 **约 18 s**（1200 步驱动 + VTK 写出）、
AFSI **38.2 s**（1200 步，峰值内存 768 MB）。两边各 100 个对齐帧，
`t = 0 .. 0.0594`。

| 指标 | 结果 |
|---|---|
| 标记点偏差（103 点，末帧）`max\|dX\|` | 2.979e-2 m = **2.86 h**（`compare.py` 汇总）；最近帧对齐为 2.27e-2 m = 2.18 h |
| **早期段 `t ≤ 0.005`（9 帧）`max\|dX\|`** | 3.65e-4 m = **0.035 h**（`max\|u\|` AFSI 0.54 / IB2d 0.43，差 26%） |
| 肌肉带长度（marker 30↔74）AFSI / IB2d | `0.04929 → 0.02759` / `0.04929 → 0.03239`（两边都收缩，AFSI 收缩更多 44% vs 34%） |
| 弹簧平均长度 AFSI / IB2d | 0.002009 / 0.001960（差 2.5%） |
| 流体 `max\|u\|` AFSI / IB2d | 8.14 / 1.93（AFSI 已接近失稳，见下） |
| 流体 `max\|du\|`（全程） | 5.85 m/s |
| 相对 L2（速度场，全程） | 2.42e+00 |

**能验证**：

* FV_LT 肌肉力律与算例自带激活模块：早期段（`t ≤ 0.005`）标记点形状几乎重合
  （`max|dX| = 0.035 h`），两边弹簧平均长度相差 2.5%；
* 收缩的**定性行为**：两边肌肉带都从 `LFO = 0.049292` 缩短
  （AFSI→0.0276、IB2d→0.0324），外壳被拉扁，符合"扇贝闭合"的力学；
* 与 demo_612/613（HeartTube 系列，同样是 FV_LT/3-element）共用同一套力律实现，
  结果特征一致。

**不能验证（如实记录）**：

* 长时间/大振幅：`t ≳ 0.08 s` 后 AFSI 的 `max|u|` 先于参考解失控
  （末帧 7.4 vs 1.67），`max|dX|` 到 2.9e-2 m（2.86 h），相对 L2 2.4 ——
  该算例在缩减网格上处于稳定边缘（见"缩减设置"表），**这段时间窗内的速度场
  与绝对轨迹不可比**；
* 完整游泳周期：官方 `Tfinal = 1.5 s`（≈4.5 个激活周期）未复现，本 demo 只跑
  激活方波 `+1` 相位的前 36%；
* 网格/`dt` 收敛性：无法在本机预算内完成（`256²` 需要 ~4.6 GB 内存，
  已被用户的内存约束排除）。

## 运行

```bash
conda activate afsi-dolfinx
python run_reference.py --tend 0.06 --dt 5e-5 --print-dump 12 --nx 96   # 参考解（~18 s）
NX=96 DT=5e-5 TFINAL=0.06 PRINT_DUMP=12 python main.py                 # AFSI（~38 s）
python compare.py
```

环境变量：`NX`（默认官方 256）、`NY`、`DT`、`TFINAL`、`PRINT_DUMP`、
`OUTPUT_PATH`（默认 `./plot`）、`GRAD_DIV`（默认 0）。

产物：`plot/afsi_result.npz`、`ib2d_reference.npz`、
`figures/compare_profiles.png`、`figures/compare_history.png`、
`figures/compare_fields.png`、`figures/compare_table.csv`。
