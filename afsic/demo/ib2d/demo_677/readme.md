# demo_677 — IB2d `Example_Scallop/Scallop_3Elem_Muscle`（扇贝 / 三元件 Hill 肌肉）

## 上游路径

* 数据：`third_party/ib2d/matIB2d/Examples/Example_Scallop/Scallop_3Elem_Muscle/`
  （`scallop.vertex/.beam/.d_spring/.3_muscle`、`input2d`、`main2d.m`、
  `Make_Scallop_Geo_and_Input_Files.m`、`test_Activation_Wave.m`、
  **`give_3_Element_Muscle_Activation.m`**）。
* 算法源码：`third_party/ib2d/pyIB2d/IBM_Blackbox/{IBM_Driver.py, Supp.py,
  please_Find_Lagrangian_Forces_On_Eulerian_grid.py}`。
* 代码基：`afsic/demo/ib2d/_template/`（弹簧/阻尼弹簧/不变梁 + 本 demo 新增的
  三元件肌肉分支）。

## 算例内容

| 项 | 值 |
|---|---|
| 流体 | `1 x 1` 周期盒、`mu = 1e-3`、`rho = 1`、`256 x 256`（官方） |
| 时间 | `dt = 5e-5`、`Tfinal = 1.5`（官方，30000 步） |
| 结构 | 103 个标记点（与 demo_676 同一外壳轮廓，`x ∈ [0.75, 0.848]`、`y ∈ [0.152, 0.25]`） |
| 不变梁 | 101 根（`kb = 1e12`） |
| 阻尼弹簧 | 153 根（`k = 1e8`、`L0 = 1/512`、**`b = 0`**，即等价线性弹簧） |
| 三元件肌肉 | **3** 条（marker 35↔69、20↔84、5↔99；`Fmax = 1e4 / 2.5e4 / 6e4`、`LFO = 0.0381 / 0.0717 / 0.1053`、`SK = 0.3`、Hill `a = 0.25`、`b = 4`） |
| 弹簧 / 目标点 / 质量点 / 外力 | 全关 |

三元件激活是**顺序**方波（`freq = 10` Hz）：`t mod 0.1 < 0.04` 时依次点火——
肌肉 1 在 `t < 0.01`、肌肉 2 在 `0.005 ≤ t < 0.015`、肌肉 3 在 `0.01 ≤ t < 0.015`
（`af = tanh(50 t)`），其余时间 `af = 0`。也就是三条肌肉在 0.015 s 内先后收缩，
推动外壳完成一次"划水"。

## 移植要点

* 三元件力律在 pyIB2d 里与 FV_LT **共用同一段代码**
  （`give_3_Element_Muscle_Force_Densities` 只是换了个激活函数），
  所以 AFSI 侧沿用模板的 `muscle_forces`，只把激活模块换成
  `give_3_Element_Muscle_Activation`（本 demo 用 `act_fn` 选择）；
* `muscle_forces` 逐条复刻：
  `v = |LF(X_h) − LF(X_h^prev)|/dt`、`Q = LF/LFO`、`F1 = exp(−((Q−1)/SK)²)`、
  `F2 = (1/Fmax)(b Fmax − a v)/(v+b)`、**`F = af · |Fmax·F1·F2|`**（注意 `abs`），
  再 `F_leader += F d̂`、`F_follower −= F d̂`；
* 激活模块 `give_3_Element_Muscle_Activation.m` 已逐行移植为
  `ib2d_input/give_3_Element_Muscle_Activation.py`，两边共用同一份。

### 发现的上游 bug / quirk

1. **MATLAB 模块与 pyIB2d 调用不匹配**：`.m` 有 **11** 个参数（最后一个是每条肌肉的
   序号 `i`）且返回 4 个输出，而 pyIB2d 的
   `give_3_Element_Muscle_Force_Densities` 只传 **10** 个参数（对全部肌肉带向量化，
   没有 `i`）。本移植采用 pyIB2d 的约定（`i` = 向量中的位置），并保留算例的力律
   （含 `abs`）。这也是 `matIB2d` 原版 blackbox 与本算例数据互相矛盾的地方
   （同 demo_613 记录的现象）。
2. **pyIB2d 漏 import**（任务书点名）：`please_Find_Lagrangian_Forces_On_Eulerian_grid.py`
   的三元件分支**调用** `give_3_Element_Muscle_Activation` 却从未 import
   （FV_LT 分支在函数内 import 了 `give_Muscle_Activation`），参考解直接 `NameError`。
   `run_reference.py` 把移植的模块注入该模块命名空间（与 demo_613 同样的补丁）。
3. **文件扩展名不一致**：pyIB2d 的 `read_Hill_3Muscle_Points` 读
   `<name>.muscle`，而算例发的是 `<name>.3_muscle`（9 列，后两列为串联/并联元件
   参数，pyIB2d 不读）。本 demo 两者都保留：`ib2d_input/scallop.muscle`
   是 `scallop.3_muscle` **前 7 列的逐字拷贝**（仅扩展名改对，`input2d` 不动）。
4. **`b = 0` 的阻尼弹簧**：算例把 `damped_springs` 打开但 153 根弹簧的阻尼系数全为 0，
   实际等价线性弹簧；AFSI 仍按 IB2d 的阻尼弹簧公式计算（模板已实现）。
5. 结构与 demo_676 相同的"标记点分辨率固定 1/512 vs 缩减流体网格"问题，
   使 `kb = 1e12` 在降网格后处于稳定边缘。

## 缩减设置（重要）

官方 `256 x 256`、`dt = 5e-5`、`Tfinal = 1.5`。本 demo 用
**`NX = 96`、`dt = 5e-5`（官方值）、`T = 0.03 s`（600 步）**，覆盖
`t ∈ [0, 0.015]` 的**一次完整顺序激活爆发**及其 0.015 s 后的松弛段。

`256²` 需要约 4.6 GB 内存（被用户的内存约束排除）；`NX = 96` 的两次运行分别
6 s（参考）/ 17.7 s（AFSI）。窗口取 `T = 0.03` 是因为更长时间后两个求解器的
`max|u|` 都迅速增大（参考解在 `t ≳ 0.025` 达 40+ m/s），进入不可信的边缘区。

## 结果

运行时间（macOS，本 demo 实测）：pyIB2d 参考 **约 6 s**、AFSI **17.7 s**
（600 步，峰值内存 746 MB）。两边各 40 个对齐帧，`t = 0 .. 0.0293`。

| 指标 | 结果 |
|---|---|
| 标记点偏差（103 点，末帧）`max\|dX\|` | 2.383e-2 m = **2.29 h** |
| 早期段 `t ≤ 0.005`（7 帧）`max\|dX\|` | 2.36e-3 m = **0.226 h**（`max\|u\|` AFSI 4.17 / IB2d 2.62） |
| 三条肌肉长度（末帧）AFSI / IB2d | 肌肉1 `0.0381→0.0135` / `→0.0215`；肌肉2 `0.0717→0.0224` / `→0.0332`；肌肉3 `0.1053→0.0281` / `→0.0575` |
| 流体 `max\|u\|` 峰值 AFSI / IB2d | 14.84 / 44.94 |
| 流体 `max\|du\|`（全程） | 50.0 m/s |
| 相对 L2（速度场，全程） | 1.01 |

**能验证**：

* 三元件肌肉机制（含 `abs` 力律、顺序激活、3 条带的窗口）：**三条肌肉都按预期顺序
  缩短**（AFSI 与 IB2d 的 LF 都从各自的 `LFO` 掉到 1/3–1/2），外壳被拉扁；
* 早期段（`t ≤ 0.005`，即肌肉 1 点火、肌肉 2 刚开始）标记点形状
  `max|dX| = 0.226 h`，两边一致；
* 与 demo_613（HeartTube 三元件，同一力律）对照：实现相同，行为特征一致。

**不能验证（如实记录）**：

* 末段速度场与绝对轨迹：`max|u|` AFSI 14.8 / IB2d 44.9（相对 L2 1.01），
  两边都已进入 `kb = 1e12` + 缩减网格的稳定边缘，**不可比**；
* AFSI 的收缩幅度明显大于参考解（三条肌肉分别多收缩约 60%/48%/105% 的长度），
  这与 676 的观察一致，说明动边界耦合在欠分辨网格下差异被放大；
* 完整游泳序列：官方 `Tfinal = 1.5 s`（15 个激活周期）未复现；
* 网格/`dt` 收敛性：本机预算内无法完成。

## 运行

```bash
conda activate afsi-dolfinx
python run_reference.py --tend 0.03 --dt 5e-5 --print-dump 15 --nx 96   # 参考解（~6 s）
NX=96 DT=5e-5 TFINAL=0.03 PRINT_DUMP=15 python main.py                 # AFSI（~18 s）
python compare.py
```

环境变量：`NX`（默认官方 256）、`NY`、`DT`、`TFINAL`、`PRINT_DUMP`、
`OUTPUT_PATH`（默认 `./plot`）、`GRAD_DIV`（默认 0）。

产物：`plot/afsi_result.npz`、`ib2d_reference.npz`、
`figures/compare_profiles.png`、`figures/compare_history.png`、
`figures/compare_fields.png`、`figures/compare_table.csv`。
