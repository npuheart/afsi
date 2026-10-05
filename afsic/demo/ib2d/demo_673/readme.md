# demo_673 — IB2d `Example_Wobbly_Beam/Beam_32`（摆动梁 / 不变梁）

## 上游路径

* 数据：`third_party/ib2d/matIB2d/Examples/Example_Wobbly_Beam/Beam_32/`
  （`BeamCurve.vertex`、`BeamCurve.beam`、`BeamCurve.target`、`input2d`、
  `main2d.m`，以及生成几何的 `BeamCurve.m`）。
* 算法源码：`third_party/ib2d/pyIB2d/IBM_Blackbox/{IBM_Driver.py, Supp.py,
  please_Find_Lagrangian_Forces_On_Eulerian_grid.py}`。
* 代码基：`afsic/demo/ib2d/_template/`（合并模板，已支持不变梁 + 目标点）。

`BeamCurve.m` 只是**生成** `.vertex/.beam/.target` 的 MATLAB 脚本（`main2d.m` 在
文件缺失时调用它），pyIB2d 驱动从不调用，故未移植；三个数据文件随上游原样附带。

## 算例内容

| 项 | 值 |
|---|---|
| 流体 | `1 x 1` 周期盒、`mu = 0.1`、`rho = 1`、**`32 x 32`（官方 = 本 demo 使用）** |
| 时间 | `dt = 5e-5`、`Tfinal = 0.05`（1000 步）（官方 = 本 demo 使用） |
| 结构 | 64 个标记点：从 `(0.25, 0.5)` 拱到 `(0.75, 0.5)`，顶点 `y ≈ 0.625` |
| 不变梁 | 62 根（IB2d 叉积扭转弹簧模型，`κ_b = 7.5e9`，参考叉积 `C = 0`） |
| 目标点 | 2 个（两端 id 0 与 63，`k = 2e8`），即把梁两端钉死 |
| 弹簧 / 质量点 / 外力 / 标量 | 全关 |

与 demo_446（从 **pyIB2d** 的 `Examples/Wobbly_Beam` 移植）物理相同：两处上游的
`BeamCurve.{vertex,beam,target}` 逐字节一致，本 demo 走 matIB2d（1 基索引）这条路。
本 demo **直接使用官方设置**（`32 x 32`、`dt = 5e-5`、`Tfinal = 0.05`，无需缩减）。

这是一根"无质量 + 近刚性"梁：`κ_b = 7.5e9` 相对 `32 x 32` 网格上的流体惯性极大，
松开后梁猛烈上下甩动，**两边流体的 `max|u|` 峰值都到 ~15 m/s**（1×1 盒内），
属于"官方算例在粗网格上仅勉强可解"的情形——这正是它作为教学算例的看点。

## 移植要点

* 不变梁力按 IB2d 原式在**半步位置**上求值（`Supp.py` / `IBM_Driver.py` 的
  `give_Me_Invariant_Beam_Lagrangian_Force_Densities`）：
  `S = (Δx)_{r→q}(Δy)_{q→p} − (Δy)_{r→q}(Δx)_{q→p}`、`K = κ(S − C)`，
  然后 `f_p += K(y_r−y_q)` 等（模板逐字复刻）；
* 目标点把两端钉在初始位置（`k = 2e8`），周期回绕；
* 结构承载用"零刚度 FE 链"承载标记点自由度（与其它 demo 相同）；
* `BeamCurve.m` 未移植（生成器脚本，驱动不调用）。

### 上游 quirk

* `print_dump = 10` 时驱动在 `cter % print_dump == 0 且 cter >= print_dump`
  写盘，时间戳是该步**步首**的时间；`main.py` 在与驱动完全相同的步上 dump，
  `compare.py` 按 `round(t/dt)` 对齐（1000 步 → 100 帧，`t = 0 .. 0.0495`）。
* 模板已支持的不变梁分支即为本算例所需；未遇到新的上游 bug。

## 结果

运行时间（macOS，本 demo 实测）：pyIB2d 参考 **约 7 s**（1000 步驱动 + VTK 写出）、
AFSI **10.9 s**（1000 步）。两边各 100 个对齐帧，`t = 0 .. 0.0495`。

| 指标 | 结果 |
|---|---|
| 标记点偏差（64 点）`max\|dX\|` | 5.626e-2 m = **1.80 h**（`compare.py` 汇总）；按最近帧对齐为 4.85e-2 m = 1.55 h |
| 早期段 `t ≤ 0.01`（20 帧）`max\|dX\|` | 7.82e-3 m = **0.25 h** |
| 中点标记点位移峰值 AFSI / IB2d | 0.2285 / 0.2212 m（差 3.3%） |
| 中点标记点 y 下拍最低点 AFSI / IB2d | 0.3964 / 0.4038（差 0.0074 m = 0.24 h） |
| 流体 `max\|u\|` 峰值 AFSI / IB2d | 15.44 / 16.16（差 4.4%），峰值时刻 `t = 0.0086` |
| 流体 `max\|du\|`（全程） | 9.598 m/s |
| 末帧 `max\|u\|` AFSI / IB2d | 2.546 / 9.360（**相位漂移**，非幅度差） |
| 相对 L2（速度场，全程） | 9.42e-1（主要由相位漂移造成） |

**能验证**：

* 不变梁 + 目标点机制：早期段（`t ≤ 0.01`）梁形状几乎重合，`max|dX| = 0.25 h`，
  中点下拍深度与回弹幅度一致（`max|u|` 峰值差 4.4%）；
* 大幅甩动的**整体幅度**一致：中点标记点位移峰值相差 3.3%，两边 `max|u|` 都在
  ~15–16 m/s；
* 与 demo_446（pyIB2d 版同一物理、独立移植）的结论互相印证：那里记录
  标记点偏差峰值 0.056 m（≈1.8 h），本 demo `compare.py` 给出 5.63e-2 m（1.80 h）。

**不能验证（如实记录）**：

* 长时间**相位**：梁是刚度为 `7.5e9` 的"近刚性"振子，微小的离散差会累积成相位差，
  `t = 0.05` 时两边的瞬时速度已明显不同相（末帧 2.55 vs 9.36 m/s），
  不能用末帧瞬时量判断机制对错；
* 瞬时速度/涡量场：`max|du| = 9.6 m/s`，相对 L2 0.94；该算例在 `32 x 32` 上
  本身欠分辨（`h = 1/32`，IB 支撑带 4h 覆盖 1/8 个盒子），IB2d 的共置投影在
  支撑带内会长出网格尺度尖峰；
* 网格收敛性：官方只提供 `32²`（`Beam_256` 是另一套数据），未做加密检验。

## 运行

```bash
conda activate afsi-dolfinx
python run_reference.py --tend 0.05 --dt 5e-5 --print-dump 10 --nx 32   # 参考解（~7 s）
NX=32 DT=5e-5 TFINAL=0.05 PRINT_DUMP=10 python main.py                 # AFSI（~11 s）
python compare.py
```

环境变量：`NX`（默认官方 32）、`NY`、`DT`、`TFINAL`、`PRINT_DUMP`、
`OUTPUT_PATH`（默认 `./plot`）、`GRAD_DIV`（默认 0；demo_446 实测 `γ=100` 略差，
故保持 0）。

产物：`plot/afsi_result.npz`、`ib2d_reference.npz`、
`figures/compare_profiles.png`、`figures/compare_history.png`、
`figures/compare_fields.png`、`figures/compare_table.csv`。
