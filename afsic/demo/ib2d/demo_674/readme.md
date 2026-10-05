# demo_674 — IB2d `Example_Wobbly_Beam/nonInv_Beam_32`（摆动梁 / 非不变梁）

## 上游路径

* 数据：`third_party/ib2d/matIB2d/Examples/Example_Wobbly_Beam/nonInv_Beam_32/`
  （`BeamCurve.vertex`、`BeamCurve.nonInv_beam`、`BeamCurve.target`、`input2d`、
  `main2d.m`，以及生成几何的 `BeamCurve.m`）。
* 算法源码：`third_party/ib2d/pyIB2d/IBM_Blackbox/{IBM_Driver.py, Supp.py,
  please_Find_Lagrangian_Forces_On_Eulerian_grid.py}`。
* 代码基：`afsic/demo/ib2d/_template/`（合并模板，已支持非不变梁 + 目标点）。

`BeamCurve.m` 只是生成数据文件的 MATLAB 脚本，pyIB2d 驱动从不调用，故未移植。

## 算例内容

| 项 | 值 |
|---|---|
| 流体 | `1 x 1` 周期盒、`mu = 0.1`、`rho = 1`、**`32 x 32`（官方 = 本 demo 使用）** |
| 时间 | `dt = 1e-5`、`Tfinal = 0.05`（5000 步）（官方 = 本 demo 使用） |
| 结构 | 64 个标记点：与 demo_673 相同的拱（`(0.25, 0.5)` → `(0.75, 0.5)`，顶点 `y ≈ 0.625`） |
| 非不变梁 | 62 组（`r = (X_p + X_r − 2X_q) − C`，`F_q += 2κ r`、`F_p, F_r −= κ r`，`κ = 1e10`，文件中 `C = (0, 0)`） |
| 目标点 | 2 个（两端 id 0 与 63，`k = 2e8`） |
| 弹簧 / 质量点 / 外力 / 标量 | 全关 |

**本 demo 直接使用官方设置**（`32 x 32`、`dt = 1e-5`、`Tfinal = 0.05`，无需缩减）。

与 demo_673（不变梁）的差别：不变梁的零力构型由生成器写入 `C`（拱形即零力），
而非不变梁在释放时**记忆生成时刻**的 `C = 0`（直线才是零力）——所以这根拱一释放
就被 62 根 `κ = 1e10` 的"扭转弹簧"往直线拉，是比 demo_673 更硬的近刚性振子。
实测两边流体速度峰值达 **50–100 m/s**（1×1 盒内），是一个"官方算例在粗网格上
远超分辨能力"的极端情形；但标记点**形状**两边仍高度一致。

## 移植要点

* 非不变梁力按 IB2d 原式在**半步位置**上求值（模板 `noninv_beam_forces` 逐字复刻
  `IBM_Driver.py` 的 `give_Me_nonInv_Beam_Lagrangian_Force_Densities`）；
* 目标点把两端钉在初始位置（`k = 2e8`），周期回绕；
* 结构与 demo_673 共用同一套管线（零刚度 FE 链承载标记点、Peskin 两阶段、4 点核）。

### 上游 quirk

1. **`nonInv_Beam_32` 里附带了一个损坏的 `BeamCurve.beam`**：第 1 行数据 5 列、
   第 2–62 行 6 列（多一个前导 `0`）。因为该算例 `beams = 0`，pyIB2d 从不读它；
   但它会让"按扩展名统一重定基（rebasing）"的脚本崩溃（`np.loadtxt` 报
   "number of columns changed from 5 to 6"），因此本 demo 的 `ib2d_input/` **不含**
   这个无用文件（demo_449 亦然）。真正使用的 `BeamCurve.nonInv_beam` 逐字节原样附带。
2. `print_dump = 50` 时驱动在 `cter % print_dump == 0 且 cter >= print_dump`
   写盘，时间戳为该步步首时间；`main.py` 在与驱动完全相同的步上 dump，
   `compare.py` 按 `round(t/dt)` 对齐（5000 步 → 100 帧，`t = 0 .. 0.04951`）。

## 结果

运行时间（macOS，本 demo 实测）：pyIB2d 参考 **约 15 s**（5000 步驱动 + VTK 写出）、
AFSI **50.1 s**（5000 步）。两边各 100 个对齐帧，`t = 0 .. 0.04951`。

| 指标 | 结果 |
|---|---|
| 标记点偏差（64 点，末帧）`max\|dX\|` | 1.687e-2 m = **0.54 h**（`compare.py` 汇总值） |
| 标记点偏差（全程）`max\|dX\|` | 4.826e-2 m = **1.54 h**；逐帧均值峰值 1.51e-2 m |
| 中点标记点 y 往返范围 AFSI / IB2d | `[0.412, 0.625]` / `[0.393, 0.627]`（幅度接近，AFSI 略浅） |
| 中点标记点位移峰值 AFSI / IB2d | 0.2129 / 0.2323 m（差 8.3%） |
| 流体 `max\|u\|` 峰值 AFSI / IB2d | 58.1 / 101.6 m/s（**都极端**，见下） |
| 流体 `max\|du\|`（全程） | 57.1 m/s |
| 相对 L2（速度场，全程） | 8.49e-1（末帧） |

**能验证**：

* 非不变梁 + 目标点机制：标记点**形状**全程紧随参考解（末帧 `0.54 h`、
  全程峰值 `1.54 h`，`h = 1/32`）；中点 y 的往返范围两边一致
  （`[0.412, 0.625]` vs `[0.393, 0.627]`），梁以 ~0.02 s 周期高频振荡；
* 与 demo_673（不变梁）对照：非不变梁因为 `C = 0` 而被拉向直线，振幅略大、
  `κ` 大 10 倍，符合两个上游算例的设计差别。

**不能验证（如实记录）**：

* 瞬时速度场：`max|u|` 两边都到 50–100 m/s，`max|du| = 57 m/s`；`32 x 32`
  网格（`h = 1/32`）对 `κ = 1e10` 的近刚性梁**远超分辨能力**，IB2d 的共置投影
  还会在 4 点支撑带内长出网格尺度尖峰，故速度场不可比；
* 相位：近刚性高频振子，微小离散差随时间累积成相位漂移；
* 网格收敛性：官方只提供 `32²`，未做加密检验。

## 运行

```bash
conda activate afsi-dolfinx
python run_reference.py --tend 0.05 --dt 1e-5 --print-dump 50 --nx 32   # 参考解（~15 s）
NX=32 DT=1e-5 TFINAL=0.05 PRINT_DUMP=50 python main.py                 # AFSI（~50 s）
python compare.py
```

环境变量：`NX`（默认官方 32）、`NY`、`DT`、`TFINAL`、`PRINT_DUMP`、
`OUTPUT_PATH`（默认 `./plot`）、`GRAD_DIV`（默认 0）。

产物：`plot/afsi_result.npz`、`ib2d_reference.npz`、
`figures/compare_profiles.png`、`figures/compare_history.png`、
`figures/compare_fields.png`、`figures/compare_table.csv`。
