# demo_671 — IB2d `Examples_Education/Pendulum/PointMass-512x512`（质量点摆 / 512²）

## 上游路径

* 数据：`third_party/ib2d/matIB2d/Examples/Examples_Education/Pendulum/PointMass-512x512/`
  （`pendulum.vertex`、`pendulum.spring`、`pendulum.target`、`pendulum.mass`、
  `input2d`，以及生成几何的 `Give_Pendulum_Geometry_PointMass.m`）。
* 算法源码：`third_party/ib2d/pyIB2d/IBM_Blackbox/{IBM_Driver.py, Supp.py,
  please_Find_Lagrangian_Forces_On_Eulerian_grid.py}`。

本目录是 demo_670（`PointMass-256x256`）的**512² 版本**：代码从 demo_670 派生，
只有 `ib2d_input/input2d` 不同。`PointMass-256x256` / `-512x512` /
`-1024x1024` 三个上游目录的 `pendulum.{vertex,spring,target,mass}` **逐字节相同**，
只有 `Nx = Ny`（256 / 512 / 1024）不同——是一组网格加密研究；本仓库把缩减后的
`NX` 也连成一串（demo_670 `NX = 64` / demo_671 `NX = 96` / demo_672 `NX = 128`）。

## 算例内容（官方设置）

| 项 | 值 |
|---|---|
| 流体 | `1 x 1` 周期盒、`mu = 0.01`、`rho = 1000`、**`512 x 512`** |
| 时间 | `dt = 2.5e-5`、官方 `Tfinal = 5.0`（200000 步） |
| 结构 | 2 个标记点：支点 marker 2 `(0.5, 2/3)`（被 target 钉死），摆锤 marker 1 `(0.33820, 0.54911)` |
| 弹簧 | 1 根线性弹簧（`k = 1e8`、`L0 = 0.2 m`），即摆杆 |
| 目标点 | 1 个（marker 2，`k = 5e7`） |
| 质量点 | 1 个（挂在 marker 1，`k = 2.5e6`、`M = 1000 kg`），即摆锤质量 |
| 重力 | 开（`gvec = (0, -1)`，`g = 9.80665`） |
| 梁 / 外力 / 标量 / 示踪 | 全关 |

初始角约偏离竖直 54°，小角度摆周期 `2π√(L/g) ≈ 0.90 s`。物理设置与 demo_670
完全一致，差别只在官方网格（`512²` vs `256²`）与本 demo 采用的重网格（`96²` vs `64²`）。

## 移植要点

质量点算法的移植（ghost 半步移动 → 弹簧力加到标记点力数组 → 半步速度 → Move →
整步速度）、以及本算例命中的上游 bug（**单行 `.spring` 的 `n, m = springs.shape`
崩溃**、**单行 `.target` 的 `aux[:, 0]` 崩溃**、`.mass` 单列索引、"只有 1 个质量点"
分支坏掉、pyIB2d 不输出 ghost、dump 标号偏移）与 demo_670 完全相同，见
[demo_670 的 readme](../demo_670/readme.md) 中"移植要点"一节；`run_reference.py`
里的四处补丁与 `main.py` 的 ghost ODE 一字未改。

`Give_Pendulum_Geometry_PointMass.m` 只是生成数据文件的 MATLAB 脚本，pyIB2d 驱动
不调用，故未移植（数据文件已原样附带）。

### ghost 算法的"复算"检验

`run_reference.py` 记录每步驱动实际使用的半步标记点位置并复算本移植的 ghost ODE：
**2001 步 `max|dXm| = 0.000e+00 m`**（与 pyIB2d 逐位一致）。

## 缩减设置（重要）

官方 `512 x 512`、`dt = 2.5e-5`、`Tfinal = 5.0`。本 demo 用
**`NX = 96`（官方约 1/5）、`dt = 5e-5`（官方 2 倍）、`T = 0.1 s`（2001 步）**。

* 网格：`512²` 在 AFSI 上单步代价约为 `96²` 的 28 倍，缩到 `NX = 96` 后单次
  运行 79 s、约 740 MB，落在预算内。上游 `256²/512²/1024²` 的加密阶梯在本仓库
  用 `NX = 64/96/128` 复现。
* `dt`：摆杆 `k = 1e8` 极硬，显式 IB 耦合稳定域窄。`dt = 1e-4` 在 `NX = 64`
  下 `t ≈ 0.3` 起失稳（见 demo_670 的 readme），`dt = 5e-5` 稳定且已收敛。

**与 demo_670 / demo_672 的对比**：同一物理在 `NX = 64 / 96 / 128` 下 AFSI 的
headline 误差分别是 `1.19e-1 h` / `5.01e-3 h` / `5.16e-2 h`（绝对值
`1.85e-3` / `5.2e-5` / `4.0e-4 m`）。本 demo（`NX = 96`）最小，但这**不是**
"AFSI 在 `NX = 96` 更准"的证据：pyIB2d 参考解本身在该网格族上就没有收敛
（`NX = 64/96/128` 之间末帧 ghost 位置变化 1.7e-3 / 9.2e-4 m），
本值更小属于两种离散误差在该网格上的部分抵消；详见 demo_672 的 readme。
三个 demo headline 误差都落在参考解自身的网格离散带（~1e-3 m）附近或以下。

## 结果

运行时间（macOS，本 demo 实测）：pyIB2d 参考 **4.8 s**（2001 步驱动，总进程 8.8 s）、
AFSI **79.0 s**（2001 步）。两边各 9 帧（含 `t = 0`），`t = 0 .. 0.10005`。

| 指标（末帧 `t = 0.10005`） | 结果 |
|---|---|
| **ghost（摆锤质量）轨迹 max\|dXm\|**（headline） | 5.221e-5 m = **5.01e-3 h** |
| ghost 速度偏差 `max\|dVm\|` | 3.83e-2 m/s |
| **ghost−marker 偏移之差**（纯弹簧耦合，机制本身） | 1.816e-4 m = **1.74e-2 h** |
| 标记点偏差（2 点） | 2.274e-4 m = 2.18e-2 h |
| 参考解末帧 ghost 位移 | 0.0244 m（相对误差 ≈ 0.21%） |
| 流体 `max\|u\|` AFSI / IB2d | 0.749 / 0.826（相对差 9.3%），`max\|du\|` = 0.537 |
| 复算检验（同一标记点轨迹下的 ghost ODE） | 2001 步 `max\|dXm\| = 0.000e+00 m` |

**能验证**：

* **质量点算法**：用 pyIB2d 自己的每步标记点轨迹复算移植的 ghost ODE，
  2001 步逐位一致（`0.0`）；
* **总体运动**：摆锤被重力驱动向下摆动，末帧 ghost 位移 `0.0244 m`，
  AFSI 轨迹偏差仅 `5.2e-5 m`（≈ 参考位移的 0.21%）；
* **网格离散带内的自洽**：AFSI 偏差小于 pyIB2d 参考解自身的网格离散带
  （`NX = 96` 与 `128` 之间 9.2e-4 m），即在该网格上两者一致。

**不能验证（如实记录）**：

* 流体瞬时速度场：`max|u|` 差 9.3%（比 demo_670 的 48.8% 好得多，但仍有）；
  IB2d 的共置 FFT 投影在标记点 4 点 delta 支撑带内会生长网格尺度尖峰，
  AFSI 的 Taylor–Hood + 插值力密度在该带内固有差异；
* 长时间的绝对轨迹：只跑到 `T = 0.1 s`（官方 5.0 s 的 1/50，约摆周期的 1/9）；
* 官方 `dt = 2.5e-5`/`Tfinal = 5` 的"完整"算例未复现（200000 步，超出本机预算）。

## 运行

```bash
conda activate afsi-dolfinx
python run_reference.py --tend 0.1 --dt 5e-5 --print-dump 250 --nx 96   # 参考解（~5 s）
NX=96 DT=5e-5 TFINAL=0.1 PRINT_DUMP=250 python main.py                 # AFSI（~79 s）
python compare.py
```

其他环境变量：`NX`（默认官方 512）、`NY`、`OUTPUT_PATH`（默认 `./plot`）、
`GRAD_DIV`（Taylor–Hood grad-div 稳定化，默认 0）。

产物：`plot/afsi_result.npz`、`ib2d_reference.npz`（含 ghost 历史 `Xm/Vm/mass_M`）、
`figures/compare_trajectories.png`（headline：ghost 轨迹 + 误差）、
`figures/compare_history.png`、`figures/compare_fields.png`、`figures/compare_table.csv`。
