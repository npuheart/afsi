# demo_496 — IB2d `Examples_First_Year_Seminar/Flow_Around_Object`

IB2d 官方算例（`matIB2d/Examples/Examples_First_Year_Seminar/Flow_Around_Object`）
在 AFSI 中的复现：**目标点（固定刚体）+ 任意外力驱动**。

## 算例内容

* `1 x 1` 周期盒、`mu = 0.01`、`rho = 1`、`128 x 128`（**官方网格**）、
  `dt = 5e-5`。官方 `T = 1.25`（25000 步）→ 本 demo **`T = 0.05`（1000 步）**，
  `print_dump = 50`（20 帧）。
* **结构**：570 个拉格朗日点（`channel.vertex`）全部由 **570 个目标点**
  锚在初始位置（`k = 1e7`）——即一个**完全固定的刚体**，
  放在通道射流中充当障碍物（"Flow Around Object"）。
* **驱动**：与 demo_495/468 同族的抛物线射流，
  `Fx = -kStiff (uX - uX_tar)`、`uX_tar = -uMax 5 tanh(t) ((Lx/2+w/2)-y)((Lx/2-w/2)-y)`，
  `kStiff = 1e4`、`w = 0.2`、`uMax = 250`，窗口
  `x in [0.1, 0.16]`、`y in [0.41, 0.59]`。

## 移植要点

1. `please_Compute_External_Forcing.m` 逐行移植为
   `ib2d_input/please_Compute_External_Forcing.py`，两个代码共用同一份；
   索引搜索用 `np.searchsorted` 复刻，并与逐行转写逐点比对（差为 `0.0`）。
2. 结构完全由目标点钉死，模板的 `target_forces` 覆盖（含 `|d| > L/2` 的
   周期折叠）；**没有弹簧/梁**，因此 `compare.py` 不打印弹簧长度。
3. 上游 `.m` 的 `zeros(Nx, Ny)` + `fx(j, i)` 写法在方形网格上碰巧正确，
   移植版统一用 `(Ny, Nx)`。

## 结果

参考运行约 79 s、AFSI 运行约 298 s（含导入，机器并行运行其它 demo）。
20 帧对齐（`t = 0 .. 0.04755`）。`compare.py` 原文：

| 指标（末帧） | compare.py 原文 |
|---|---|
| 标记点最大偏差 | `max|dX| 5.923e-06 m = 7.58e-04 h (570 Lagrangian points)` |
| 速度峰值 AFSI / IB2d | `8.4453e-01 / 8.8274e-01 (4.33e-02 relative)` |
| 速度场偏差 | `max|du| 1.391e-01 = 8.42e-01 max|u|`（带外）; `1.224e-01 (0.86 max|u|)`（带内）; `relative L2 1.475e-01` |
| 力场偏差（带外） | `max|df| 8.924e+01 away from the band, |f|max 7.170e+02` |
| 外力窗口 | `184 nodes, max|u| AFSI 0.5783 / IB2d 0.5726 (9.98e-03 relative); forcing check |f_x - F_x^module(u^AFSI)| <= 8.368e-02 on the core nodes` |

**能验证**：**刚体完全不动**（目标点偏差 5.9e-6 m = 7.6e-4 h，即两边都把
结构钉死了）；外力窗口内速度峰值一致到 1.0e-2；外力模块与 AFSI 力场自洽
（残差 8.4e-2）。
**不能验证（如实记录）**：物体尾迹中的瞬时速度场逐点值（`max|du|` 达
0.84 `max|u|`，相对 $L_2$ 1.5e-1）。物体是"尖角"离散点集，绕流分离点在
两种离散下不同；`T = 0.05` 也只覆盖射流开启（`tanh(t)`）的头 0.05 s。

## 运行

```bash
conda activate afsi-dolfinx
python run_reference.py --tend 0.05 --print-dump 50              # 官方 128²，~79 s
TFINAL=0.05 PRINT_DUMP=50 python main.py                         # ~298 s
python compare.py
```
