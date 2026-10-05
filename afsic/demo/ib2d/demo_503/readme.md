# demo_503 — IB2d `Example_VIV_Cylinder/Single_VIV_Cylinder_512`

IB2d 官方算例（`matIB2d/Examples/Example_VIV_Cylinder/Single_VIV_Cylinder_512`）
在 AFSI 中的复现：**弹簧 + 不变梁 + 目标点 + 任意外力脉动射流**（VIV 高分辨率变体）。

## 算例内容

* `1 x 1` 周期盒、`mu = 1e-3`、`rho = 1`、官方 `512 x 512` →
  本 demo **`NX = 128`（`Ny = 128`）**、`dt = 2.5e-5`（官方值）；
  官方 `T = 0.9`（36000 步）→ 本 demo **`T = 0.01`（400 步）**、
  `print_dump = 100`（4 帧）。
* **结构**：2330 个拉格朗日点：**662 根弹簧**、**440 根不变梁**、
  **1890 个目标点**（与 128 变体同一几何，只是 `ds` 更细）。
* **驱动**：其 `.m` 与 `Single_VIV_Cylinder_128`（demo_500）**逐字节相同**：
  窗口 `x in [0.1, 0.16]`、`y in [0.41, 0.59]`，
  `uX_tar = -uMax 5 sin(freq_P 2 pi t) ((Lx/2+w/2)-y)((Lx/2-w/2)-y)`、
  `freq_P = 10`、`rest = 2*period`、`w = 0.2`、`uMax = 75`、`kStiff = 1e4`。
  见 demo_500 的移植版（同一份推导）。

## 移植要点

1. 外力模块逐行移植（与 demo_500 相同），与逐行转写逐点比对（差 `0.0`）。
2. 上游 `zeros(Nx, Ny)` + `fx(j, i)` 在方形网格上正确；移植版用 `(Ny, Nx)`。
3. **缩减**：`NX = 128`（官方 512）、`T = 0.01`（官方 0.9）。

## 结果

参考运行约 134 s；AFSI 模拟耗时 101.1 s（墙钟另含导入）。
4 帧对齐（`t = 0 .. 0.00753`）。`compare.py` 原文：

| 指标（末帧） | compare.py 原文 |
|---|---|
| 标记点最大偏差 | `max|dX| 9.952e-04 m = 1.27e-01 h (2330 Lagrangian points)` |
| 速度峰值 AFSI / IB2d | `1.7513e+00 / 3.4530e+00 (4.93e-01 relative)` |
| 速度场偏差 | `max|du| 9.943e-01 = 7.12e-01 max|u|`（带外）; `2.556e+00 (0.26 max|u|)`（带内）; `relative L2 6.177e-01` |
| 力场偏差（带外） | `max|df| 1.758e+03 away from the band, |f|max 5.833e+03` |
| 外力窗口 | `184 nodes, max|u| AFSI 1.6637 / IB2d 1.5039 (1.06e-01 relative); forcing check |f_x - F_x^module(u^AFSI)| <= 1.745e+00 on the core nodes` |
| 弹簧平均长度 | `AFSI 0.022673 / IB2d 0.022673 (4.83e-06 relative)` |

**能验证**：圆柱运动学（0.13 h）、**弹簧平均长度 4.8e-6**、
**外力窗口内速度峰值 1.1e-1**、外力模块与 AFSI 力场自洽
（残差 1.7，相对 `|f|max = 5.8e3` 约 3e-4）。
**不能验证（如实记录）**：瞬时速度峰值（相对差 0.49）与全场速度
（相对 $L_2$ 0.62）：`T = 0.01` 只有 1/9 个脉冲（上游 `T = 0.9`），
涡街未形成。

## 运行

```bash
conda activate afsi-dolfinx
python run_reference.py --tend 0.01 --print-dump 100 --nx 128    # ~134 s
TFINAL=0.01 PRINT_DUMP=100 NX=128 python main.py                 # 101.1 s
python compare.py
```
