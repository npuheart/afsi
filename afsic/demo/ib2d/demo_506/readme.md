# demo_506 — IB2d `Example_VIV_Cylinder/Single_VIV_Cylinder_512_Rectangle`

IB2d 官方算例（`matIB2d/Examples/Example_VIV_Cylinder/Single_VIV_Cylinder_512_Rectangle`）
在 AFSI 中的复现：**弹簧 + 不变梁 + 目标点 + 任意外力脉冲射流**
（矩形水槽 + 半周期占空比的脉冲）。

## 算例内容

* 矩形盒 `Lx = 1.0`、`Ly = 0.25`、`mu = 1e-3`、`rho = 1`、
  官方 `512 x 128` → 本 demo **`NX = 128`（`Ny = 32`）**、
  `dt = 2.5e-5`（官方值）；官方 `T = 1.0` → 本 demo **`T = 0.01`（400 步）**、
  `print_dump = 100`（4 帧）。
* **结构**：2398 个拉格朗日点：**695 根弹簧**、**462 根不变梁**、
  **1936 个目标点**。
* **驱动**（`please_Compute_External_Forcing.m` → `ib2d_input/*.py`）：
  矩形盒下移 `yShift = 0.375`，窗口 `x in [0.1, 0.108]`、
  `y in [0.035, 0.225]`，`uX_tar = -uMax 5 sin(freq_P 2 pi t)
  ((Lx/2-yShift+w/2)-y)((Lx/2-yShift-w/2)-y)`，`freq_P = 10`、
  `rest = 10*period`；**与 256 矩形变体的唯一差别**：脉冲条件是
  `t_mod <= period/2`（半周期占空比，上游 copy-paste，保留原样）。
  上游 `.m` 这里正确使用 `zeros(Ny, Nx)`。

## 移植要点

1. 外力模块逐行移植（含 `HALF_PERIOD = True` 的忠实保留），
   与逐行转写逐点比对（差 `0.0`）。
2. **缩减**：`NX = 128`（官方 512）、`T = 0.01`（官方 1.0）。

## 结果

参考运行约 47 s；AFSI 模拟耗时 18.4 s（墙钟另含导入）。
4 帧对齐（`t = 0 .. 0.00753`）。`compare.py` 原文：

| 指标（末帧） | compare.py 原文 |
|---|---|
| 标记点最大偏差 | `max|dX| 5.008e-04 m = 6.41e-02 h (2398 Lagrangian points)` |
| 速度峰值 AFSI / IB2d | `1.5958e+00 / 4.0283e+00 (6.04e-01 relative)` |
| 速度场偏差 | `max|du| 1.478e+00 = 6.33e-01 max|u|`（带外）; `3.803e+00 (0.06 max|u|)`（带内）; `relative L2 9.066e-01` |
| 力场偏差（带外） | `max|df| 5.324e+03 away from the band, |f|max 6.435e+03` |
| 外力窗口 | `24 nodes, max|u| AFSI 1.5958 / IB2d 1.1123 (4.35e-01 relative); forcing check |f_x - F_x^module(u^AFSI)| <= 2.133e+00 on the core nodes` |
| 弹簧平均长度 | `AFSI 0.022651 / IB2d 0.022651 (1.05e-06 relative)` |

**能验证**：圆柱运动学（6.4e-2 h）、**弹簧平均长度 1.1e-6**、
外力模块与 AFSI 力场自洽（残差 2.1，相对 `|f|max = 6.4e3` 约 3e-4）。
**不能验证（如实记录）**：速度场（相对 $L_2$ 0.91；峰值差 0.60）：
矩形盒只有 32 行节点、`dy = 0.0078`，脉冲靶速峰值 `5 uMax = 375 m/s`，
`T = 0.01` 只覆盖 1/5 个脉冲（半周期占空比）。

## 运行

```bash
conda activate afsi-dolfinx
python run_reference.py --tend 0.01 --print-dump 100 --nx 128    # ~47 s
TFINAL=0.01 PRINT_DUMP=100 NX=128 python main.py                 # 18.4 s
python compare.py
```
