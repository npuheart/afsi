# demo_501 — IB2d `Example_VIV_Cylinder/Single_VIV_Cylinder_256_Rectangle`

IB2d 官方算例（`matIB2d/Examples/Example_VIV_Cylinder/Single_VIV_Cylinder_256_Rectangle`）
在 AFSI 中的复现：**弹簧 + 不变梁 + 目标点 + 任意外力脉冲射流**（VIV 的
"矩形水槽"变体）。

## 算例内容

* 矩形盒 `Lx = 1.0`、`Ly = 0.25`、`mu = 1e-3`、`rho = 1`、
  官方 `256 x 64` → 本 demo **`NX = 128`（`Ny = 32`）**、`dt = 2.5e-5`（官方值）；
  官方 `T = 1.0`（40000 步）→ 本 demo **`T = 0.01`（400 步）**、
  `print_dump = 100`（4 帧）。
* **结构**：1194 个拉格朗日点：**341 根弹簧**、**226 根不变梁**、
  **968 个目标点**——弹性支承的圆柱。
* **驱动**（`please_Compute_External_Forcing.m` → `ib2d_input/*.py`）：
  矩形盒把抛物线剖面**下移** `yShift = 0.375`：

  ```
  uX_tar = -uMax 5 sin(freq_P 2 pi t) ((Lx/2-yShift+w/2)-y)((Lx/2-yShift-w/2)-y),
  freq_P = 10, rest = 10*period, w = 0.2, uMax = 75, kStiff = 1e4,
  窗口 x in [0.1, 0.108], y in [0.035, 0.225],
  ```

  上游 `.m` 这里正确地用了 `zeros(Ny, Nx)`（矩形网格上必须）。

## 移植要点

1. 外力模块逐行移植，与逐行转写逐点比对（差 `0.0`）。
2. **缩减**：`NX = 128`（官方 256）、`T = 0.01`（官方 1.0）。
   `x` 窗口 `[0.1, 0.108]` 要求 `dx <= 0.008`，`NX = 128` 恰好让窗口剩一列节点。

## 结果

参考运行约 137 s；AFSI 模拟耗时 33.3 s（墙钟另含 1-3 min 解释器/库导入）。4 帧对齐
（`t = 0 .. 0.00753`）。`compare.py` 原文：

| 指标（末帧） | compare.py 原文 |
|---|---|
| 标记点最大偏差 | `max|dX| 4.894e-04 m = 6.26e-02 h (1194 Lagrangian points)` |
| 速度峰值 AFSI / IB2d | `1.5958e+00 / 3.9782e+00 (5.99e-01 relative)` |
| 速度场偏差 | `max|du| 1.507e+00 = 6.21e-01 max|u|`（带外）; `3.753e+00 (0.06 max|u|)`（带内）; `relative L2 9.016e-01` |
| 力场偏差（带外） | `max|df| 5.189e+03 away from the band, |f|max 6.300e+03` |
| 外力窗口 | `24 nodes, max|u| AFSI 1.5958 / IB2d 1.1241 (4.20e-01 relative); forcing check |f_x - F_x^module(u^AFSI)| <= 2.105e+00 on the core nodes` |
| 弹簧平均长度 | `AFSI 0.023097 / IB2d 0.023097 (2.95e-06 relative)` |

**能验证**：圆柱运动学（6.3e-2 h）、**弹簧平均长度 3.0e-6**、
外力模块与 AFSI 力场自洽（残差 2.1，相对 `|f|max = 6.3e3` 约 3e-4）。
**不能验证（如实记录）**：速度峰值与全场速度场（相对差 0.6 / $L_2$ 0.90）。
原因：矩形盒只有 32 行节点、`dy = 0.0078`，而脉冲靶速峰值
`5 uMax = 375 m/s`；`T = 0.01` 又只有 1/3 个脉冲（周期 0.1 s）。
上游 `256 x 64` + `T = 1.0` 在本机不可行。

## 运行

```bash
conda activate afsi-dolfinx
python run_reference.py --tend 0.01 --print-dump 100 --nx 128    # ~180 s
TFINAL=0.01 PRINT_DUMP=100 NX=128 python main.py                 # ~280 s
python compare.py
```
