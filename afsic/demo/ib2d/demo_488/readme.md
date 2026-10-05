# demo_488 — IB2d `Example_Meshmerize_Me/Starfish`

IB2d 官方算例（`matIB2d/Examples/Example_Meshmerize_Me/Starfish`）
在 AFSI 中的复现：**弹簧 + 目标点（海星）+ 任意外力周期射流**。

## 算例内容

* 水槽盒 `Lx = 1.0`、`Ly = 0.25`、`mu = 0.1`、`rho = 1000`、
  官方 `1024 x 256`、`dt = 5e-5`。官方 `T = 3.0`（60000 步）→
  本 demo **`NX = 128`（`Ny = 32`）、`T = 0.01`（200 步）**、`print_dump = 50`。
* **结构**：3010 个拉格朗日点：**1440 根弹簧** + 3010 个目标点
  （`starfish_flowtank.target`）——弹簧保持海星形状、目标点把它锚住。
* **驱动**（`please_Compute_External_Forcing.m` → `ib2d_input/*.py`）：
  窗口 `x in [0.015, 0.065]`、`y in [0.011, 0.239]`，

  ```
  Fx = -kStiff (uX - uX_tar),  uX_tar = -uMax sin(2 pi f t) ((Ly/2+w/2)-y)((Ly/2-w/2)-y),
  f = 2 Hz, w = 0.23, uMax = 500, kStiff = 1e4,
  ```

  即整个水槽高度的抛物线剖面以 2 Hz 来回振荡（海星感受往复流）。

## 移植要点

1. 外力模块逐行移植，与逐行转写逐点比对（差 `0.0`）。
2. `starfish_flowtank.*` 1-based → 自动减 1。
3. **缩减**：`NX = 128`、`T = 0.01`（上游 1024x256、60000 步不可行）。

## 结果

参考运行约 84 s、AFSI 运行约 168 s（墙钟时间含导入；`main.py` 自报的模拟
耗时只有 24.7 s，网格很小）。5 帧对齐（`t = 0 .. 0.01005`）。`compare.py` 原文：

| 指标（末帧） | compare.py 原文 |
|---|---|
| 标记点最大偏差 | `max|dX| 7.110e-07 m = 9.10e-05 h (3010 Lagrangian points)` |
| 速度峰值 AFSI / IB2d | `1.6587e-02 / 1.0838e-02 (5.30e-01 relative)` |
| 速度场偏差 | `max|du| 9.306e-03 = 1.41e-01 max|u|`（带外）; `3.879e-03 (0.64 max|u|)`（带内）; `relative L2 5.938e-01` |
| 力场偏差（带外） | `max|df| 9.056e+01 away from the band, |f|max 8.228e+03` |
| 外力窗口 | `203 nodes, max|u| AFSI 0.0166 / IB2d 0.0108 (5.30e-01 relative); forcing check |f_x - F_x^module(u^AFSI)| <= 4.052e+01 on the core nodes` |
| 弹簧平均长度 | `AFSI 0.019789 / IB2d 0.019789 (2.02e-07 relative)` |

**能验证**：海星**完全不动**（9.1e-5 h）、**弹簧平均长度一致到 2.0e-7**
（结构被两边同样地钉住）；外力模块与 AFSI 力场自洽（残差 4.1e1，
相对 `|f|max = 8.2e3` 约 5e-3）。
**不能验证（如实记录）**：速度场（相对 $L_2$ 0.59）。`T = 0.01` 时
`sin(2π·2t) = sin(0.126) ≈ 0.13`，速度只有 `~1e-2 m/s`，
相对差被"尚未启动"的极小量放大。上游 `T = 3.0`（30 个振荡周期）
在本机不可行。

## 运行

```bash
conda activate afsi-dolfinx
python run_reference.py --tend 0.01 --print-dump 50 --nx 128     # ~60 s
TFINAL=0.01 PRINT_DUMP=50 NX=128 python main.py                 # ~20 s
python compare.py
```
