# demo_502 — IB2d `Example_VIV_Cylinder/Single_VIV_Cylinder_256`

IB2d 官方算例（`matIB2d/Examples/Example_VIV_Cylinder/Single_VIV_Cylinder_256`）
在 AFSI 中的复现：**弹簧 + 不变梁 + 目标点 + 任意外力脉动射流**（VIV 方盒变体）。

## 算例内容

* `1 x 1` 周期盒、`mu = 1e-3`、`rho = 1`、官方 `256 x 256` →
  本 demo **`NX = 128`（`Ny = 128`）**、`dt = 2.5e-5`（官方值）；
  官方 `T = 1.0`（40000 步）→ 本 demo **`T = 0.01`（400 步）**、
  `print_dump = 100`（4 帧）。
* **结构**：1194 个拉格朗日点：**341 根弹簧**、**226 根不变梁**、
  **968 个目标点**——弹性支承的圆柱。
* **驱动**（`please_Compute_External_Forcing.m` → `ib2d_input/*.py`）：
  窄入流窗口 `x in [0.1, 0.108]`、`y in [0.41, 0.59]`，

  ```
  uX_tar = -uMax 5 sin(freq_P 2 pi t) ((Lx/2+w/2)-y)((Lx/2-w/2)-y),
  freq_P = 10, rest = 10*period（长间歇）, w = 0.2, uMax = 75, kStiff = 1e4,
  ```

  上游 `.m` 里 `zeros(Nx, Ny)` + `fx(j, i)` 在该算例是**方形**网格
  （`256 x 256`）所以碰巧正确；移植版统一用 `(Ny, Nx)`。

## 移植要点

1. 外力模块逐行移植，与逐行转写逐点比对（差 `0.0`）。
2. **缩减**：`NX = 128`（官方 256；`x` 窗口需 `dx <= 0.008`）、
   `T = 0.01`（官方 1.0）。

## 结果

参考运行约 132 s；AFSI 模拟耗时 101.1 s（墙钟另含导入）。
4 帧对齐（`t = 0 .. 0.00753`）。`compare.py` 原文：

| 指标（末帧） | compare.py 原文 |
|---|---|
| 标记点最大偏差 | `max|dX| 3.957e-04 m = 5.07e-02 h (1194 Lagrangian points)` |
| 速度峰值 AFSI / IB2d | `1.5976e+00 / 3.3424e+00 (5.22e-01 relative)` |
| 速度场偏差 | `max|du| 9.175e-01 = 7.26e-01 max|u|`（带外）; `3.112e+00 (0.07 max|u|)`（带内）; `relative L2 8.875e-01` |
| 力场偏差（带外） | `max|df| 4.054e+03 away from the band, |f|max 5.146e+03` |
| 外力窗口 | `23 nodes, max|u| AFSI 1.5976 / IB2d 1.2488 (2.79e-01 relative); forcing check |f_x - F_x^module(u^AFSI)| <= 2.123e+00 on the core nodes` |
| 弹簧平均长度 | `AFSI 0.023097 / IB2d 0.023097 (3.08e-06 relative)` |

**能验证**：圆柱运动学（5.1e-2 h）、**弹簧平均长度 3.1e-6**、
外力模块与 AFSI 力场自洽（残差 2.1，相对 `|f|max = 5.1e3` 约 4e-4）。
**不能验证（如实记录）**：速度峰值（相对差 0.52）与全场速度
（相对 $L_2$ 0.89）。`T = 0.01` 只有 1/3 个脉冲（周期 0.1 s，
上游 `T = 1.0` 覆盖 10 个脉冲），涡街远未形成；`dx = 0.0078` 与圆柱
直径同量级。

## 运行

```bash
conda activate afsi-dolfinx
python run_reference.py --tend 0.01 --print-dump 100 --nx 128    # ~132 s
TFINAL=0.01 PRINT_DUMP=100 NX=128 python main.py                 # 101.1 s
python compare.py
```
