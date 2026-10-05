# demo_504 — IB2d `Example_VIV_Cylinder/Single_VIV_Cylinder_Damped_Springs_256`

IB2d 官方算例（`matIB2d/Examples/Example_VIV_Cylinder/Single_VIV_Cylinder_Damped_Springs_256`）
在 AFSI 中的复现：**弹簧 + 阻尼弹簧 + 不变梁 + 目标点 + 任意外力脉动射流**
（带阻尼支承的 VIV）。

## 算例内容

* `1 x 1` 周期盒、`mu = 1e-3`、`rho = 1`、官方 `256 x 256` →
  本 demo **`NX = 128`（`Ny = 128`）**、`dt = 2.5e-5`（官方值）；
  官方 `T = 1.0`（40000 步）→ 本 demo **`T = 0.01`（400 步）**、
  `print_dump = 100`（4 帧）。
* **结构**：1194 个拉格朗日点：**339 根弹簧**（`k = 2e7`）、
  **2 根阻尼弹簧**（`kappa = 2e4`、`b = 5`，`viv_geo.d_spring`）、
  **226 根不变梁**、**968 个目标点**。阻尼弹簧让圆柱的振动被耗散
  （对比 demo_502 的无阻尼变体）。
* **驱动**（`please_Compute_External_Forcing.m` → `ib2d_input/*.py`）：
  与 demo_502（`Single_VIV_Cylinder_256`）**逐字节相同的模块**：
  窗口 `x in [0.1, 0.108]`、`y in [0.41, 0.59]`，`freq_P = 10`、
  `rest = 10*period`、`w = 0.2`、`uMax = 75`、`kStiff = 1e4`。

## 移植要点

1. 外力模块逐行移植（同 demo_502），与逐行转写逐点比对（差 `0.0`）。
2. **阻尼弹簧**由模板的 `damped_spring_forces` 覆盖（IB2d
   `give_Me_Damped_Spring_Lagrangian_Force_Densities`：`F = k(L-L0) + b v`）；
   本算例 `update_damp_springs = 0`，不需要用户模块。
3. **缩减**：`NX = 128`、`T = 0.01`。

## 结果

参考运行约 132 s；AFSI 模拟耗时 101.2 s（墙钟另含导入）。
4 帧对齐（`t = 0 .. 0.00753`）。`compare.py` 原文：

| 指标（末帧） | compare.py 原文 |
|---|---|
| 标记点最大偏差 | `max|dX| 3.934e-04 m = 5.04e-02 h (1194 Lagrangian points)` |
| 速度峰值 AFSI / IB2d | `1.5976e+00 / 3.3426e+00 (5.22e-01 relative)` |
| 速度场偏差 | `max|du| 9.175e-01 = 7.26e-01 max|u|`（带外）; `3.112e+00 (0.07 max|u|)`（带内）; `relative L2 8.875e-01` |
| 力场偏差（带外） | `max|df| 4.055e+03 away from the band, |f|max 5.147e+03` |
| 外力窗口 | `23 nodes, max|u| AFSI 1.5976 / IB2d 1.2487 (2.79e-01 relative); forcing check |f_x - F_x^module(u^AFSI)| <= 2.122e+00 on the core nodes` |
| 弹簧平均长度 | `AFSI 0.022840 / IB2d 0.022840 (4.06e-08 relative)` |

**能验证**：圆柱运动学（5.0e-2 h）、**弹簧平均长度 4.1e-8（机器精度级）**、
外力模块与 AFSI 力场自洽（残差 2.1，相对 `|f|max = 5.1e3` 约 4e-4）。
**不能验证（如实记录）**：速度场（相对 $L_2$ 0.89）。与 demo_502 同为
"只跑了 1/3 个脉冲" 的限制；注意本算例在 `T = 0.01` 内阻尼还没明显作用
（末帧标记点偏差与无阻尼变体几乎相同）。

## 运行

```bash
conda activate afsi-dolfinx
python run_reference.py --tend 0.01 --print-dump 100 --nx 128    # ~132 s
TFINAL=0.01 PRINT_DUMP=100 NX=128 python main.py                 # 101.2 s
python compare.py
```
