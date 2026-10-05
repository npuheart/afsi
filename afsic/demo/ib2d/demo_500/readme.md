# demo_500 — IB2d `Example_VIV_Cylinder/Single_VIV_Cylinder_128`

IB2d 官方算例（`matIB2d/Examples/Example_VIV_Cylinder/Single_VIV_Cylinder_128`）
在 AFSI 中的复现：**弹簧 + 不变梁 + 目标点（弹性支承的圆柱）+ 任意外力脉动来流**
（涡激振动 VIV 的最小算例）。

## 算例内容

* `1 x 1` 周期盒、`mu = 0.01`、`rho = 1`、**官方 `128 x 128` 与
  `dt = 2.5e-5` 原样保留**；官方 `T = 0.9`（36000 步）→
  本 demo **`T = 0.01`（400 步）**、`print_dump = 100`（4 帧）。
* **结构**：570 个拉格朗日点：**164 根弹簧**、**108 根不变梁**、
  **462 个目标点**——一个被弹性支承（弹簧 + 目标点回位）的圆柱，
  在来流中会横向振动（VIV）。
* **驱动**（`please_Compute_External_Forcing.m` → `ib2d_input/*.py`）：
  窗口 `x in [0.1, 0.16]`、`y in [0.41, 0.59]`，

  ```
  Fx = -kStiff (uX - uX_tar),  uX_tar = -uMax 5 sin(freq_P 2 pi t) ((Lx/2+w/2)-y)((Lx/2-w/2)-y),
  freq_P = 10 Hz, 脉冲占空比：on 一个周期、off 两个周期（rest = 2*period）,
  w = 0.2, uMax = 75, kStiff = 1e4,
  ```

  即周期为 `0.1 s` 的抛物线射流脉冲。**注意**：上游 `T = 0.9` 覆盖 9 个脉冲，
  本 demo 的 `T = 0.01` 只覆盖**第一个脉冲的前 10%**。

## 移植要点

1. 外力模块逐行移植，与逐行转写逐点比对（差 `0.0`）。
2. 上游 `.m` 里 `zeros(Nx, Ny)` + `fx(j, i)`（j = y 索引）在方形网格上碰巧
   正确（该算例 `128 x 128` 恰好方形），移植版统一用 `(Ny, Nx)`。
3. `viv_geo.*` 1-based → 自动减 1。
4. **只缩减 `Tfinal`**：网格与 `dt` 都是官方值。

## 结果

参考运行约 137 s；AFSI 模拟耗时 101.1 s（墙钟另含 1-3 min 解释器/库导入，
400 步）。
4 帧对齐（`t = 0 .. 0.00753`）。`compare.py` 原文：

| 指标（末帧） | compare.py 原文 |
|---|---|
| 标记点最大偏差 | `max|dX| 3.595e-04 m = 4.60e-02 h (570 Lagrangian points)` |
| 速度峰值 AFSI / IB2d | `1.5657e+00 / 1.6882e+00 (7.25e-02 relative)` |
| 速度场偏差 | `max|du| 8.919e-01 = 4.72e-01 max|u|`（带外）; `1.184e+00 (0.30 max|u|)`（带内）; `relative L2 4.213e-01` |
| 力场偏差（带外） | `max|df| 1.274e+03 away from the band, |f|max 7.403e+03` |
| 外力窗口 | `184 nodes, max|u| AFSI 1.5657 / IB2d 1.4652 (6.86e-02 relative); forcing check |f_x - F_x^module(u^AFSI)| <= 1.607e+00 on the core nodes` |
| 弹簧平均长度 | `AFSI 0.024042 / IB2d 0.024042 (1.16e-05 relative)` |

**能验证**：圆柱的运动学（标记点 4.6e-2 h）、**弹簧平均长度 1.2e-5**、
速度峰值 7.3e-2、**外力窗口内速度峰值 6.9e-2**、外力模块与 AFSI 力场自洽
（残差 1.6，相对 `|f|max = 7.4e3` 约 2e-4）。
**不能验证（如实记录）**：全场速度场逐点值（相对 $L_2$ 0.42）。
`T = 0.01` 只有 1/30 个脉冲周期（上游 `T = 0.9`），
圆柱刚开始被推、涡街尚未形成；且 `dz = 0.0078` 与圆柱直径同量级。

## 运行

```bash
conda activate afsi-dolfinx
python run_reference.py --tend 0.01 --print-dump 100 --nx 128    # ~130 s
TFINAL=0.01 PRINT_DUMP=100 NX=128 python main.py                 # ~260 s
python compare.py
```
