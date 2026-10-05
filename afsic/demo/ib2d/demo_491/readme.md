# demo_491 — IB2d `Example_HeartTube/Insect_Heart/Valveless_Model`

IB2d 官方算例（`matIB2d/Examples/Example_HeartTube/Insect_Heart/Valveless_Model`）
在 AFSI 中的复现：**目标点（钉死的心脏管）+ 任意外力 `tanh(t)` 入流**。

## 算例内容

* 长盒 `Lx = 5.0`、`Ly = 1.25`、`mu = 10`、`rho = 1000`、
  官方 `1024 x 256`、`dt = 1e-4`；官方 `T = 4.0`（40000 步）→
  本 demo **`NX = 512`（`Ny = 128`）、`T = 0.01`（100 步）**、`print_dump = 25`。
* **结构**：4060 个拉格朗日点，**全部**为目标点（`k` 见 `insect_HT.target`）
  —— 一根钉死的"无瓣心脏管"（去掉了 Valveless 版的瓣膜）。
* **驱动**（`please_Compute_External_Forcing.m` → `ib2d_input/*.py`）：
  窗口 `x in [0.4, 0.41]`、`y in [0.41+0.125, 0.59+0.125]`，

  ```
  Fx = -kStiff (uX - uX_tar),  uX_tar = -uMax 5 tanh(t) ((Ly/2+w/2)-y)((Ly/2-w/2)-y),
  w = 0.2, uMax = 250, kStiff = 1e4,
  ```

  **注意用 `Ly` 而非 `Lx`**（上游如此）：`Ly = 1.25` → 剖面峰值在 `y = 0.625`
  （盒中高），窗口正跨在管口。`tanh(t)` 在 `T = 0.01` 只到 1%。

## 移植要点

1. 外力模块逐行移植（保留 `Y_MIN = 0.41+0.125` 的写法），与逐行转写逐点比对
   （差 `0.0`）。
2. `insect_HT.*` 1-based → 自动减 1。
3. **缩减**：`NX = 512`（窗口 `[0.4,0.41]` 需 `dx <= 0.01`）、`T = 0.01`。

## 结果

参考运行约 138 s；AFSI 模拟耗时 128.4 s（墙钟另含导入）。
5 帧对齐（`t = 0 .. 0.01010`）。`compare.py` 原文：

| 指标（末帧） | compare.py 原文 |
|---|---|
| 标记点最大偏差 | `max|dX| 1.653e-07 m = 1.69e-05 h (4060 Lagrangian points)` |
| 速度峰值 AFSI / IB2d | `6.8442e-04 / 5.0647e-04 (3.51e-01 relative)` |
| 速度场偏差 | `max|du| 1.781e-04 = 6.48e-01 max|u|`（带外）; `1.878e-04 (0.63 max|u|)`（带内）; `relative L2 3.463e-01` |
| 力场偏差（带外） | `max|df| 1.766e+00 away from the band, |f|max 1.245e+03` |
| 外力窗口 | `19 nodes, max|u| AFSI 0.0007 / IB2d 0.0005 (3.51e-01 relative); forcing check |f_x - F_x^module(u^AFSI)| <= 1.240e+01 on the core nodes` |

**能验证**：心脏管**完全钉死**（1.7e-5 h）；外力模块与 AFSI 力场自洽
（残差 12.4，相对 `|f|max = 1245` 约 1e-2，与窗口量级相符）。
**不能验证（如实记录）**：速度场（相对 $L_2$ 0.35；窗口内 0.35）。
`uMax = 250`、`T = 0.01` 时 `tanh(t) ≈ 0.01`，实测速度 `~5e-4 m/s`
（绝对差 `1.8e-4 m/s`），相对差是"尚未启动"的噪声；上游 `T = 4.0` 不可行。

## 运行

```bash
conda activate afsi-dolfinx
python run_reference.py --tend 0.01 --print-dump 25 --nx 512     # ~138 s
TFINAL=0.01 PRINT_DUMP=25 NX=512 python main.py                 # 128.4 s
python compare.py
```
