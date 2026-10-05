# demo_490 — IB2d `Example_HeartTube/Insect_Heart`

IB2d 官方算例（`matIB2d/Examples/Example_HeartTube/Insect_Heart`）
在 AFSI 中的复现：**弹簧 + 目标点（心脏管）+ 任意外力脉动入流**。

## 算例内容

* 长盒 `Lx = 6.0`、`Ly = 1.5`、`mu = 1.0`、`rho = 1000`、
  官方 `1024 x 256`、`dt = 2.5e-4`。官方 `T = 4.0`（16000 步）→
  本 demo **`NX = 256`（`Ny = 64`）、`T = 0.05`（200 步）**、`print_dump = 50`。
* **结构**：7068 个拉格朗日点：**1305 根弹簧**（`k` 见 `insect_HT.spring`）、
  **6198 个目标点**（`insect_HT.target`，按上游 index 表选择的一部分点）
  —— 一根被弹性连接并被目标点固定的"心脏管"。
* **驱动**（`please_Compute_External_Forcing.m` → `ib2d_input/*.py`）：
  窗口 `x in [0.29, 0.315]`、`y in [0.65125, 0.84875]`（管的口部），

  ```
  Fx = -kStiff (uX - uX_tar),  uX_tar = uMax 5 |sin(2 pi t)| ((MP+w/2)-y)((MP-w/2)-y)/scale,
  MP = 0.75, w = 0.1975, scale = ((MP+w/2)-MP)((MP-w/2)-MP) = -w^2/4,
  uMax = 5, kStiff = 1e4,
  ```

  `|sin|` 使入流每 1 s 出现**两次**正脉冲（生理上像心脏的搏动），
  `/scale` 归一化让窗口中心的靶速恰为 `5 uMax |sin(2 pi t)|`。

## 移植要点

1. 外力模块逐行移植，与逐行转写逐点比对（差 `0.0`）。
2. `insect_HT.*` 为 1-based，`ib2d_io` 自动减 1。
3. **缩减**：`NX = 256`（`dx = 0.0234`，窗口 `[0.29, 0.315]` 里只剩
   一列节点，这已经是能允许的最粗网格）、`T = 0.05`（上游 16000 步不可行）。

## 结果

参考运行约 93 s；AFSI 模拟耗时 64.7 s（墙钟另含 1-3 min 解释器/库导入；
机器并行运行其它 demo）。4 帧对齐（`t = 0 .. 0.03775`）。`compare.py` 原文：

| 指标（末帧） | compare.py 原文 |
|---|---|
| 标记点最大偏差 | `max|dX| 7.621e-05 m = 3.25e-03 h (7068 Lagrangian points)` |
| 速度峰值 AFSI / IB2d | `4.6749e-01 / 1.8412e-01 (1.54e+00 relative)` |
| 速度场偏差 | `max|du| 2.903e-01 = 5.77e-01 max|u|`（带外）; `2.753e-01 (0.50 max|u|)`（带内）; `relative L2 1.337e+00` |
| 力场偏差（带外） | `max|df| 2.872e+03 away from the band, |f|max 5.654e+04` |
| 外力窗口 | `9 nodes, max|u| AFSI 0.4675 / IB2d 0.1841 (1.54e+00 relative); forcing check |f_x - F_x^module(u^AFSI)| <= 3.319e+02 on the core nodes` |
| 弹簧平均长度 | `AFSI 0.030421 / IB2d 0.030425 (1.46e-04 relative)` |

**能验证**：心脏管**几乎不动**（3.3e-3 h）、**弹簧平均长度 1.5e-4**
（管的几何两边一致）；外力模块与 AFSI 力场自洽（残差 3.3e2，
相对 `|f|max = 5.7e4` 约 6e-3）。
**不能验证（如实记录）**：速度场（相对 $L_2$ 1.34；窗口内靶速差 1.54 倍）。
原因：上游要求 `1024 x 256` 才把 [0.29,0.315] 的窄口分辨开，
在 `256 x 64` 上窗口**只有 9 个节点**、`dy = 0.023`，
而 `5 uMax |sin|` 的靶速可达 25 m/s；`kStiff = 1e4` 的罚力对离散化
极敏感，两种实现给出的局部速度完全不同。同 demo_485，
本例**不适合**在缩减网格上验证速度场。

## 运行

```bash
conda activate afsi-dolfinx
python run_reference.py --tend 0.05 --print-dump 50 --nx 256     # ~93 s
TFINAL=0.05 PRINT_DUMP=50 NX=256 python main.py                 # 64.7 s
python compare.py
```
