# demo_489 — IB2d `Example_Meshmerize_Me/Dragonfly_Wing`

IB2d 官方算例（`matIB2d/Examples/Example_Meshmerize_Me/Dragonfly_Wing`）
在 AFSI 中的复现：**目标点（钉死的翅脉网络）+ 任意外力窄缝射流**。
其外力模块与 `Insect_Wing_Veins`（demo_487）**逐字节相同**（只差一个空行），
上游唯一的差别是 `Tfinal`（本算例 `T = 10.0`，而 `Wing_Veins` 是 `1.0`）。

## 算例内容

* 盒 `Lx = 0.5`、`Ly = 0.25`、`mu = 0.01`、`rho = 1000`、
  官方 `1024 x 512`、`dt = 5e-5`；官方 `T = 10.0`（200000 步）→
  本 demo **`NX = 256`（`Ny = 128`）、`T = 0.02`（400 步）**、`print_dump = 100`。
* **结构**：**32823** 个拉格朗日点（`wing_vein.vertex`，与 demo_487 同一份几何）
  **全部**是目标点 —— 整张翅脉网络被完全钉死。
* **驱动**：窄缝窗口 `x in [0.015, 0.017]`、`y in [0.1927, 0.1978]`，
  `Fx = -kStiff (uX - uX_tar)`，`uX_tar = uMax tanh(2t) ((MP+w/2)-y)((MP-w/2)-y)/scale`，
  `MP = 0.19525`、`w = 0.0051`、`scale = -w^2/4`、`kStiff = 1e4`、`uMax = 0.1`。

## 移植要点

1. 外力模块逐行移植（与 demo_487 的移植版相同），与逐行转写逐点比对（差 `0.0`）。
2. `wing_vein.*` 为 1-based，`ib2d_io` 自动减 1。
3. **缩减**：`NX = 256`（窄缝下界）、`T = 0.02`（上游 200000 步不可行）。

## 结果

参考运行约 343 s；AFSI 模拟耗时 240.0 s（墙钟另含 1-3 min 解释器/库导入；
32k 标记点很贵）。4 帧对齐（`t = 0 .. 0.01505`）。`compare.py` 原文：

| 指标（末帧） | compare.py 原文 |
|---|---|
| 标记点最大偏差 | `max|dX| 2.500e-08 m = 1.28e-05 h (32823 Lagrangian points)` |
| 速度峰值 AFSI / IB2d | `1.3589e-04 / 8.9708e-05 (5.15e-01 relative)` |
| 速度场偏差 | `max|du| 3.068e-06 = 9.66e-01 max|u|`（带外）; `4.618e-05 (0.49 max|u|)`（带内）; `relative L2 8.535e-01` |
| 力场偏差（带外） | `max|df| 1.540e-19 away from the band, |f|max 2.911e+01` |
| 外力窗口 | `3 nodes, max|u| AFSI 0.0001 / IB2d 0.0001 (5.15e-01 relative); forcing check |f_x - F_x^module(u^AFSI)| <= 0.000e+00 on the core nodes` |

**能验证**：翅脉网络**完全钉死**（1.3e-5 h）；**外力模块与 AFSI 力场在核心
节点上完全一致（残差恰好 0.0）**；带外力场为 0（`1.5e-19`）。
**不能验证（如实记录）**：速度场。`uMax = 0.1`、`T = 0.02` 时
`tanh(2t) ≈ 0.04`，实测速度 `~1e-4 m/s`（绝对差 `3e-6 m/s`），
相对差 0.85 纯属噪声量级。上游 `T = 10.0` 才能看到翅脉周围的真实流动。

## 运行

```bash
conda activate afsi-dolfinx
python run_reference.py --tend 0.02 --print-dump 100 --nx 256    # ~343 s
TFINAL=0.02 PRINT_DUMP=100 NX=256 python main.py                 # 240.0 s
python compare.py
```
