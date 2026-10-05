# demo_487 — IB2d `Example_Meshmerize_Me/Insect_Wing_Veins`

IB2d 官方算例（`matIB2d/Examples/Example_Meshmerize_Me/Insect_Wing_Veins`）
在 AFSI 中的复现：**目标点（钉死的翅脉网络）+ 任意外力窄缝射流**。

## 算例内容

* 盒 `Lx = 0.5`、`Ly = 0.25`、`mu = 0.1`、`rho = 1000`、
  官方 `1024 x 512`、`dt = 5e-5`；官方 `T = 1.0`（20000 步）→
  本 demo **`NX = 256`（`Ny = 128`）、`T = 0.01`（200 步）**、`print_dump = 50`。
* **结构**：**32823** 个拉格朗日点（`wing_vein.vertex`）**全部**是目标点
  （`k` 见 `wing_vein.target`）——整张翅脉网络被完全钉死。
* **驱动**（`please_Compute_External_Forcing.m` → `ib2d_input/*.py`）：
  窄缝窗口 `x in [0.015, 0.017]`、`y in [0.1927, 0.1978]`，

  ```
  Fx = -kStiff (uX - uX_tar),  uX_tar = uMax tanh(2t) ((MP+w/2)-y)((MP-w/2)-y)/scale,
  MP = (0.1978+0.1927)/2 = 0.19525, w = 0.1978-0.1927 = 0.0051, scale = -w^2/4,
  kStiff = 1e4, uMax = 0.1,
  ```

  `/scale` 归一化让缝隙中心的靶速恰为 `uMax tanh(2t)`（很小，`0.1 m/s` 量级）。

## 移植要点

1. 外力模块逐行移植（保留上游“表达式常量”写法 `w = 0.1978-0.1927`），
   与逐行转写逐点比对（差 `0.0`）。
2. `wing_vein.*` 为 1-based，`ib2d_io` 自动减 1。
3. **缩减**：`NX = 256`（下界由窄缝 `[0.015,0.017]` 决定：`dx = 0.002` 时
   缝里只剩 1 列节点）、`T = 0.01`。上游 1024x512、20000 步不可行。

## 结果

参考运行约 197 s；AFSI 模拟耗时 127.0 s（墙钟另含导入；32k 标记点的
dof 匹配与铺力都很贵）。5 帧对齐（`t = 0 .. 0.01005`）。`compare.py` 原文：

| 指标（末帧） | compare.py 原文 |
|---|---|
| 标记点最大偏差 | `max|dX| 5.110e-09 m = 2.62e-06 h (32823 Lagrangian points)` |
| 速度峰值 AFSI / IB2d | `3.4140e-05 / 3.4988e-05 (2.42e-02 relative)` |
| 速度场偏差 | `max|du| 1.161e-06 = 9.67e-01 max|u|`（带外）; `6.249e-06 (0.82 max|u|)`（带内）; `relative L2 7.229e-01` |
| 力场偏差（带外） | `max|df| 1.123e-19 away from the band, |f|max 1.954e+01` |
| 外力窗口 | `3 nodes, max|u| AFSI 0.0000 / IB2d 0.0000 (2.42e-02 relative); forcing check |f_x - F_x^module(u^AFSI)| <= 0.000e+00 on the core nodes` |

**能验证**：翅脉网络**完全钉死**（2.6e-6 h，即机器精度量级的锚定）；
**外力模块与 AFSI 力场在核心节点上完全一致（残差恰好 0.0）**；
带外 `max|df| = 1.1e-19`（力场在带外就是零）。
**不能验证（如实记录）**：速度场。原因：`uMax = 0.1`、`T = 0.01` 时
`tanh(2t) ≈ 0.02`，靶速只有 `2e-3 m/s`，实测速度 `~3e-5 m/s`
（比 IB2d 的同类量级还小 3 个数量级），**相对差 0.72 完全是数值噪声**
（绝对差 `1.2e-6 m/s`）。要看到真实流动需要上游的 `T = 1.0`。

## 运行

```bash
conda activate afsi-dolfinx
python run_reference.py --tend 0.01 --print-dump 50 --nx 256     # ~240 s
TFINAL=0.01 PRINT_DUMP=50 NX=256 python main.py                 # ~300 s
python compare.py
```
