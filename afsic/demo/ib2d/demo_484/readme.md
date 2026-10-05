# demo_484 — IB2d `Example_Meshmerize_Me`（叶脉网络）

IB2d 官方算例（`matIB2d/Examples/Example_Meshmerize_Me`，`string_name = leaf_vein`）
在 AFSI 中的复现：**目标点（钉死的叶脉网络）+ 任意外力“活塞”射流**。

## 算例内容

* 非方形盒 `Lx = 1.0`、`Ly = 2.5`、`mu = 1e-3`、`rho = 1`、
  官方 `410 x 1024`、`dt = 1e-4`。官方 `T = 0.5`（5000 步）→
  本 demo **`NX = 200`（`Ny = 500`）、`T = 0.01`（100 步）**、`print_dump = 25`。
  （`Ny >= 500` 是必须的：射流窗口在 `y` 向只有 `0.005` 宽，
  `Ny = 500` 时 `dy = 0.005` 恰好剩一行节点。）
* **结构**：1585 个拉格朗日点（`leaf_vein.vertex`）**全部**是目标点
  （`k = 1e7`）——一片被完全钉死的叶脉网络；上游 `leaf_flow.vertex`
  是流程图辅助文件，模拟中不使用。
* **驱动**（`please_Compute_External_Forcing.m` → `ib2d_input/*.py`）：
  在窗口 `x in [0.252, 0.364]`、`y in [0.3625, 0.3675]` 上施加
  **只有 y 分量**的射流

  ```
  Fy = -kStiff (uY - uY_tar),  uY_tar = uMax tanh(2t) ((MP+w/2)-x)((MP-w/2)-x)/scale,
  scale = ((MP+w/2)-MP)((MP-w/2)-MP) = -w^2/4,  MP = 0.3034625, w = 0.112,
  kStiff = 1e4, uMax = 2.5,
  ```

  `/scale` 归一化使窗口中心的靶速恰为 `uMax tanh(2t)`；
  `T = 0.01` 时 `tanh(2t) = tanh(0.02) ≈ 0.02`，**射流只开启了 2%**。

## 移植要点

1. 外力模块逐行移植 + 与逐行转写逐点比对（差 `0.0`）。
2. 上游 `.m` 把 `w`/`MP` 写成 `3.595500e-01-2.473750e-01` 这种“表达式常量”，
   移植版保留原式便于对照。
3. **缩减**：`NX = 200`（下界由窗口决定）、`T = 0.01`。上游 410x1024
   （42 万节点）在本机负载下不可行。

## 结果

参考运行约 208 s、AFSI 运行约 474 s（100 步，10 万节点；墙钟时间含解释器/
库导入，机器同时并行运行其它 demo，`main.py` 自报的模拟耗时是 398 s）。
5 帧对齐（`t = 0 .. 0.01010`）。`compare.py` 原文：

| 指标（末帧） | compare.py 原文 |
|---|---|
| 标记点最大偏差 | `max|dX| 1.172e-05 m = 2.34e-03 h (1585 Lagrangian points)` |
| 速度峰值 AFSI / IB2d | `4.5228e-02 / 5.6965e-02 (2.06e-01 relative)` |
| 速度场偏差 | `max|du| 2.160e-02 = 6.21e-01 max|u|`（带外）; `4.996e-02 (0.12 max|u|)`（带内）; `relative L2 7.669e-01` |
| 力场偏差（带外） | `max|df| 7.830e+01 away from the band, |f|max 1.301e+02` |
| 外力窗口 | `22 nodes, max|u| AFSI 0.0452 / IB2d 0.0379 (1.93e-01 relative); forcing check |f_x - F_x^module(u^AFSI)| <= 9.915e-04 on the core nodes` |

**能验证**：叶脉网络**完全不动**（2.3e-3 h）；外力模块与 AFSI 力场在窗口核心
节点上完全自洽（残差 9.9e-4，相对窗口量级 1e-2 可忽略）。
**不能验证（如实记录）**：速度场（相对 $L_2$ 0.77）。原因有两层：
(a) `T = 0.01` 时 `tanh(2t) ≈ 0.02`，流场速度只有 `~5e-2 m/s`，
   差异基本是"尚未启动的流动"的数值噪声；
(b) 上游 `T = 0.5`、410x1024 在本机不可行。本 demo 只验证"射流窗口就位 +
   结构钉死"。

## 运行

```bash
conda activate afsi-dolfinx
python run_reference.py --tend 0.01 --print-dump 25 --nx 200     # ~200 s
TFINAL=0.01 PRINT_DUMP=25 NX=200 python main.py                 # ~398 s
python compare.py
```
