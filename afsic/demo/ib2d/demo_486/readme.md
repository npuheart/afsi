# demo_486 — IB2d `Example_Meshmerize_Me/old`（旧版叶脉网络）

IB2d 官方算例（`matIB2d/Examples/Example_Meshmerize_Me/old`，`string_name = leaf_vein`）
在 AFSI 中的复现：**目标点（钉死的叶脉网络）+ 任意外力射流**（与 demo_484
同族，但外力公式与窗口不同）。

## 算例内容

* `1 x 1` 周期盒、`mu = 1e-3`、`rho = 1`、官方 `512 x 512`、`dt = 1e-4`。
  官方 `T = 0.5`（5000 步）→ 本 demo **`NX = 192`（`Ny = 192`）、
  `T = 0.02`（200 步）**、`print_dump = 50`。
  （`Ny >= 200` 附近是窗口要求：射流窗口在 `y` 向宽 `0.005`。）
* **结构**：2335 个拉格朗日点，全部为目标点（`k = 1e7`）——钉死的叶脉网络。
* **驱动**（`please_Compute_External_Forcing.m` → `ib2d_input/*.py`）：
  窗口 `x in [0.41, 0.545]`、`y in [0.115, 0.12]`，只有 y 分量：

  ```
  Fy = -kStiff (uY - uY_tar),  uY_tar = -uMax 5 tanh(t) ((MP+w/2)-x)((MP-w/2)-x),
  MP = 0.4772, w = 0.1486, uMax = 250, kStiff = 1e4,
  ```

  与 demo_484（新版 `Example_Meshmerize_Me`）的关键差别：**没有 `/scale`
  归一化**，所以窗口中心的靶速是 `5 uMax w^2/4 ≈ 69 m/s`；`tanh(t)` 在
  `T = 0.02` 只到 0.02。

## 移植要点

1. 外力模块逐行移植（保留上游写法 `w = 0.551500e-01-4.029000e-01`，
   `midPoint = (5.515e-1+4.029e-1)/2`），与逐行转写逐点比对（差 `0.0`）。
2. `leaf_vein.*` 为 1-based，`ib2d_io` 自动减 1。
3. **缩减**：`NX = 192`、`T = 0.02`。

## 结果

参考运行约 207 s；AFSI 模拟耗时 262.6 s（墙钟另含 1-3 min 解释器/库导入；
机器并行运行其它 demo）。5 帧对齐（`t = 0 .. 0.02010`）。
`compare.py` 原文：

| 指标（末帧） | compare.py 原文 |
|---|---|
| 标记点最大偏差 | `max|dX| 3.032e-05 m = 5.82e-03 h (2335 Lagrangian points)` |
| 速度峰值 AFSI / IB2d | `1.2700e-01 / 2.7426e-01 (5.37e-01 relative)` |
| 速度场偏差 | `max|du| 7.327e-02 = 7.33e-01 max|u|`（带外）; `2.281e-01 (0.17 max|u|)`（带内）; `relative L2 7.158e-01` |
| 力场偏差（带外） | `max|df| 1.708e+02 away from the band, |f|max 2.779e+02` |
| 外力窗口 | `26 nodes, max|u| AFSI 0.1270 / IB2d 0.1107 (1.47e-01 relative); forcing check |f_x - F_x^module(u^AFSI)| <= 3.002e-03 on the core nodes` |

**能验证**：叶脉**完全不动**（5.8e-3 h）；窗口内速度峰值差 15%；
外力模块与 AFSI 力场自洽（残差 3.0e-3，相对窗口量级约 1e-3）。
**不能验证（如实记录）**：全场速度（相对 $L_2$ 0.72）。
`T = 0.02` 时 `tanh(t) ≈ 0.02`，射流刚起步，全场速度很小，
差异主要是"尚未启动"的噪声；与 demo_484 同一限制。

## 运行

```bash
conda activate afsi-dolfinx
python run_reference.py --tend 0.02 --print-dump 50 --nx 192     # ~90 s
TFINAL=0.02 PRINT_DUMP=50 NX=192 python main.py                 # ~263 s
python compare.py
```
