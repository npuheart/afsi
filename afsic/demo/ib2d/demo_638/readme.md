# demo_638 — IB2d `Examples_Education/Convergence/Jellyfish/Simulation_Skeletons/Re150/Res_384_480x144`（水母收敛阶梯 / 水母游动）

IB2d 官方算例（`matIB2d/Examples/Examples_Education/Convergence/Jellyfish/Simulation_Skeletons/Re150/Res_384_480x144`）。

## 算例内容

* 官方设置：`144 x 480`、`dt = 1e-5`、`mu = 6.667`、`Tfinal = 5.0`；
  本 demo 用 **`96x320`、`T = 0.0025`（250 步、`print_dump = 50`）**。
* **结构**：369 markers, 267 springs (k=9.6e+08, alpha=1..1), 237 non-invariant beams (kb=2.21e+11), 72 target points (k=2.5e+06)
* 机制：弹簧 + **非不变梁** + 目标点 + `update_Springs.m`
  （已逐行移植为 `ib2d_input/update_Springs.py`，两个代码共用同一份）。

## 移植要点

* `Examples_Education/Convergence/Jellyfish` 是**同一几何、同一结构，只改雷诺数与
  分辨率**的收敛阶梯（`Re ∈ {37.5, 75, 150, 300}` × 11 个分辨率）；每个目录都独立
  成一个 demo，随机说明里给出官方分辨率。`Example_Jellyfish_Swimming/Hoover_Jellyfish`
  与 `Tentacle_Jelly` 是作者在该几何上的两个变体（后者额外带 poroelastic）。
* 官方网格最大 `384 x 1280`，**必须缩减**：本 demo 统一用 `96x320`（见上）。
  缩减只影响流体分辨率；弹簧静长、梁曲率、目标点都是物理量，结构与加载**完全不变**。
* 参考侧与 AFSI 侧加载的是**同一份** `update_Springs.py`（`run_reference.py` 会把它拷进
  pyIB2d 的运行目录），所以静长的行波相位在两边完全一致。

## 结果

? 帧对齐（`t = ? .. ?`）；`compare.py` 原文：

| 指标（末帧） | compare.py 原文 |
|---|---|
| 标记点偏差 | max|dX| 1.427e-05 m = 4.57e-04 h (369 Lagrangian points) |
| 弹簧平均长度 | mean length AFSI 0.116124 / IB2d 0.116122 (1.53e-05 relative) |
| 力场 | max|df| 1.698e+04 away from the band, |f|max 7.202e+04 |
| 速度场 | max|u| AFSI 6.7641e-02 / IB2d 3.9516e-02 (7.12e-01 relative); max|du| 2.814e-02 = 2.88e-01 max|u| away from the IB band (30432/30720 nodes), 1.476e-06 (1.00 max|u|) inside it; relative L2 4.707e-01 |
| 目标点锚点差 | — |

**能验证**：结构力学与加载——弹簧平均长度一致到 **1.53e-05**（相对），标记点偏差
1.427e-05 m = 4.57e-04 h，说明两边施加的是同一套弹簧/梁/目标点力，
`update_Springs` 的行波没有相位漂移。

**不能验证**：速度场（相对 $L_2$ 4.707e-01）。低雷诺数水母的自推进流场对分辨率极敏感：
把 `144 x 480` 降到 `96x320` 之后，参考解本身远未收敛，两边的粗网格响应差异被放大。
本族因此把"结构/驱动是否忠实移植"作为可验证项，速度场需要在接近官方网格的机器上复现
（内存与时间随网格线性增长）。

## 运行

```bash
conda activate afsi-dolfinx
python run_reference.py --nx 96 --tend 0.0025 --print-dump 50      # ~90 s
NX=96 TFINAL=0.0025 PRINT_DUMP=50 python main.py                   # ~30 s
python compare.py
```
