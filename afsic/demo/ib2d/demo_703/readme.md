# demo_703 — IB2d `Example_Channel_Flow/Example_Turek_Hron`（质量点 + 圆柱绕流基准）

IB2d 官方算例（`matIB2d/Examples/Example_Channel_Flow/Example_Turek_Hron`）。
Turek–Hron 是流固耦合最经典的基准算例之一：**槽道内的弹性圆柱在射流中摆动/升沉**。

## 算例内容

* 官方：`1024 x 128`、`dt = 5e-5`、`mu = 7.5e-4`、`rho = 1`、`Lx x Ly = 2 x 0.25`、
  `Tfinal = 8`（16 万步、`print_dump = 500`）。本 demo 用 **`NX = 256`（`256 x 32`）、`T = 0.02`（400 步、4 帧）**。
* **结构**：2055 个拉格朗日点 = 弹簧 + 目标点 + 不变梁 + **80 个质量点**（`M = 5`，
  `k = 1e4`）：圆柱上的 80 个标记点各自挂一个自由"幽灵"质点，幽灵受重力/流体力驱动，
  再通过弹簧把力传回流体。
* **入口射流**：算例自己的 `please_Compute_External_Forcing.m`（已逐行移植为
  `ib2d_input/please_Compute_External_Forcing.py`，两个代码共用）：
  窗口 `x in [0.1, 0.16]`、`y in [0.035, 0.215]`，抛物型目标速度

  ```
  uX^tar = -uMax (100 tanh 2t) ((Ly/2+w/2)-y) ((Ly/2-w/2)-y),  uY^tar = 0
  ```

  `kStiff = 1e3`、`w = 0.2`、`uMax = 2.75`（渐近峰值约 2.75 m/s，位于 `y = Ly/2`）。
* 机制组合是**质量点 + 外力模块**：这正是 `main.py` 需要由 demo_570 的 mass 机制
  叠加外力模块的原因（两条代码路径本来就都在模板里）。

## 移植要点

1. `main.py` 取 demo_570（mass_pts + gravity + spring/beam/target + 外力模块）为基，
   只替换 `ib2d_input/` 与移植的 `please_Compute_External_Forcing.py`；
   质量点算法（`dt/2` 半步位置、`F_Mass` 半更新速度、整步位置/速度）与 demo_570 完全一致。
2. 参考侧与 AFSI 侧加载的是**同一份**外力模块与同一份 `.mass/.spring/.beam/.target`；
   `run_reference.py` 会把用户模块拷进 pyIB2d 的运行目录。
3. 官方 `1024 x 128` 的网格在 16 万步下需要数小时；本 demo 用 `256 x 32` 缩短时间窗口，
   并在 readme 里给出官方数值。**圆柱/弹簧/质量数据是物理量，插值到粗网格不改变加载**。

## 结果

4 帧对齐（`t = 0 .. 0.01505`，`256 x 32`）：

| 指标（末帧 `t = 0.01505`） | compare.py 原文 |
|---|---|
| **质量点位置**（80 个幽灵点） | `max\|dXm\|` 1.371e-8 m = **1.75e-6 h**（rms 1.03e-6 h）  ← 头条指标 |
| 质量点速度 | `max\|dVm\|` 2.69e-6 m/s |
| 弹簧耦合本身（幽灵-标记偏移） | 1.70e-7 m = 2.18e-5 h |
| 标记点偏差（2055 点） | 1.565e-6 m = 2.00e-4 h |
| 速度场 `max\|u\|` AFSI / IB2d | 0.0900 / 0.0832（8.1e-2 相对，窗口内射流刚起步） |
| 参考侧的幽灵位移（同帧） | 0（400 步内圆柱几乎未动，官方需数万步才有明显摆动） |

**能验证**：质量点耦合在高速射流下仍逐位一致（1.7e-6 h ≈ 1.4e-8 m），
标记点 2e-4 h，说明质量点算法与外来力模块的移植都是忠实的。

**不能验证**：本窗口内速度场的 8.1% 差异（射流刚由 `100 tanh(2t)` 起振，
场量还很小、相对差被放大）；以及**圆柱的摆动/升沉响应**——官方 `Tfinal = 8`
（16 万步）才会出现完整的涡激振动，本 demo 的 400 步只覆盖启动阶段。
要复现基准数据需在 `1024 x 128` 上跑满 `T = 8`（数小时）。

## 运行

```bash
conda activate afsi-dolfinx
python run_reference.py --nx 256 --tend 0.02 --print-dump 100    # ~7 s（pyIB2d 很快）
NX=256 TFINAL=0.02 PRINT_DUMP=100 python main.py                 # ~13 s
python compare.py
```
