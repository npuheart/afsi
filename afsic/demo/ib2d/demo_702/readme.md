# demo_702 — IB2d `Example_Boussinesq/Falling_Sphere`（质量点 + Boussinesq 浮力 + 被动标量）

IB2d 官方算例（`matIB2d/Examples/Example_Boussinesq/Falling_Sphere`）。
**这是本仓库唯一同时用到 mass_pts + gravity + Boussinesq + concentration 的算例**，
因此它也是"最多机制叠加"的一个 demo。

## 算例内容

* 官方：`128 x 128`、`dt = 1.25e-5`、`mu = 0.25`、`rho = 1`、`Tfinal = 2`（16 万步、`print_dump = 800`）。
  本 demo 用 **`128 x 128`（官方网格）、`T = 0.02`（1600 步、`print_dump = 100`、17 帧）**。
* **结构**：1502 个拉格朗日点：弹簧 + 目标点 + 不变梁 + **174 个质量（幽灵）点**（`M = 0.25`）。
* **重力**：`gravity = 1`、`gvec = (0, -1)`、`g = 9.80665`，只作用在质量点上。
* **Boussinesq**：`boussinesq = 1`、`expansion_coeff = 0.25`；浮力项按下式加到欧拉力上
  （`please_Form_Boussinesq_Forcing_Terms`）：`Fx += rho fBouss_X C`、`Fy += rho fBouss_Y C`，
  其中 `fBouss = exp_Coeff * 0.981 * gvec ⊗ mask`，用的是**上一步**的浓度场。
* **标量**：`concentration = 1`、`kDiff = 1e-5`，初始场由算例自带的
  `Give_Me_Initial_Concentration` 生成（`C = 0.5 (tanh(80((y-0.2)-0.5Lx-0.001cos 2πx)) + 1)`）。

## 移植要点

1. **上游 bug：`boussinesq.mesh` 根本不存在**。算例把 `boussinesq = 1` 打开，
   pyIB2d 的 Boussinesq 分支要读 `boussinesq.mesh`（列出浮力作用的所有欧拉节点），
   但该目录**既没有这个文件，生成脚本 `Make_Tub_Geometry_and_Initial_Concentration.m`
   也从不写它**（只写 `.vertex/.mass/.spring/.target/.beam/.concentration`）。
   照上游跑，两个代码都会立刻 `FileNotFoundError: boussinesq.mesh`。
   本 demo 用 `make_input.py` 重建该掩码：生成脚本自己调用
   `give_Me_Indices_To_Apply_Force(xMin=0,xMax=Lx,yMin=0,yMax=Lx)`，即**全域掩码**，
   本脚本据此写出全部节点对。
2. 顺带发现第二个上游怪癖：MATLAB 的索引是 1-based，而 pyIB2d 的
   `please_Form_Boussinesq_Forcing_Terms` **没有减 1** 就直接 `mat[yInd, xInd] = 1`——
   1-based 的全域掩码会撞到 `index Nx` 报 `IndexError`（已在 `make_input.py` 里说明，
   写成 0-based；已发布的 demo_577 的掩码同样带一个节点的偏移）。
3. 镜像合并：`main.py` 由 demo_576（mass_pts + 标量 + gravity）与 demo_577（Boussinesq）
   合并而成，浮力项插在外力之后、流体求解之前，用的浓度场与 pyIB2d 一致（上一步的值）。

## 结果

17 帧对齐（`t = 0 .. 0.02001`，官方网格 `128 x 128`，1600 步）：

| 指标（末帧 `t = 0.02001`） | compare.py 原文 |
|---|---|
| **质量点位置**（174 个幽灵点） | `max\|dXm\|` 1.123e-5 m = **1.44e-3 h**（rms 1.12e-3 h） |
| 质量点速度 | `max\|dVm\|` 1.11e-3 m/s |
| 弹簧耦合本身（幽灵-标记相对位移） | 2.19e-7 m = **2.80e-5 h** |
| 标记点偏差（1502 点） | 1.144e-5 m = 1.46e-3 h |
| 速度场 `max\|u\|` AFSI / IB2d | 0.1669 / 0.1700（**1.78e-2** 相对）、`max\|du\|` 5.69e-3 |
| 浓度 `max\|dC\|` | 3.38e-4（相对 $L_2$ **1.96e-5**） |
| 浓度总量 | 守恒到 1e-5 相对 |

**能验证**：四种机制叠加下全部成立——幽灵质点轨迹（1.4e-3 h）、弹簧耦合（2.8e-5 h）、
浮力-浓度耦合与标量输运（相对 $L_2$ 2e-5）、速度场（1.8%）。

**不能验证**：无（本算例的窗口内每一项都在误差范围内）。需要注意
`T = 0.02` 只覆盖 1600 步（官方 16 万步）：更长时间后球体下落到
`Tfinal = 2` 的过程中，标量界面会被拉伸出极薄的层，那时需要更高的分辨率
（本 demo 已按官方网格运行，只缩短了时间）。

## 运行

```bash
conda activate afsi-dolfinx
python make_input.py --nx 128                                  # 重建 boussinesq.mesh（上游缺失）
python run_reference.py --nx 128 --tend 0.02 --print-dump 100   # ~200 s
NX=128 TFINAL=0.02 PRINT_DUMP=100 python main.py                # ~128 s
python compare.py
```
