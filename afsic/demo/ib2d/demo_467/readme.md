# demo_467 — IB2d 任意外力接口 `arb_ext_force`（空槽道内的调制抛物射流）

IB2d 官方算例（`matIB2d/Examples/Example_Channel_Flow/Example_Parabolic_Flow_In_Empty_Channel`）
在 AFSI 中的复现，同时是 AFSI **任意外力（`arb_ext_force`）机制**的第一个算例。

## 算例内容

`1 x 1` 周期盒、`mu = 10`、`rho = 1`、`128 x 128`、`dt = 1e-4`、`T = 0.015`
（151 步；pyIB2d 用 `while current_time < Tfinal` 累加浮点，比 `round(T/dt)` 多走一步，
本移植复刻了这个步数）。

* **结构**：两条 231 点的水平"槽道壁"，分别在 `y = 0.4` 与 `y = 0.6`，
  `x in [0.05, 0.9484]`（拉氏间距 `ds = Lx/(2Nx) = dx/2`），两端留口（进/出流）；
  462 个标记点**全部**是 `k = 1e7` 的目标点（钉死），因此结构本身几乎不动，
  本算例考察的是**流场**。
* **外力**：算例自带的 `please_Compute_External_Forcing.m` 在
  `x in [0.1, 0.16]`、`y in [0.41, 0.59]` 的窗口内施加**速度目标型罚力**

  $$F_x = -k\,(u_x - u_x^{\rm tar}),\qquad
  u_x^{\rm tar} = -u_{\max}\,5|\sin(10\pi t)|\,
  \Big(\tfrac{L_x}{2}+\tfrac{W}{2}-y\Big)\Big(\tfrac{L_x}{2}-\tfrac{W}{2}-y\Big),$$

  `k = 1e4`、`W = 0.2`、`u_max = 250`（即中心峰值 `5 * 0.01 * 250 = 12.5 m/s`），
  10 Hz 调制；`F_y = 0`。注意 `k dt / rho = 1` **恰好在罚力的边际增益上**，
  这一点是下面"壁面窄带伪速度"的关键。

## 移植要点：`arb_ext_force`

IB2d 的接口是每步调用运行目录下的模块：

```python
Fx, Fy, first, inds = please_Compute_External_Forcing(
    dt, current_time, x, y, grid_Info, uX, uY, first, inds)
```

`grid_Info = [Nx, Ny, Lx, Ly, dx, dy, supp, Nb, ds]`（与 matIB2d 一致，
`ds = Lx/(2Nx)`），`x`、`y` 是节点坐标 `0:dx:Lx-dx`，`uX`、`uY` 是 `(Ny, Nx)` 的
Eulerian 速度，返回的 `Fx`、`Fy` 是**节点力密度**，在流体求解前加到 Eulerian 力上。

1. `ib2d_input/please_Compute_External_Forcing.py` 是 `.m` 的**逐行移植**
   （`np.searchsorted` 复刻 `.m` 的"首个 `x >= xMin` / 末个 `x <= xMax`"搜索，
   0-based 索引重排；**AFSI 与 pyIB2d 参考跑的是同一个文件**，因此对比只反映
   求解器与 IB 机制之差）。原算例没有 pyIB2d 版本，只有 `.m`。
2. AFSI 侧（`main.py`）：IB2d 的流体网格是**节点共置**的（`(i h, j h)`），
   与 AFSI `IBMesh` 的 Q1 顶点自由度**同一套格点**，所以外力可以直接在
   Q1 节点自由度上求值：把 `solver.u_n` 插值到 Q1 空间 → 组装 `(Ny, Nx)` 数组 →
   调用模块 → 把返回的 `(Fx, Fy)` 填回 Q1 函数，再**插值到 Q2** 后加到
   `solver.f`（`solver.f` 是 Taylor–Hood 的 Q2 力空间）。
   周期盒的镜像自由度用 `(i mod Nx, j mod Ny)` 映射，保证力场本身周期。
3. 目标点力、$X_h$ 半步位置、两阶段流体求解、以及**与 pyIB2d 同一 dump 步**
   的帧对齐约定都与 demo_459–466 相同。

## 结果

参考运行 ~21 s（151 步）、AFSI 运行 ~9 s，两边各 4 帧（含 `t=0` 初态）。

| 指标（末帧 `t = 0.0151`） | 已发布 `dt = 1e-4` | 加密 `dt = 5e-5` |
|---|---|---|
| 标记点偏差（壁面钉死） | 1.86e-5 m = **2.4e-3 h** | 1.85e-5 m = 2.4e-3 h |
| 速度场 `max\|du\|`（壁面窄带外） | 9.6e-2 = 6.9% `max\|u\|` | 2.1e-2 = **1.5%** `max\|u\|` |
| 速度场相对 $L_2$ 误差（窄带外） | 5.5e-2 | **2.2e-2** |
| 槽道核心流量 $\int_{0.42}^{0.58} u_x\,dy$ @ `x=0.5` | 2.9e-2 相对差 | **1.5e-2** |
| 壁面窄带（4 节点 $\delta$ 支撑）`max\|du\|` | 1.11 = 0.80 `max\|u\|` | **2.3e-2 = 0.02** `max\|u\|` |
| Eulerian 力场 `max\|df\|` | 2.7e4（`\|f\|` 峰值 1.1e5） | 4.6e3 |
| 罚力公式核对 | `\|f_x + k(u_x-u_x^{\rm tar})\| <= 2.7e2`（0.25% `\|f\|_{\max}`） | 1.3e2 |

流场本身的一致性很好：中心线上两边分别是 `1.3929 / 1.3961 m/s`（差 **0.03%**），
整条剖面（`x = 0.125` 的射流截面与 `x = 0.5` 的槽道截面）都对到亚百分比；
`max|u|` 逐帧 0.5%/0.1%/0.03% 逐次收敛（`0.4938/0.4974`、`0.9555/0.9618`、
`1.3929/1.3961`）。注意施加速度目标 5.71 m/s 而中心线只到 1.39 m/s：
$\mu = 10$ 的黏性耗散与压力梯度与罚力竞争，两边都如此。

### 壁面窄带伪速度（本 demo 的额外发现）

`k dt / rho = 1` 时，IB2d 的**共置 FFT 投影**在浸入边界的 4 节点 $\delta$ 支撑带里
长出一个网格尺度尖峰（`x=0.5`、`j=49..54`）：

| `u_x` @ `x=0.5` | j=49 | j=50 | j=51 | j=52 | j=53 | j=54 |
|---|---|---|---|---|---|---|
| IB2d, `dt=1e-4` | -0.1521 | **-0.5854** | **-1.1464** | -0.7477 | 0.0555 | 0.3547 |
| IB2d, `dt=5e-5` | -0.0594 | -0.0617 | -0.0337 | 0.0765 | 0.2322 | 0.3762 |
| AFSI, `dt=1e-4` | -0.0688 | -0.0691 | -0.0368 | 0.0681 | 0.2186 | 0.3641 |
| AFSI, `dt=5e-5` | -0.0681 | -0.0656 | -0.0298 | 0.0732 | 0.2194 | 0.3634 |

AFSI 的 Q1 解在窄带内**几乎不随 `dt` 变化**（-0.037 → -0.030），参考里的 -1.15
在 `dt` 减半后塌到 -0.034（与 AFSI 一致），即该尖峰是**参考格式在罚力边际增益
（`k dt = 1`）下的网格尺度伪模态**，不是 AFSI 漏了什么。因此上表把误差显式分到
"窄带外 / 窄带内"；`compare.py` 会在存在 `dt/2` 参考时自动做这组交叉核对。
（同类"罚力柔度/边际增益"现象见 demo_460：`k=1e8` 目标点的柔度差随罚力刚度线性缩放。）

## 运行

```bash
conda activate afsi-dolfinx
python run_reference.py                       # pyIB2d 参考 -> ib2d_reference.npz（~21 s）
python main.py                                # AFSI -> plot/afsi_result.npz（~9 s）
python compare.py                             # 对比 + figures/（含力场与剖面图）

# 可选的 dt/2 交叉核对（compare.py 会自动识别并打印第二组表）
python run_reference.py --dt 5e-5 --out ib2d_reference_dt5e-5.npz --run ib2d_run_dt5e-5
DT=5e-5 OUTPUT_PATH=./plot_dt5e-5 python main.py
```

环境变量（`main.py`）：`TFINAL`、`DT`、`NX`/`NY`、`PRINT_DUMP`、
`GRAD_DIV`（默认 0）、`OUTPUT_PATH`（默认 `plot/`）、
`IB2D_EXAMPLE`（默认 `ib2d_input/`）。

## 文件

* `ib2d_input/`：`input2d`、`channel.vertex`、`channel.target`（matIB2d 1-based 原样）
  以及 `.m` 的 Python 移植 `please_Compute_External_Forcing.py`。
* `main.py`：AFSI 端（外力 + 钉死目标点 + 两阶段求解 + 帧对齐 dump）。
* `run_reference.py` / `ib2d_reference.py`：pyIB2d 参考运行与 VTK → npz（含 `f` 力场）。
* `compare.py`：帧对齐、窄带内/外误差分解、剖面/历史/场图与 CSV 表。
