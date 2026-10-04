# demo_473 — IB2d `Tracers_In_Channel_Flow`（射流槽道里的被动示踪粒子）

IB2d 官方算例（`matIB2d/Examples/Example_Tracers/Tracers_In_Channel_Flow`）。
**新机制：被动示踪粒子（tracers）**——不产生任何力，只随流体平流。

## 算例内容

* `1 x 1` 周期盒、`mu = 0.01`、`rho = 1`、`64 x 64`、`dt = 5e-5`；
  官方 `T = 0.5`（10000 步），本 demo 默认 `T = 0.25`（5000 步）。
* **结构**：286 个钉死目标点（`k = 1e7`）描出槽道壁。
* **示踪粒子**：**110 个**被动点，用与标记点相同的 IB 核平流、并按 IB2d 规则
  回卷进盒（`X_t^{n+1} = (X_t^n + dt U_h(X_t^n)) mod L`），完全不反馈给流体。
* **外力**：`x in [0.1, 0.13]`、`y in [0.41, 0.59]` 的 `tanh(t)` 缓启抛物射流
  （`k = 1e4`、`u_max = 200`）。

## 结果（`GRAD_DIV = 2.5`）

参考运行 ~100 s、AFSI 运行 ~105 s，各 51 帧。

| 指标（末帧 `t = 0.25`） | $\gamma = 0$ | $\gamma = 2.5$ |
|---|---|---|
| 施力窗口 `max\|u\|` AFSI / IB2d | 2.3737 / 2.3542（0.83%） | 2.3273 / 2.3542（1.14%） |
| 速度场相对 $L_2$ 误差（窄带外） | 2.06e-1 | **6.9e-2** |
| 流量（中列、窗口行）相对差 | 2.0e-1 | **4.2e-2** |
| 标记点偏差 | 7.9e-2 h | **6.8e-2 h** |
| 110 个示踪粒子 `max\|ΔX_t\|` | 5.7 h | **1.4 h** |
| 罚力公式核对（`\|f\|_{\max}` 1.2e3） | ≤0.23（0.02%） | ≤0.31 |

* **γ 的作用**：本例的射流在被壁面围住的槽道里传播 0.4 m，$\gamma = 0$ 时的**散度误差**
  会让下游射流衰减过头（$L_2$ 2.1e-1、流量差 20%）；$\gamma = 2.5$ 修正后 $L_2$ 与流量都降 3 倍
  （与 demo_451 的阻抗泵同一结论，见其 readme）。
* **示踪粒子**：逐点不可复现（拉格朗日混沌，和 demo_451 一致），但 $\gamma = 2.5$ 时
  110 点最大偏差降到 1.4 h（γ=0 时 5.7 h），说明主要偏差来源是速度场而非平流公式。

## 运行

```bash
conda activate afsi-dolfinx
python run_reference.py --tend 0.25 --print-dump 100     # ~100 s
GRAD_DIV=2.5 TFINAL=0.25 PRINT_DUMP=100 python main.py   # ~105 s
python compare.py
```

> 上游 bug（本 demo 的额外发现）：pyIB2d 自己的 tracer 更新调用
> `please_Move_Lagrangian_Point_Positions(Uh, Vh, xT, yT, xT, yT, x, y, dt, grid_Info, 0)`
> 少了 4 个位置参数（`porous_Yes..F_Poro`），直接跑会 `TypeError`。
> `run_reference.py` 用一个小 dispatcher 把参数补齐（标记点调用不受影响），
> 与 demo_451 的做法一致。
