# demo_587 — IB2d `Example_HeartTube/Electromechanical_Pumping`（电生理驱动的心管泵）

IB2d 官方算例（`matIB2d/Examples/Example_HeartTube/Electromechanical_Pumping`）在 AFSI 中的复现。
这是心管（HeartTube）家族里**第一个真正用到 `electro_phys`（FitzHugh–Nagumo 电生理）机制**的算例：
一维 FHN 反应–扩散模型的膜电位被写进**跨管弹簧的刚度**，把管子周期性捏合 → 泵送。

## 1. 算例内容

| 项目 | 值（官方 `input2d`） |
|---|---|
| 周期盒 / 网格 | `5 x 5`，`64 x 64`，`mu = 1`，`rho = 1` |
| 时间 | `dt = 1e-4`，`T = 4`（40000 步） |
| 结构 | 156 个标记点（上下各 78 点，`x in [1,4]`，`y = 2.5 ± 0.5`） |
| 弹簧 | 232 根（`k = 1e7`，静长 `ds = 0.0390625`）；最后 78 根是**跨管弹簧**（`k = 1e-3`、静长 0） |
| 梁 | 152 根不变梁（`kb = 7.5e7`，`C = 0`） |
| 目标点 | 4 个角点（`k = 1e6`，钉死） |
| 开关 | `electro_phys = 1`，`update_springs = 0`，无肌肉/无标量/无外力 |

`HeartTube.m` 里弹簧的顺序是：先沿上下壁的 154 根（`s -> s+1`），再 78 根跨管弹簧（`s -> s+78`），
所以**弹簧第 155–232 行就是那 78 根跨管弹簧**，与 `FitzHugh_Nagumo_1d.m` 里的
`ind_Start = 155`、`ind_End = 232` 完全对应。

## 2. 移植的机制（`electro_phys`）

官方驱动在 IBM 时间循环**之前**先一次性解一维 FitzHugh–Nagumo：

```
dv/dt = D v_xx + v (v - a)(v - 1) - w + I(t),    dw/dt = eps (v - gamma w)
```

* 参数：`D = 10`、`a = 0.3`、`gamma = 1`、`eps = 0.001`、`I_mag = 0.05`；
* 空间：`numPtsAlongTube = 78` 个 IBM 管点、`factor = 4` 倍加密（`N = 311`，`L = 500`），
  **非周期** Laplacian（`.m` 里周期那两行被注释掉了：两端为 0）；
* 激活电流 `I(t)`：10 个脉冲（`Np = 10`），脉冲间隔 `T_final/Np = 0.4 s`，
  注入窗口 `j = floor(0.25N) ... floor(0.3N)`，宽度 `dp = 脉冲间隔/50 = 0.008 s`；
* 显式前向 Euler，`numFHN = 10` 个 FHN 步对应 1 个 IBM 步；
* 时间被放大 2500 倍（`dt -> dt*2500`，`T -> T*2500`）。

然后在**每一步、力计算之前**，驱动把该步的膜电位写进弹簧刚度：

```matlab
springs_info(155:232, 3) = ( 1.6e1 * electro_potential(ePhys_Ct, :) ).^4;
```

`ePhys_Ct` 从 1 开始、每步 +1，所以 IBM 第 `k` 步用的是 `v` 历史第 `k` 行（`t = (k-1)dt`）。
电位 `v ∈ [0, 1.26]`，因此 `k_cross = (16 v)^4` 峰值约 `5.6e4`（加载时管径从 1.0 收缩到 0.35，随后松弛）。

AFSI 侧 `main.py` 完全照此实现：`ib2d_input/FitzHugh_Nagumo_1d.py` 逐行移植
（时间循环保留、空间轴向量化），`apply_electro(step)` 写 `sp_k[154:232] = (16 v[step])**4`，
位置与顺序都按驱动来（半步位置之后、力之前）。

## 3. 移植要点与上游疑点

1. **上游疑点（本算例最重要的发现）**：例子里自带的 `update_Springs.m` 注释说
   第 3 列是刚度、**第 4 列是静长**，但 `IBM_Driver.m` 的 electro 块写的是
   `springs_info(...,3)`，而 matIB2d 的弹簧数组布局是 `[p1, p2, k, L_rest, alpha]`
   （见 `please_Find_Lagrangian_Forces_On_Eulerian_grid.m:415-417`）。也就是说
   **被调制的是弹簧刚度而不是静长**，与 `update_Springs.m` 的注释（以及该目录里那份
   从未被调用的 `update_Springs.m`）不一致。本移植按**真实代码**（刚度）实现。
2. **另一个上游疑点**：`FitzHugh_Nagumo_1d.m` 先 `vStore = zeros(NT_IBM, N+1)`，
   但存储计数器最后写到第 `NT_IBM + 1` 行 —— MATLAB 会静默**自动扩行**，
   所以最终是 `NT_IBM + 1` 行。本移植按真实行数分配（并且只存
   `compute_IBM_Potential_At_IBM_Lag_Pts` 真正用到的每 4 点，物理上等价）。
   同时 `.m` 还存了 `wStore`（从不读取），本移植只存膜电位。
3. **pyIB2d 根本没有实现 `electro_phys`**：它把 flag 读进 `electro_phys_Yes` 后就不再使用
   （`IBM_Driver.py:213`，全文再无引用）。因此**参考解不可能"原样"跑出来**。
   本 demo 的做法是：让参考运行打开其**文档化的 `update_springs` 钩子**
   （`ib2d_input/update_Springs.py`），用与 `main.py` **同一份** FHN 模块把同样的
   `(16v)^4` 写进弹簧刚度。于是"参考"= pyIB2d 的流体/结构求解器 + 完全相同的电生理耦合，
   二者唯一的差别是 FSI 离散本身。这一点在下面"不能验证的部分"里如实说明。
4. **时间对齐**：pyIB2d 与 AFSI 都用 `while current_time < T: current_time += dt`，
   本 demo 照抄该循环（而不是 `round(T/dt)`）以保证步数一致；帧对齐沿用
   demo_478 的约定（驱动第 `k` 帧是第 `k*print_dump + 1` 步的状态，其 VTK 时间戳落后一步，
   属于上游标注瑕疵）。
5. **最小镜像分支**：pyIB2d 的 `give_Me_Spring_Lagrangian_Force_Densities` 里
   `|dx| > Lx/2` 的写法是 `dx = sign(dx)*(Lx - sign(dx)*dx)`，它把符号又乘回去了
   （对 `dx = -4.8`、`Lx = 5` 给出 `-0.2` 而不是 `+0.2`，方向上等于把弹簧力反号）。
   本例里**没有一根弹簧会触发该分支**（跨管弹簧跨度约 1.0 < `Lx/2 = 2.5`，壁弹簧跨度 `ds`），
   所以两种写法在本算例完全等价；AFSI 侧用的是正确的最小镜像
   `d -= round(d/L)*L`，并在此如实记录这一上游写法差异。
6. `T = 4 s` 是官方的完整设置（40000 步）；本 demo 的验证运行**截断到 `T = 0.45 s`**
   （一个完整的脉冲周期 0.4 s 加一点松弛），并在 readme 里明确记录。
   注意 FHN 的脉冲间隔是 `T_final/10`，所以**电生理的 horizon 必须保持 4 s**
   （`EPHYS_TFINAL`，默认取 `input2d` 里的 `Tfinal`），只截断流体计算；这样脉冲时序与官方一致。

## 4. 结果（`compare.py`）

参考运行 **83 s**（4500 步，64²；含 T=4 的 FHN 预计算），AFSI 运行 **482 s**（4500 步，
`GRAD_DIV = 0`）。两边各 91 帧（`print_dump = 50`，`t = 0 … 0.45 s`）。

| 指标（`t = 0.45 s`） | 结果 |
|---|---|
| **电生理耦合**：78 根跨管弹簧的 `(16v)^4` | AFSI 与参考**逐位相同**（`max\|Δ\| = 0`） |
| 管径（跨管弹簧平均长度，静息 1.0） | AFSI 0.3456–1.0000 / IB2d 0.3576–1.0000；逐帧 `max\|Δw\| = 4.1e-2` |
| 收缩相位 | 两侧管径极小值同在 `t ≈ 0.105`（第一个脉冲后收缩 65%） |
| 标记点偏差（156 点） | 中位 2.74e-2 m = **0.35 h**，最大 6.6e-2 m = 0.85 h |
| 末帧速度场 | `max\|u\|` AFSI 4.75 / IB2d 4.65，`max\|Δu\| = 0.66` |
| 收缩过程中 `max\|u\|` | 峰值约 70 m/s（`u dt/h ≈ 0.09`，两种离散都在分辨率边缘） |

**能验证**：① 电生理耦合本身**精确**（写进弹簧刚度的那 78 个数完全一致）；
② 管径的收缩深度（65%）与相位（极小值时间）在 1e-2 量级内一致；
③ 标记点轨迹在 0.35 h 量级（`h = Lx/Nx = 0.078`）。

**不能验证（如实记录）**：
* **瞬时速度场**（`max\|Δu\| = 0.66`，约为峰值的 14%）。原因与 demo_478/474 相同：
  剧烈运动的高速管壁（`max\|u\| ≈ 70 m/s`，`u dt/h ≈ 0.9` 的 CFL 边缘）在
  "FFT 投影 + 共置罚力"（IB2d）与 "Taylor–Hood + 插值力密度"（AFSI）两种离散下不同。
* **"参考解是纯 pyIB2d"**：因为 pyIB2d 没有 electro 实现，参考运行里的电生理耦合
  是由本 demo 提供的 `update_Springs.py` 注入的（与 `main.py` 共享同一份代码）。
  也就是说这个 demo 验证的是 **FSI + 耦合**，而**电生理求解器本身**是靠
  `FitzHugh_Nagumo_1d.py` 与逐行 MATLAB 转写的对拍验证的（见下）。

电生理求解器的独立验证（`FitzHugh_Nagumo_1d.py` vs 1-based MATLAB 逐行转写）：
`T = 0.02 / 0.04 / 0.08` 三种设置下 `max|Δv| = 8.7e-19 / 1.7e-18 / 3.5e-18`（浮点舍入）。
（`accumulate t = t + dt` 与 `t = k*dt` 会改变脉冲开关落在哪一步，本移植按 MATLAB 的累加方式实现。）

## 5. 运行

```bash
conda activate afsi-dolfinx
python run_reference.py --tend 0.45 --print-dump 50     # pyIB2d 参考（~83 s）
TFINAL=0.45 PRINT_DUMP=50 python main.py               # AFSI（~482 s）
python compare.py                                      # 表格 + figures/
```

环境变量：`TFINAL`、`DT`、`NX`/`NY`、`PRINT_DUMP`、`OUTPUT_PATH`、`GRAD_DIV`（默认 0）、
`EPHYS_TFINAL`（电生理 horizon，默认 `input2d` 的 `Tfinal = 4`）。
`--nx`/`--dt`/`--print-dump`/`--tend` 用于参考；`--ephys-tend` 单独控制参考侧的电生理 horizon。

> 运行时间是在本机同时有多个其它 dolfinx 作业时测得的，绝对时间仅供参考。
