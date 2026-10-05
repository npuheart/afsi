# demo_614 — IB2d `Example_HeartTube/128x128_Electromechanical_Pumping_w_Ca_Dynamics`（高分辨率钙驱动电生理泵）

IB2d 官方算例（`matIB2d/Examples/Example_HeartTube/128x128_Electromechanical_Pumping_w_Ca_Dynamics`）
在 AFSI 中的复现。它是 demo_588（`Electromechanical_Pumping_w_Ca_Dynamics`）的**高分辨率变体**：
机制完全一样（钙动力学 → 一维 FHN → 把 `(16v)^4` 写进跨管弹簧刚度），
只是管子更长、电生理分辨率更高、`dt` 小 10 倍。

## 1. 算例内容

| 项目 | 值（官方 `input2d`） |
|---|---|
| 周期盒 / 网格 | `5 x 2.5`，`128 x 64`，`mu = 1`，`rho = 1` |
| 时间 | `dt = 1e-5`，`T = 4`（**400000 步**） |
| 结构 | 310 个标记点（上下各 155，`y = 1.25 ± 0.5`）；**463 根弹簧**（`k = 1e7`；前 308 根沿壁、后 155 根是跨管弹簧）、306 根不变梁（`kb = 7.5e7`）、4 个目标点 |
| 电生理 | `numPtsAlongTube = 155`、`ind_Start = 309`、`ind_End = 463`、`numFHN = 1`、`D = 10`（与 demo_588 相同，钙驱动电流 `I = 0.5 Caf`） |

跨管弹簧正好是**第 309–463 行**（`(N-2) = 308` 之后的 155 根），与 `.m` 里的索引一致。

## 2. 机制与移植

与 demo_588 **完全相同**：

* `Calcium_Dynamics.py`（Williams 2008 质量作用定律，刺激周期 0.5 s，与 `T_final` 无关）；
* `FitzHugh_Nagumo_1d.py` 一维 FHN（`D = 10`、`a = 0.3`、`eps = 1e-3`、非周期 Laplacian、
  前向 Euler、`numFHN = 1` 即每个 IBM 步走一个 FHN 步、155 个管点）；
* 每步把 `springs_info(309:463, 3) = (16 v)^4` 写进**刚度**（同样的上游疑点：注释说第 4 列是静长，实际写第 3 列）；
* pyIB2d 没有 `electro_phys`，参考运行通过其 `update_springs` 钩子注入同一份模块。

> **上游额外疑点（`higher_resolution/` 子目录）**：该子目录是另一个（更晚的）变体
> （`D = 0.001`、`mu = 3.1`、`T = 2.1`），但它的 `HeartTube.spring` **只有 308 根弹簧**
> （没有跨管弹簧），而它的 `FitzHugh_Nagumo_1d.m` 仍然写 `ind_Start = 309, ind_End = 463`。
> 在 MATLAB 里 `springs_info(309:463,3) = ...` 会把弹簧矩阵**自动扩到 463 行**，
> 新增行的 leader/follower 为 0 → 力计算里 `dx = dy = 0`、`LF = 0` → 除零 NaN。
> 也就是说该子目录按原样**跑不出结果**；本 demo 只移植父目录（可运行的版本）。

> 另注：父目录里有一个 `HeartTube.concentration`（`kDiff = 5e-5`），但 `input2d` 的
> `concentration = 0`，所以该文件**未被使用**；移植时逐字拷贝但不读取。

## 3. 验证设置（重要：大幅截断）

官方设置 = `dt = 1e-5` × `T = 4 s` = **400000 步**；仅电生理的历史矩阵就有约 0.5 GB，
两侧都不可能在本机跑完。本 demo 的验证运行：

* `NX = 64`（`NY = 32`，`h = 0.078`，官方 128×64）；
* `TFINAL = 0.008 s`（**800 步**）——钙刺激在 `t ∈ [0, 0.005]` 打开，
  因此这 8 ms 覆盖"刺激开启 + 膜电位建立 + 跨管弹簧刚度从 0 涨到 9e3"的完整过程；
* `print_dump = 50`（17 帧）。

钙刺激周期是物理时间 0.5 s，与 `T_final` 无关，所以截断**不改变**刺激时序
（`EPHYS_TFINAL` 默认跟随 `TFINAL`）。

## 4. 结果（`compare.py`）

参考与 AFSI 各 17 帧，全部对齐（`t = 0 … 0.008 s`）。AFSI 800 步实测 **40 s**。

| 指标（末帧 `t = 0.0080 s`） | 结果 |
|---|---|
| **电生理耦合**：155 根跨管弹簧的 `(16v)^4` | AFSI 与参考**逐位相同**（`max\|Δ\| = 0`），峰值 1.585e5 |
| 管径（155 根跨管弹簧平均长度，静息 1.0） | AFSI **0.93797** / IB2d **0.93809**；逐帧 `max\|Δw\| = 1.2e-3`（**0.13%**） |
| 刚度增长历史 | 两侧同步：`t = 0.0035` 时 52.6，`t = 0.005` 时 3604，`t = 0.008` 时 8940 |
| 标记点偏差（310 点） | 中位 7.8e-4 m = **1.0e-2 h**，最大 3.0e-2 m = 0.38 h（`h = 0.0781`） |
| 末帧速度场 | `max\|u\|` AFSI 153.6 / IB2d 208.5，`max\|Δu\| = 171.6` |

**能验证**：① 更细/更长的电生理（155 管点、`numFHN = 1`）与耦合——写进弹簧刚度的 155 个数
**逐位相同**；② 收缩的起段——管径 `max|Δw| = 1.2e-3`（0.13%），刚度增长历史逐帧一致，
标记点中位偏差仅 **0.01 h**（本例的运动在前 8 ms 非常温和）。

**不能验证（如实记录）**：
* **瞬时速度场**：`max|Δu| = 171.6`（与峰值同量级）。刚度从 0 涨到 9e3 后管壁加速极快，
  末帧 `max|u|` 两侧分别 154 / 209 m/s（IB2d 侧更高）——这属于"高刚度 + 高速度"下两种
  离散（FFT 投影 + 共置罚力 vs Taylor–Hood + 插值力密度）的差异，与 demo_478/587/588 同类。
* **分辨率与时长**：`NX` 与 `TFINAL` 都远小于官方（见 §3），因此不是官方设置的复现。
* 参考侧的电生理由本 demo 的钩子注入（pyIB2d 无实现），验证的是**求解器 + 耦合**。

## 5. 运行

```bash
conda activate afsi-dolfinx
python run_reference.py --tend 0.008 --nx 64 --print-dump 50   # pyIB2d 参考
TFINAL=0.008 NX=64 PRINT_DUMP=50 python main.py                # AFSI（~40 s）
python compare.py
```

环境变量：`TFINAL`、`DT`、`NX`/`NY`、`PRINT_DUMP`、`OUTPUT_PATH`、`GRAD_DIV`（默认 0）、
`EPHYS_TFINAL`（默认跟随 `TFINAL`）。
