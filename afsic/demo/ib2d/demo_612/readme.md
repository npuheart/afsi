# demo_612 — IB2d `Example_HeartTube/Muscle_Model`（FV_LT 肌肉心管）

IB2d 官方算例（`matIB2d/Examples/Example_HeartTube/Muscle_Model`）在 AFSI 中的复现。
这就是 demo_448（pyIB2d `HeartTube_Muscle`）的 **matIB2d 孪生版**：同一几何、同一肌肉参数，
只是文件是 1-based 编号、并多看了一眼 `Muscle_Model/input2d` 的官方设置。

## 1. 算例内容

| 项目 | 值（官方 `input2d`） |
|---|---|
| 周期盒 / 网格 | `5 x 5`，`128 x 128`，`mu = 0.1`，`rho = 1` |
| 时间 | `dt = 5e-4`（**见下方注意**），`T = 0.25`（500 步） |
| 结构 | 310 个标记点（上下各 155，`x in [1,4]`，`y = 2.5 ± 0.5`）；308 根弹簧（`k = 1e7`，静长 `ds = 0.0195`）、306 根不变梁（`kb = 7.5e7`）、4 个目标点、**153 条 FV_LT 肌肉带**（`i -> i+155`，`Fmax = 1e5`、`LFO = 1`、`SK = 0.3`、`a = 0.25`、`b = 4`） |
| 激活 | 算例自带的 `give_Muscle_Activation.m`：10 Hz **行波方波**（激活区 = 管长去掉两端各 25% buff，方波宽为激活区 1/10） |

数据文件和 demo_448 的 pyIB2d 版**逐个数值相同**，只有编号基不同（已核对）。

## 2. 机制与移植

肌肉力律（与 demo_448/466 完全相同，逐字复刻）：

```
Fm = a_f(t, x_i) * Fmax * exp(-((Q-1)/SK)^2) * (1/P0)(b P0 - a vc)/(vc + b),
Q = LF/LFO,  P0 = Fmax,  vc = |LF(X_h) - LF(X_h^{prev})|/dt
```

`a_f` 直接 import 算例自带的激活函数（移植为 `ib2d_input/give_Muscle_Activation.py`，
与 demo_448/466 里的那份逐字节相同），`run_reference.py` 把同一份文件拷进参考运行目录，
所以两个代码用的是**同一个激活函数**。结构力（弹簧/不变梁/目标点/肌肉）都在
半步位置 `X_h` 求值，收缩速度用上一半步位置（驱动的 `xLag_P` 记账）。

> **注意（dt）—— demo_448 的同款问题**：官方 `dt = 5e-4` 下 AFSI 的显式耦合
> **4 步内发散**（`mean_width` 从 1.0 飙到 3.4e3、`|F|max` 到 7e11），而 pyIB2d 在同
> `dt` 下稳定 —— 两侧显式稳定域不同。demo_448 已经记录过同一现象并把双侧统一下调到
> `dt = 1e-4`；本 demo 照做（`--dt 1e-4` / `DT=1e-4`，双侧一致，仍是同参数对照）。

> **编号与帧对齐**：`ib2d_io.load_example(..., index_base=1)` 把 1-based 文件重编号为 0-based；
> `run_reference.py` 重写 `.spring`（2 列）、`.beam`（3 列）、`.target`（1 列）、`.muscle`（2 列）。
> 帧对齐沿用 demo_478 的约定（驱动第 `k` 帧 = 第 `k*print_dump + 1` 步的状态）。

## 3. 结果（`compare.py`）

参考 20 帧、AFSI 21 帧，对齐 20 帧。AFSI 运行参数：`DT=1e-4`（与参考一致）、
`NX=64`（`h = 0.0781`，`TFINAL=0.1` 截断，1000 步，实测 **172 s**）；
参考 `--tend 0.1 --dt 1e-4 --nx 64 --print-dump 50`。

| 指标（末帧 `t = 0.0951 s`） | 结果 |
|---|---|
| 管径（153 条肌肉带平均长度，静息 1.0） | AFSI **0.6250** / IB2d **0.6167**（行波把管子收缩约 38%）；逐帧 `max\|Δw\| = 2.1e-2` |
| 收缩（激活）相位 | 两侧管径极小值同在 `t = 0.0951`；管径历史逐帧同步（`t = 0.02` 时 0.953/0.966，`t = 0.05` 时 0.895/0.916） |
| 长度-张力因子 `F1` 均值 | AFSI 0.5333 / IB2d 0.5324；`max\|ΔF1\| = 2.6e-1`（出现在激活前沿最陡处） |
| 标记点偏差（310 点） | 中位 4.3e-2 m = **0.55 h**，最大 6.4e-2 m = 0.82 h（`h = 0.0781`） |
| 末帧速度场 | `max\|u\|` AFSI 26.4 / IB2d 29.6，`max\|Δu\| = 14.9` |

**能验证**：行波激活的 FV_LT 肌肉驱动本身——管径收缩幅度（0.625/0.617）、收缩相位
（极小值同一帧）、管径历史逐帧一致（`max|Δw| = 2.1e-2`，约为收缩行程的 5%）、
标记点在 0.55 h（中位）量级。

**不能验证（如实记录）**：
* **瞬时速度场**（`max|Δu| = 14.9`，与峰值同量级）与逐点标记轨迹的尾部
  （最大偏差 0.82 h，出现在行波前沿经过处的高加速度段）。
* **分辨率与时间步都被下调**：官方 `dt = 5e-4`（发散，见上）与 `128×128` 网格；
  本 demo 用 `dt = 1e-4` + `NX = 64`（`h` 与 IB2d 的 `ds` 都随之变化），
  并且只跑 `T = 0.1 s`（官方 0.25 s）——因此**不是**对官方设置的全参数复现。

## 4. 运行

```bash
conda activate afsi-dolfinx
python run_reference.py --tend 0.1 --dt 1e-4 --nx 64 --print-dump 50   # pyIB2d 参考
DT=1e-4 TFINAL=0.1 NX=64 PRINT_DUMP=50 python main.py                  # AFSI
python compare.py
```

环境变量：`TFINAL`、`DT`、`NX`/`NY`、`PRINT_DUMP`、`OUTPUT_PATH`、`GRAD_DIV`（默认 0）。
