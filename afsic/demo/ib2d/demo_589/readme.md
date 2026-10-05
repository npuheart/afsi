# demo_589 — IB2d `Example_HeartTube/EM_Pump_w_Calcium_and_Muscles`（钙电生理直接驱动 FV_LT 肌肉）

IB2d 官方算例（`matIB2d/Examples/Example_HeartTube/EM_Pump_w_Calcium_and_Muscles`）在 AFSI 中的复现。
心管家族里**唯一同时打开 `electro_phys` 和 `FV_LT_muscle`** 的算例：此时驱动不再调制弹簧刚度，
而是把电生理信号直接当作**肌肉激活系数**。

## 1. 算例内容

| 项目 | 值（官方 `input2d`） |
|---|---|
| 周期盒 / 网格 | `5 x 5`，`64 x 64`，`mu = 1`，`rho = 1` |
| 时间 | `dt = 1e-4`，`T = 4`（40000 步） |
| 结构 | 156 个标记点（上下各 78）；154 根弹簧（`k = 1e7`，**没有跨管弹簧**）、152 根不变梁、4 个目标点、**76 条 FV_LT 肌肉带**（`s -> s+78`，`Fmax = 1e5`、`LFO = 1`、`SK = 0.3`、`a = 0.25`、`b = 4`） |
| 开关 | `electro_phys = 1`，`FV_LT_muscle = 1`，`update_springs = 0` |

## 2. 移植的机制

驱动里的判断是

```matlab
if ( electro_phys_Yes == 1 && muscles_Yes == 0 )      % ← 本算例不满足
    springs_info(ePhys_Start:ePhys_End, 3) = (16 v)^4;
end
```

所以本算例**跳过** demo_587/588 的"改刚度"路径。电生理（钙动力学 + 钙驱动的一维 FHN，
`Calcium_Dynamics.m` + `FitzHugh_Nagumo_1d.m`，78 个管点、每 IBM 步 10 个模型步，
与 demo_588 完全相同的模块）只被用来给肌肉的激活系数赋值
（`give_Ca_ElectroPhys_Muscle_Activation.m`）：

```
a_f = 1e1 * v(t, leader node)
Fm  = a_f * Fmax * exp(-((Q-1)/SK)^2) * (1/P0)(b P0 - a vc)/(vc + b),  Q = LF/LFO, P0 = Fmax
```

其中 `v(t, ·)` 是膜电位（`a_f` 峰值约 **12.5**，而 demo_612/592 的行波激活只有 0/1），
`vc` 是肌肉带的收缩速度。力施加在 `leader`/`follower` 两点上，与 FV_LT 模型一致。

> 注意 `.m` 里 `electro_potential(floor(ct+1), ith_muscle)` 的第二个下标传的是
> **leader 节点编号**（`id_Leader`），而电位矩阵的列正是沿管的节点序号，
> 所以"第 i 条肌肉用第 i 个节点的电位"是**有意为之**（不是 off-by-one）。
> 本移植在 `give_Muscle_Activation.py` 里从算例自带的 `HeartTube.muscle` 读出 leader 编号，
> 按同样的规则取列。

## 3. 移植要点

1. **pyIB2d 没有 electro 分支**：它的 FV_LT 路径只 import 运行目录下的
   `give_Muscle_Activation`，并用 `(v, LF, LFO, SK, a, b, Fmax, current_time, xPt, xLag)`
   的向量化签名调用。因此本 demo 把示例的
   `give_Ca_ElectroPhys_Muscle_Activation.m` 移植成**同名同签名**的
   `ib2d_input/give_Muscle_Activation.py`，`run_reference.py`（pyIB2d）与 `main.py`
   **共用这一份文件**，于是参考解里也包含完全相同的电生理/激活/力律。
   leader 节点列由 `HeartTube.muscle` 得到（对参考侧 0 基、`ib2d_input` 里 1 基文件自动判别）。
2. 激活模块的数值与 MATLAB 逐行转写**完全一致**（`give_Muscle_Activation` 与
   `1e1*v[row, id_Leader]` 直接求值对比，`max` 相对差 `0.0`，在 `T = 0.02` 的第 1/51/201 步检验）。
3. `Calcium_Dynamics.py` 与 MATLAB 逐行转写逐位一致；钙驱动 FHN 与逐行转写差 `1.1e-16`（见 demo_588）。
4. 帧对齐沿用 demo_478 的约定（驱动第 `k` 帧 = 第 `k*print_dump + 1` 步的状态）。
5. `T = 4 s` 是官方完整设置；本 demo 截断到 **`T = 0.02 s`**（`dt = 1e-4`，200 步）
   并把 `NX` 从 64 降到 **32**（`h = 0.15625`），只展示钙刺激打开后的第一个收缩起段，
   已在 readme 记录（`TFINAL`/`NX` 均可覆盖）。

## 4. 结果（`compare.py`）

参考与 AFSI 各 11 帧（`print_dump = 20`），对齐 10 帧（`t = 0 … 0.0181`）。
（本轮运行主机被多个 dolfinx 作业挤占，wall time 不代表算法开销：AFSI 200 步实测 **9.7 s** CPU。）

| 指标（末帧 `t = 0.0181 s`） | 结果 |
|---|---|
| 管径（76 条肌肉带平均长度，静息 1.0） | AFSI 0.8526 / IB2d 0.8622；逐帧 `max\|Δw\| = 1.1e-2` |
| 收缩相位 | 两侧极小值同在 `t = 0.0181`（仍在收缩中） |
| 长度-张力因子 `F1` 均值 | AFSI 0.773 / IB2d 0.768；`max\|ΔF1\| = 8.0e-2` |
| 标记点偏差（156 点） | 中位 1.4e-2 m = **0.09 h**，最大 2.6e-2 m = 0.17 h（`h = 0.15625`） |
| 末帧速度场 | `max\|u\|` AFSI 58.9 / IB2d 58.1，`max\|Δu\| = 25.3` |

**能验证**：① 电生理 + 激活系数（与 MATLAB 逐行转写一致到 0）；② 肌肉驱动的收缩——
管径（0.853/0.862）、长度-张力因子、收缩相位在 1e-2 量级内一致；
③ 标记点轨迹在中位 **0.09 h**、最大 0.17 h。

**不能验证（如实记录）**：
* **瞬时速度场**：`max|Δu| = 25.3`，与峰值同量级。收缩速度极大
  （`max|u| ≈ 59 m/s`，`u dt/h = 0.038`... 即 AFSI 侧 `max|u| dt / h` 约 0.038 但管壁在
  肌肉激活开启瞬间仍有强加速度），两种离散的差异被放大。
* 参考解里的电生理/激活是本 demo 注入的（原因见 §3.1），因此验证的是
  **求解器 + 耦合**，而不是"pyIB2d 自带的实现"（不存在）。

## 5. 运行

```bash
conda activate afsi-dolfinx
python run_reference.py --tend 0.02 --nx 32 --print-dump 20     # pyIB2d 参考
TFINAL=0.02 NX=32 PRINT_DUMP=20 python main.py                  # AFSI
python compare.py
```

环境变量：`TFINAL`、`DT`、`NX`/`NY`、`PRINT_DUMP`、`OUTPUT_PATH`、`GRAD_DIV`（默认 0）、
`EPHYS_TFINAL`（默认跟随 `TFINAL`）。
