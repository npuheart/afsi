# demo_493 — IB2d `Example_Wobbly_Beam/Beam_256`

IB2d 官方算例（`matIB2d/Examples/Example_Wobbly_Beam/Beam_256`）在 AFSI 中的复现。
它只用**不变梁 + 目标点**两种机制，因此模板无需任何用户模块。

## 算例内容

* `1 x 1` 周期盒、`mu = 0.02`、`rho = 1`、官方 `256 x 256`、`dt = 5e-5`。
  官方 `T = 1.25`（25000 步）→ 本 demo **`NX = 128`、`T = 0.05`（1000 步）**，
  `print_dump = 50`（20 帧）。
* **结构**：384 个拉格朗日点构成一条 "wobbly"（初始形状左右起伏）的曲线；
  **382 根不变梁**（`kb = 5e11`、目标曲率 `C = 0`，即梁"想变直"）
  + **2 个钉死目标点**（`k = 1.75e8`）固定两端。
* 与 IB2d 的 `Beam_256` 一样：梁被弹直后来回颤振（wobble），流体被拖着走。

## 移植要点

1. **无需移植任何 `.m` 模块**：`input2d` 只打开 `beams`/`target_pts`，
   模板的 `beam_forces`（不变梁 = IB2d 的扭转弹簧形式）与 `target_forces`
   已实现，并且 `main.py` 在启动时把向量化力与逐行循环对照（机器精度）。
2. `BeamCurve.beam` / `BeamCurve.target` 为 1-based，`ib2d_io.load_example`
   自动减 1（`index_base=1`）。
3. 唯一改动是**降低分辨率 + 缩短 Tfinal**（见"运行"）：官方网格 256²、
   25000 步在这个机器负载下跑不完。

## 结果

参考运行约 48 s、AFSI 运行约 314 s（含解释器/库导入，机器同时并行运行其它
demo，见"运行"）。20 帧对齐（`t = 0 .. 0.04755`）。以下为 `compare.py`
自己输出的原文：

| 指标（末帧） | compare.py 原文 |
|---|---|
| 标记点最大偏差 | `max|dX| 1.074e-02 m = 1.38e+00 h (384 Lagrangian points)` |
| 速度峰值 AFSI / IB2d | `4.4887e+00 / 4.6376e+00 (3.21e-02 relative)` |
| 速度场偏差 | `max|du| 2.972e-01 = 9.36e-01 max|u|`（带外）; `2.117e-01 (0.95 max|u|)`（带内）; `relative L2 4.825e-02` |
| 力场偏差（带外） | `max|df| 3.335e+02 away from the band, |f|max 8.683e+02` |

**能验证**：结构运动学（标记点 1.38 h）、速度峰值 3.2%、速度场相对 $L_2$
4.8e-2 —— 即"梁怎么摆、流场整体多快"两边一致。
**不能验证（如实记录）**：瞬时速度场逐点值（`max|du|` 达 0.94 `max|u|`）与
力场（带外相对差约 0.4）。梁刚度 `kb = 5e11` 极高，弯曲波在 IB2d
（FFT 投影 + 共置罚力）与 AFSI（Taylor–Hood + 插值力密度）中的传播不同，
细网格上的高频振荡无法逐点重合。

## 运行

```bash
conda activate afsi-dolfinx
python run_reference.py --tend 0.05 --print-dump 50 --nx 128     # ~48 s
TFINAL=0.05 PRINT_DUMP=50 NX=128 python main.py                 # ~314 s
python compare.py
```

（`NX=128` 同时把 `Ny` 按盒子长宽比取整为 128。）
