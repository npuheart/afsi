# demo_468 — IB2d `Example_1_Wall_Flow_Around_Cylinder`（单壁 + 圆柱绕流的任意外力驱动）

IB2d 官方算例（`matIB2d/Examples/Example_Channel_Flow/Example_1_Wall_Flow_Around_Cylinder`）
在 AFSI 中的复现，沿用 demo_467 建立的 `arb_ext_force` 模板。

## 算例内容

`1 x 1` 周期盒、`mu = 0.01`、`rho = 1`、`32 x 32`、`dt = 1e-4`、`T = 0.125`
（1250 步）。

* **结构**：85 个标记点，**全部**为 `k = 1e7` 的钉死目标点——一条 58 点的壁
  （`y = 0.4`，`x in [0.05, 0.94]`，点距 `dx/2`）加一个 27 点的圆柱
  （半径 `0.0333`，圆心 `(0.2998, 0.5)`）。注意圆柱半径仅 ≈1.07 `h`
  （`h = 1/32`），本算例的网格**故意粗到几乎无法分辨圆柱**。
* **外力**：算例的 `please_Compute_External_Forcing.m` 在
  `x in [0.1, 0.16]`、`y in [0.41, 0.59]` 施加速度目标型罚力

  $$F_x = -k\,(u_x - u_x^{\rm tar}),\qquad
  u_x^{\rm tar} = -u_{\max}\,5\tanh(t)\,
  \Big(\tfrac{L_x}{2}+\tfrac{W}{2}-y\Big)\Big(\tfrac{L_x}{2}-\tfrac{W}{2}-y\Big),$$

  `k = 1e2`（比 demo_467 软两个量级）、`W = 0.2`、`u_max = 250`，即抛物线射流
  随时间 `tanh(t)` 缓慢开启；`F_y = 0`。罚力增益 `k\,dt/\rho = 1e-2` 远离边际值，
  因此**不会**出现 demo_467 那种壁面窄带伪速度。

## 移植要点

与 demo_467 完全相同的 `arb_ext_force` 机制（节点力密度 → Q1 节点求值 →
Q2 插值叠加到 `solver.f`），外加可选的弹簧/梁（本算例只有目标点）。本 demo 起
模板做了通用化：

* `main.py` 从算例文件里按需装配弹簧/梁/目标点，并从被移植的模块里读出力窗口
  与常数（打印用）；
* `compare.py` 的所有网格相关量（力窗口、目标速度剖面、IB 窄带）都由**模块本身**
  与**目标点坐标**导出，因此同一份 `compare.py` 可直接用于本族其他算例；
  它还会自动配对 `ib2d_reference<TAG>.npz`/`plot<TAG>/` 做交叉核对。

## 结果

参考运行 ~10 s、AFSI 运行 ~5 s，两边各 13 帧（对齐 13 帧）。

| 指标（末帧 `t = 0.12`） | 原样 `32 x 32`, `dt = 1e-4` | 加密 `64 x 64`, `dt = 5e-5` |
|---|---|---|
| 标记点偏差（壁 + 圆柱钉死） | 2.30e-5 m = **7.4e-4 h** | 9.3e-6 m = 6.0e-4 h |
| `max\|u\|` AFSI / IB2d | 0.8617 / 0.9548（**低 10%**） | 0.8186 / 0.8236（**低 0.6%**） |
| 施力窗口 `max\|u\|` 相对差 | 4.8e-2 | **6.0e-3** |
| 速度场相对 $L_2$ 误差（窄带外） | 1.66e-1 | **3.8e-2** |
| `max\|du\|`（窄带外） | 2.12e-1 = 0.22 `max\|u\|` | 6.4e-2 = 7.7e-2 `max\|u\|` |
| 槽道流量（中列、窗口行）相对差 | 1.8e-1 | **5.3e-2** |
| 力场 `max\|df\|`（窄带外，`\|f\|_{\max}` 4.7e2） | 13.8 | 3.4 |
| 罚力公式核对（`f_x` vs 模块按 AFSI 速度算出的力） | ≤6.2e-2 | ≤2.2e-2 |

**结论**：原样网格上两边的差异（$L_2$ 1.7e-1）不是实现错误，而是**粗网格上的
离散差异**——圆柱半径只有 1.07 `h`，Q1 有限元与 IB2d 共置 FFT 投影在这种
"几乎不可分辨的障碍物"上只有一阶一致的解；把网格加倍（圆柱 2.1 `h`）后
$L_2$ 误差降到 3.8e-2、`max|u|` 差 0.6%。标记点（钉死的壁与圆柱）在两种分辨率下
都稳定在 $7\times10^{-4}h$ 量级，力场的公式核对稳定在 $10^{-2}$（`|f|` 峰值的
万分之几）。`compare.py` 报出的 `window max|u|` 时间序列（原样网格：
`0.019/0.014, ..., 0.819/0.824`）显示两边在整段过程中同步增长，偏差从早期的
30% 单调收敛到 10%，与"网格尺度差异随流动变光滑而减小"一致。

## 运行

```bash
conda activate afsi-dolfinx
python run_reference.py                  # pyIB2d 参考 -> ib2d_reference.npz（~10 s）
python main.py                           # AFSI -> plot/afsi_result.npz（~5 s）
python compare.py                        # 对比 + figures/（自动带 64^2 交叉核对）

# 加密交叉核对（compare.py 会自动识别）
python run_reference.py --nx 64 --dt 5e-5 \
    --out ib2d_reference_n64.npz --run ib2d_run_n64
NX=64 DT=5e-5 PRINT_DUMP=100 OUTPUT_PATH=./plot_n64 python main.py
```

环境变量（`main.py`）：`TFINAL`、`DT`、`NX`/`NY`、`PRINT_DUMP`、`GRAD_DIV`（默认 0）、
`OUTPUT_PATH`（默认 `plot/`）、`IB2D_EXAMPLE`（默认 `ib2d_input/`）。

## 文件

* `ib2d_input/`：`input2d`、`channel.vertex`、`channel.target`（matIB2d 1-based 原样）
  以及 `.m` 的 Python 移植 `please_Compute_External_Forcing.py`。
* `main.py` / `run_reference.py` / `ib2d_reference.py` / `compare.py`：同 demo_467
  的家族模板（本 demo 起通用化）。
