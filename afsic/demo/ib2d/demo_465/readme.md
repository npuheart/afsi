# demo_465 — IB2d `Moving_Circle`（状态插值：线性 vs 三次）

IB2d 教育算例对（`matIB2d/Examples/Examples_Education/Interpolation/Moving_Circle/`
的 `Linear_Interp` 与 `Cubic_Interp`）在 AFSI 中的复现。两个算例**只差插值函数**，
因此用一个移植 + `INTERP` 环境变量同时覆盖，并各自与自己的 pyIB2d 参考对比。

## 算例内容

`1 x 1` 周期盒、`mu = 10`、`rho = 1`、`32 x 32`、`dt = 1e-4`、`T = 0.1`（1000 步）。
结构是 64 点圆（半径 0.2），全部为 `k = 1e7` 目标点；锚点在三个预设位置之间往返：

$$\mathbf X_{\rm A}=(0.35,0.25),\quad \mathbf X_{\rm B}=(0.85,0.45),\quad \mathbf X_{\rm C}=(0.55,0.75)$$

周期 $2(t_1+t_2) = 0.06$ s（$t_1=0.01$：A↔B；$t_2=0.02$：B↔C），
每个相位内

$$\mathbf X = \mathbf S_1 + g(s)\,(\mathbf S_2 - \mathbf S_1),\qquad s = \frac{t-t_{\rm phase}}{t_{\rm phase\ 时长}}$$

| 方案 | $g(s)$ |
|---|---|
| `INTERP=linear`（默认） | $g(s) = s$ |
| `INTERP=cubic` | 分段三次：$s\le0.25$ 时 $a_3s^3$；$s\le0.925$ 时 $\sum b_i s^i$；之后 $\sum c_i s^i$（两端**函数值**连续，圆仍精确经过 A/B/C） |

$T=0.1$ s 覆盖 1.67 个周期，圆的行程（A↔C）为 0.45 m = **14.4 h**（h = 1/32），
即一步走 1.2 h 量级——与 demo_459 同属"快运动浸入边界"。

## 结果

参考运行每个方案约 5 s、AFSI 每个方案约 8 s（各 200/201 帧）：

| 指标 | `linear` | `cubic` |
|---|---|---|
| 标记点偏差 | 最大 **8.30e-4 m = 0.027 h**，中位 1.0e-4 m | 最大 **6.08e-4 m = 0.019 h**，中位 1.4e-4 m |
| 圆心轨迹 | 最大差 3.7e-4 m | 最大差 2.7e-4 m |
| 圆心-锚点滞后 | ≤ 1.19e-2 m（瞬态），稳态 ~1e-3 m | ≤ 4.9e-3 m |

结论：**两种插值方案都在 0.02 h 量级内复现参考**，圆经过 A/B/C 的相位与轨迹一致；
这一对算例的价值在于它同时检验了"多状态插值"机制（demo_460 的两状态插值之外的
一般情形）与两种时间参数化的差异：三次参数化在相位切换处的运动更"缓入缓出"，
两侧的圆心-锚点滞后也随之不同（1.2e-2 vs 4.9e-3 m），但两种方案下两个求解器都一致。

## 实现

1. 一个 `.m` 的 Python 移植通过 `INTERP` 环境变量选择 $g(s)$，参考运行与 AFSI 侧共用
   同一套状态文件与相位划分；`ib2d_io.load_example(..., index_base=1)` 重编号
   `matIB2d` 的 1-based 文件，`run_reference.py` 写出重编号后的 `.target` 并把
   `State_A/B/C.pts` 与移植模块拷进运行目录。
2. 结构用闭合零刚度 FE 环承载 64 个标记点自由度；输出帧与 pyIB2d 落在**同一步**。

## 运行

```bash
conda activate afsi-dolfinx
python run_reference.py --out ib2d_reference_linear.npz --run ib2d_run_linear
INTERP=cubic python run_reference.py --out ib2d_reference_cubic.npz --run ib2d_run_cubic
python main.py                                     # linear -> plot/
INTERP=cubic OUTPUT_PATH=./plot_cubic python main.py
python compare.py                                  # 两个方案一起对比 + figures/
```

环境变量：`TFINAL`、`DT`、`NX`、`PRINT_DUMP`、`GRAD_DIV`、`INTERP`（`linear`/`cubic`）、
`OUTPUT_PATH`。
