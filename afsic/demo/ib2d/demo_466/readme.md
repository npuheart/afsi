# demo_466 — IB2d `Lymphangion`（行波激活的 FV_LT 肌肉管）

IB2d 官方算例（`matIB2d/Examples/Example_Lymphangion`）在 AFSI 中的复现。
这是"带肌肉驱动的管"算例：**153 条 FV_LT 肌肉带**由**行波方波激活**驱动管壁蠕动。

## 算例内容

`5 x 5` 周期盒、`mu = 1`、`rho = 1`、`128 x 128`、`dt = 5e-4`、`T = 0.25`（500 步）。
结构是 310 点淋巴管（$x\in[1,4]$、$y\in[2,3]$）：弹簧（$k=10^7$）+
306 根不变梁 + 4 个角点目标点 + **153 条肌肉带**（标记点 $i$ 与 $i+155$，
`Fmax = 1e5`、`LFO = 1`、`SK = 0.3`、`a = 0.25`、`b = 4`）。

肌肉力用 IB2d 的 FV_LT（Hill + 长度-张力）模型

$$F = a_f(t,x)\cdot F_{\max}\exp\!\Big[-\Big(\tfrac{Q-1}{SK}\Big)^2\Big]\cdot
\frac{1}{P_0}\frac{bP_0 - av}{v + b},\qquad Q = \frac{L_F}{L_{FO}}$$

其中激活系数 $a_f$ 由算例自带的 `give_Muscle_Activation.m` 给出：**10 Hz 行波方波**，
激活区长度为管长的 50%（两端各留 25% buff），方波宽为激活区的 1/10。
该模块与 demo_448（pyIB2d `HeartTube_Muscle`）中的激活函数**逐字节相同**，
因此本移植直接复用 demo_448 的实现与肌肉力公式。

> **上游开关疑点（本 demo 的额外发现）**：该 `matIB2d` 算例自带
> `Lymphangion.muscle` 与 `give_Muscle_Activation.m`，几何脚本也会打印肌肉文件，
> 但 `input2d` 里 `FV_LT_muscle = 0`——**按原样运行得到的是"无肌肉"的被动管**。
> 本移植默认把它打开（`MUSCLE=1`），并将 `MUSCLE=0` 作为"原样"对照保留。
> 这与 demo_455 发现的反向问题（梁开关拼写错误导致结构力静默为零）属同类上游瑕疵。

## 结果

参考运行 12 s、AFSI 运行 27 s，两边各 50 帧：

| 指标 | 结果 |
|---|---|
| 管径（平均跨径，肌肉带长度均值） | 1.0132 → AFSI **0.5977** / IB2d **0.5957**（收缩 41%）；逐帧最大差 **2.8e-3**（0.5%） |
| 激活相位 | 两侧管径极小值同在 $t=0.245$ |
| 标记点偏差 | 中位 **7.1e-3 m = 0.18 h**，最大 1.6e-2 m = 0.42 h（$t=0.005$ 的启动瞬态）；$t\ge0.05$ 后 ≤ 0.32 h |
| 速度场 | 末帧 `max\|u\|` AFSI 4.39 / IB2d 4.91，`max\|Δu\|` 1.86 m/s |

结论：**行波激活的 FV_LT 肌肉驱动移植正确**——管径的收缩幅度与相位、以及逐帧
管径（0.5% 以内）都与参考一致；标记点逐点偏差在 0.2 h 量级，主要来自启动瞬态。

## 实现

1. `matIB2d` 1-based 文件 → `ib2d_io.load_example(..., index_base=1)`；
   `run_reference.py` 对 `.spring`（2 列）、`.beam`（3 列）、`.target`（1 列）、
   `.muscle`（2 列）分别重编号，并把算例自带的 `give_Muscle_Activation.py`
   拷进参考运行目录（pyIB2d 从那里 import）。
2. AFSI 侧肌肉力、弹簧/梁/目标点、以及 `X_h` / `X_h_prev` 的肌肉速度差分
   （`v = |L_F(X_h) - L_F(X_h^{prev})|/dt`）与 demo_448 相同；默认 $\gamma = 2.5$。
3. 输出帧与 pyIB2d 落在**同一步**（帧对齐约定同 demo_459–465）。

## 运行

```bash
conda activate afsi-dolfinx
python run_reference.py                    # pyIB2d 参考（肌肉开）-> ib2d_reference.npz（~12 s）
MUSCLE=0 python run_reference.py --out ib2d_reference_nomuscle.npz   # 原样（被动管）
python main.py                             # AFSI -> plot/afsi_result_g2.5.npz（~27 s）
python compare.py                          # 对比 + figures/
```

环境变量：`TFINAL`、`DT`、`NX`、`PRINT_DUMP`、`GRAD_DIV`（默认 2.5）、
`MUSCLE`（默认 1）、`OUTPUT_PATH`。
