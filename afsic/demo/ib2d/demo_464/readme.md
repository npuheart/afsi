# demo_464 — IB2d `Update_Springs_Peristaltic_Pumping`（蠕动泵：行波驱动静长）

IB2d 官方算例（`matIB2d/Examples/Example_HeartTube/Update_Springs_Peristaltic_Pumping`）
在 AFSI 中的复现。与 demo_451（pyIB2d `Tracers_In_Impedance_Pump`）**同一根
HeartTube 几何**（156 点：弹簧 $k=10^7$ + 不变梁 + 4 个角点目标点，$5\times5$ 盒、
$\mu=1$、$64\times64$、dt=1e-3、T=2），但换成**蠕动（peristaltic）行波**驱动。

## 算例内容

`.spring` 文件末尾 76 根"跨径"弹簧（把标记点 $i$ 与 $i+78$ 连起来）的静长由
`update_Springs.m` 按行波改变：

$$\mathrm{RL}_i = \begin{cases}
d - 0.95\sin\!\big(\frac{2\pi}{L}(x_i - x_M - vt)\big), & \text{若正弦}\ge 0\\
d, & \text{否则}
\end{cases}
\qquad d=1,\ L=2,\ f=4\,\mathrm{Hz},\ v=Lf=8,\ t=\mathrm{rem}(t,1/f)$$

其中 $x_i$ 是**当前**构型下该弹簧主端标记点的 $x$，$x_M = \texttt{xLag}(39)$
（该 156 点管的第 39 号点，MATLAB 索引）。与 demo_451 的阻抗泵（固定一组弹簧
同相位收缩）不同，这里的收缩是**空间传播**的，而且是本仓库第一个
`update_Springs` 规律**依赖当前标记点坐标**（"柔性驱动"）的算例。

> 上游细节：$t=0$ 时 $\sin(\pi(x_i-x_M))$ 在整根管上取 $\pm1$，即**一半的泵弹簧
> 一开始就处在最小静长 0.05**（规律的初值不连续）。因此两个求解器在
> $t\lesssim 0.3$ s 都要经历一次强瞬态，逐点比较在该区段意义有限。

## 实现

1. `matIB2d` 的 1-based 文件 → `ib2d_io.load_example(..., index_base=1)`；
   `run_reference.py` 对 `.spring`（2 个编号列）、`.beam`（3 个）、`.target`（1 个）
   分别重编号，并把 `.m` 的 Python 移植 `update_Springs.py` 拷进参考运行目录
   （pyIB2d 从那里 import）。
2. AFSI 侧实现同一条行波规律，**在驱动时序上完全对齐**：驱动函数读到的是
   **全步**标记点位置（与 `IBM_Driver` 中 `update_Springs(dt, ct, xLag, yLag, ...)`
   一致），且在该步力装配之前调用；`.m` 的 Python 移植与 AFSI 侧实现对同一构型
   逐位一致（`0.00e+00`）。
3. 默认 $\gamma = 2.5$（demo_451 对该几何的取值）；$\gamma=0$ 会数值崩溃（NaN）。
4. 输出帧与 pyIB2d 落在**同一步**（帧对齐约定同 demo_459–463）。

## 结果

参考运行 27 s、AFSI 运行 32 s，两边各 201 帧：

| 指标 | 结果 |
|---|---|
| 标记点偏差（全程） | 中位 **5.2e-3 m = 0.067 h**；最大 1.75e-1 m = 2.24 h（在 $t=0.14$，初始强瞬态内） |
| 标记点偏差（$t\ge0.3$ s） | 最大 **2.46e-2 m = 0.32 h**，中位 4.6e-3 m = 0.059 h |
| 管宽（射流特征） | AFSI 0.9345–1.1938、IB2d 0.9593–1.1925；瞬态后逐帧差 ~1e-3–1e-2（如 t=1.121：0.9993/0.9955） |
| 收缩相位 | 管宽极小值 AFSI $t=0.051$ / IB2d $t=0.061$（瞬态内），之后逐周期同步 |
| 速度场 | 末帧 `max\|u\|` AFSI 14.04 / IB2d 12.88，`max\|Δu\|` 2.64 m/s |

结论：**行波（位置相关）静长驱动移植正确**——瞬态之后两侧的管宽轨迹逐帧一致到
$10^{-3}$ 量级、标记点偏差 ≤0.32 h（中位 0.06 h），蠕动收缩相位同步；初始
$t<0.3$ s 的 2.24 h 峰值来自上游规律在 $t=0$ 的不连续初值（一半泵弹簧直接跳到
最小静长），两侧都经历同一瞬态、但瞬时响应速度不同。

## 运行

```bash
conda activate afsi-dolfinx
python run_reference.py          # pyIB2d 参考 -> ib2d_reference.npz（~27 s）
python main.py                   # AFSI        -> plot/afsi_result_g2.5.npz（~32 s）
python compare.py                # 对比 + figures/
```

环境变量：`TFINAL`、`DT`、`NX`、`PRINT_DUMP`、`GRAD_DIV`（默认 2.5）、`OUTPUT_PATH`。
