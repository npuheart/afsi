# demo_455 — Rubberband with Beams（不变梁橡皮筋，IB2d）

参考算例：`Examples/Rubberband_with_Beams`。**规范版在 MATLAB 套件里**：
`matIB2d/Examples/Example_Standard_Rubberband/Rubberband_with_Beams`
（`input2d` 为 `beams = 1`）。pyIB2d 副本
`pyIB2d/Examples/Rubberband_with_Beams` 的 `input2d` 有一处**标志位笔误**
（见 §2），几何/梁文件两版一致，已拷贝为 `ib2d_input/`（0-based，无需索引换算）。

## 1. 物理设置

249 个标记点组成的**闭合椭圆环**（x 半轴 0.2、y 半轴 0.4，中心 (0.5, 0.5)），
相邻三点 `(i-1, i, i+1 mod 249)` 用 **不变梁**（IB2d "beams"，叉积形式扭转弹簧，
$\kappa = 10^7$）连接：

$$
S = (x_r-x_q)(y_q-y_p) - (y_r-y_q)(x_q-x_p),\qquad K = \kappa\,(S - C)
$$

力按 IB2d `give_Me_Beam_Lagrangian_Force_Densities` 的符号分配到 p/q/r 三个节点
（`main.py:beam_forces`，与 `main.py` 里逐根循环的校验差 $1.5\times10^{-16}$）。

`.beam` 文件第 5 列是生成时刻的参考叉积 $C \approx -1.28518\times10^{-6}$
（249 行几乎逐位相同，相对散度 $3\times10^{-9}$）。对半径为 $R$、$N$ 等分的圆，
$S = -R^2\sin^2(2\pi/N)\cdot$(缓变因子)；反解 $C$ 得
$R = 0.282886 \approx \sqrt{0.2\times0.4} = 0.282843$，即**静止态是等面积圆**。
所以椭圆环被释放后，由扭转弹簧驱动在"椭圆 ↔ 圆"之间摆动（这是 demo_444
弹簧环的同族算例，但驱动量是弯曲刚度而非线弹簧）。

流体/耦合与其它 ib2d demo 完全一致：周期 Q2/Q1 Taylor–Hood、
Peskin (2002) 两阶段格式（`PeskinRK2Solver`）、4 点 Peskin 核、
$\mathrm{ds} = \min(L_x/2N_x, L_y/2N_y)$、`IBMesh(order=1)`
（网格顶点 = IB2d 笛卡尔网格）；结构力在**半步位置** $X^{n+1/2}$ 上用 numpy
按 IB2d 原式求值（零刚度的 FE 链只承载标记点自由度）。

参数（`ib2d_input/input2d`）：$\rho = 1$、$\mu = 0.0025$、$64\times64$、
$\mathrm{dt} = 10^{-2}$、$T_{\rm final} = 25$（2500 步）。

## 2. ⚠️ pyIB2d 副本的 `input2d` 标志位笔误（本次移植核查发现）

pyIB2d 副本写的是

```
beams = 0               % 0 (for no) or 1 (for yes)
nonInv_beams = 1        % 0 (for no) or 1 (for yes) % non-Invariant Beams
```

而 pyIB2d 解析器（`IBM_Blackbox/please_Initialize_Simulation.py`）认的键名是
`beams` / `nonInvariant_beams`（逐键匹配，**匹配不到就取默认 0**）。
`nonInv_beams` 不在键表里，于是：

* **用 pyIB2d 原样跑这个算例，结构力全部为 0**（`beams = 0`、
  `nonInvariant_beams = 0`）：`IBM_Driver.main` 打印的 `FIBER MODEL INCLUDES:`
  是空列表，环根本不受力，算的是一潭静止流体；
* 目录里也只有 `rubberband.beam`（5 列：p1,p2,p3,$\kappa$,C），
  没有 `.nonInv_beam`（6 列：再加参考二阶差分），从文件格式看就是**不变梁**。

复核证据（在 pyIB2d 副本目录里直接调用它自己的解析器）：

```
$ python -c "import sys; sys.path.insert(0,'../../IBM_Blackbox'); \
             import please_Initialize_Simulation as P; \
             fp,gp,tp,lsp,op,lnp = P.please_Initialize_Simulation(); \
             print('beams =',lsp[4],' nonInvariant_beams =',lsp[6])"
beams = 0.0  nonInvariant_beams = 0.0
```

因此本 demo 按 **不变梁**移植，`ib2d_input/input2d` 只改了这一段（改为
`beams = 1`、`nonInvariant_beams = 0`，并在文件里加注释说明）。
`run_reference.py` 与其它 demo 一样把 lag 开关**按位置**传给
`IBM_Driver.main`（与 MATLAB 版 `main2d.m` 一致），故参考结果不受该笔误影响。

## 3. 结果

### 3.1 AFSI vs pyIB2d（64×64，dt=1e-2，T=25）

![shapes](figures/compare_shapes.png)

![history](figures/compare_history.png)

| $t$ | dXmax | dXmax$/h$ | 面积 AFSI | 面积 IB2d | 半轴 AFSI $b_1,b_2$ | 半轴 IB2d $b_1,b_2$ |
|---:|---:|---:|---:|---:|---|---|
| 2.0 | 1.43e-2 | 0.92 | 0.25059 | 0.25095 | 0.3050, 0.2472 | 0.3036, 0.2403 |
| 6.0 | 1.70e-2 | 1.09 | 0.25292 | 0.25194 | 0.3048, 0.2387 | 0.2954, 0.2374 |
| 12.0 | 1.42e-2 | 0.91 | 0.25709 | 0.25334 | 0.3044, 0.2427 | 0.3086, 0.2392 |
| 18.0 | 2.42e-2 | 1.55 | 0.26047 | 0.25422 | 0.3050, 0.2456 | 0.3152, 0.2433 |
| 24.8 | 3.18e-2 | **2.03** | 0.26352 | 0.25501 | 0.3057, 0.2482 | 0.3212, 0.2473 |

（$h = 1/64 = 0.015625$；参考每 `print_dump = 20` 步输出一次，最后一次是
$t = 24.8$，故对照表也止于该时刻；`compare_table.csv` 为完整逐步表。）

全程标记点最大偏差 **均值 1.24 h / 峰值 2.04 h**，t ≤ 4 内均值 0.83 h。
环的"塌—张"相位与短半轴 $b_2$ 基本重合；偏差主要来自长半轴 $b_1$：
**IB2d 参考的长轴在 t ≳ 6 后持续缓慢膨胀（0.295 → 0.321），而 AFSI 稳定在
0.305 附近**，面积因此出现系统性差异（IB2d +1.5% vs AFSI +4.9%）。
即：不能用这个长时算例宣称 AFSI 保体积优于参考——参考本身在 64² 上并不保面积。

### 3.2 这不是时间步误差（AFSI 与参考都已被 dt 收敛）

| 量（$t = 24.8$） | IB2d 64², dt=1e-2 | IB2d 64², dt=5e-3 | AFSI 64², dt=1e-2 | AFSI 64², dt=5e-3 |
|---|---:|---:|---:|---:|
| 面积 | 0.25501 | 0.25494 | 0.26352 | 0.26346 |
| 半轴 $(b_1,b_2)$ | (0.3212, 0.2473) | (0.3213, 0.2469) | (0.3057, 0.2482) | (0.3057, 0.2482) |

两侧把 dt 减半后结果都到小数点后 4 位不变，故偏差来自**流体/耦合离散**
（AFSI 为有限元 Q2/Q1 + Peskin 核，IB2d 为谱方法）而非时间推进。
复现（时间步研究）：

```bash
DT=0.005 OUTPUT_PATH=plot_dt5e3 PRINT_DUMP=40 python main.py      # AFSI dt 减半
python run_reference.py --dt 0.005 --print-dump 40 --out ref_dt5e3.npz   # 参考 dt 减半
```

顺带记录一条参考侧的经验：**这个算例没法用加密参考来仲裁**——在 128² 上，
pyIB2d 参考的环会持续外扩并在 t ≈ 8.8 s 冲出周期盒后 `NaN`
（x 范围 0.26 → 1.00），且 dt = 1e-2 与 dt = 5e-3 在发散前的轨迹逐帧一致
（如 t = 8.6 s 时差 < 1e-3），说明这不是时间步失稳而是参考自身的空间失稳。
所以本文只与随算例发布的 64²/dt=1e-2 参考对照（它已被 dt 减半验证为时间收敛）。

### 3.3 grad-div γ 的取舍

| 方案 | dXmax$/h$ 均值 | dXmax$/h$ 峰值 | 面积（$t=24.8$） | 面积相对漂移 |
|---|---:|---:|---:|---:|
| γ = 0（默认） | **1.24** | **2.04** | 0.26352 | +4.9% |
| γ = 2.5 | 1.32 | 2.34 | 0.25143 | **+0.05%** |

两种度量给出相反排序：γ = 2.5 把环的面积几乎锁死（漂移 0.05%，远小于参考的
1.5%），代价是长轴被压到 0.297（参考 0.321），标记点偏差略大；γ = 0 的标记点
偏差更小，但面积多涨 3.4%。由于本算例的"正解"（参考）自身在 64² 上面积不守恒，
不同 γ 对应的是**不同的收敛极限**，本 demo 因此默认 γ = 0（与参考的标记点轨迹
最接近），并在 §3.3 同时保留 γ = 2.5 的曲线供对照：

```bash
GRAD_DIV=2.5 python main.py     # -> plot/afsi_result_g2.5.npz
GRAD_DIV=2.5 python compare.py ib2d_reference.npz figures_g2p5
```

与 demo_446/449（梁类算例）的结论一致：梁的弯曲刚度已经被 IB 核分辨得比较
充分，grad-div 在这类算例里不是必需项。

## 4. 运行

```bash
conda activate afsi-dolfinx
cd afsic/demo/ib2d/demo_455

# (1) pyIB2d 参考（首次运行会自动 clone third_party/ib2d；本机约 55 s）
python run_reference.py                    # -> ib2d_reference.npz（125 帧）

# (2) AFSI（64×64，2500 步，本机约 37 s）
python main.py                             # -> plot/afsi_result_g0.npz + XDMF
TFINAL=0.5 PRINT_DUMP=10 python main.py    # 冒烟

# (3) 对比与出图
python compare.py                          # -> figures/（形状、历史、涡量）

# 可选：分辨率 / 时间步 / 稳定项覆盖
NX=128 DT=0.005 python main.py             # 加密网格须同时减小 dt（见 §3.2）
GRAD_DIV=2.5 python main.py                # grad-div 对照
```

`main.py` 支持的环境变量：`IB2D_EXAMPLE`、`NX`/`NY`、`DT`、`TFINAL`、
`PRINT_DUMP`、`GRAD_DIV`、`OUTPUT_PATH`。
`run_reference.py` 支持 `--tend/--dt/--print-dump/--src/--out/--ib2d/--run`
（`--src` 可用于换分辨率跑参考）。

## 5. 文件

- `main.py`          AFSI 运行（不变梁力 + 两阶段流固耦合；输出 npz + XDMF）
- `run_reference.py` pyIB2d 参考（按位置传 lag 开关：`beams = 1`）
- `ib2d_io.py`       0-based 输入读取（`input2d`/`.vertex`/`.beam` 等）
- `ib2d_reference.py` VTK → npz 转换（与其它 demo 同布局）
- `compare.py`       对比表 + `figures/`（形状、面积/半轴历史、涡量场）
- `ib2d_input/`      算例输入（`input2d` 已按 §2 修正 `beams` 开关）
