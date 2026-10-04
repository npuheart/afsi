# demo_456 — Concentration Diffusion（被动标量输运，IB2d）

参考算例：`pyIB2d/Examples/Concentration_Diffusion`（生成脚本 `Diffusion_Chamber.py`）。
输入文件逐字拷贝为 `ib2d_input/`（0-based，无需索引换算）。

## 1. 物理设置

1×1 周期盒，$\rho=1$、$\mu=10$、$64\times64$、$\mathrm{dt}=10^{-4}$、
$T_{\rm final}=0.5$（5000 步）、每 200 步输出一次。两个要素：

* **结构**：496 个标记点组成的方形"腔壁"（边长跨 $[1/64,\,63/64]$，间距
  $\mathrm{ds}=1/128$），每个点都是**目标点**（`chamber.target`，$k=10^7$），
  锚点 = 初始位置；
* **背景浓度**：`chamber.concentration` 第一行是扩散系数 $k=0.025$，其余是
  $64\times64$ 的初始浓度场——中心 $(31..33)^2$ 共 9 个网格点为 $1.0$，其余为 $0$
  （$\int C = 9$）。

**本算例中腔壁恰好被钉在锚点上，因此 IB 力恒等于零、流体速度恒等于零**
（参考侧实测 `max|u| = 0`、标记点位移 0；AFSI 侧腔壁位移 $3\times10^{-16}$）。
即这是 IB2d 的"background flow item"：**纯扩散基准**，结构只是背景。
它的价值在于被动标量场有解析解：二维点源扩散是 Gauss 分布
$C(r,t)=M/(4\pi k t)\exp(-r^2/(4kt))$，各坐标方差满足

$$
\sigma^2(t)=\sigma^2(0)+2kt
$$

（周期盒下要加上像源求和，见 §3.2），可以同时校验 AFSI 与参考的标量求解器。

## 2. AFSI 实现

* 流体/结构管线与其它 ib2d demo 完全一致：周期 Q2/Q1 Taylor–Hood +
  Peskin (2002) 两阶段（`PeskinRK2Solver`）、4 点核、$\mathrm{ds}=\min(L_x/2N_x,L_y/2N_y)$、
  零刚度 FE 链承载标记点自由度、结构力用 IB2d 目标点公式在 $X^{n+1/2}$ 求值；
* **被动标量**：在同一个周期盒上解

  $$\frac{\partial C}{\partial t}+\mathbf u\cdot\nabla C=k\nabla^2 C$$

  用 **Q1 有限元**：因网格为 $(N_x,N_y)$ 个四边形单元，Q1 自由度正好落在 IB2d
  的 $64\times64$ 网格节点上，于是直接复用 `PeskinRK2Solver` 的周期约化
  `periodic_prolongation`（$C_{\rm full}=P\,C_{\rm red}$），与流体用的是**同一套
  周期性处理**。时间推进为显式"集中质量"Galerkin：

  $$
  r = -A\,C - k\,S\,C,\qquad
  C_{\rm red} \mathrel{+}= \mathrm{dt}\,\frac{P^{\mathsf T} r}{m_{\rm red}},\qquad
  C_{\rm full} = P\,C_{\rm red}
  $$

  其中 $A_{ij}=\int(\mathbf u\cdot\nabla\varphi_j)\varphi_i$（对流项，用
  IB 插值到标量节点上的流体速度装配，即标量对流体单向耦合）、
  $S_{ij}=\int\nabla\varphi_j\cdot\nabla\varphi_i$，$m_{\rm red}$ 为集中质量。
  初始条件取 IB2d 网格场的节点值（逐位一致）。
* 标量步长的稳定性：显式 Euler 要求 $\mathrm{dt} < 2/(k\lambda_{\max})$，
  其中 $\lambda_{\max}$ 由 $S_{\rm red}x=\lambda M_{\rm red}x$ 用 `eigsh` 求得；
  本算例 $\mathrm{dt}<4.88\times10^{-3}$，取 $\mathrm{dt}=10^{-4}$ 有很大裕量
  （程序会打印该判据并断言）。
* 每步都重新装配对流矩阵（本算例 $\mathbf u\equiv 0$）。

## 3. 结果（64×64, dt=1e-4, T=0.5, 26 帧）

![history](figures/compare_history.png)

![field](figures/compare_field.png)

### 3.1 与 pyIB2d 参考逐帧比较

| $t$ | 总质量 AFSI / IB2d | $C_{\max}$ AFSI / IB2d | $\sigma^2$ AFSI | $\sigma^2$ IB2d | dCmax | 相对 $L_2$ |
|---:|---|---|---:|---:|---:|---:|
| 0.04 | 9.00000 / 9.00000 | 0.16866 / 0.16510 | 2.16276e-3 | 2.16776e-3 | 3.6e-3 | 1.5e-2 |
| 0.12 | 9.00000 / 9.00000 | 0.05763 / 0.05721 | 6.16276e-3 | 6.16776e-3 | 4.2e-4 | 5.2e-3 |
| 0.24 | 9.00000 / 9.00000 | 0.02898 / 0.02887 | 1.21624e-2 | 1.21674e-2 | 1.1e-4 | 2.7e-3 |
| 0.40 | 9.00000 / 9.00000 | 0.01743 / 0.01739 | 2.01303e-2 | 2.01352e-2 | 3.9e-5 | 1.6e-3 |
| **0.50** | 9.00000 / 9.00000 | 0.01395 / 0.01393 | 2.50179e-2 | 2.50228e-2 | **2.5e-5** | **1.3e-3** |

* 两侧**总质量都严格守恒**（9.000000，逐帧不变）；
* 方差逐帧差 $5\times10^{-6}$（相对 $2\times10^{-4}$），末态场的最大逐点差
  $2.5\times10^{-5}$（峰值的 0.18%）、相对 $L_2$ 差 $1.3\times10^{-3}$；
* 早期 $(t\lesssim0.1)$ 场差最大（$3.6\times10^{-3}$）——初始 9 点 δ 状团块只有
  3 个节点宽，Q1 Galerkin 与差分的上风格式对它的初始展平不同；随场变光滑
  两者迅速收敛。
* 腔壁位移 $3.1\times10^{-16}$（与参考同样"绝对静止"）。

### 3.2 解析校验：为什么两者都比 $\sigma_0^2+2kt$ 低 0.55%

$\sigma^2(t)=\sigma^2_0+2kt$ 是无界区域的结论。本算例是**周期盒**，正解应为
初始 9 个点源 Gauss 分布**加上所有像源**：

$$
C_{\rm exact}(\mathbf x,t)=\sum_{\rm src}\sum_{n,m\in\mathbb Z}
\exp\!\left(-\frac{|\mathbf x-\mathbf x_{\rm src}-(nL_x,mL_y)|^2}{4kt}\right)
$$

`compare.py` 直接算出该和（$|n|,|m|\le2$）并在同一节点测度下取方差：

| $t$ | $\sigma^2$ 精确（含像源） | $\sigma^2_0+2kt$（无界公式） | AFSI 相对误差 | IB2d 相对误差 |
|---:|---:|---:|---:|---:|
| 0.04 | 2.162760e-3 | 2.162760e-3 | $-7\times10^{-15}$ | +2.3e-3 |
| 0.20 | 1.0162735e-2 | 1.0162760e-2 | $-1.7\times10^{-6}$ | +4.9e-4 |
| 0.40 | 2.0132715e-2 | 2.0162760e-2 | $-1.2\times10^{-4}$ | +1.2e-4 |
| 0.50 | 2.5023858e-2 | 2.5162760e-2 | $\mathbf{-2.4\times10^{-4}}$ | $-4.4\times10^{-5}$ |

（`compare.py` 会自动打印并画出这三条曲线，`compare_table.csv` 有全部 26 帧。）

* **AFSI 相对精确解的最大相对误差 $2.4\times10^{-4}$**（出现在末态），IB2d 为
  $4.3\times10^{-3}$（$t=0.02$）/ $4.4\times10^{-5}$（末态）；
* 早期（$t\lesssim 0.1$）AFSI 的方差与无界公式 $\sigma_0^2+2kt$ **逐位吻合**
  （相对误差 $10^{-14}$ 量级）：Q1 集中质量 Galerkin 在均匀周期网格上精确
  保持二阶矩的增长率；随 $t$ 增大方差略低于 $2kt$，且降幅与含像源精确解一致
  ——那是周期盒的真实物理（像源回流），不是离散误差；
* 无界公式在 $t=0.5$ 偏 **+0.55%**，两套离散都跟着含像源精确解走。因此这个小
  算例不能用来宣称谁的扩散算子更准（AFSI 早期更准、IB2d 末态更准，量级都远
  小于像源效应），但足以确认两套离散都在 $10^{-4}$ 量级复现解析解。

### 3.3 与 demo_453/454 的对照

453/454 里 grad-div 对"收缩环"是必需的（伪散度导致漏流/崩溃）；本算例流体
严格静止（$\mathbf u\equiv0$，IB 力为 0），grad-div 无对象，故
`PeskinRK2Solver` 用默认 $\gamma=0$。

## 4. 运行

```bash
conda activate afsi-dolfinx
cd afsic/demo/ib2d/demo_456

# (1) pyIB2d 参考（首次运行会自动 clone third_party/ib2d；本机约 28 s）
python run_reference.py                    # -> ib2d_reference.npz（26 帧，含 C）

# (2) AFSI（64×64，5000 步，本机约 76 s）
python main.py                             # -> plot/afsi_result.npz + XDMF
TFINAL=0.002 PRINT_DUMP=10 python main.py  # 冒烟

# (3) 对比、解析校验与出图（含精确像源解）
python compare.py                          # -> figures/ + compare_table.csv
```

`main.py` 支持的环境变量：`IB2D_EXAMPLE`、`NX`/`NY`、`DT`、`TFINAL`、
`PRINT_DUMP`、`OUTPUT_PATH`（注意加密网格要同时减小 $\mathrm{dt}$ 以满足标量
显式稳定性判据，程序会断言）。
`run_reference.py` 支持 `--tend/--dt/--print-dump/--src/--out/--ib2d/--run`。

## 5. 文件

- `main.py`          AFSI 运行（IB 流体 + 腔壁目标点 + Q1 被动标量输运）
- `run_reference.py` pyIB2d 参考（lag 开关按位置传：`targets = 1`、`concentration = 1`）
- `ib2d_io.py`       0-based 输入读取（本 demo 新增 `read_concentration`）
- `ib2d_reference.py` VTK → npz 转换（本 demo 新增 `concentration.*.vtk` → `C`）
- `compare.py`       逐帧对比 + 精确像源解校验 + `figures/`
- `ib2d_input/`      算例输入（逐字拷贝；其中 `input2d` 的 `nonInv_beams = 0`
  是 pyIB2d 的未知键名，取值 0 且本结构只有目标点，不影响结果）
