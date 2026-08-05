# demo_422 — 单块（monolithic）浸没边界有限元 IBFE：方腔顶盖驱动 + 浸没弹性圆盘

FEniCSx（dolfinx 0.10.0）对 `afsi/a.cpp`（deal.II 参考实现）的复刻。

忠实复现 Boffi–Gastaldi–Heltai (2007)、Heltai–Costanzo (2012)、
Roy–Heltai–Costanzo (2015) 的浸没边界有限元方法（IBFE），基准算例
`LDCFlow_Ball_DGP_INH1`：单位方腔 [0,1]²、顶盖以 U=(1,0) 驱动、
弹性圆盘初始位于 (0.6,0.5)、半径 R=0.2，不可压 neo-Hookean 材料。

与算子分裂的浸没方法不同，这里**流体速度 u、压力 p 与固体位移 W 在同一个
3×3 块系统中联合求解**（单块 / monolithic）：每个时间步是后向欧拉，
用 Newton 迭代求解

$$
\begin{bmatrix}
K & B^T & -A_{uW} \\
B & 0 & 0 \\
-M_{fs}^T & 0 & \tfrac{1}{\Delta t}M_s
\end{bmatrix}
\begin{bmatrix} du \\ dp \\ dW \end{bmatrix}
=
\begin{bmatrix} -R_u \\ -R_p \\ -R_W \end{bmatrix}
$$

其中残差
$$R_u = K u + B^T p - f_{el}(W),\quad R_p = B u,\quad R_W = \tfrac{1}{\Delta t}M_s(W-W^n) - M_{fs}^T u,$$
$f_{el}$ 是扩散到背景网格的弹性力，$A_{uW}=df_{el}/dW$ 是其切向（混合刚度）。
相互作用/混合质量 $M_{fs}$ 每个时间步重建一次（几何冻结在 $W^n$，半隐式），
$f_{el}$ 与 $A_{uW}$ 在 Newton 循环内随当前 $W$ 重新组装。压力是延伸到固体
内部的流体变量，在整个域（含圆盘）上强制不可压，与论文一致。

## 方法要点（两套非匹配网格）

* **背景（Euler 网格）**：方腔三角形网格，速度 P2（向量）+ 压力 P1（标量，
  Taylor–Hood）。论文用 Q2/FE_DGP(1)，P2/P1 是标准稳定的 FEniCSx 对应，物理一致。
* **固体（Lagrange 网格）**：gmsh 生成的圆盘，位移 P2（向量）。
* **两种耦合算子**（IBFE 的核心，只在两网格之间传递）：
  1. **扩散 Jᵀ（固体→背景）**：弹性体力按
     $f_{el,i} = -\int_{\Omega_s}(P F^T):\nabla_x\phi_i^{bg}\,dX$ 扩散到背景，
     其中 $P=\mu(F-F^{-T})$、$F=I+\nabla_X W$，切向
     $A_{uW,ij}=df_{el,i}/dW_j$ 由背景测试函数在映射到背景网格的固体积分点
     处求值得到；
  2. **插值 J（背景→固体）**：流体速度用混合质量投影到固体
     $M_s u_s = M_{fs}^T u$，圆盘随流体运动 $dW/dt = u_s$。
* **实现机制**：固体积分点定位到背景网格。因为背景是**规则矩形三角剖分**
  （每方格按固定对角线一分为二），在 `__init__` 一次性建立
  `square_to_cell[i,j,t]` 查找表，运行时用 O(1) 解析公式
  `(⌊x/dx⌋, ⌊y/dy⌋)` + 对角线判定定位，**完全跳过包围盒树碰撞检测**
  （interaction 阶段 ~3× 提速，64×64 每步省 ~30ms）。实测 11526 个固体积分
  点中 99.99% 与 `compute_colliding_cells` 逐位一致；仅当积分点恰好落在背景
  cell 边界/对角线上（本算例 1 个点）归属相邻 cell——这是固有歧义（bb_tree
  对边界点也只返回“第一个”碰撞 cell），因 P2 基函数梯度跨 cell 不连续，
  只影响切线 $A_{uW}$（解差 ~1e-5，$f_{el}$ 逐位一致）。非规则网格或 MPI
  下自动回退包围盒树。参考坐标做仿射逆映射，背景 P2 基函数值与梯度用
  basix 在映射点 tabulate（物理梯度 `grad_x φ = grad_ξ φ · J⁻¹`）。这正是
  论文强调的“固体函数与背景测试函数做内积”。

## 代码结构（已模块化）

```
demo_422/
  config.py     # MPI 助手 + make_config（环境变量/CLI 配置）
  meshgen.py    # 背景方腔网格 + gmsh 圆盘网格
  linops.py     # PETSc<->scipy 转换、MumpsFactor、仿射映射助手
  immersed.py   # ImmersedFEM 类（组装/耦合/Newton/求解）
  main.py       # 命令行入口（薄 CLI）
  validate.py   # 验证脚本
  bench_vs_336.py, bench_422_breakdown.py   # 效率基准
```

## 运行

环境：`conda activate afsi-dolfinx`（dolfinx 0.10.0）。

```bash
cd afsic/demo/demo_422

python main.py                      # 完整基准（64x64，810 步，T=8.1s）
python main.py --steps 10 --nx 16   # 快速冒烟测试
```

环境变量覆盖：

| 变量 | 默认 | 说明 |
|---|---|---|
| `NX` / `NY` | 64 | 背景网格分辨率（方腔 [0,1]²，三角形） |
| `SOLID_H` | 0.0125 | 圆盘网格特征尺寸 |
| `STEPS` | 810 | 时间步数 |
| `DT` | 0.01 | 时间步长 |
| `T` | 8.1 | 总时间（与 STEPS 取其一） |
| `MU_S` | 0.1 | 固体剪切模量 μᵉ（不可压 neo-Hookean P=μ(F−F⁻ᵀ)） |
| `RHO_S` | 1.0 | 固体密度 |
| `PIN` | 0 | 1=钉住圆盘中心（准静态演示） |
| `SCHEME` | 0 | 0=单块 3×3；3=约化 2×2（精确 Schur 消去 W） |
| `FROZEN` | 2 | 冻结 Jacobian 准 Newton：0=完全 Newton（每迭代分解）；1=每步分解一次；2=跨步复用（默认，停滞自适应重分解+失败回退） |
| `LINEAR_SOLVER` | direct | monolithic 线性求解器：`direct`=MUMPS 直接分解（默认）；`gmres`=FGMRES(50) + 块 LDU 预条件（见下） |
| `FLUID_SOLVER` | mumps | gmres 模式下流体鞍点求解器：`mumps`=直接分解（2D 快，默认）；`amg`=GAMG（K 与压力 Schur 补 S_p 都用 AMG，**3D 可扩展路径**，无直接分解） |
| `OUT` | 10 | 输出间隔（步） |
| `OUTPUT` | output | 输出目录 |

## 方案（`SCHEME`）

* **0（默认）单块 3×3**：完整 monolithic 系统，Newton + 稀疏直接 LU
  （**MUMPS**，经 PETSc KSP preonly+LU 调用，不可用时回退 SuperLU_DIST）。
  鞍点用 $10^{-8}M_p$ 正则化 (1,1) 块并钉一个压力自由度——只影响中间线性
  求解，Newton 收敛解用的是精确残差，不受影响。牛顿二次收敛
  （`|dX|: 1e-1 → 1e-5 → 1e-14`）。
  64×64 下每次 Newton 约 0.3 s，完整 810 步基准约十几分钟。
* **3 约化 2×2（精确 Schur 消去）**：把 W 用精确 $M_s^{-1}$（稀疏因子，
  $M_s$ 时不变）消去：
  $$K_{\sim}=K-\Delta t\, A_{uW}M_s^{-1}M_{fs}^T,\qquad
  \begin{bmatrix}K_{\sim}&B^T\\B&0\end{bmatrix}\begin{bmatrix}du\\dp\end{bmatrix}
  =\begin{bmatrix}-R_u-\Delta t A_{uW}M_s^{-1}R_W\\-R_p\end{bmatrix},$$
  再回代 $dW=\Delta t\,M_s^{-1}(M_{fs}^T du-R_W)$。消去精确，Newton 仍二次
  收敛，收敛解与 3×3 完全相同（已在 16×16 上验证两方案轨迹逐位一致）。
  **注意**：精确消去使 $K_\sim$ 变为稠密的 $n_u\times n_u$ 块，仅适合粗网格
  （大网格请用方案 0）。a.cpp 的方案 3/4 用 $M_s$ 的对角近似代替 $M_s^{-1}$，
  在软盘基准上 Newton 发散（对角与一致质量逆差 O(1)），故此处一律用精确消去。

## 验证（已通过）

1. **耦合自检**（对应 a.cpp `verify_coupling`）：把线性背景场 g=(1,0) 与
   (x,y) 投影到固体，P2 精确复现线性场 → 误差 1e-15（机器精度）。
2. **弹性力与切向**：`f_el(W=0)=0`；切线有限差分一致性相对误差 7.7e-9；
   刚体平移（无应变）`f_el≈0`（5.9e-20）；小转动 `f_el~O(θ²)`。
3. **Newton 二次收敛**（切向正确）。
4. **流体求解器**：同网格 P2/P1 稳态 Stokes 方腔流场与顺时针主涡一致
   （探针：(0.5,0.1) u_x=-0.056、(0.5,0.5) u_x=-0.19、(0.5,0.9) u_x=+0.48，
   右壁 u_y<0、左壁 u_y>0，max|u|=1）。
5. **轨迹**：圆盘被顺时针主涡夹带运动（初始位于 (0.6,0.5)，主涡中心约
   (0.53,0.75)，该处流场向左略向下），环绕运动而非径直撞壁，与论文基准
   一致；变形尺度符合物理（μ_s=0.1 相对流体应力尺度 0.01 → 应变 ~10%）。
6. **方案一致性**：SCHEME=0 与 SCHEME=3 轨迹逐位一致（~5e-15）。
7. **触壁鲁棒性**：软盘（μ_s=0.1）在粗网格下长时间运行可能被夹带贴壁
   （物理合理）；出界固体积分点跳过+警告，与 a.cpp 行为一致，不崩溃。
8. **边界条件（blocked↔expanded 自由度映射）**：P2 向量空间
   `locate_dofs_topological` 返回 **blocked 索引**（0..n_u/2-1），而 monolithic
   矩阵用 **expanded 全局自由度**（dof = block*2 + comp）。速度 BC 行/列清零、
   lid 初值/提升都必须用 expanded 索引（block→2b、2b+1）。修正后腔体涡流与
   dolfinx 原生块组装及 legacy FEniCS 完全一致。

## 性能（64×64，优化后）

耦合组装已全面向量化（interaction 1.5s→0.06s、Mfs 0.6s→0.06s、弹性力 0.03s→
0.001s/迭代、切向 A_uW 0.59s→0.11s），`apply_bc` 去掉 LIL 往返（1.7s→0.01s）。
配合 **跨步冻结 Jacobian 准 Newton（FROZEN=2，默认）**：monolithic 矩阵只分解
一次并在后续若干时间步复用（仅弹性力每迭代重算；停滞时自适应重分解，失败则
回退完全 Newton 保证鲁棒性），MUMPS 分解成本被摊销。

| 配置 | 每步 | 每模拟秒 |
|---|---|---|
| 原始版（每 Newton 分解，64×64） | 10.0 s | — |
| apply_bc 快速 CSR 版 | 4.73 s | — |
| 向量化耦合 + 每步冻结（FROZEN=1） | 1.15 s | — |
| **跨步冻结（FROZEN=2，dt=0.01）** | **0.48 s** | 48.2 s |
| FROZEN=2，dt=0.05 | 0.58 s | 11.5 s |
| FROZEN=2，dt=0.1 | 0.75 s | **7.5 s** |

dt=0.01 下约 50 步才需一次分解。单步较原始版提速 **~21×**。

## 与 demo_336（显式 Chorin 算子分裂）效率对比

同一物理（64×64、μ=0.01、μ_s=0.1、圆盘 R=0.2）：

* demo_336（显式，dt=0.005，受 CFL 限制）：0.056 s/步，~11.3 s/模拟秒；
* demo_422（隐式 monolithic，FROZEN=2）：
  - dt=0.01（与 336 等时间精度）：0.48 s/步，~48 s/模拟秒（等精度下较慢，
    因为隐式每步做全耦合 Newton）；
  - **dt=0.1（隐式无条件稳定，可跑 20× 显式 CFL 的 dt）**：~7.5 s/模拟秒，
    **比 demo_336 快 ~1.5×**，且圆盘轨迹与 dt=0.01 参考误差 <1.5%。

**结论**：隐式方案的优势不在等精度小步长，而在**无条件稳定**——它能用
显式无法达到的大时间步（dt=0.05~0.1），从而在每模拟秒墙钟耗时上反超
算子分裂显式方案，同时没有分裂误差、完全隐式。基准脚本 `bench_vs_336.py`
在 64×64 下同进程跑两算例核心循环并给出 dt 缩放对比。

## 大弹性系数（mu_s）下的稳定性优势（实测）

固体变硬时，隐式方案的优势被放大——显式方案受 CFL/刚性问题限制，隐式
无条件稳定。实测（32×32，20 步）：

| 方案 | μ_s=0.1 | μ_s=10 |
|---|---|---|
| 336（显式）dt=0.005 | 稳定 | 稳定 |
| 336（显式）dt=0.02 | 稳定 | **爆炸（max\|coords\|≈5.8e8）** |
| 336（显式）dt=0.05 | 稳定 | **爆炸（≈4.1e5）** |
| **422（隐式）dt=0.05** | 稳定 | **稳定（max\|W\|=0.2）** |

即：μ_s=10 时显式 336 的稳定性上限收紧到 dt≤0.01，而隐式 422 在 dt=0.05
依然稳定。**固体越硬，显式 CFL 限制越紧，隐式大时间步稳定性优势越被放大**
（经典"隐式方法赢在刚性问题"）。

### 墙钟时间转折点（诚实结论）

按"隐式超过显式耗时的区间不再计算"的原则，实测（32×32 与 64×64）：

| μ_s | 336 最大稳定 dt | 336 每模拟秒 | 422 (dt=0.1) 每模拟秒 | 谁快 |
|---|---|---|---|---|
| 0.1 | 0.1 | ~11 s (64×64) | **7.5 s** | **422** |
| 1 | 0.05 | 更小 | 更大 | 336 |
| 10 | 0.01 | 更小 | 更大（每步 3.95s@64×64） | 336 |
| 100 | 0.002 | 更小 | **NaN/圆盘出域（不计算）** | 336 |

**隐式也适当减小 dt 的结果**（32×32，两方案各自最优稳定 dt）：减小 dt
确实让隐式在 μ_s=100 不再发散（dt=0.005 可跑，"不爆炸"目标达成），但其
每模拟秒墙钟仍全面落后（μ_s=10 时隐式 107s vs 显式 1.0s；μ_s=100 时隐式
521s vs 显式 3.0s）——32×32 显式可到 dt=0.1 极便宜，而 monolithic 每步成本
高一个量级，dt 调优无法弥补。

**结论**：隐式的墙钟优势只出现在**软盘 + 大时间步 + 较细网格**（64×64、
dt=0.1 时快 ~1.5×）。随 μ_s 增大，**当前直接法的隐式每步成本反而上升**
（刚性耦合使冻结 Jacobian 更快失效→每步重分解/重启），即使把隐式 dt 也
相应减小，本算例中"mu_s 越大隐式墙钟越占优"仍不成立——这正是块预条件
迭代求解器（常数预条件、每步成本与耦合强度近似无关）要解决的：让隐式每步
成本不再随 μ_s 上升，刚性极限下的墙钟优势才可能出现。

## 大网格扩展性（422，隐式，dt=0.01）

| 网格 | 流体 dofs (u+p) | μ_s=0.1 每步 | μ_s=10 每步 |
|---|---|---|---|
| 64×64 | 37507 | 0.59 s | 3.95 s |
| 96×96 | 83907 | 0.84 s | 6.24 s |
| 128×128 | 148739 | 1.21 s | 10.3 s |

全部稳定（隐式）；跨步冻结使 MUMPS 分解摊销，单步成本近似随网格线性增长。

## 块预条件迭代求解器（LINEAR_SOLVER=gmres，FGMRES(50) + 块 LDU）

monolithic Jacobian

$$A=\begin{bmatrix}K & B^T & -A_{uW}\\ B & s_{11} & 0\\ -M_{fs}^T & 0 & \tfrac{1}{\Delta t}M_s\end{bmatrix}$$

用**块 LDU 预条件**（借鉴大规模 FSI 分析的 block-LDU 做法）：耦合块保留在
$L$、$U$ 中，只丢二阶修正 $A_{uW}\,\tfrac{\Delta t}{M_s}\,M_{fs}^T$：

$$P=L\,\tilde D\,U,\quad
L=\begin{bmatrix}I&0&-A_{uW}\,(\tfrac{\Delta t}{M_s})\\ 0&I&0\\0&0&I\end{bmatrix},\quad
\tilde D=\mathrm{diag}(\begin{bmatrix}K&B^T\\B&s_{11}\end{bmatrix},\ \tfrac{1}{\Delta t}M_s),\quad
U=\begin{bmatrix}I&0&0\\0&I&0\\ -(\tfrac{\Delta t}{M_s})M_{fs}^T&0&I\end{bmatrix}$$

外层用 **FGMRES(50)（右预条件、柔性）** 而非左 GMRES（scipy 无 FGMRES，本实现
自写：存预条件向量 $z_j=Mv_j$、Givens 旋转增量跟踪残差、可提前终止）。柔性
预条件的价值：流体鞍点内部的迭代/AMG 求解每次应用略不相同，FGMRES 保持稳定。
流体 Stokes 块与固体质量块都是常数；每次 Krylov 迭代 = 1 次流体鞍点求解 +
2 次 $M_s$ 回代 + 2 个耦合 matvec，**不做任何 monolithic 分解**。

**流体鞍点求解器可插拔**（`FLUID_SOLVER`）：

* `mumps`（默认，2D 快）：流体鞍点 $[K\ B^T;\ B\ s_{11}]$ 一次 MUMPS 分解。
* `amg`（**3D 可扩展路径，无任何直接分解**）：流体鞍点用内层 FGMRES + 块对角
  预条件，**速度块 $K$ 与压力 Schur 补 $S_p = B\,\mathrm{diag}(K)^{-1}B^T$ 各
  用 PETSc GAMG 求解**（Elman/Silvester/Wathen 风格）。$K$、$S_p$ 只解到松
  容差（预条件的一部分），外层 FGMRES 吸收误差。
* **固体块 $M_{ww}=(1/dt)M_s$ 始终精确处理**：小且常数，MUMPS 分解一次。

32×32 实测（块对角 vs 块 LDU，收敛到 rtol=1e-8 的迭代数）：

| μ_s | 块对角 | 块 LDU (FGMRES) |
|---|---|---|
| 0.1 | 6 | **4** |
| 10 | 12 | **~4** |

解与 MUMPS 直接法一致（≤1e-9）。`FLUID_SOLVER=amg` 端到端（16×16, μ_s=10）：
Newton 3 次收敛，**解与 mumps 版一致到 6.4e-12**（机器精度）。2D 下 AMG 路径
因多层迭代 + Python 开销比 MUMPS 慢（预期，仅验证 3D 正确性）；3D 下 MUMPS
不可行而 AMG+Krylov 可行。

**诚实结论**：LDU 在中低 μ_s 稳定减少迭代，但在 μ_s=100 下预条件残差**停滞在
~3e-3 相对水平**（无法到 1e-8）。原因：本算例的 (1,3) 耦合块是**随 μ_s 缩放的
弹性刚度** $-A_{uW}$，被丢弃的 $A_{uW}\frac{\Delta t}{M_s}M_{fs}^T\sim
\mu_s\frac{\Delta t}{\rho_s}$，在大 μ_s 下相对流体算子 $K$ 不再是小量——而大
规模 FSI 文档里被丢的 $A_s$ 是纯几何插值（不随刚度缩放），所以文档的"丢二阶
修正"只在弱耦合成立。要让隐式每步成本真正与 μ_s 无关，需要在预条件中显式
处理弹性刚度（如把 $A_{uW}$ 并入 Schur 块的块三角预条件），这超出了本 demo
的 LDU 范围。


**从大规模分析中真正可借鉴、可迁移到 3D 的**：
1. **FGMRES(50)+块 LDU**：预条件随 Jacobian 变化用 FGMRES（右预条件）比
   普通 GMRES 更稳；块结构（1 流体鞍点 + 2×$M_{ww}^{-1}$）正是大问题的形态。
2. **流体块用 AMG + 压力 Schur 补 $S_p$ 预条件**：文档对 $A$ 与 $S_p$ 都
   用 AMG 代替直接分解——这是 3D 下唯一可行的流体预条件（MUMPS 在 3D 不可行）。
3. **固体块 $M_{ww}$ 精确处理**：$M_s$ 小且常数，直接分解一次即可；文档也是
   这么做的（相对流体，固体块极小）。
4. **结构拓扑差异**：文档的 monolithic 把固体弹性放进 (3,3) 块，而本实现
   （deal.II a.cpp 的 IBFE 形态）把弹性力放进流体动量方程，导致 (1,3) 耦合
   随 μ_s 缩放——这是"隐式每步随 μ_s 变慢"的根源，LDU 只缓解不根治。

当前为 scipy 实现（有 Python 开销，2D 小规模比摊销后的直接法慢）；生产版应
为 PETSc KSP FGMRES + Python PC（把每迭代的 1 次流体回代+2 次 $M_s$ 回代降到
~ms 级），流体块换 AMG，这是通往 3D/大规模可扩展求解器的路径。

## 输出

* `output/velocity.xdmf`、`output/pressure.xdmf`：背景流场（P1 插值输出，
  XDMF 要求函数阶数=网格阶数）；
* `output/solid.xdmf`：圆盘位移场 W（参考网格，ParaView 用 *Warp by vector*
  查看变形/运动）；
* `output/solid_deformation.txt`：`t cx cy |disp| max|W| A_ref A_def max(J-1)
  max(1-J)` 逐时间步报告（A 守恒与 J≈1 可检验不可压）。

## 与 a.cpp（deal.II）的差异

* 背景单元 P2/P1 三角（a.cpp 为 Q2/FE_DGP(1) 四边形）：标准 FEniCSx 稳定对，
  物理一致。
* 固体网格用 gmsh 生成（a.cpp 用 `hyper_ball` + 全局加密）。
* 线性求解用 PETSc **MUMPS**（`MumpsFactor`，回退 SuperLU_DIST；a.cpp 用
  GMRES + 多重网格块预条件）。本实现关注方法的正确复刻；求解器可替换。
* a.cpp 的 scheme 1/2/4（预条件/外推实验）未单独复刻；scheme 0 与 scheme 3
  （精确消去）已实现并验证。
