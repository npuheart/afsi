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
* **实现机制**：固体积分点用包围盒树 + `compute_colliding_cells` 定位到背景
  网格，参考坐标做仿射逆映射，背景 P2 基函数值与梯度用 basix 在映射点
  tabulate（物理梯度 `grad_x φ = grad_ξ φ · J⁻¹`）。这正是论文强调的
  “固体函数与背景测试函数做内积”。

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

## 性能

耦合组装（固体积分点定位 + Mfs + f_el/A_uW）已向量化：32×32 下每次
Newton 迭代的弹性组装约 0.1 s（相对逐点循环 ~100× 提速）。单步耗时主要由
monolithic 稀疏 LU 决定（64×64 每次 Newton ~0.3 s）。

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
