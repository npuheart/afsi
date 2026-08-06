# demo_422 线性求解与预处理器

monolithic 3×3 块 Jacobian（后向欧拉 + Newton）：

$$
A=\begin{bmatrix}
K & B^T & -A_{uW}\\
B & s_{11} & 0\\
-M_{fs}^T & 0 & \tfrac{1}{\Delta t}M_s
\end{bmatrix},\qquad
A\begin{bmatrix}du\\dp\\dW\end{bmatrix}=\begin{bmatrix}-R_u\\-R_p\\-R_W\end{bmatrix}
$$

其中 $s_{11}=10^{-8}M_p$ 为鞍点正则化，(1,3) 耦合块 $-A_{uW}$ 是弹性刚度切向（随 $\mu_s$ 缩放），
(3,1) 块 $-M_{fs}^T$ 是纯几何插值。两种线性求解模式由 `LINEAR_SOLVER` 选择。

## 1. `LINEAR_SOLVER=direct`（默认）——MUMPS 直接分解 + 冻结因子

不是 Krylov 预条件，而是**稀疏直接 LU**：`linops.MumpsFactor` 把 scipy csr 转成 PETSc AIJ，
用 KSP `preonly`+LU（优先 MUMPS，回退 SuperLU_DIST）做一次分解，接口同 `splu`。

配合跨步冻结准 Newton（`FROZEN=2`，默认），**该 LU 因子跨时间步复用**，作为 Newton 迭代的
"冻结近似逆"：只每迭代重算弹性力，停滞时自适应重分解、失败回退完全 Newton。dt=0.01 下约
50 步才分解一次，64×64 单步 0.48 s（较每 Newton 分解版快 ~21×）。

> 关键点：冻结的是**修改后的 Newton**，收敛解与完全 Newton 相同（~1e-11），不改变隐式解。

## 2. `LINEAR_SOLVER=gmres`——FGMRES(50) + 右块 LDU 预条件（3D 可扩展路径）

外层用 **FGMRES(50)（右预条件、柔性）**，预条件是**常数块算子**：

$$
P = L\,\tilde D\,U,\qquad
\tilde D=\mathrm{diag}\Big(\begin{bmatrix}K&B^T\\B&s_{11}\end{bmatrix},\ \tfrac{1}{\Delta t}M_s\Big),
$$

$$
L=\begin{bmatrix}I&0&-A_{uW}\,\tfrac{\Delta t}{M_s}\\0&I&0\\0&0&I\end{bmatrix},\quad
U=\begin{bmatrix}I&0&0\\0&I&0\\-\tfrac{\Delta t}{M_s}M_{fs}^T&0&I\end{bmatrix}.
$$

耦合块保留在 $L$、$U$ 中，**只丢二阶修正** $A_{uW}\,\tfrac{\Delta t}{M_s}M_{fs}^T$
（文献 monolithic FSI 的常规做法），故 $P^{-1}A$ 靠近单位阵。流体鞍点 $[K\ B^T;\ B\ s_{11}]$
与固体质量 $M_s$ 都是常数，各分解一次；每迭代成本 = **1 次流体鞍点求解 + 2 次 $M_s$ 回代 +
2 个耦合 matvec**，无任何 monolithic 分解。

**流体鞍点求解器可插拔（`FLUID_SOLVER`）**：

| 模式 | 流体鞍点如何处理 | 适用 |
|---|---|---|
| `mumps`（默认） | 一次 MUMPS 分解 | 2D 快 |
| `amg` | 内层 FGMRES + 块对角预条件：速度块 $K$ 与压力 Schur 补 $S_p=B\,\mathrm{diag}(K)^{-1}B^T$ 各用 PETSc **GAMG** 松容差求解 | **3D 路径**，无任何直接分解 |

固体块 $M_{ww}=\tfrac{1}{\Delta t}M_s$ 小且常数，**始终用 MUMPS 精确处理**。
柔性预条件（FGMRES 而非 GMRES）是因为 `amg` 模式下预条件每次应用略不同。

## 诚实局限

块 LDU 在 $\mu_s=100$ 时预条件残差**停滞在 ~3e-3**（无法到 1e-8）。根因：本 IBFE 公式把弹性力
放进流体动量方程，使 (1,3) 耦合块随 $\mu_s$ 缩放，被丢项 $A_{uW}\tfrac{\Delta t}{M_s}M_{fs}^T
\sim \mu_s\tfrac{\Delta t}{\rho_s}$ 相对流体算子 $K$ 不再是小量；而大规模 FSI 文献中 $A_s$ 是
纯几何插值（不随刚度缩放），"丢二阶修正"只在弱耦合成立。要让每步成本真正与 $\mu_s$ 无关，
需把 $A_{uW}$ 并入 Schur 块的块三角预条件（超出本 demo 范围）。

## 代码位置

* `linops.py`：`MumpsFactor`（直接 LU）、`GAMGSolver`（GMRES+GAMG 近似逆）、
  `IterativeFluidSaddle`（内层 FGMRES+块对角，amg 路径）、`fgmres`（自写柔性 GMRES）。
* `immersed.py`：`_build_block_preconditioner`（常数预条件，惰性构建一次）、
  `_block_gmres_solve`（`p_inv` 实现 $P^{-1}$ 的三步回代）、`solve_monolithic_gmres`（冻结 Jacobian 的 gmres 时间步）。
* 谱/迭代研究：`_prec_research.py`（$P^{-1}A$ 谱随 $\mu_s$ 的变化 + FGMRES 残差历史）。

---

# 预处理器研究计划（2026-08-05）

## 0. 先做剖析：0.48 s/步 这个数字不对劲

64×64 MAC 网格下，$u$ 约 $2\times64\times65\approx8.3\times10^3$、$p$ 约 $4.1\times10^3$，加上固体 $W$ 总计 $O(1.5\times10^4)$ 自由度。冻结 LU 之后每步只剩三角回代 + 残差装配。这个规模的 2D 稀疏矩阵，MUMPS 一次回代应在 $1\text{–}5$ ms，一次数值分解也就 $20\text{–}50$ ms。即使每步 5 次 Newton，线性代数总计 $\lesssim 30$ ms，分解摊到每步 $\approx 1$ ms。**剩下 $\approx 0.45$ s 花在别处。**

而且"较每 Newton 分解版快 ~21×"反推出原来 $\approx 10$ s/步——$1.5\times10^4$ 的 2D 问题不该这样。这个数字更像 **scipy CSR → PETSc AIJ 重建 + MUMPS analysis 阶段每次 Newton 都重做**，而不是分解本身贵。

所以第一件事是把单步时间拆成五桶计时：(1) 残差/弹性力装配，(2) $M_{fs}$、$A_{uW}$ 重建，(3) scipy→PETSc 转换 + `KSPSetOperators`，(4) `KSPSolve` 本身，(5) 其余 Python 开销。我赌 (1)+(2)+(3) 占 80% 以上。如果是，下面 §1 能给 $5\text{–}10\times$，比任何预条件改进都大。

## 1. 工程提速（不动算法）

**(a) 固定稀疏模式 + 符号分析只做一次。** 结构每步只移动 $O(\Delta t|u|)$，delta 核支撑固定。按"当前构型 ± 若干格"取稀疏模式的**超集**，一次性预分配 PETSc AIJ，之后每次只 `MatSetValuesCSR` 就地更新数值（多余位置填 0），`KSPSetOperators` 走 `SAME_NONZERO_PATTERN`。这既砍掉 analysis（常占分解 $30\text{–}50\%$），也彻底消掉 scipy→PETSc 的重建。

**(b) spread/interp 矩阵向量化。** 2D 4 点 Peskin 核每个拉格朗日节点 16 个非零。用 numpy 一次性算出全部 $(i,j)$ 索引与权重（张量积 + `ravel_multi_index`）再喂 `coo_matrix`。纯 Python 循环版与向量化版差 $50\text{–}100\times$；若已向量化，再上 numba 只剩 $2\text{–}3\times$。

**(c) MUMPS 参数。** `ICNTL(7)=5`（METIS）或 `ICNTL(28)=2`+`ICNTL(29)=2`（PT-SCOTCH）；3D 路径上 `ICNTL(35)=2` 开 BLR 低秩压缩（`CNTL(7)=1e-8`）常有 $2\text{–}3\times$。2D 收益有限，3D 明显。

## 2. 冻结 Newton → 冻结**预条件**（算法上最划算的一步）

现在 $A_0$ 用作固定点迭代的近似逆：$x_{k+1}=x_k-A_0^{-1}R(x_k)$，线性收敛率 $\rho\approx|I-A_0^{-1}A(x_k)|$。构型漂移或 $\mu_s$ 大时 $\rho\to1$ ⇒ 停滞 ⇒ 触发重分解。

**把同一个 $A_0$ 改当 Krylov 预条件**：对当前 Jacobian 解 $A(x_k)\delta=-R(x_k)$，右预条件 GMRES(20)，$P=A_0$。每次 Krylov 迭代 = 1 次三角回代（和现在固定点一步同价）+ 1 次 matvec（便宜，甚至可 matrix-free：$Av\approx[R(x+\epsilon v)-R(x)]/\epsilon$，连 Jacobian 都不用装配）。收益：

- GMRES 直接消掉最坏的几个特征值，所需迭代通常是固定点的 $1/2\text{–}1/3$；
- 恢复不精确 Newton 的超线性收敛，配 Eisenstat–Walker 强制项 $\eta_k$ 自适应内层容差；
- $\rho\to1$ 也不停滞 ⇒ 重分解间隔从 50 步拉到几百步。

改动量很小——你 `linops.py` 里自写的 `fgmres` 就是现成的，把 `x -= lu.solve(R)` 换成 `x -= fgmres(A_k, R, M=lu.solve, tol=eta_k)`。

**更轻的替代**（<50 行）：在现有固定点迭代上加 Anderson 加速，depth $m=3\text{–}5$，对线性收敛的修改 Newton 通常省一半迭代。

**(d) 时间步外推初值。** $x^{n+1,(0)}=2x^n-x^{n-1}$（或 $W$ 用 $W^n+\Delta t,u^n$），通常直接省一次 Newton，$20\text{–}30\%$。

## 3. 修 $\mu_s$ 停滞：把附加刚度并进 Schur 块

记 $\mathcal F=\begin{bmatrix}K&B^T\\B&s_{11}\end{bmatrix}$，$C=\begin{bmatrix}-A_{uW}\\0\end{bmatrix}$，$E=\begin{bmatrix}-M_{fs}^T&0\end{bmatrix}$，$D=\tfrac{1}{\Delta t}M_s$，则**精确** Schur 补为

$$
\mathcal S=\mathcal F-CD^{-1}E=\begin{bmatrix}K-\Delta t\,A_{uW}M_s^{-1}M_{fs}^{T}&B^{T}\\[2pt]B&s_{11}\end{bmatrix}.
$$

（符号请按你自己的约定核一下——IBFE 里这一项应给出半正定的"附加刚度"贡献。）现在的 $\tilde D$ 取 $\mathcal S\approx\mathcal F$，丢的正是这一项。你笔记里已经把"速度块加修正 $A_{uW}\Delta t M_s^{-1}M_{fs}^T$"列为下一步了，这里的关键是**它可以显式组装**：

- 用**集中质量** $M_s\to\mathrm{diag}(M_s)$，三元积退化成两次稀疏矩阵乘，scipy 一行；
- 稀疏度：$A_{uW}$、$M_{fs}$ 都是 Euler×Lagrange 的 delta-核矩阵，乘出来是 Euler×Euler，带宽 ≈ 核支撑自卷积（4 点核 ⇒ 2D 约 $7\times7$ stencil），且只在结构附近非零；
- 组装频率与冻结因子同频（结构慢），成本摊掉。

**先试便宜版**（10 行，值得先跑）：只取该项的对角或行和集中，$\tilde K=K+\mathrm{diag}(\Delta t A_{uW}M_s^{-1}M_{fs}^T)$，SIMPLE 风格。若 $3\times10^{-3}$ 的平台掉到 $10^{-6}$ 以下，说明主要是对角效应，你以极小代价拿到 $\mu_s$ 无关性；不掉再上完整三元积。

代价是 $\tilde K$ 不再常数，`mumps` 路径要跟着重分解；但 `amg` 路径下 GAMG setup 也能跨步复用，**这条修正在 3D/AMG 路径上反而更便宜**。

**和辅助空间方向的接口**：$\Delta t\,A_{uW}M_s^{-1}M_{fs}^T$ 正是 $\Pi\bar A\Pi^*$ 形式的 spread–刚度–interpolate 三明治。把它并进 $\tilde K$ 之后，$\tilde K$ 的可扩展求解就等同于你辅助空间课题的核心目标——demo_422 可以直接当那条线的 2D 试验台。

## 4. 流体鞍点内层（3D 路径的主要杠杆）

**(a) Schur 近似换 Cahouet–Chabard。** 现在的 $S_p=B\,\mathrm{diag}(K)^{-1}B^T$ 在 $K=\tfrac{\rho}{\Delta t}M+\mu L$ 下只捕捉了 $\tfrac{\Delta t}{\rho}L_p$ 那一半，缺粘性那一半。换成

$$
S_p^{-1}\approx\frac{1}{\mu}M_p^{-1}+\frac{\rho}{\Delta t}L_p^{-1},
$$

对 $\Delta t$ 与 $\mu$ 都鲁棒。$M_p^{-1}$ 用集中质量或 Chebyshev，$L_p$ 常数、GAMG setup 一次复用。这和你 q2q1 项目里列的 next step 是同一件事，在 IBFE 里更急（有效时间/粘性尺度跨度更大）。

**(b) 内层去掉 Krylov。** 外层已是柔性 FGMRES，内层不必"松容差求解"。直接 1–2 次 GAMG V-cycle 作 $K^{-1}$、1 次 Chebyshev-Jacobi 作 $M_p^{-1}$、1 次 V-cycle 作 $L_p^{-1}$，省掉嵌套 Krylov 的正交化开销。外层迭代略涨，单次迭代便宜很多，总时间通常净赢。

**(c) 结构网格上 GMG > GAMG，** 通常 $2\text{–}4\times$（setup 近零、无粗算子构造）。`adaptive-fem-v2` 的 Block-GMG 是现成的，这是 3D 路径最终该落到的形态。

**(d)** $\tfrac{1}{\Delta t}M_s$ 若用集中质量就是对角阵，直接除法，不必走 MUMPS。

## 5. 零碎

- $s_{11}=10^{-8}M_p$ 只是防奇异，别当 Schur 近似用；这么小的正则化也会让 MUMPS pivoting 变累，可试 `ICNTL(24)=1` + 静态 pivoting 阈值。
- 预条件里可用增广拉格朗日：$K\to K+\gamma B^TM_p^{-1}B$、Schur $\to-\gamma^{-1}M_p$，$\gamma\sim O(1)$ 能显著压外层迭代（代价是 $K$ 块需要专门的 AL 平滑子）。
- BE 是一阶，换 BDF2 可在同精度下放大 $\Delta t$。但耦合强度 $\sim\mu_s\Delta t/\rho_s$ 随 $\Delta t$ 增长，所以顺序必须是**先修 §3 再放大 $\Delta t$**。

---

## 优先级

| 措施 | 预期收益 | 改动量 | 备注 |
|---|---|---|---|
| §0 五桶剖析 | 定位用 | 30 min | 先做，可能直接改写结论 |
| §1a 固定模式 + 一次符号分析 | $2\text{–}5\times$ | 半天 | 若 (3) 桶大则最高 |
| §1b spread 装配向量化 | $2\text{–}10\times$ | 半天 | 若 (2) 桶大则最高 |
| §2d 外推初值 | $20\text{–}30\%$ | 10 行 | 无风险 |
| §2 冻结 LU 当预条件 | $1.5\text{–}3\times$ + 鲁棒性 | 1 天 | 复用已有 `fgmres` |
| §3 附加刚度对角版 | 解 $\mu_s$ 停滞 | 10 行 | **先跑这个再决定完整版** |
| §3 完整三元积 | $\mu_s$ 无关 | 2–3 天 | 通往辅助空间课题 |
| §4a Cahouet–Chabard | 3D 内层 $2\text{–}5\times$ | 1 天 | 只在 amg 路径 |
| §4b/c 单 V-cycle + GMG | 3D $2\text{–}4\times$ | 数天 | 3D 路径终态 |

如果只能做一件事：**先剖析**。如果只能做两件：剖析 + §3 的对角版——前者大概率给你最大的墙钟时间收益，后者解掉文档里唯一那个真正的方法论障碍。

---

# 执行记录（2026-08-05）

## §0 五桶剖析 —— 完成（修正文档假设）

64×64, dt=0.01, FROZEN=2, mu_s=0.1, 20 步（2.0 次 Newton/步，2 次分解），每步 279 ms：

| 桶 | ms/步 |
|---|---|
| (4) KSPSolve 回代（2×/步 × 77ms） | 158（负载放大） |
| (2) Mfs + A_uW 重建 | 42.5 |
| (3) scipy→PETSc + KSP setup（2 次摊销） | 28.6 |
| (5) interaction+build+bc+rhs | 28.3 |
| (1) 残差 + 弹性力 | 22.0 |

**修正**：文档赌 (1)+(2)+(3) 占 80%，但**当前负载（~13）下回代被放大到 77ms（占 57%）**。
隔离测量：MUMPS 回代真实 min=6ms/median=27ms；**scipy→PETSc 转换 179ms/次**（隐藏成本，
冻结下摊销小，但 mu_s 大/频繁分解时会暴涨——印证 §1a）。真实负载下 (1)+(2)+(3)≈93ms 才是
大头，其中 (2) Mfs+A_uW 42.5ms 最大（已向量化，剩余 coo→csr 排序）。

## §3 附加刚度并进 Schur —— 对角版验证**无效**（记录机制）

- **加号/减号对角版都无效**（mu_s=100 停滞 7e-3 → 1.4e-2 / 7.8e-2），完整三元积（diag M_s）也差。
- **机制**：$P_{11}=D_{11}+CD^{-1}E$，L·U 展开里的 $CD^{-1}E$ 用**精确** $D^{-1}=(1/\Delta t)M_s^{-1}$。
  若 $D_{11}=K-\mathrm{diag}$ 修正，残留 $A_{uW}(full-diag)\frac{\Delta t}{M_s}M_{fs}^T$ 项（随 μ_s）——
  **集中质量与精确 $M_s^{-1}$ 不一致**。
- **集中质量统一版**（Minv 与 D11 都用 diag）：$P_{11}=K$ 精确（重建差 1.4e-17），但 $P_{33}$ 用
  lumped 逆（~25）≠ $A_{33}=(1/\Delta t)M_s$（~0.04），固体块误差经耦合放大 → 谱 6.6–14.5 更差。
- **根因**：需 $D_{11}$ 修正用精确 $M_s^{-1}$（稠密，64×64 时 $A_{uW}M_s^{-1}$ ≈ 2.1 GB，不可扩展）。
  ⇒ **§3 任何 diag 近似都不能解决 μ_s 停滞**；完整版不可扩展。与 readme"诚实局限"一致：
  IBFE 拓扑的 (1,3) 弹性刚度耦合是根本障碍。

## §2 冻结 LU 当 Krylov 预条件 —— 验证**有效**，已接入为 `FROZEN=3`（最有价值）

把 `dX = lu.solve(rhs)`（固定点）换成 `dX = fgmres(A_k, rhs, M=lu.solve)`（Krylov 预条件），
matvec 用当前 Jacobian $A_k$（每 Newton 迭代装配；文档建议 matrix-free 可省）：

| 场景 (8 步) | 固定点 frozen=2 | Krylov frozen=3 |
|---|---|---|
| dt=0.05, μ_s=1 | 77 解, **18 restart**, 2 分解 | **24 解, 0 restart**, 1 分解 |
| dt=0.1, μ_s=0.1 | 80 解, **21 restart**, 1 分解 | **24 解, 0 restart**, 1 分解 |

解与 frozen=2 一致（≤3.4e-13，机器精度）。30 步 wall-time：frozen=2 59.3s vs frozen=3
**20.1s（2.9×）**。**关键**：GMRES matvec 用当前 $A_k$（精确），预条件 $A_0^{-1}$ 陈旧只影响
迭代数、**不导致停滞/restart** ⇒ 冻结因子全程复用（0 restart），同时线性解数减少 3.2×——
相当于 frozen=1 的收敛性 + frozen=2 的低分解成本。已实现为 `FROZEN=3`（`immersed.py`
`solve_monolithic`：fgmres(A_k, rhs, M=lu.solve, restart=50)；fgmres 不收敛时自动重分解）。

## §4a Cahouet–Chabard Schur 近似 —— 完成（amg 路径，3D 内层）

`_assemble_fluid_blocks` 新增 $L_p$（压力 Laplacian）；`IterativeFluidSaddle` 传
`Mp/Lp/rho/eta/dt` 时用
$S_p^{-1}\approx\frac{1}{\mu}M_p^{-1}+\frac{\rho}{\Delta t}L_p^{-1}$
（$M_p^{-1}$ 集中质量对角，$L_p^{-1}$ GAMG、pin 一个压力 dof）。amg 路径默认启用。
验证：流体鞍点解与 MUMPS 一致 ~1.5e-6（GAMG 松容差水平），端到端 amg+CC 跑通
Newton 3 次。2D 当前参数下与旧 $S_p$ 精度相当（预期）；价值在 3D 大尺度跨度
（补上 $B\,\mathrm{diag}(K)^{-1}B^T$ 缺的粘性半）。

## §2 matrix-free matvec —— 完成（可选 `MATRIX_FREE=1`，2D 不划算）

frozen=3 加 `MATRIX_FREE=1`：FGMRES matvec 用残差差分
$Av\approx[R(X+\varepsilon v)-R(X-\varepsilon v)]/(2\varepsilon)$（$\varepsilon=10^{-5}$
最优），BC 行模拟单位行。**关键坑**：`_residual` 用 `self.X` 而非传入 X，必须写显式残差
函数；且 matrix-free 给的是**未 BC 的 Jacobian**（与 raw 一致 4e-11），需手动设 BC 行。
验证：单线性解与显式 A_k 一致 6e-13（端到端 1e-10），eps=1e-5 最优。
**2D 下更慢**（每 matvec 2 次 force 组装，单解 11ms vs 显式 4ms），迭代数略增（24→26）。
价值在**免装配 Jacobian**（3D 重写 / 不可微本构时可用），默认关闭。

## §4b 单 V-cycle（内层去 Krylov）—— 验证**不适用**（GAMG 对 P2 向量质量差）

`GAMGSolver` 加 `vcycle=True`（preonly + GAMG，1 次 V-cycle）。实测（32×32 流体鞍点，
Cahouet-Chabard S_p）：**单 V-cycle 不收敛**（err 4.3），内层 max_it 扫描：

| 内层 max_it | 外层 FGMRES 迭代 | 解 err |
|---|---|---|
| 1 (preonly) | 10 | 2.95 |
| 3 | 10 | 0.35 |
| 5 | 10 | 1.8e-3 |
| 100 (松收敛) | 5 | 1.5e-6 |

**结论**：GAMG 一次 V-cycle 对 2D P2 向量流体块 $K$ 的质量太差（代数聚集不匹配块结构），
"1–2 次 V-cycle 作 $K^{-1}$"的假设在此不成立，需要多次迭代。⇒ §4c 的 **GMG（几何多网格）**
才是 2D/3D 结构网格的正解（几何插值算子精确，不像 GAMG 代数聚集）；实现需层级网格 +
P2/P1 层间插值/限制算子（P2 的粗边中点→细 dof 插值较繁琐），3D 重写时配合现成 Block-GMG
框架落地。

## 5. 块 LU 分解与约化系统（2026-08-06 新增，符号已数值核对）

### 5.1 块 LU（用户公式，已验证与代码逐项一致）

记三行式为 $[A\ B\ A_s;\ B^T\ 0\ 0;\ M_u\ 0\ M_w]$（对照代码：$A=K$、$B_{用户}=B^T_{代码}$、
$A_s=-A_{uW}$、$M_u=-M_{fs}^T$、$M_w=\tfrac{1}{\Delta t}M_s$），其块 LU 为

$$
L=\begin{bmatrix}I&0&0\\B^TA^{-1}&I&0\\M_uA^{-1}&M_uA^{-1}BS^{-1}&I\end{bmatrix},\quad
U=\begin{bmatrix}A&B&A_s\\0&-S&-B^TA^{-1}A_s\\0&0&S_w\end{bmatrix},
$$

$$
S_w=M_w-M_u\underbrace{\left(A^{-1}-A^{-1}BS^{-1}B^TA^{-1}\right)}_{\mathcal{P}=
\left.\begin{bmatrix}A&B\\B^T&0\end{bmatrix}^{-1}\right|_{(1,1)}}A_s .
$$

$S= B^TA^{-1}B$，$\mathcal P$ 是流体鞍点逆的 (1,1) 块。此 LU 我已手动逐项展开核对，
与代码 `solve_monolithic_reduced` 的约化一致（见 5.2）。

### 5.2 符号核对结论（16×16 网格数值验证）

**代码事实**（`immersed.py`）：残差 $R_u=Ku+B^Tp-f_{el}$、$R_p=Bu$、
$R_W=\tfrac{1}{\Delta t}M_s(W-W^n)-M_{fs}^Tu$；$f_{el}=-\int(PF^T):\nabla\phi\,dX$、
$A_{uW}=\partial f_{el}/\partial W$（装配带负号）；monolithic (1,3) 块 $=-A_{uW}$、(3,1) 块 $=-M_{fs}^T$。

1. **$W$ 是位移**（$F=I+\nabla W$），第三行是**运动学方程**（`dW/dt = u_s`，盘随流体运动），
   不是固体动量方程——用户的猜测正确，$\Delta t$ 因子确实出现（弹性项 $\sim\mu_s\Delta t$）。
2. **约化公式正确**：消元 $W$ 得
   $\tilde K=K-\Delta t A_{uW}M_s^{-1}M_{fs}^T$，与用户
   $\tilde A=A-A_sM_w^{-1}M_u$ 逐项一致（双重负号抵消，符号相同）。
3. **但正号（$+\mu_s\Delta t K_{s0}$，椭圆）要求耦合块是变分伴随**：若 spread=interp 的伴随
   $J=M_s^{-1}M_{fs}^T,\ J^*=M_{fs}M_s^{-1}$，则 $A_{uW}\to -J^*\mu_s K_{s0}$，约化修正变成
   $+\mu_s\Delta t\,J^*K_{s0}J$——**对称 PSD**（K_s0、M_s^{-1} PSD ⇒ 三明治 PSD），
   $\tilde K$ 保持椭圆，正是用户期望的形式。
4. **demo 的耦合块非伴随**（数值，16×16，nu=2178, ns=630）：

   | μ_s | demo 修正 sym(C) min eig | 伴随一致 sym(C_adj) min eig |
   |---|---|---|
   | 0.1 | −3.5e-4 | −1e-17（PSD） |
   | 1 | −3.5e-3 | −9e-17 |
   | 10 | −3.5e-2 | −1e-15 |
   | 100 | −3.5e-1 | −8e-15 |

   `C_demo = -dt A_uW M_s^{-1} Mfs^T`（代码实际用的修正）对称部分**负定且随 μ_s 线性增长**；
   伴随一致版 `C_adj = μ_s dt M_fs M_s^{-1} K_s0 M_s^{-1} M_fs^T`（K_s0 为固体侧弹性刚度，
   单独 UFL 装配）在所有 μ_s **PSD**。⇒ 负号来自非伴随耦合块（直接弱形式 spread ≠
   Mfs 的伴随），**不是** W 是位移/速度的问题。

### 5.3 后果：大 μ_s 是格式（变分一致性）问题，不是预处理问题——用户直觉正确

- **约化算子失去正定性**（带 BC）：min eig $\tilde K$：μ_s=0.1→2.82e-4（=K）、10→2.80e-4、
  **100→−2.18e-1**、**1000→−3.38**；修正幅度 max|C|/max|K| 在 μ_s=100 已达 2.3。
- **实际运行**（16×16，seeding 顶盖，3 步）：μ_s=0.1 两方案一致（|X|=9.374）；
  **μ_s=100 完整 3×3 与约化 2×2 都爆**（|X|≈1.3e5 与 6e5，不一致）——Newton 发散。
  这与 §2"诚实局限"（block-LDU 在 μ_s=100 停滞 7.2e-3）同源：都是 (1,3) 弹性刚度随 μ_s
  增长 + 非伴随插值导致的格式层不稳定，不是单纯预条件缺陷。

### 5.4 求解新思路

1. **速度块用精确约化算子（唯一实测 μ_s 鲁棒）**：见 §6——`K~ = K − dt A_uW M_s⁻¹ M_fsᵀ`
   （真实符号、精确 M_s⁻¹）作流体鞍点 (1,1) 块，迭代数恒定 ~15–18（μ_s 0.1→1000）。
   稠密（M_s⁻¹），2D 可用；3D 走 matvec 精确版（见 §6 思路）。
2. **$S_w$ 作固体块预条件**：用户 $S_w=(1/\Delta t)M_s-M_{fs}^T\mathcal P A_{uW}$ 在
   μ_s=0.1 实测 **SPD**（[5.9e-3, 7.1e-2]），捕捉流体加质量；可替换 (3,3) 块的
   $(1/\Delta t)M_s$。注意仅限稳定区（大 μ_s 已失去正定）。
3. **格式修正（3D 前置）**：§6 实验证明伴随一致（PSD）版预条件在 μ_s=100 **失败**
   ——当前非伴随公式下必须用真实符号的约化算子；若要在 3D 拿到稀疏可扩展的 μ_s 鲁棒
   预条件，唯一路径是把耦合块改成伴随一致（spread 走 $J^*$），但那会改变 Jacobian
   （不再逐位等于 deal.II）。取舍记录见 §6。
4. 约化系统作**精确**求解器只适用粗网格（$\tilde K$ 稠密）；作**预条件**（§6 V3）2D 可行。

> 研究工具：`_prec_adjoint.py`（非伴随 C_demo vs 伴随一致 C_adj 的 PSD/定号跨 μ_s 扫描）、
> `_prec_mu_sweep.py`（本节迭代数实验）。

## 6. μ_s-鲁棒预条件实验（2026-08-06，实测）

**问题**：随 μ_s 增长，什么预条件能让 FGMRES 迭代数不变？

**设置**：16×16 网格（nu=2178, np=289, ns=630），monolithic 3×3 A（当前 A_uW + BC），
随机 RHS，FGMRES(50) rtol=1e-8 maxiter=1000。五种子变体都是块 LDU（L/U 耦合保留），
仅流体鞍点 (1,1) 速度块不同：

| 变体 | 速度块 K~ | 稀疏性 |
|---|---|---|
| V0 基线 | K | 稀疏（现实现） |
| V3 精确约化 | K − dt A_uW M_s⁻¹ M_fsᵀ | **稠密** |
| V2 伴随稠密 | K + μ_s dt M_fs M_s⁻¹K_s0 M_s⁻¹ M_fsᵀ | 稠密 |
| V1 伴随集中 | K + μ_s dt M_fs diag(M_s)⁻¹K_s0 diag(M_s)⁻¹ M_fsᵀ | 稀疏 |
| V4 集中真实符号 | K − dt A_uW diag(M_s)⁻¹ M_fsᵀ | 稀疏 |

（K_s0 = 固体侧弹性刚度，单独 UFL 装配。）**FGMRES 迭代数**：

| μ_s | V0 | **V3** | V2 | V1 | V4 |
|---|---|---|---|---|---|
| 0.1 | 15 | 15 | 15 | 26 | 15 |
| 1 | 15 | 15 | 20 | 73 | 15 |
| 10 | 20 | 15 | 58 | 354 | 18 |
| 100 | 发散 | **18** | 发散 | 发散 | 发散 |
| 1000 | 发散 | **17** | 发散 | 发散 | 发散 |

（"发散"= 1000 次内未收敛。）

**结论**：

1. **可能**——V3（精确约化算子，真实符号 $K-dt A_{uW}M_s^{-1}M_{fs}^T$）把迭代数
   **恒定在 15–18**，μ_s 从 0.1 到 1000 不变。机理：块 LDU 的 L/U 已保留耦合，唯一被丢的
   就是二阶项 $A_sM_{ww}^{-1}M_{wu}=dt A_{uW}M_s^{-1}M_{fs}^T$；把它**精确**放回 (1,1) 块，
   预条件就完整复现 A 的 μ_s 依赖 ⇒ $P^{-1}A$ 谱与 μ_s 无关。
2. **所有稀疏/近似版本都失败**（μ_s≥100 发散）：集中质量 V4（差 $A_{uW}(M_s^{-1}-\mathrm{diag}^{-1})M_{fs}^T\sim\mu_s$）、
   伴随 PSD V1/V2（真实修正是**不定**的，PSD 近似对不上谱——再次印证 §5 非伴随结论）。
3. **稠密是 2D 的代价，3D 的障碍**：精确 M_s⁻¹ 使 K~ 稠密。3D 可扩展路径 = **matvec 精确版**：
   不组装 K~，流体鞍点内层 Krylov 用算子
   $K~ v = Kv - dt\,A_{uW}\big(M_s^{-1}(M_{fs}^T v)\big)$（A_uW、M_fs 稀疏带 + M_s⁻¹ 走已有
   稀疏因子，M_s 小故廉价）——精确、免稠密存储，代价是嵌套 Krylov。
4. **若要 3D 稀疏 PSD 版成立，必须改格式为伴随一致耦合**（spread 走 $J^*$），但那会改变
   Jacobian、不再逐位等于 deal.II——与"符号正确（结果同 deal.II）"矛盾，需用户决策。

### 6.1 V3 精确机理（块级验证，μ_s=100，`_verify_PinvA.py`）

记流体鞍点 $S=\begin{bmatrix}K~&B^T\\B&s_{11}\end{bmatrix}$、$F=S^{-1}$。块 LDU 预条件
$P^{-1}$（L/U 耦合精确）作用到 A 上，$(1,1)$ 块为

$$
X_{11}=F_{11}(K-A_sM_{ww}^{-1}M_{wu})+F_{12}B .
$$

- **V3** 取 $K~=K-A_sM_{ww}^{-1}M_{wu}=K-dt\,A_{uW}M_s^{-1}M_{fs}^T$（精确约化算子），则
  $X_{11}=F_{11}K~+F_{12}B=I$（因 $F=S^{-1}$ 且 $K~$ 正是消元后的 (1,1)）——**μ_s 项被精确消掉**。
  剩余扰动全部 μ_s 无关：$(3,1)$ 块 = $2M_{ww}^{-1}M_{wu}=-2dt\,M_s^{-1}M_{fs}^T$
  （实测 max|X31−2M_ww⁻¹M_wu|=7.6e-3，|2M_ww⁻¹M_wu|max=2.4e-2）+ BC 局部性。
  实测谱：Re∈[+0.135,+1.00]，全正实部，mean|λ−1|=0.010（紧簇在 1）。
- **V0** 取 $K~=K$，则 $X_{11}=I-F_{11}^0 A_sM_{ww}^{-1}M_{wu}$ 含
  $A_sM_{ww}^{-1}M_{wu}\sim\mu_s$（随 μ_s 增长）→ 谱散开。实测 max|X11−I|=2.84、
  Re∈[−1.93,+3.75]（出现负实部）→ 发散。

### 6.2 M_ww 换成集中质量是否可行？（2026-08-06，`_lump_ww.py`）

**结论：不建议换——a.cpp 已试过并否决；且会改变解。但无需换也能稀疏可扩展（a.cpp scheme 5）。**

- **参考实现（a.cpp）**：全质量 $M_s u_s=M_{fs}^Tu$（一致投影，line 68-71）与 demo 一致。
  a.cpp **scheme 3/4 正是对角 $M_s^{-1}$**（$K~=K-dt\,A_{uW}D_s^{-1}M_{fs}^T$，$D_s=\mathrm{diag}(M_s)$），
  注释明言其"把 Newton 限制到线性收敛、软盘发散"（line 1365-1366, 1585）——demo 注释同。
  **正解是 scheme 5**（line 1576-1712）：精确 $M_s^{-1}$ 走 **ILU（隐式）**，$K~$ 保持隐式
  算子不显式组装，约化 2×2 + 块对角预条件（K 上 multigrid、Mp 上 Jacobi）。
- **我的实测（16×16）**：**一致集中**（$A'_{33}=\tfrac1{\Delta t}\mathrm{diag}(M_s)$ + 约化
  $K~_lump=K-dt\,A_{uW}\mathrm{diag}(M_s)^{-1}M_{fs}^T$）下，V3 稀疏版**确实 μ_s 鲁棒**：
  迭代数 15→18 恒定（μ_s 0.1→1000），且 $C_{lump}$ 只 **0.83% 稠密**（稀疏带）、固体块全对角。
  **但解保真度差**：随机 RHS 相对差 13.7%，光滑 RHS 下固体块差 ~89%——因为把运动学插值从
  一致投影（$M_s^{-1}M_{fs}^T$）换成了集中插值（$\mathrm{diag}(M_s)^{-1}M_{fs}^T$），
  是真实的格式改变，解不再等于全质量版（偏离 deal.II）。
- **此前 V4/V1 失败的原因**：只把**预条件里**的 $M_s^{-1}$ 集中化、实际 $A_{33}$ 保持全质量——
  **不一致**，约化算子与真实 Schur 差 $O(\mu_s)$。一致性要求连 $A_{33}$ 一起换。
- **正确路径（不牺牲解）**：保持全质量，$M_s^{-1}$ 走已有稀疏因子（M_s 小且良态，
  同 a.cpp ILU），$K~$ 作为**隐式算子** $K~v=Kv-dt\,A_{uW}(M_s^{-1}(M_{fs}^T v))$，
  流体鞍点内层 Krylov（块对角：K 上 AMG/MG + Mp 上 Jacobi）——即 a.cpp scheme 5 的移植，
  就是 §6-3 待办。

### 6.3 外层 FGMRES 保证什么？（正确性澄清，2026-08-06，`_prec_lump_on_full.py`）

**FGMRES 保证"它被解的那个系统的解"正确，不等于"全系统的解"正确。**

- 解在**全系统 A** 上（集中版只当预条件）：FGMRES 保证解 $=A^{-1}b$ 正确。实测 μ_s≤10
  时相对误差 ~1e-9 ✓。但**集中预条件在全系统上不 μ_s 鲁棒**：μ_s=100 发散——因为附加刚度里
  用了 $\mathrm{diag}(M_s)^{-1}$（$K~_lump$），与全系统精确消元需要的 $M_s^{-1}$ 差
  $A_{uW}(M_s^{-1}-\mathrm{diag}^{-1})M_{fs}^T\sim\mu_s$，X11=I 的精确抵消被破坏（同 V4 机理）。
- 解在**集中系统 A'** 上（我 §6.2 的 `A_lump+V3`）：FGMRES 保证 $A'^{-1}b$，但 $A'\ne A$
  → 解偏离 ~89%（固体块）——"收敛快"是假象：它在解**被换掉的系统**，不是正确性。
- **结论**：lumping 不能同时拿到"正确 + 稀疏 + μ_s 鲁棒"。只有两条路：
  ① 稠密 V3（全 $M_s^{-1}$，2D）；② a.cpp scheme 5 隐式 $M_s^{-1}$（嵌套 Krylov，3D）。
  外层 FGMRES 保证正确性的前提是**解全系统**，而全系统上的 μ_s 鲁棒只有精确 $K~$ 能做到。

### 6.4 matvec-V3：不用写出稠密矩阵，能否推广 3D？（2026-08-06，`_matvec_v3.py`）

**能——这就是 a.cpp scheme 5**。$K~=K-dt\,A_{uW}M_s^{-1}M_{fs}^T$ 作为**隐式算子**（从不组装）：

$$
K~ v = Kv - dt\,A_{uW}\big(M_s^{-1}(M_{fs}^T v)\big),\qquad M_s^{-1}\ \text{走稀疏因子}
$$

流体鞍点 $[K~ B^T; B\ s_{11}]$ 用**内层 GMRES** 解（每内层迭代 = 2 稀疏 matvec + 1 小 $M_s^{-1}$
回代），外层 FGMRES 解全系统 A。

**实测（8×8，nu=578）**：
- K~ 算子 matvec 与稠密矩阵一致 **1e-16**；
- μ_s=0.1：外层 3 次 + 内层 6 次收敛，解正确 1.6e-12（全质量、一致投影，不偏离 deal.II）；
- **μ_s=100：外层不收敛，内层 >20000 次**——内层预条件用普通 K（a.cpp 的块对角 MG(K)+Mp），
  附加刚度 $K~-K\sim\mu_s$ 使 $S_0^{-1}S$ 远离单位阵。

**结论**：
1. **不写稠密矩阵：可行**（a.cpp scheme 5 现成）。**3D 结构上可行**：全部操作稀疏
   （K/A_uW/Mfs 带、M_s 小走稀疏因子，无稠密存储）。
2. **μ_s 鲁棒性转移到了内层求解**——这是关键。稠密 V3 的 μ_s 鲁棒来自鞍点的**精确** LU；
   matvec 版换成内层迭代后，若内层预条件不含附加刚度，μ_s→∞ 时内层发散。
3. **3D 大 μ_s 的真正难点 = 内层 μ_s 鲁棒预条件**（辅助空间 $\Pi\bar A\Pi^*$，同 §3/§6）：
   内层速度块需含附加刚度（如 K~ 上的 MG/AMG 或加法项）。基准 μ_s=0.1 下 a.cpp 的普通
   块对角已够用（实测内层 6 次）——所以 a.cpp scheme 5 对基准 OK，大 μ_s 是开放问题。

**内层为什么发散（`_inner_spectrum.py`，8×8 谱证据）——不是 M_s 难处理**：
内层 GMRES 解 $S=[K~ B^T; B s_{11}]$（$K~=K+$附加刚度），预条件 $S_0^{-1}=[K B^T; B s_{11}]^{-1}$。
收敛性由 $S_0^{-1}S=I+S_0^{-1}(S-S_0)$ 决定，其中 $S-S_0=\mathrm{diag}(-dt\,A_{uW}M_s^{-1}M_{fs}^T,0)\sim\mu_s$。
$M_s$ 在 $K~$ 里走稀疏因子**精确处理**，不是问题；**问题是附加刚度随 μ_s 增长而普通 K 预条件不含它**：

| μ_s | $S_0^{-1}S$ 特征值 Re 区间 | max\|λ−1\| | 内层迭代 |
|---|---|---|---|
| 0.1 | [+0.996, +1.005] | 0.01 | 6 ✓ |
| 100 | [−3.33, +6.40]（负实部） | 5.40 | >20000 ✗ |

μ_s=100 时 $S_0^{-1}S$ 谱散开且出现负实部 → 内层 GMRES 发散。修复=内层预条件含附加刚度
（辅助空间），与外层 V0 失效是同一个 μ_s 机理，只是挪到了内层。

> **澄清"处理 M_s 就行"（易混淆点）**：M_s 精确（稀疏因子）是**必要条件**，不是充分条件。
> 两个版本里 M_s 都是精确处理的（稠密 V3 用稠密逆、matvec 版用稀疏因子，等价）。
> **稠密 V3 之所以"处理 M_s 就够"**：鞍点 $[K~ B^T; B s_{11}]$ 用**稠密 LU 精确解**，
> 没有内层预条件问题，X11=I 直接成立 → μ_s 鲁棒。
> **matvec 版不够**：M_s 同样精确，但鞍点换成**内层迭代**，迭代需要预条件，而普通 K
> 预条件不含附加刚度 → 新需求暴露。所以问题**不在 M_s**，在鞍点求解方式
> （精确 LU vs 迭代+预条件）。

## 待办（未执行）

- §1a 固定稀疏模式 + 一次符号分析（scipy→PETSc 179ms/次，频繁分解场景价值高）
- §1b spread 装配向量化（Mfs/A_uW 已向量化，剩余 coo→csr，收益有限）
- §1c MUMPS 参数（ICNTL(7)=5 METIS、3D 上 ICNTL(35)=2 BLR）
- §4c GMG（几何多网格，结构网格 2–4×，3D 终态）
- §6-3 matvec 精确版 μ_s 鲁棒预条件（流体鞍点内层 Krylov + 算子 K~，3D 路径）——**最高优先级**
- §6-4 耦合块伴随一致化（改格式，需用户决策：结果将偏离 deal.II）
- §5.4-2 S_w 固体块 Schur 预条件（稳定区）
- §5 零碎（ICNTL(24)、增广拉格朗日、BDF2）
