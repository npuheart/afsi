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

## 待办（未执行）

- §1a 固定稀疏模式 + 一次符号分析（scipy→PETSc 179ms/次，频繁分解场景价值高）
- §1b spread 装配向量化（Mfs/A_uW 已向量化，剩余 coo→csr，收益有限）
- §1c MUMPS 参数（ICNTL(7)=5 METIS、3D 上 ICNTL(35)=2 BLR）
- §2d 外推初值（linear 已实现 = §2d 的 $2x^n-x^{n-1}$；速度外推与之等价）
- §2 matrix-free matvec（Av ≈ [R(x+εv)-R(x)]/ε，免去每迭代装配 A_k）
- §4a Cahouet–Chabard Schur 近似（3D 内层）
- §4b/c 单 V-cycle + GMG（3D 终态）
- §5 零碎（ICNTL(24)、增广拉格朗日、BDF2）
