# multiphase — AFSI 的两相流（扩散界面）能力

IB2d 与 AFSI 原有的求解器（`ChorinSolver` / `IPCSSolver` / `PeskinRK2Solver`）
都假设**单一密度、单一黏度**，没有 VOF / level-set / Cahn–Hilliard / 表面张力，
因此上游**没有任何真正的多相流算例**（见 `../readme.md` 的说明）。这里补上这一能力：
一个自带的**扩散界面（diffuse-interface）两相流求解器** + 4 个可复现的验证算例。

## 求解器：`two_phase.py`

单文件、串行、二维、结构化四边形网格（Q1 顶点 = 网格节点，与 IB2d 算例一致）。
速度 `CG2`（Taylor–Hood）、压力 `CG1`、相场 `CG1`。

**1. 相场（Allen–Cahn 方程，半隐式 IMEX）**

```
d phi/dt + u.grad(phi) = M [ eps lap(phi) - W'(phi)/eps ] ,
W(phi) = phi^2 (1-phi)^2 / 4 ,  W'(phi) = phi(1-phi)(1-2phi)/2
```

* 平衡界面剖面：`phi(d) = 0.5 (1 - tanh(d / (2 sqrt(2) eps)))`，界面厚度 `4 eps`
  （算例里取 `eps = 1~2 h`）。
* 双势阱用**稳定化分裂** `W'(phi^{n+1}) ~ W'(phi^n) + s (phi^{n+1} - phi^n)`，
  `s = 0.75 >= max|W''| = 1/2`，于是每步只需求一个对称正定线性系统；
  再对结果做 `clip(0,1)` 兜底（readme 里明确写出这一点）。
* 对流显式、扩散隐式。

**2. 变物性**：`rho(phi) = rho_1 + (rho_2 - rho_1) phi`（`mu` 同理），逐节点求值，
进入投影的每一项；可选重力浮力 `(rho(phi) - rho_1) g`。

**3. 表面张力（CSF）**：`f_st = sigma kappa grad(phi)`，
`kappa = -div(grad(phi)/|grad(phi)|)`。两个关键实现细节：

* 曲率用**不分解形式**（不做 `grad/|grad|` 之后再求散度）：
  `kappa = -(lap phi)/|grad phi| + (grad phi . Hess(phi) . grad phi)/|grad phi|^3`。
  否则在界面外 `|grad phi|` 指数小，先除后微分会把网格噪声放大 1/h 倍：
  直接对 CG1 单元梯度做散度时，半径 0.25 的液滴曲率会飙到 `±35`（精确值 4），
  运行几步就发散。
* 曲率与力都在**结构网格上用中心差分**计算并做 2~3 次欠松弛平滑；
  测试算例里界面带上的曲率是 `3.95 ± 0.5`（精确 4）。

**4. 投影（经典 Chorin，与 AFSI `ChorinSolver` 同型，系数带 `1/rho`）**

```
1)  rho (u* - u^n)/dt + rho (u^n . grad)u^n - div(mu (grad u* + grad u*^T)) = f_st + f_g
2)  div( (1/rho) grad p^{n+1} ) = div(u*)/dt
3)  u^{n+1} = u* - dt (1/rho) grad p^{n+1}
```

压力的自然边界是**齐次**的 `(1/rho) dp/dn = 0`（同 `ChorinSolver`）：`u*` 已在边界
自由度的意义上满足 `u.n = u_bc.n`，齐次法向导数让修正不再改变 `u.n`，不需要任何
面通量记账。（把教科书里的 `(1/rho) dp/dn = (u* - u_bc).n/dt` 塞进去，离散相容性会
依赖边界通量的求积，压力解随后会放大网格尺度模态。）
另外压力**不能同时出现在预测步**：两边都算等于把压力算了两遍，几步之内就发散——
这是本模块调试中最容易踩的坑。

`rho(phi)`、`mu(phi)` 每步都变，所以承载它们的两个矩阵每步在**不变的稀疏结构**上
重新装配（必须 `zeroEntries()` 再装配，否则 `assemble_matrix` 是**累加**的——
同样是踩过的坑）；当 `rho_1 == rho_2` 且 `mu_1 == mu_2`（纯表面张力）时跳过重装，
速度快很多。

## 算例一览

| 目录 | 验证对象 | 解析参照 | 结论 |
|---|---|---|---|
| [`demo_801_laplace/`](demo_801_laplace/readme.md) | 静液滴 Laplace 定律 | `dp = sigma / R` | ✅ 定量：`NX=128` 误差 **0.89%**，一阶收敛 |
| [`demo_802_shear_droplet/`](demo_802_shear_droplet/readme.md) | 剪切流中液滴形变 | Taylor `D = Ca(19λ+16)/(16λ+16)` | ⚠️ 定性：`D` 偏大 1.5~2 倍（界面厚度 + 有界域约束），已给出趋势与测量标定 |
| [`demo_803_relaxation/`](demo_803_relaxation/readme.md) | 变形液滴松弛时间 | Maffettone–Minale `tau` | ✅ 定量：三个 `λ` 都在 **6~18%** 以内 |
| [`demo_804_rayleigh_taylor/`](demo_804_rayleigh_taylor/readme.md) | 密度分层 RT 失稳（+ 静水压力） | `sqrt(A_t g k)` / 黏性修正 | ◐ 量级：增长率 1.02（理论 1.65）；静水 `dp/dy` 精确 |

**能验证**：表面张力（静力学积分 → Laplace 定律）、表面张力与黏性耗散的**动平衡**
（松弛时间）、变密度 + 重力的静水压力梯度、密度分层不稳定性增长。

**不能验证 / 已知局限**（都写进各 demo 的 readme）：

* **Allen–Cahn 不守恒**：曲率驱动下液滴会缓慢收缩 `dR/dt ~ -M eps / R`，
  所以要取小 `M`（`0.002~0.005`）。需要严格守恒时应该换成 Cahn–Hilliard。
* **剪切形变偏大**：`demo_802` 测得 `D/Taylor ≈ 1.5~2`，随界面变薄（`eps=2h→1h`：
  2.08→1.53）与液滴变大而下降，说明是**扩散界面厚度 + 有界域约束**的联合效应；
  在本次可承受的网格规模下没有跑到收敛，因此只作定性结论。
* **寄生流**：`demo_801` 给出残余速度（`Ca_parasitic ~ 3e-4`）作为 CSF 寄生电流的
  量级指标。
* 只支持**串行、二维、结构化网格**；没有周期边界；黏性项显式/隐式混合仅一阶时间精度。

## 运行

```bash
conda activate afsi-dolfinx
cd afsic/demo/multiphase

python demo_801_laplace/main.py --sweep --plot        # 6 个 (NX, eps) 组合
python demo_802_shear_droplet/main.py --plot          # 单个 Ca 算例
python demo_803_relaxation/main.py --sweep --plot     # lambda = 0.5/1/2
python demo_804_rayleigh_taylor/main.py --plot        # RT 增长
```

每个算例把结果写到自己的 `output/*.csv`（+ `--plot` 时的 PNG），
运行时间与数字见各 readme 的结果表。
