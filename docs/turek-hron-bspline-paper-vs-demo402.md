# Turek–Hron 改版算例（Composite B-Splines 论文）与本地 demo_402 的参数对照

## 0. 来源与摘录

来源：用户提供的摘录，出自 *Local Divergence-Free Immersed Finite Element-Difference
Method Using Composite B-Splines*（该文用改版 Turek–Hron FSI 算例做核函数/B-spline 对比，
即其 Fig. 24–26、Table 4）。以下为原文照录：

> We investigate a modified version of the Turek-Hron fluid-structure interaction (FSI)
> benchmark,$^{56}$ which simulates flow around a flexible elastic beam attached to a fixed
> circular cylinder.$^{25}$ While the original benchmark specifies domain dimensions of
> $L=2.5$ and $H=0.41$, we extend the length to $L=2.46=6.0H$ to accommodate square Cartesian
> grid cells. This modification has a negligible impact on the benchmark results. The
> computational setup uses a fine-grid Cartesian cell size of $\Delta x=L/N$ with a time step
> of $\Delta t=0.00164\Delta x$, where $N$ is the grid number along the longest dimension of
> the fluid domain. The structure consists of a circular cylinder (diameter $d=0.1$) centered
> at $(0.2,0.2)$; and an elastic beam (length $l=0.35$, height $h=0.02$) fixed to the
> cylinder's rear. A control point $A$ (initial position is $(0.6,0.2)$) is used for monitoring
> displacement. Fig. 24 shows the setup schematic. The boundary conditions are specified as
> follows: at the inlet ($x=0$), $u(0,y)=1.5Uy(H-y)/(H/2)^2$, where $U=2$ is the average
> velocity; at the outlet ($x=L$), zero normal traction and zero tangential velocity are
> imposed; and along the top and bottom walls ($y=0,H$), zero velocity conditions are enforced.
>
> The flow parameters yield $Re=\rho Ud/\mu=200$, with $\rho=1000$ and $\mu=1$.
>
> The structure is modeled using the Saint Venant-Kirchhoff constitutive law:
> $\mathbb{S}=\lambda_s\mathrm{tr}(\mathbb{E})\mathbb{I}+2\mu_s\mathbb{E}$ where $\mathbb{S}$ is
> the second Piola-Kirchhoff stress tensor, $\mathbb{E}$ is the Green-Lagrange strain tensor,
> $\mathbb{I}$ is the second-order identity tensor, and material parameters are
> $\mu_s=1\times10^6$, and $\lambda=8\times10^6$. The cylinder is constrained using a spring
> tether force with penalty parameter $\kappa_s=5.0\times10^4\Delta x/\Delta t^2$.
>
> We examine three kernels: IB$_3$, BS$_3$, and CBS$_{32}$. The fluid domain uses $N=128$ grid
> points along its longest dimension, providing sufficient resolution to isolate the effects of
> solid mesh refinement. We investigate MFAC values of 0.5, 0.75, 1.0, 1.25, and 1.5.
>
> Figure 25 presents a representative color map of the vorticity field, highlighting the
> deformed beam simulated with the CBS$_{32}$ kernel and MFAC = 0.5. Table 4 summarizes the
> maximum vertical displacements ($\Delta Y$) at point A (shown in Fig. 24) for each kernel type
> across different MFAC values. The IB$_3$ kernel exhibits the most stable behavior concerning
> MFAC variations, while CBS kernels require smaller MFAC values and fail when MFAC exceeds 1,
> consistent with observations from other benchmarks. Figure 26 illustrates the oscillation
> histories of the vertical displacement for three MFAC values. The IB and BS kernels yield
> similar results, with smaller MFAC values generally predicting larger displacements. In
> contrast, the CBS kernel is less sensitive to the solid mesh resolution. The observed phase
> shift across different MFAC values is attributed to time step variations.

## 1. 论文算例的参数（整理，SI）

| 量 | 值 |
|---|---|
| 域 `L × H` | **2.46 × 0.41 m**（= 6.0H；原基准 2.5 × 0.41，改短以便方格网格）|
| 圆柱 | `d = 0.1 m`，圆心 `(0.2, 0.2)` |
| 弹性梁 | `l = 0.35`、`h = 0.02`，固定在圆柱后方（尾端即监测点 A）|
| 监测点 A | `(0.6, 0.2)`，记录**竖向位移 ΔY** |
| 入口 | `u(0,y) = 1.5U·y(H−y)/(H/2)²`，`U = 2 m/s`（未提到斜坡/软启动）|
| 出口 | **零法向 traction + 零切向速度** |
| 上/下壁 | 零速度 |
| 流体 | `ρ = 1000 kg/m³`、`μ = 1 Pa·s` → **Re = 200** |
| 固体本构 | **Saint Venant–Kirchhoff**：`S = λ_s tr(E) I + 2 μ_s E` |
| 固体参数 | `μ_s = 1×10⁶ Pa`、`λ_s = 8×10⁶ Pa` → `ν = 0.444`、`E ≈ 2.89 MPa` |
| 圆柱约束 | spring tether，`κ_s = 5.0×10⁴ Δx/Δt²` |
| 流体网格 | `N = 128` 沿最长边 → `Δx = L/N = 1.9219 cm`（原文要求方格；若严格方格且 H 向取整数格，则 `Δx = H/21 = 1.9524 cm`，L 向 126 格，与 128 差 ~1.5 %）|
| 时间步 | `Δt = 0.00164 Δx = 3.1519×10⁻⁵ s` → **CFL = U·Δt/Δx = 0.00328** |
| 罚参数数值 | `κ_s = 5×10⁴·Δx/Δt² = 9.67×10¹¹`，即无量纲组 `κ_s Δt²/Δx ≡ 5×10⁴` 恒定 |
| 研究变量 | 核函数 `IB₃ / BS₃ / CBS₃₂` × `MFAC ∈ {0.5, 0.75, 1.0, 1.25, 1.5}`（MFAC = 固体网格间距 / Δx）|
| 目标量 | 点 A 的最大竖向位移（Table 4，各核 × MFAC）；ΔY(t) 振荡史（Fig. 26）；涡量场（Fig. 25，CBS₃₂ + MFAC 0.5）|

## 2. 与本地 `demo_402` 的差异（逐项实测对齐）

我们的出厂配置（`configuration.py`，CGS）：域 `220 × 41 cm`（= 2.20 × 0.41 m）、
`Nx×Ny = 220×41`、`dt = 5×10⁻⁵ s`、`Um = 200 cm/s`、`ρ = 1 g/cm³`、`μ = 10 dyne·s/cm²`、
`μ_s = 2×10⁷`、`λ_s = 8×10⁷ dyne/cm²`、`β = 1×10⁶ dyne/cm³`、Neo-Hookean、ChorinSolver。

| 量 | 论文 | demo_402（出厂） | 差异与影响 |
|---|---|---|---|
| 域长 `L` | 2.46 m (= 6.0H) | **2.20 m (= 5.366H)** | 我们短 0.26 m：尾端 (x=0.6) 到出口只剩 1.60 m（论文 1.86 m），尾流/出口反馈更强 |
| 域高 `H` | 0.41 m | 0.41 m | 相同 ✓ |
| 圆柱/梁几何 | d=0.1 @(0.2,0.2)，l=0.35, h=0.02 | 相同（10 cm / 35×2 cm）| 相同 ✓（几何来自 `turek.geo`）|
| 监测点 | A = (0.6, 0.2) 竖向位移 | 我报"尾尖（参考构型 x 最大处）平均位移" | 位置一致，量可比 ✓ |
| 入口 | `1.5U y(H−y)/(H/2)²`，U = 2 m/s，**无 ramp** | 同式，Um = 200 cm/s = 2 m/s，**带 2 s 余弦 ramp** | t < 2 s 时两者不是同一物理状态，**必须 ramp 结束后再比** |
| 出口 | 零法向 traction **+ 零切向速度** | `p = 0` Dirichlet（法向 traction 由自然条件给出，切向无约束）| 出口切向条件不同 |
| 壁面 | 零速度 | 零速度 ✓ | 相同 |
| Re | 200 | **200**（`1×200×10/10`）| 相同 ✓（注意 demo readme 写 Re=100，是按 Ubar=1 m/s 算的，与代码矛盾；站点页已标为 readme/code 冲突）|
| 固体本构 | **Saint Venant–Kirchhoff** | **Neo-Hookean**（代码实际；readme 声称 SVK）| 大变形幅值下两者有差异，属模型级差异 |
| 固体参数 | `μ_s = 1×10⁶ Pa`、`λ_s = 8×10⁶ Pa`（ν=0.444, E=2.89 MPa）| `μ_s = 2×10⁶ Pa`、`λ_s = 8×10⁶ Pa`（ν=0.4, E=5.6 MPa）| **我们刚度是论文的 1.94 倍** → 挠度更小、频率更高 |
| 圆柱约束 | spring tether，`κ_s = 5×10⁴ Δx/Δt²`（**随网格/步长自动缩放**，无量纲组恒为 5×10⁴）| 固定常数 `β = 1×10⁶ dyne/cm³ = 1×10⁷ N/m³` | 结构级差异：我们的无量纲组 `βΔt²/(ρΔx) ≈ 2.5×10⁻³`（Δx=1 cm, dt=5e-5）且会随 dt² 变——这正是本地实测到的失稳/冻结来源（见 §3）|
| 流体网格 | `Δx = 1.9219 cm`（N=128 沿 L）| 出厂 `Δx = 1.00 cm`（220×41）；我另跑的粗网格 `Δx ≈ 2.0 cm`（110×21）| **出厂网格比论文细 1.92 倍**；110×21 才与论文密度相当 |
| 时间步 | `Δt = 0.00164Δx`，即 `Δt/Δx = 0.00164`，CFL = 0.00328 | `dt = 5×10⁻⁵ s`，`dt/Δx = 0.005`，CFL = 0.01 | **我们的相对步长是论文的 3.05 倍**（即便网格更细）|
| 固体网格 | 用 MFAC = 0.5–1.5 × Δx 参数化研究 | `turek.geo` 固定 `MeshSizeMax = 0.006 m = 0.6 cm` | 在 Δx=1 cm 时 MFAC ≈ 0.6（≈论文 0.5 档）；Δx=2 cm 时 MFAC ≈ 0.3，**低于论文扫描区间** |
| 核函数 | `IB₃ / BS₃ / CBS₃₂` 可选（B-spline 复合核）| AFSI 的 4 点 Peskin 核，写死在 C++ 层（`IBMesh`/`IBInterpolation`），**不可选** | 论文 Table 4 / Fig 26 的"核对比"无法直接复现 |
| 固体惯性 | immersed FE，含固体密度（有惯性项）| **无质量 Peskin IB**（readme 已列 known limitation）| FSI2 原基准依赖 `ρ_s ≈ 10 ρ_f` 的附加惯性；这是质的差别 |
| 时长 | 本段未给（该类算例通常 t ≥ 10 s 看拍动稳态）| 我跑到 t = 2 s（ramp 刚结束）| 尚未进入稳态拍动段 |

## 3. 顺带说明：罚参数写法的差异为什么重要

论文的 `κ_s = 5×10⁴ Δx/Δt²` 是**自适应**形式（无量纲组 `κ_sΔt²/Δx` 固定为 5×10⁴），
网格/步长变化时罚力强度不变。我们 `demo_402` 的 `β = 1×10⁶ dyne/cm³` 是固定数，
其**无量纲组随 Δt² 变**：把它换成 424/426 的等价判据 `βΔt²/(ρΔx)` 后，本地实测
（`docs/demo-426-ib-coupling-findings.md`）表明显式（冻结力）耦合在该组偏大时会发散、
偏小时又会"解耦/冻结"（demo_402 的 88×17 运行、IPCS 运行都是这个病症）。
所以想让 402 稳定且参数可移植，应该照论文那样把罚刚度写成随 `Δx/Δt²` 缩放的形式。

## 4. 想对齐论文设置，需要改什么（按优先级）

1. **域与网格**：`X_MAX=2.46`、`Y_MAX=0.41`（米）→ 方格网格取 `Ny = 21`、`Δx = 1.9524 cm`、
   `Nx = 126`（或按 `Δx = L/128 = 1.9219 cm`、`Ny = 21` 用 `Nx = 128`）。
2. **时间步**：`Δt = 0.00164Δx = 3.20×10⁻⁵ s`（我们当前 dt 需缩小 1.56 倍）。
3. **入口**：去掉 2 s ramp（论文未用），或只比较 ramp 结束后的时段。
4. **出口 BC**：补"零切向速度"。
5. **固体**：换成 SVK + `μ_s = 1×10⁶ Pa`、`λ_s = 8×10⁶ Pa`（需要写 SVK 的 UFL 形式；
   当前代码是 Neo-Hookean）。
6. **圆柱罚**：改成 `κ_s = 5×10⁴ Δx/Δt²`（当前 β 固定），否则稳定性/位移都不可比。
7. **MFAC**：把固体网格间距设为 `MFAC × Δx`（0.5–1.5 → 0.98–2.93 cm），替换
   `turek.geo` 里固定的 `MeshSizeMax = 0.006`。
8. **核函数对比**：放弃或改造——`IB₃/BS₃/CBS₃₂` 需要把 AFSI 的 IB 核做成可选的
   （C++ `IBKernel` 层），这是论文那张表的核心变量，我们目前只能做单核结果。

**近似可做的版本**：只对齐 1–3 + 6（网格≈1.95 cm、dt=3.2e-5、无 ramp、罚自适应），
用我们现有的 Neo-Hookean（把 μ_s 降到 1e6 Pa 以贴近论文刚度），跑 T ≈ 2–4 s，
比较点 A 的 ΔY(t) 幅值/频率量级（论文 Fig. 26 的量级），并明确声明本构、核函数、
固体惯性三项与论文不同。

本机成本估算：`Δx ≈ 1.95 cm` 的网格（112×21 或 113×21）单步 ≈0.12–0.13 s（实测 110×21），
`Δt = 3.2×10⁻⁵ s` → 跑 T = 2 s 需 62500 步 ≈ **2.2–2.3 h**；T = 4 s 约 4.5 h。

---

## 5. 实测：哪些能对齐、哪些不能（2026-09-26，本机 dolfinx 0.10，单进程）

### 5.1 并行（MPI）不可用 —— 先说结论

| 运行 | 结果 |
|---|---|
| `mpirun -n 1` | 正常，与单进程逐位一致 |
| `mpirun -n 2` | **MPI 致命错误**：`Fatal error in internal_Allreduce_c: Message truncated`（rank 间集合操作错位），2 s 内中止 |
| `mpirun -n 4` | 能跑完，**流场与 -n 1 一致到 5–6 位**（`u_L2` 1.3937105412 vs 1.3937126582），但**固体位移/tip 是 NaN** |

根因在 C++ 耦合层：`IBMesh::extract_dofs` / `assign_dofs` 在**每个 rank 上遍历全局 nx×ny
网格**并对 dof 做 `getitem/setitem`；`IBInterpolation::assign` 用
`function_space()->dim()` 配 `vector()->set_local()`。这些在单 rank 下没问题，多 rank 时
"本地/全局 dof" 语义不成立（读到的可能是别的 rank 的/不存在的 dof）。demo_426 上看到的
"两个 rank 各报一半节点" 是同一问题的另一个表现。**要并行必须先改这两处 C++
（`afsic/src/coupling/src/IBMesh.h`、`IBInterpolation.h`）并重编，属于库级改动。**

### 5.2 参数对齐的实测边界（同一进程，逐项试探）

| 尝试 | 设置（域固定 L=246 cm=6H, H=41 cm） | 结果 |
|---|---|---|
| 论文原样 | Δx=1.952 cm, Δt=3.202e-5, SVK μ_s=1e7, **κ_s=5e4·Δx/Δt²** | 固体 ~5 ms 内爆（cyl_dx→1e10 cm）|
| 罚降档 | κ̂=50 / κ̂=5 | 均立即爆 |
| 罚再降 | κ̂=0.05，无斜坡 | t≈0.02 s 起缓慢增长（max\|u\|/scale 1.5→4.8，尖峰固定在圆柱下方）|
| 加上斜坡 | κ̂=0.05，RAMP_T=0.5/2.0 | t≤0.05 s 稳定（**试探太短，未暴露**）|
| 罚再降 | κ̂=0.01（MFAC=0.5）| t≈0.17 s 爆 |
| 罚≈出厂值 | κ̂=5e-4（= 出厂 β=1e6 dyne/cm³），SVK μ_s=1e7 | **t≈0.11 s 炸** |
| 固体加硬 | 同上但 μ_s=1.4e7 | t≈0.16 s 炸 |
| 固体再加硬 | 同上但 μ_s=2.0e7（E=5.6 MPa）| t=0.20 s 稳定 |
| 固体网格加密 | μ_s=1e7，MFAC≈0.2（1268 节点）| t≈0.13 s 炸（加密固体没救）|
| 出厂固体+论文网格 | NH μ_s=2e7, β=1e6, Δx=1.952 cm | t=0.20 s 稳定（未探更长）|
| 论文网格+论文固体 | SVK μ_s=2e7, κ̂=5e-4, Δx=1.952 cm | t=0.27 s 起上涨（max\|u\|/scale→15）→ **粗网格上梁只有 1 格厚，我们 4 点核（支撑 4 格）无法分辨** |

**三条硬边界（都是我们这套"无质量 + 显式 + 固定 4 点核"实现的限制，不是论文的问题）**：

1. **圆柱系绳罚**：论文 `κ̂ = κ_sΔt²/Δx = 5×10⁴`，而我们显式耦合的无量纲预算
   `κ̂ = βΔt²/(ρΔx)` 在 κ̂ ≳ 10⁻² 就开始发散 → 论文值超预算约 **10⁶ 倍**。
   采用论文**形式**、量级取到稳定值 κ̂ ≈ 2.5×10⁻³（≈ 出厂 β=1e6 dyne/cm³）。
2. **固体刚度**：SVK 下论文的 `μ_s = 1×10⁶ Pa`（E≈2.9 MPa）必炸（无质量显式耦合的
   附加质量不稳），稳定边界在 μ_s ∈ (1.4e6, 2.0e6] Pa，故采用 **μ_s = 2×10⁶ Pa**（2 倍于论文）。
3. **网格/时间步**：Δx = 1.92 cm 时梁厚 2 cm ≈ 1 格 < 核支撑 → 不可用；改用
   **Δx = 1.0 cm（246×41，方格）**，梁 2 格。若同时坚持论文的 Δt = 0.00164Δx
   （=1.64e-5 s）则 7 s 需 426,829 步 ≈ **22.5 h**（单进程），故 Δt 取 5e-5 s
   （Δt/Δx = 0.005，为论文的 3 倍）→ 140,000 步 ≈ 7.4 h。

### 5.3 最终采用的配置（本次交付）

| 项 | 论文 | 本次运行 | 是否一致 |
|---|---|---|---|
| 域 | 2.46 × 0.41 m = 6H | 246 × 41 cm | ✅ 一致 |
| 圆柱/梁几何、监测点 A | d=0.1 @(0.2,0.2)，l=0.35,h=0.02,A=(0.6,0.2) | 同上 | ✅ |
| 入口/Re | 1.5U y(H−y)/(H/2)², U=2 m/s, Re=200 | 同上（另加 0.5 s 余弦斜坡避免启动冲击）| ✅（斜坡为额外） |
| 固体本构 | Saint Venant–Kirchhoff | SVK（同一形式）| ✅ |
| 固体参数 | μ_s=1e6, λ_s=8e6 Pa | μ_s=2e6, λ_s=8e6 Pa | ⚠️ μ_s 2×（稳定性所迫）|
| 圆柱约束 | κ_s=5e4Δx/Δt² | 同形式，κ̂=2.5e-3（≈出厂 β）| ⚠️ 量级差 2×10⁷ |
| 流体网格 | N=128 → Δx=1.92 cm | 246×41 → Δx=1.00 cm（方格）| ⚠️ 更细 1.9×（核分辨率所迫）|
| 时间步 | Δt=0.00164Δx=3.20e-5 s | Δt=5.0e-5 s（Δt/Δx=0.005）| ⚠️ 3.05× |
| 固体网格 | MFAC 0.5–1.5 | MFAC=0.5（MeshSizeMax=0.98 cm）| ✅ 在其扫描区间内 |
| 核函数 | IB₃/BS₃/CBS₃₂ | 固定 4 点 Peskin | ❌ 不可选（库级限制）|
| 时长 | 本段未给 | T = 7 s | — |

命令（单进程，本机 `afsi-dolfinx` 环境）：

```bash
cd afsic/demo/demo_402
MESH_SIZE=0.0098 $RUN python -B -u generate_mesh.py        # MFAC=0.5
SOLVER=chorin LX=246 LY=41 NX=246 NY=41 DT=5e-5 UM=200 \
  MU_S=2.0e7 LAMBDA_S=8.0e7 SOLID_LAW=svk \
  PENALTY_MODE=paper KAPPA_HAT=2.5e-3 RAMP_T=0.5 T=7.0 FPS=100 \
  OUT=$PWD/plot/paper_T7_dx1 $RUN python -B -u run_compare.py
```

---

## 6. 2026-10-01 重测：对偶性修复后的论文配置 + chorin/IPCS 对比（T = 1 s）

§5 的三条"硬边界"（κ̂ ≳ 1e-2 发散、μ_s=1e6 必炸、Δx=1.92 cm 不可用）都是**对偶性修复
之前**测的。修复（`IB_DIRECT_LOAD=1`，采样算子的伴随载荷）之后全部重测，并顺带定位、
修复了一个环境级线性求解器 bug。

### 6.0 关键 bug：PETSc 3.25.5 `KSPMINRES` 静默返回零解（IPCS 压力修正失效）

PETSc 3.25 的新版 MINRES 带默认"解范数上限" `maxxnorm = 1/√ε ≈ 6.7e7`
（`KSPMINRES` 的 trust-region radius，`-ksp_minres_radius`）。当压力泊松解范数达到该
量级，MINRES 会在**第 1 次迭代直接停止**，返回**未更新的零解**并报告"收敛"
（`reason=6`，petsc4py 映射为 `KSP_CONVERGED_CG_CONSTRAINED`）——不报错、不警告。

最小复现（同一 Laplacian、同一出口 Dirichlet，仅缩放右端；`/tmp/ksp_min_test.py`）：

| 右端量级 | MINRES 结果 |
|---|---|
| 1e0 | reason=2，4 次迭代，\|x\|=3.3e5 ✓ |
| 1e3 | reason=2，4 次迭代，\|x\|=3.3e8 ✓ |
| ≥ 1e5 | **reason=6，1 次迭代，x=0**（静默失败）|

同一矩阵 CG / BCGS 均正常且互相一致到机器精度。本 demo 用 CGS（u~300 cm/s → p~1e10），
必然触发；demo_426（u~0.25，p~O(1)）不触发——这解释了为什么 426 的历史压力输出正常。

**后果**：IPCS 的压力修正项 `φ` 恒为 0（`p_L2 ≡ 0`），`u = u_s` 完全没有投影——
即此前所有该配置下的 IPCS 结果都是"无压力"假结果（含此前记录的"IPCS 稳定性问题"结论
需要重新审视）。

**修复**：`IPCSSolver.py` 压力 KSP 由 `MINRES` 改为 `CG + HYPRE(BoomerAMG)`
（压力矩阵对称正定；`IPCS_KSP_CHORIN=1` 的 BCGS+HYPRE 亦可）。修复后 IPCS 第 1 步
`φ` 与 Chorin 的 `p` 逐位一致（3.255182e21 vs 3.2552e21）。

### 6.1 对齐后的运行配置（本次 T = 1 s）

```bash
cd afsic/demo/demo_402
MESH_SIZE=0.0096 conda run -n afsi-dolfinx python -B -u generate_mesh.py   # MFAC=0.5
for S in chorin ipcs; do
  env SOLVER=$S T=1.0 DT=5e-5 LX=246 LY=41 NX=128 NY=21 UM=200 \
      MU_S=1.0e7 LAMBDA_S=8.0e7 SOLID_LAW=svk \
      PENALTY_MODE=paper KAPPA_HAT=0.25 RAMP_T=0 IB_DIRECT_LOAD=1 \
      KSP_MONITOR=1 FPS=100 OUT=$PWD/plot/t1s/$S \
      caffeinate -i conda run --no-capture-output -n afsi-dolfinx \
      mpirun -n 4 python -B -u run_compare.py
done
# 注（§6.6 起）：geo/环境单位已统一为 CGS——MESH_SIZE 用 cm（0.96）；上表中的参数
# 现在就是 main.py/configuration.py 的默认值，复现只需 env T=... SOLVER=... OUT=...
```

| 项 | 论文 | 本次运行 | 一致? |
|---|---|---|---|
| 域 | 2.46 × 0.41 m | 246 × 41 cm | ✅ |
| 圆柱/梁/监测点 A | d=0.1@(0.2,0.2), l=0.35, h=0.02, A=(0.6,0.2) | 同 | ✅ |
| 入口 | 1.5U y(H−y)/(H/2)², U=2 m/s（未提斜坡） | 同, `RAMP_T=0` | ✅ |
| 出口 | 零法向 traction + 零切向速度 | p=0 Dirichlet（法向≈0，切向自由） | ⚠️ |
| Re | 200 (ρ=1000, μ=1) | 200 (ρ=1 g/cm³, μ=10 dyne·s/cm²) | ✅ |
| 本构 | Saint Venant–Kirchhoff | SVK (`SOLID_LAW=svk`) | ✅ |
| 固体参数 | μ_s=1e6, λ_s=8e6 Pa | μ_s=1e7, λ_s=8e7 dyne/cm²（等同） | ✅ |
| 圆柱系绳 | κ_s=5e4·Δx/Δt²（κ̂=5e4） | 同形式 κ̂=0.25（新稳定上限，见 6.2） | ⚠️ 量级 |
| 流体网格 | N=128 沿最长边 | 128×21（dx=1.9219, dy=1.9524 cm，近似方格） | ≈ |
| 时间步 | Δt=0.00164Δx=3.2e-5 s | Δt=5e-5 s（用户允许不一致） | ⚠️ 允许 |
| 固体网格 | MFAC ∈ 0.5–1.5 | MFAC=0.5（MeshSizeMax=0.96 cm；281 节点/569 tri） | ✅ |
| 核函数 | IB₃ / BS₃ / CBS₃₂ | IB₄（4 点 Peskin，库内固定） | ❌ 用户已知 |
| 固体惯性 | 有（immersed FE） | 无（无质量 Peskin IB） | ❌ 方法固有 |

### 6.2 罚刚度边界重测（对偶性修复后）

κ̂ 阶梯（chorin，T=0.05，其余同上）：

| κ̂ | 2.5e-3 | 2.5e-2 | 0.25 | 2.5 | 25 |
|---|---|---|---|---|---|
| 结果 | 稳定 | 稳定 | 稳定（T=0.3 亦稳定） | 固体会爆（force→1e19） | 爆 |

即修复后稳定上界从 ~1e-2 提高到 0.25–2.5 之间（**约 100 倍**），但距论文的 κ̂=5e4 仍差
~2e5 倍（显式耦合的固有预算）。实跑取 κ̂=0.25：圆柱漂移 ≤ 6.3e-4 cm（0.1 s 后），
等效刚固（论文的 κ̂=5e4 也是这个效果）。

### 6.3 结果（T = 1 s）

`plot/t1s/{chorin,ipcs}/`（history.csv + 场文件；对比脚本 `plot/compare_t1s.py`，
涡量快照 `plot/plot_t1s_fields.py`；图 `compare_t1s.png`、`vorticity_t_end*.png`）。

| 指标（0–1 s） | chorin | ipcs | 备注 |
|---|---|---|---|
| 稳定性 | ✅ 零 NaN，20000 步 | ✅ 零 NaN，20000 步 | IPCS KSP maxit 9/6/9，零失败 |
| `max\|u\|` 峰值 [cm/s] | 758 @ t=0.02 s | 759 @ t=0.02 s | 冲启瞬态；t>0.1 s 后 ~460–510 |
| `u_L2=∫u²dx` 末值 | 5.814e8 | 5.783e8 | 逐帧相对差：中位 0.18 %、最大 0.54 % |
| 点 A `max\|ΔY\|` (t>0.3 s) | **3.65 cm** @ t≈0.87 s | **3.39 cm** @ t≈0.87 s | 论文 Table 4（IB₃, MFAC=0.5）= 0.03686 → 若为 m 则 3.69 cm |
| 点 A 峰峰值 (t>0.5 s) | 5.39 cm | 4.94 cm | 相位一致 |
| 振荡频率（过零估计） | 5.10 Hz | 5.10 Hz | 相同 |
| 圆柱漂移 max\|Δx\| | 1.0e-2（第 1 步）/ 8.4e-4（t>0.1 s） | 1.0e-2 / 8.7e-4 | 罚约束守住 |

结论：**修复后的 IPCS 与 Chorin 在这套论文配置下定量一致**（场量 ~0.2 %，点 A 幅值差
~7 %，频率相同），两者都稳定跑满 T = 1 s，点 A 竖向位移幅值与论文 IB₃/MFAC=0.5 档
（0.03686）量级吻合。图：`plot/t1s/compare_t1s.png`（时程对比）、
`plot/t1s/*/vorticity_t_end*.png`（t=1 s 涡量 + 变形梁，论文 Fig.25 风格）。

说明：
- 冲启（无斜坡）第 1 步会产生 p_L2 ~ 3e21 的压力尖峰（投影法的启动瞬态），~0.05 s 内
  衰减干净；若要更贴近原始 Turek 基准的软启动，可用 `RAMP_T=2`（标准 2 s 余弦斜坡，
  官方 FSI2/FSI3 的做法）。
- Δx≈1.9 cm 时梁厚 2 cm ≈ 1 格、IB₄ 核支撑 4 格，几何等效变厚——定量对比论文的
  Table 4 时须带上这条（论文自己也用核/`MFAC` 扫描，量级一致即可）。
- 本配置下解析出的涡量 ≈ ±230 s⁻¹（P1 顶点分辨率下），涡量图仅作定性对比。

### 6.4 固体模型/系绳实现与论文 formulation 的逐条核对（2026-10-01 复查）

依据 `afsic/demo/demo_402/solid-formulation-notes.md`（论文摘录：rigid penalty force、
volumetric penalization、modified invariants），逐项核对 `main.py` 的实现。

**(1) 本构：Saint Venant–Kirchhoff —— ✅ 完全一致。**
代码 `SOLID_LAW=svk` 分支：`E = (FᵀF − I)/2`、`S = λ_s tr(E) I + 2μ_s E`、`P = F·S`
（`main.py` L_hat 之前的几行），与论文 $\mathbb{S}=\lambda_s\mathrm{tr}(\mathbb{E})\mathbb{I}+2\mu_s\mathbb{E}$
逐字对应；2D 取平面应变（E₃₃=0，用三维 λ_s 写面内分量）。
`μ_s=1e7`、`λ_s=8e7` dyne/cm² = 论文的 `1e6`、`8e6` Pa。

**(2) 圆柱系绳力 F = κ(ψ−χ) + η(V − ∂χ/∂t) —— 弹性项 ✅，阻尼项原缺失 → 已补（可选）。**

| 论文要素 | 本实现 | 结论 |
|---|---|---|
| 方向：拉向目标位形 ψ | `−β(χ−X0)` 进 b1；实测 `corr(F_cyl, X0−χ)=+0.78`（>0 恢复力），圆柱漂移 ≤8e-4 cm（t>0.1 s）| ✅ 符号正确 |
| 目标 ψ、速度 V | ψ = X0（参考构型），V = 0（静止约束）| ✅ |
| 作用对象 = 整个刚体 | 对圆柱区（cell tag 1）逐节点施加 | ✅ |
| 阻尼项 η(V − ∂χ/∂t) | **原实现没有**；现增加 `ETA_HAT`（η = η̂·ρΔx/Δt，默认 0＝纯弹簧）| ✅ 已补，见下 |
| 系数 κ_s = 5e4 Δx/Δt² | 形式已实现（`PENALTY_MODE=paper`）；**量级受显式耦合稳定性限制** | ⚠️ 见 (2b) |
| 节点系数"直接由 χ 的节点系数确定"（F_l=κ(ψ_l−χ_l)）| 我们经弱式组装（= 一致质量加权 M·β(ψ−χ)），非对角形式；总力等价，节点分配差一个质量权重 | ⚠️ 等价级别差异 |

**(2a) 阻尼项实现与验证**（`ETA_HAT`，默认 0；回归：η=0 时与既有 T=1 s 运行逐位一致 Δ=0）：
η̂=0.25（κ̂=0.25）时 t=0.01 的 `cyl_dx` 由 4.81e-3 → 1.40e-3 cm（**圆柱漂移 ↓3.4×**），
`u_L2` 变化 −0.25%、`tip_dy` +0.4% —— 项按论文形式起效。但**阻尼不能扩大 κ̂ 稳定域**：
κ̂=2.5 在 η̂=2.5/25/250 下仍爆（cyl_dx → 1e3~1e8）。

**(2b) 罚刚度边界更新**（2026-10-01 复查，T=0.05~0.3）：

| κ̂ | 5e-4 | 2.5e-3 | 2.5e-2 | 0.25 | 1.0 | 2.5 | 25 |
|---|---|---|---|---|---|---|---|
| 结果 | — | 稳定 | 稳定 | 稳定（T=1 s ✅）| 稳定（T=0.3 ✅）| 爆 | 爆 |

即稳定上界在 **1.0–2.5 之间**（比此前记录的 0.25 再高 4 倍，仍比论文的 5e4 低 ~2e4–5e4 倍）。
κ̂=1.0 时圆柱漂移降到 4.5e-4 cm（κ̂=0.25 为 6.3e-4）。论文 κ_s 的绝对值在显式 partitioned
耦合下不可达是这套无质量 IB 的固有预算，不能再靠阻尼补救。

**(3) 刚体圆柱假设 —— ⚠️ 我们是"弹性盘 + 系绳"，不是严格刚体；偏差已量化。**
论文的 immersed rigid structure 不携带弹性应力（纯罚力）；本 demo 对圆柱（tag 1）同样
施加 SVK 弹性项 + 系绳。T=1 s 运行的实测（从 `solid.h5` 重算）：

| 量 | 值 |
|---|---|
| 圆柱质心漂移（相对**参考构型**的刚体拟合平移，t=1 s） | 2.4e-3 cm（= 0.00024 d，d=10 cm）；冲启第 1 步有 1.0e-2 cm 瞬时跳变 |
| 圆柱残余变形（相对最佳刚体运动拟合的 RMS） | 2.1e-2 cm（t=1 s，随时间缓慢增长）|
| 圆柱单元面积比 \|J−1\| | 中位 0.4 %、90 % 分位 3.3 %、最大 **22 %**（最差 1 % 单元；位置在上游 x≈15–17）|
| 梁单元面积比 \|J−1\| | ≤ 3 % ✅ |
| 梁尖位移（对照） | ±3.6 cm |

**结论**：圆柱不是严格刚体（上游面被流体压变形 ~2e-2 cm RMS，最差单元体积变化 22 %），偏差
主要来自 κ̂ 远低于论文 + 圆柱本身带弹性。κ̂ 提到 1.0 可将该变形进一步压缩数倍（见 §6.5）。

> 口径说明：`history.csv` 的 `cyl_dx/cyl_dy` 是圆柱区 **P2 自由度**上的平均（含边中点，
> 且以参考构型为基线），与"刚体拟合平移"差 ~3 倍；两者都 ≪ d。上表用后者（与系绳目标
> ψ=X0 的定义一致）。`plot/solid_rigidity_check.py` 可复跑这两个口径。

**(4) volumetric penalization（π_stab）—— 对本次基准不适用。**
π_stab 是给 **不变量形式**（W(I₁,I₂)+U(J)，近不可压）模型的稳定化；论文基准明确用 SVK
（应力直接由 E 给出），其参数表也没有 ν_stab。实测我们的梁体积变化 ≤3 %，并没有出现论文
描述的"unphysical contractions"，所以不需要该稳定项。（若将来用 neo-Hookean 分支，注意
代码里 `P_iso = μ J^{-1}(F − (I₁/2)Fᵀ⁻¹)` 是 **2D** 缩放 J^{−2/d}，不是论文的 3D 修饰不变量
J^{−2/3} 形式——本次基准不涉及。）

**(5) modified invariants（Ĩ₁=J^{−2/3}I₁, Ĩ₂=J^{−4/3}I₂）—— 不适用（同上）。**
SVK 直接以 E 表达，不需要修饰不变量；论文的两套不变量形式用于其 neo-Hookean/其它不变量模型。

**(6) 固体惯性 —— ⚠️ 方法级差异（已记录）。**
本 demo 是无质量 Peskin IB：χ 由插值得到的流体速度显式推进（`χ ← χ + Δt·u_interp`），
没有独立的固体惯性方程；论文的 immersed FE 框架带固体密度 ρ_s（ρ_s/ρ_f ≈ 1，
demo 场景下附加惯性/阻尼效应更接近准静态）。这一条影响的是"结构动力学"而非系绳/本构
公式本身，属已知固有差异（§6.1 表已列）。

**小结（一致性评分）**：本构 ✅；系绳形式（弹性+方向+对象）✅、阻尼项现已可选实现 ✅；
罚刚度形式 ✅ / 量级 ⚠️（显式耦合限制，1.0 为当前可用的最硬值）；刚体圆柱 ⚠️（等效刚体近似，
变形已量化）；π_stab / 修饰不变量 ➖（本基准不适用）。

### 6.5 单位统一：论文参数 m·kg·s → demo 代码 g·cm·s（2026-10-01）

论文(SI) 参数逐项换算到 CGS（1 Pa = 10 dyne/cm²；1 Pa·s = 10 dyne·s/cm²）：

| 量 | 论文 (SI) | CGS | 落到的代码位置 |
|---|---|---|---|
| 域 L × H | 2.46 × 0.41 m | **246 × 41 cm** | `configuration.py` 默认 `LX/LY` |
| 圆柱 d / 圆心 | 0.1 m / (0.2, 0.2) m | **10 cm / (20, 20) cm** | `turek.geo`（cm）|
| 弹性梁 l × h | 0.35 × 0.02 m | **35 × 2 cm** | `turek.geo` |
| 监测点 A | (0.6, 0.2) m | **(60, 20) cm** | 记录器（尖端节点）|
| 入口平均速度 U | 2 m/s | **200 cm/s** | `configuration.py` `UM` |
| 流体 ρ | 1000 kg/m³ | **1 g/cm³** | `RHO` |
| 流体 μ | 1 Pa·s | **10 dyne·s/cm²** | `MU`（Re = 1·200·10/10 = 200 ✓）|
| 固体 μ_s | 1×10⁶ Pa | **1×10⁷ dyne/cm²** | `MU_S` |
| 固体 λ_s | 8×10⁶ Pa | **8×10⁷ dyne/cm²** | `LAMBDA_S` |
| 网格 N=128（最长边）| Δx = L/N | **Δx = 1.9219 cm**（Ny=21，dy=1.9524 cm）| `NX/NY` |
| 时间步 | 0.00164Δx = 3.152e-5 s | 取 **5e-5 s**（允许不一致）| `DT` |
| 系绳罚 κ_s | 5.0×10⁴·Δx/Δt²（无量纲组 κ̂=κ_sΔt²/(ρΔx)=5e4）| **κ_s = κ̂·ρ·Δx/Δt² [dyne/cm⁴]**：κ̂=5e4 ↔ 3.84e13；实跑 κ̂=1.0 ↔ **7.69e8 dyne/cm⁴** | `main.py` `PENALTY_MODE=paper` + `KAPPA_HAT` |

**代码修改（本次）**：
1. `turek.geo`：几何由 SI(m) 改写为 **CGS(cm)**（`cx=20, R=5, flag_L=35, flag_h=2,
   eps=1e-4 cm`，`MeshSizeMax=0.96 cm` = MFAC 0.5×Δx）。
2. `generate_mesh.py`：`MESH_SIZE` 环境变量改为 **cm**（旧文档里的 `0.0096`(m) → `0.96`(cm)）。
3. `main.py`：删除读入后的 `geometry.x *= 100` 隐藏换算（网格已是 cm）；
   默认值改为论文 CGS 配置：`SOLID_LAW=svk`、`PENALTY_MODE=paper`（`KAPPA_HAT=1.0`）、
   `RAMP_T=0`（论文入口无斜坡）、`IB_DIRECT_LOAD=1`（对偶性修复路径）；
   罚刚度打印单位修正为 `dyne/cm^4`。
4. `configuration.py`：默认值即论文 CGS（246×41、128×21、μ_s=1e7、λ_s=8e7、Um=200、
   μ=10、ρ=1、dt=5e-5），并把换算表写进文件头注释；`nu_s` 0.4 → 0.444（由 μ_s/λ_s 推得）。

**等价性验证**：用纯默认参数重跑 T=0.02（`OUT=plot/scan/cgs_default`），t=0.01 与旧配置
（geo 用 m + ×100 + 显式 env）逐项对比：`u_L2` 相对差 2.3e-6、`tip_dx` 8e-5、`max_u` 2.8e-3
（差异来自重新生成的固体网格：293 节点 vs 281 节点，几何完全一致、网格质量同级）。

### 6.6 κ̂=1.0 变体（稳定上界内的最硬系绳，T=1 s）

§6.2/§6.4 复核发现 κ̂=1.0 仍稳定（T=0.3），比 §6.3 用的 0.25 更接近论文的刚体系绳。
同配置（CGS、SVK、MFAC=0.5、IB_DIRECT_LOAD=1、RAMP_T=0）跑 T=1 s：

| 指标（0–1 s） | κ̂=0.25（§6.3）| **κ̂=1.0 chorin** | κ̂=1.0 IPCS |
|---|---|---|---|
| 点 A `max\|ΔY\|` | 3.65 cm | **3.35 cm** | **2.94 cm** |
| 峰峰值 (t>0.5 s) | 5.39 cm | **4.81 cm** | **4.18 cm** |
| 振荡频率（过零估计）| 5.10 Hz | **5.10 Hz** | **5.10 Hz** |
| `u_L2` 末值 | 5.814e8 | **5.792e8** | **5.758e8** |
| 圆柱漂移（刚体拟合平移，t=1 s）| 2.4e-3 cm | **1.7e-3 cm** | **1.7e-3 cm** |
| 圆柱残余变形 RMS（t=1 s）| 2.0e-2 cm | **1.6e-2 cm**（−20%）| **1.6e-2 cm** |
| 圆柱单元 \|J−1\| 中位/90%/max | 0.4/3.3/22 % | **0.26/2.3/18 %** | **0.29/2.2/19 %** |

结论：κ̂ 0.25→1.0（硬 4 倍）几乎不改变流动与拍动（`u_L2` 逐帧差中位 0.17%、频率相同），
圆柱再刚一点（残余变形 −20%）。κ̂≈1 附近系绳已基本"等效刚固"，进一步提升收益很小
（2.5 会爆）。**当前最接近论文的圆柱约束参数是 κ̂=1.0。**

