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
