# Demo 336 — 方腔驱动圆盘（multi-direct-frocing-elastic：弹性固体 direct-forcing 版）

在 `multi-direct-frocing`（mdf 刚体版）基础上**增加固体的推进方程**，并采用
**真正的 direct-forcing**：圆盘是**带惯性的弹性固体**（可压缩 neo-Hookean），
流体在标记处被直接力强制匹配固体速度（无滑移），反作用力（added-mass 项）喂回
固体动量方程。

## 固体方程（动量方程 + 本构）

- **动量方程（总拉格朗日，带惯性）**：
  $$\rho_s\frac{\partial^2\mathbf X_s}{\partial t^2}=
  \nabla_X\cdot\mathbf P(\mathbf F)+\mathbf f^{\text{fluid}\to\text{solid}}$$
- **本构（可压缩 neo-Hookean + Kelvin-Voigt 粘性）**：
  $$\mathbf P(\mathbf F)=\mu_s(\mathbf F-\mathbf F^{-T})+
  \lambda_s\,\ln(\det\mathbf F)\,\mathbf F^{-T},\qquad \mathbf F=\nabla\mathbf X_s$$
  $$\sigma^{\text{visc}}=2\,\mu_s^{\text{visc}}\,\operatorname{sym}(\nabla V_s),
  \qquad \mu_s^{\text{visc}}=\mu_f=0.01 \text{（默认，与 IBFE readme 一致）}$$
  注：标准 IBFE 常把固体当粘弹性体（弹性 + 与流体相同的粘性）；本仓库 demo_402
  的 readme 也写明 "the solid is added with the same viscosity with fluid"（但其代码
  实际只实现了弹性 + 惩罚，未真正加粘性）。本版显式实现了该粘性项，可用
  `MU_S_VISC` 覆盖（设 0 关闭 = 纯弹性，运动更快）。
- **无滑移（direct-forcing 约束）**：
  $$\mathbf u(\mathbf X_s)=\frac{\partial\mathbf X_s}{\partial t},\qquad
  \mathbf F_{IBM,l}=\frac{\mathbf V_{s,l}-\mathbf U_l}{\Delta t}$$

流体求解器与 mdf 版一致：AB2 预测 + 压力泊松 $\nabla^2p=\frac{2}{3\Delta t}\nabla\cdot U$
+ L2 投影；闭合方腔，顶盖 $U=(1,0)$ 滑动，压力钉角点。

## 每步算法（分区显式，固液耦合在固体速度更新中隐式处理）

```
1) 流体预测（AB2，无 IBM 力）                      → u*
2) 插值 u* 到固体节点                              → U_l
3) 弹性内力 F_int = ∫P(F):∇δv dX
4) 固体推进（added-mass 隐式，稳定）:
     无粘性: (M_HRZ + ρ_f·diag(V_node)) V_s^{n+1}
                = M_HRZ V_s^n + ρ_f·diag(V_node)·U_l − dt·F_int
     有粘性: (diag(M_HRZ+ρ_f V_node) + dt·K_visc) V_s^{n+1} = 同上右端
         （K_visc = ∫2μ_s^visc sym(∇δu):sym(∇δv)；粘性必须隐式，显式会失稳）
     X_s^{n+1} = X_s^n + dt·V_s^{n+1}
5) 直接力（目标 = 固体速度）:
     F_IBM = (V_s^{n+1} − U_l)/dt · ΔV_l → 扩散 → f_IBM
     u = u* + dt·f_IBM
6) 压力泊松 + L2 投影
```

关键点：
- 固体**带惯性**（$\rho_s$），速度 $V_s$ 由动量方程解出，不再是"无质量随流体平流"；
- **added-mass 项 $\rho_f V_{\text{node}}$** 把固液耦合在 $V_s$ 更新里隐式处理 →
  稳定（对 $\rho_s\sim\rho_f$ 无需子迭代）；
- 质量用 **HRZ 正定集中**（P2 行求和集中会给顶点≈0/负质量，不能用）；
- 固体网格用**准均匀点 + Delaunay**（避免圆心细长三角扇——那种极细单元对抖动
  极敏感、易翻转）。

## 固体与流体方程是否相对解耦？

**是。** 本方案是**分区显式（partitioned / staggered）耦合**：每步先独立推进流体，
再独立推进固体（动量方程），只通过界面插值（$U_l$）与直接力扩散（$f_{IBM}$）交换
一次信息，不把两者联立成一个方程组（monolithic 求解）。

注意：这里的"隐式"只在**固体速度子步内部**（把 $\rho_f V_{\text{node}}$ 放在
$V_s^{n+1}$ 一侧），不是固液联立求解——固液仍是分区、相对解耦的。

## 文件

| 文件 | 说明 |
|------|------|
| `configuration.py` | 方腔/弹性圆盘/本构参数，`STEPS`/`NX`/`NY` 覆盖 |
| `main.py` | AB2 流体 + 惯性弹性固体 direct-forcing 时间循环 |
| `output/` | `velocity.xdmf/.h5`、`pressure.xdmf/.h5`、`solid.xdmf`(位移场)、`solid_force.xdmf`、`forces.csv` |

## 运行

```bash
conda activate afsi-dolfinx
cd afsic/demo/demo_336/multi-direct-frocing-elastic
python main.py                 # 默认 400 步 (T=1.0 s)
STEPS=100 python main.py       # 短程冒烟验证
```

ParaView：`solid.xdmf` 用 **Warp by vector**（位移场 `u`）查看圆盘变形；
`solid_force.xdmf` 查看直接力/节点力。

## 默认参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 方腔 | $1\times1$ | $N_x=N_y=128,\ h\approx7.8$ mm，Re=100 |
| $\Delta t$ | 0.0025 s | |
| 弹性圆盘 | $(0.6,0.5),\ r=0.2$ | P2，~433 节点（Delaunay 三角化） |
| $\rho_s$ | 1.0 | 固体密度（= ρ_f，added-mass 稳定） |
| $\mu_s$ / $\lambda_s$ | 0.05 / 0.5 | 可压缩 neo-Hookean |
| $\mu_s^{\text{visc}}$ | 0.01 | Kelvin-Voigt 固体粘性（= 流体 μ；`MU_S_VISC` 覆盖，0=关） |

## 验证结果（128×128，默认参数）

- **600 步（t=1.5s）**：全程无 NaN；流场正常发展（u_L2→0.059，与刚体版 t=1.5s
  ≈0.054 一致）；固体平滑变形（|disp|max 0→0.22），`det_min` 缓慢降至 0.956
  （未翻转），体积近似守恒（膨胀 <0.7%），直接力 Fx 收敛→0。
- **对比修复前**：原"无质量运动学平流"版 t≈0.1~1.2s 即网格翻转出 NaN；本版带
  惯性 + added-mass 隐式 + Delaunay 网格后稳定窗口大幅延长。

## 固体对流体的影响（back-effect）——为什么看起来"被流体带动"

实测（t=1.5s，600 步，u_L2 = 全局流体动能）：

| 情形 | u_L2 | 相对无固体 | 说明 |
|------|------|-----------|------|
| 纯方腔（无固体） | 0.0612 | — | 参照 |
| 软固体（$\rho_s=1$，默认） | 0.0588 | **−4%** | 轻软体基本随流走，反作用小 |
| 重固体（$\rho_s=50$） | 0.0484 | **−21%** | 重物不易被拖动→挡流，反作用明显 |

**结论**：不是没有固体对流体的影响，而是**轻软体($\rho_s\sim\rho_f$、小 $\mu_s$)本来就被
流场带走**，$F_{IBM}=(V_s-U_l)/dt\approx0$，反作用趋近于零——这是正确物理（"柔性体随流"
现象）。要让圆盘**明显影响流体**（像障碍物挡流），把密度比调大即可：

```bash
RHO_S=50 python main.py     # 重固体：u_L2 比无固体低 21%，有持续 Fx
RHO_S=1  python main.py     # 默认：轻软体随流（back-effect 小）
SOLID_ACTIVE=0 python main.py  # 纯方腔参照
```

默认取 $\rho_s=1$（轻软体、back-effect 小但演示"随流变形"）；想看强双向耦合用
`RHO_S` 调大。

## 为什么 DF 圆盘比 IBFE 运动更快（固体粘性）

IBFE（以及本仓库 demo_402 的 readme 表述）把固体当作**粘弹性**体：弹性 + 与流体
相同的粘性（$\mu_s^{\text{visc}}=\mu_f=0.01$）。direct-forcing 版默认没有固体粘性
（纯弹性）。本版已实现 Kelvin-Voigt 粘性（`mu_s_visc`，默认 0.01=流体 μ，可关）。

**A/B 实测（$\rho_s=1/\mu_s=0.2$，128×128，5s）：**

| 量 | 无粘性 (visc=0) | 粘性 0.01 | 说明 |
|----|-----------------|-----------|------|
| 顶边触壁 (y>0.99) | t=3.14s | t=3.16s | 几乎相同 |
| 质心 @4.5s | (0.433,0.873) | (0.425,0.872) | 差异 <0.01 |
| 结局 | **t=4.89s 网格翻转 NaN** | **跑满 5s, det_min=0.70** | 关键差异 |

**结论（修正直觉）：**
- **$\mu_s^{\text{visc}}=0.01$ 对圆盘"整体运动"几乎无影响**（轨迹重合）。原因：固体
  粘性阻尼的是**内部变形速率**（偏量应变率 $\operatorname{sym}(\nabla V_s)$），而圆盘
  主要被主涡**近似刚体地平动/公转**，几乎不产生应变率 → 粘性应力≈0 → 不减速。
  所以"粘性让固体跟随更慢"在 0.01 这个量级**不成立**。
- **它真正的作用是稳定性**：抑制变形场高频分量 → 无粘性在 t=4.89s 单元翻转出 NaN
  （$\det F\to0$），加粘性后**撑过 4.89s 跑满 5s**（det_min 仍 0.70，未翻转）。
- 因此 **IB 与 DF 触壁时间差（IB 顶部 0.98@5s vs DF >1.0@4.8s）不是由 0.01 固体粘性
  造成的**，更可能来自分辨率（IB 用 64×64 vs DF 128×128）与固体推进方式
  （IB 是运动学平流 vs DF 动量方程）等差异。

> 稳定性注意：Kelvin-Voigt 粘性项**必须隐式处理**（并入固体更新 LHS 解小系统）。
> 显式（forward Euler）处理会在几步内放大高频速度分量导致网格翻转（实测
> $t\approx0.02$s 即翻）。

## 已知局限

- **软弹性体在持久剪切流中持续拉伸**：方腔主涡持续剪切圆盘，软材料（本默认参数）
  会被不断拉伸（|disp|max 随时间近似线性增长），最终仍可能单元翻转（$\det F\to0$）
  → NaN。`main.py` 检测到 NaN 会优雅终止并提示；增大 $\mu_s$（更硬）或缩短 $T$ 可
  缓解拉伸，但更硬时弹性波 CFL 更紧、added-mass 更敏感，需调小 $\Delta t$。
- **近不可压缩 + P2 体积锁定**：$\lambda_s$ 过大时 $P_2$ 位移单元体积锁定，可用
  F-bar 或混合 u-p 单元缓解。
- **流体域从原点出发、单进程、IBM 力积分 $\propto h$ 不收敛**：同 mdf/刚体版
  （见 `multi-direct-frocing/readme.md`）。

## 后续可扩展

- **半隐式/强耦合（子迭代或 monolithic）**：每步内重复"插值-推进-施力"数次，或
  把固液整体联立求解，彻底去除显式耦合限制
- **F-bar / 混合 u-p**：缓解近不可压缩体积锁定，允许更大的 $\lambda_s$
- **固液密度比 $\rho_s/\rho_f$ 扫描**：验证 added-mass 隐式处理的适用范围
