# Demo 336 — 方腔驱动圆盘（multi-direct-frocing-elastic：弹性固体 direct-forcing 版）

在 `multi-direct-frocing`（mdf 刚体版）基础上**增加固体的推进方程**，并采用
**真正的 direct-forcing**：圆盘是**带惯性的弹性固体**（可压缩 neo-Hookean），
流体在标记处被直接力强制匹配固体速度（无滑移），反作用力（added-mass 项）喂回
固体动量方程。

## 固体方程（动量方程 + 本构）

- **动量方程（总拉格朗日，带惯性）**：
  $$\rho_s\frac{\partial^2\mathbf X_s}{\partial t^2}=
  \nabla_X\cdot\mathbf P(\mathbf F)+\mathbf f^{\text{fluid}\to\text{solid}}$$
- **本构（可压缩 neo-Hookean）**：
  $$\mathbf P(\mathbf F)=\mu_s(\mathbf F-\mathbf F^{-T})+
  \lambda_s\,\ln(\det\mathbf F)\,\mathbf F^{-T},\qquad \mathbf F=\nabla\mathbf X_s$$
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
     (M_HRZ + ρ_f·diag(V_node)) V_s^{n+1}
         = M_HRZ V_s^n + ρ_f·diag(V_node)·U_l − dt·F_int
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

## 验证结果（128×128，默认参数）

- **600 步（t=1.5s）**：全程无 NaN；流场正常发展（u_L2→0.059，与刚体版 t=1.5s
  ≈0.054 一致）；固体平滑变形（|disp|max 0→0.22），`det_min` 缓慢降至 0.956
  （未翻转），体积近似守恒（膨胀 <0.7%），直接力 Fx 收敛→0。
- **对比修复前**：原"无质量运动学平流"版 t≈0.1~1.2s 即网格翻转出 NaN；本版带
  惯性 + added-mass 隐式 + Delaunay 网格后稳定窗口大幅延长。

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
