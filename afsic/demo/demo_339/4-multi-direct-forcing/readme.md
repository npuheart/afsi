# Multi-direct forcing（迭代直接力法）圆柱绕流 — DFIBMFoam 算法移植

在 AFSI（FEniCSx/dolfinx）中复现 **MsureCFD/DFIBMFoam**（Mi et al. 2025,
*Computers and Electronics in Agriculture* 237: 110625）的
**multi-direct forcing（迭代直接力）IBM** 算法。几何/物理参数与 `demo_339`
其余实现统一：DFG 2D-3、Re=100、固定圆柱（SI 单位）。

## 算法（与 DFIBMFoam 逐条对应）

### 1. 预测步（AB2 对流 + 3/2-1/2 半隐式扩散）
$$
\frac{U^*-U^n}{\Delta t}
+ \frac{3}{2}(U^n\cdot\nabla)U^n - \frac{1}{2}(U^{n-1}\cdot\nabla)U^{n-1}
= \frac{3}{2}\nu\nabla^2 U^* - \frac{1}{2}\nu\nabla^2 U^n + \frac{1}{2}\nabla p^n
$$

### 2. Multi-direct forcing（每步 `n_iter=10` 次迭代，体积力 `IBMf` 累加）
对第 $k$ 次迭代：
$$
\tilde U^{(k)} = U^* - \tfrac{3}{2}\Delta t\,\nabla p^n + \Delta t\,\mathbf{f}^{(k-1)}
$$
$$
U_l^{(k)} = \sum_{ij}\tilde U^{(k)}_{ij}\,\phi\!\Big(\tfrac{r_x}{\Delta x}\Big)\phi\!\Big(\tfrac{r_y}{\Delta y}\Big)
\qquad\text{(Peskin 4点δ核插值)}
$$
$$
\mathbf F_l^{(k)} = \frac{\mathbf U_l^d - U_l^{(k)}}{\Delta t},
\qquad
\mathbf f^{(k)}_{ij} = \mathbf f^{(k-1)}_{ij} + \sum_l \mathbf F_l^{(k)}\,\phi_x\phi_y\,\frac{\Delta V_l}{\Delta x\Delta y}
$$
其中拉格朗日体积 $\Delta V_l=\Delta s_l\sqrt{\Delta x\Delta y}$（$\Delta s_l$ 为弧长）。
**插值用的是 $\tilde U$ 而非 $U^*$**——已把待施加的压力梯度减掉，无滑移满足得更好。

### 3. 投影步
$$
U = U^* + \Delta t\,\mathbf{f}_{\mathrm{IBM}},\qquad
\nabla^2 p^{n+1} = \frac{2}{3\Delta t}\nabla\cdot U,\qquad
U^{n+1} = U - \tfrac{3}{2}\Delta t\,\nabla p^{n+1}
$$
（速度修正用 L2 投影保持 Dirichlet 边界；DFIBMFoam 里是直接代数减法）

## 与 demo_339 其他实现的区别

| | 本 demo (mdf) | `direct_forcing/`（旧） |
|---|---|---|
| 时间推进 | AB2 分步（2 阶） | 一阶 Chorin |
| 施力 | **真体积力进入动量方程**（迭代累加） | 后验速度掩码（力被 α 归一化压没） |
| 核 | Peskin 4点 δ（复用 afsic） | 同左 |
| 标记 | 圆柱**边界**标记 + Δs·h 体积加权 | 圆盘内部全部 DOF |
| 压力 | 压力泊松**看见**圆柱（∇·U 驱动） | 压力看不见圆柱 |
| Cd | 体积力积分（量级合理但**网格依赖**，见下文收敛性验证） | 标记求和代理（8× 虚高） |

## 文件

| 文件 | 说明 |
|------|------|
| `configuration.py` | 统一 SI 参数 + IBM 参数（n_markers/n_iter） |
| `main.py` | AB2 分步求解器 + multi-direct forcing 时间循环 |
| `output/` | 运行输出：`velocity.xdmf/.h5`、`pressure.xdmf/.h5`（可导入 ParaView） |
| `readme.md` | 本文档 |

## 运行

```bash
conda activate afsi-dolfinx
cd afsic/demo/demo_339/4-multi-direct-forcing
python main.py                 # 完整 10000 步 (T=10 s)，输出到 output/
STEPS=2500 python main.py      # 短程验证 (t=2.5 s)
```

## 已知局限（继承自 DFIBMFoam / afsic IBM）

- **必须均匀笛卡尔网格**（`IBMesh` 结构化索引，与 DFIBMFoam `findCell_` 同理）
- **单进程**（`IBMesh` 构建的 hash→dof 映射假设全局编号；dolfinx MPI 多进程下需重写）
- **固定圆柱**（U^d=0）。DFIBMFoam 的振荡/游动运动学可通过改 `desiredIbpVel` 接入
- 边界-only 标记（忠实 DFIBMFoam）；圆柱内部速度场不受显式约束

## 验证记录（历史，N=2500 步，t=2.5s，与 body_fitted 对比）

| 量 | body_fitted | mdf (disk) | mdf (boundary) |
|----|-------------|-----------|----------------|
| $u_{L2}$ | 0.9728 | 1.1027 | 1.1051 |
| $p_{L2}$（运动学） | 0.1319 | 0.0841 | 0.0879 |
| $C_d$ | 2.664 | 4.091 | 4.388 |

中心线 $y=0.2$ 探针（disk 模式）：

| x | body $u_x$ | mdf $u_x$ | 备注 |
|---|-----------|-----------|------|
| 0.18（圆柱内） | 洞(NaN) | 0.188 | 内部未完全清零 |
| 0.30（尾流） | +0.046 | −0.154 | 回流区偏长 |
| 0.40 | +0.961 | −0.213 | 差异大 |
| 0.50 | 1.045 | 0.020 | |
| 1.00 | 1.061 | 1.077 | 远场接近 |
| 1.50 | 1.061 | 1.264 | 下游恢复偏快 |

### 结论（如实）

- **算法移植正确**：AB2 分步 + multi-direct forcing 运行稳定，体积力进入动量方程，
  $C_d$ 量级物理合理（4.1，对比旧 `direct_forcing` 的 24 是质的改善），压力为运动学压力。
- **量化上尚未与 body_fitted 吻合**（$C_d$ 偏高 ~54%）：
  1. **圆柱内部未完全置零**（disk 模式中心仍 ~0.19）——迭代次数/标记密度不够，或投影步扰动；
  2. **漫界面效应**（Peskin 4点核支撑 ~4 格）使有效圆柱偏大 → 阻力偏高；
  3. **下游速度恢复过快**（x=1.5 处 1.26 vs 1.06），可能源于内部泄漏或质量守恒细节。
- 这是 **diffuse-interface IBM 的固有特性**（不如贴体网格锐利），DFIBMFoam 原版用于
  振荡/细长体时影响较小；对固定粗圆柱基准需调参或加内部掩码。

## Cd 收敛性验证（与官方贴体教程对比，2026-08 结论）

历史验证（2026-08）：把本 demo 的 multi-direct forcing 求解器与官方 dolfinx 贴体教程
（CN+AB2 IPCS，**表面应力积分** Cd）在**相同物理**下对比
（ρ=1、μ=0.001、Re=100、sin 入口 $U=1.5\sin(\pi t/8)$、dt=0.001、t=0.3s、300 步）：

| 分辨率 | mdf Cd（体积力积分） | tutorial Cd（表面应力积分） | uL2 差 | 尾流 x=0.3 速度差 |
|---|---|---|---|---|
| 110×21 | 0.439 | 0.291 | 2.1% | −23.8% |
| 220×41 | 0.268 | 0.292 | 1.6% | −11.4% |
| 440×82 | 0.157 | 0.292 | 1.4% | −6.6% |

**结论**：

1. **速度场收敛且两方法一致**：主流区差 ~1–3%，随加密趋近；尾流 x=0.3 差随加密收敛
   （−23.8% → −6.6%）。说明本 demo 的流场求解（IBM 无滑移施力）是可靠的。
2. **体积力积分 Cd 不收敛**：mdf 的 Cd 随加密下降（0.439→0.268→0.157，趋向 0），
   而 tutorial 稳定在 0.29。
3. **根因（不是流场错，是 Cd 算法错）**：$C_d=-2\int f_{\mathrm{IBM}}\,dV/(\bar U^2 D)$，
   每标记权重 $dV=\Delta s\cdot h$，所有标记 $\sum_l dV = 2\pi r\cdot h \propto h$，
   **总力积分正比于网格尺度 $h$** → 网格越细 Cd 越小。这是 **diffuse-interface IBM
   力积分的固有网格依赖**（Mittal & Iaccarino 2005 综述），同样影响 `main.py` 的 Cd 输出。

**Cd 的正确算法（对 IBM）**：

- ✅ **控制体积动量平衡**（momentum-deficit）：取包围圆柱的矩形控制体，积分进出动量通量
  + 表面压力 + 粘性 → 阻力 = 动量收支差。不依赖 IBM 力，网格无关，**最推荐**。
- ✅ **包络面应力积分**：在圆柱外取半径 $r+\delta$ 的包络面（$\delta$ 大于扩散界面宽度），
  在面上积压力 + 粘性应力。
- ✅ **表面应力积分**（需贴体网格，tutorial 用此法）。
- ❌ **直接力积分 $\int f_{\mathrm{IBM}}\,dV$**：当前 `main.py` 所用，网格依赖（∝h），
  不推荐作为定量 Cd。

> 因此 `main.py` 打印的 Cd 只适用于**固定网格下的量级参考**（220×41 时 2.53 在合理量级），
> 不能作为网格无关的定量结果；定量 Cd 应改用控制体积动量平衡或包络面应力积分。

### 可调参数（见 configuration.py）

| 参数 | 作用 |
|------|------|
| `marker_mode` | `"boundary"`（边界环，DFIBMFoam 原版）或 `"disk"`（填充圆盘） |
| `marker_spacing_h` | 内部标记间距（默认 0.5h；disk 模式） |
| `n_iter` | 每步施力迭代（默认 10） |
| `mask_interior` | **每步把圆柱内部速度硬置零（默认 True）**——保证实体固体，否则流体穿过"空心"圆柱，抑制卡门涡街 |

---

## 卡门涡街（10 s 运行，T=10 s / 10000 步）

**关键修复**：`mask_interior=True`。此前（仅 boundary 标记）圆柱内部漏流
（中心 u≈0.19-0.61），流体穿过空心圆柱，边界层分离和涡脱落被破坏，**没有涡街**。
加内部掩码后圆柱为实体，**涡街出现**。

t>6s 准稳态段（`output/velocity.xdmf` 尾流探针 u_y 时间序列）：

| 尾流探针 (y=0.2) | u_y 峰峰 | 周期 | St |
|------------------|----------|------|-----|
| x=0.5 | 0.110 | 0.190s | 0.526 |
| x=0.6 | 0.126 | 0.193s | 0.519 |
| x=0.8 | 0.113 | 0.193s | 0.519 |

- **Cl 振荡**：均值 −0.053、半幅 ≈0.015（DFG 参考 Cl 幅值 0.0106，量级吻合）
- **Cd ≈ 2.53**（几乎恒定）、**St ≈ 0.52**
- ⚠️ 与 DFG 2D-3 参考（Cd≈5.57、St≈0.3）有差距。

### 与 body_fitted 10s 对比（同网格 220×41，均跑满 10 s）

| 指标 | body_fitted（一阶 Chorin） | mdf（AB2 + 直接力） |
|------|----------------------------|---------------------|
| 卡门涡街 | **不明显**：尾流 u_y 始终为负（无交替），峰峰仅 ~0.005 | **清晰**：u_y 正负交替，峰峰 0.126 |
| 数值稳定性 | t>6.8s 周期性爆表（u_y 尖峰达 **-5811 m/s**，Cd 尖峰 1e91） | 全程稳定 |
| Cd | ~2.2-3.7（含尖峰，不可靠） | 2.53（恒定） |
| St | 无法判定（无干净涡街） | 0.52 |

**结论**：在这套 220×41 粗网格 + 一阶 Chorin 下，body_fitted **反而不能形成干净的卡门涡街**
（数值耗散大 + 后期周期爆表）；mdf（AB2 二阶 + 直接力）**涡街更清晰、更稳定**。
两者都达不到 DFG 精细网格参考（Cd 5.57 / St 0.3）——这是**粗网格 + 低阶的共性限制**，
需加密网格（如 440×82）才能定量逼近。
