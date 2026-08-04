# Demo 336 — 方腔驱动圆盘（multi-direct-frocing-elastic：弹性固体版）

在 `multi-direct-frocing`（mdf 刚体版）基础上**增加固体的推进方程**：圆盘不再是
刚体，而是一块**弹性固体**（可变形、有本构），由方腔流场驱动、变形并反馈力给流体。

## 固体方程（推进方程 + 本构）

- **推进方程（运动学平流，每步显式一步）**：
  $$\mathbf X_s^{n+1} = \mathbf X_s^n + \mathbf V_s^n\,\Delta t,\qquad
  \mathbf V_s^n = \text{流体速度插值到固体节点 (fluid\_to\_solid)}$$
- **本构（可压缩 neo-Hookean 型，总拉格朗日）**：
  $$\mathbf P(\mathbf F) = \mu_s(\mathbf F-\mathbf F^{-T}) +
  \lambda_s\,\ln(\det\mathbf F)\,\mathbf F^{-T},\qquad \mathbf F=\nabla\mathbf X_s$$
- **节点力（弱形式，对参考构型积分）**：
  $$\mathbf F_{\text{solid}} = -\int \mathbf P(\mathbf F):\nabla\delta\mathbf v\,
  \mathrm dX$$
  组装后经 `solid_to_fluid` 扩散回流体体积力场 $f_{\text{IBM}}$，流体获得弹性冲量
  $U = U^* + \Delta t\,f_{\text{IBM}}$。

流体求解器与 mdf 版一致：AB2 预测 + 压力泊松 $\nabla^2p=\frac{2}{3\Delta t}\nabla\cdot U$
+ L2 投影；闭合方腔，顶盖 $U=(1,0)$ 滑动，压力钉角点。

## 固体与流体方程是否相对解耦？

**是。** 本方案是**分区显式（partitioned / staggered）耦合**：

1. 先独立推进流体（AB2 预测，此时不含固体力）；
2. 再把流体速度插值到固体 → 独立推进固体（运动学平流）→ 由本构算弹性力；
3. 最后把弹性力扩散回流体（替换式 $f_{\text{IBM}}$）→ 投影修正。

每步内**先流体后固体依次求解，只通过界面插值/力扩散交换一次信息**，不把两者联立成
一个方程组（monolithic 求解）——这就是"相对解耦"。

代价：显式耦合对**重固体（$\rho_s\gg\rho_f$）或硬/近不可压缩固体**存在失稳风险
（added-mass / 网格翻转），见下方「已知局限」。

## 文件

| 文件 | 说明 |
|------|------|
| `configuration.py` | 方腔/弹性圆盘/本构参数，`STEPS`/`NX`/`NY` 覆盖 |
| `main.py` | AB2 流体 + 弹性固体推进/弹性力耦合时间循环 |
| `output/` | `velocity.xdmf/.h5`、`pressure.xdmf/.h5`、`solid.xdmf`(位移场)、`solid_force.xdmf`、`forces.csv` |

## 运行

```bash
conda activate afsi-dolfinx
cd afsic/demo/demo_336/multi-direct-frocing-elastic
python main.py                 # 默认 400 步 (T=1.0 s)，软化本构
STEPS=100 python main.py       # 短程冒烟验证
```

ParaView：`solid.xdmf` 用 **Warp by vector**（位移场 `u`）查看圆盘变形；
`solid_force.xdmf` 查看节点弹性力。

## 默认参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 方腔 | $1\times1$ | $N_x=N_y=128,\ h\approx7.8$ mm，Re=100 |
| $\Delta t$ | 0.0025 s | |
| 弹性圆盘 | $(0.6,0.5),\ r=0.2$ | P2，769 节点 / 1408 三角 |
| $\mu_s$ / $\lambda_s$ | 0.02 / 0.2 | 软化本构（稳定窗口更长、变形明显） |
| 原始刚度 | 0.1 / 10.0 | 与原始 demo_336 一致，但稳定窗口短 |

## 验证结果（128×128，软化本构）

- 600 步（t=1.5s）测试：流场正常发展（u_L2→0.058），固体随流变形
  （|u|max 从 0 → 0.19），体积守恒（vol≈0.1256 恒定），t≈1.23s 前无 NaN。
- 原始刚度（0.1/10.0）：t≈0.1~0.4s 即网格翻转出 NaN（见局限）。

## 已知局限

- **无质量弹性固体 + 分区显式耦合的固有失稳**：固体节点逐点随流体速度平流，在
  方腔剪切流中被持续剪切，最终某单元翻转（$\det F\to0$）→ `ln/inv` 出 NaN。
  硬/近不可压缩时更早（原始参数 t≈0.1~0.4s），软化后 t≈1.2s。仓库对
  demo_339 `ibfe` 已记录同类现象（"显式耦合 ~200 步后固体力 NaN，固有特性"）。
  `main.py` 检测到 NaN 会优雅终止并提示；调软本构或缩短 $T$ 可延长稳定窗口。
- **无固体惯性**：推进方程是运动学平流（$\mathbf X_s += \mathbf V_s\Delta t$），
  不求解质量-加速度（$\rho_s\int\delta v\cdot\ddot X$）。这与原始
  `fsi_paralell.py` 一致；若要加惯性（更稳、可承受重固体），需在固体方程中加入
  质量项并做 Newmark/中心差分时间积分（见「后续可扩展」）。
- **流体域从原点出发、单进程、IBM 力积分 $\propto h$ 不收敛**：同 mdf/刚体版
  （见 `multi-direct-frocing/readme.md`）。

## 后续可扩展

- **加固体惯性**：质量矩阵 + 显式中心差分/Newmark，固液密度比可调，稳定性大幅提升
- **半隐式/强耦合（monolithic 或子迭代）**：每步内迭代几次"插值-推进-施力"，
  缓解显式失稳
- **F-bar / 体积锁定缓解**：近不可压缩时用 F-bar 或混合 u-p 单元避免 $P_2$ 单元
  体积锁定与翻转
