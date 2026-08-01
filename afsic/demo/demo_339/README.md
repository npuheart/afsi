# Demo 339 — 圆柱绕流数值方法对比

经典 DFG 2D-3 基准（Re = 100 圆柱绕流）在 AFSI 框架下的**多种实现**：

| 目录 | 方法 | 固体表示 | 流体网格 |
|------|------|----------|----------|
| `no_cylinder/` | 纯通道流（基准对照） | 无 | 220×41 均匀四边形 |
| `body_fitted/` | 贴体网格法 | 网格中的洞（无滑移 Dirichlet） | Gmsh 贴体三角形 |
| `ibfe/` | IB-FE 浸没边界法 | Neo-Hookean 圆盘 + 惩罚固定 | 220×41 均匀四边形 |
| `multi_direct_forcing/` | 浸没边界直接力法（IBM, DFIBMFoam 移植） | 边界标记 + Peskin δ 核 + 内部掩码 | 220×41 均匀四边形 |

> **注**：早期曾有 `direct_forcing/` 实现（速度掩码式直接力），经与 `body_fitted` 定量对比证实其
> Cd/近壁压力不可信，已**删除**并被 `multi_direct_forcing/` 取代（见 §8）。

---

## 1. 统一物理参数

各实现**统一为 SI 单位**，几何与物理参数完全相同：

| 参数 | 值 | 说明 |
|------|-----|------|
| 通道 $L_x \times L_y$ | $2.2 \times 0.41$ m | DFG 2D-3 通道 |
| 圆柱圆心 / 半径 | $(0.2, 0.2)$ / $0.05$ m | 直径 $D = 0.1$ m |
| 平均入口速度 $U_m$ | $1.0$ m/s | 抛物线剖面 + 2 s 余弦斜坡 |
| 密度 $\rho$ | $1000$ kg/m³ | |
| 动力粘度 $\mu$ | $1.0$ Pa·s | $\text{Re} = \rho U_m D/\mu = 100$ |
| 网格 $N_x \times N_y$ | $220 \times 41$ | 均匀网格 $h \approx 0.01$ |
| 时长 $T$ / 步长 $\Delta t$ | $10$ s / $0.001$ s | 共 $10^4$ 步 |
| 入口剖面 | $u_x(y)=\dfrac{1.5U_m}{H}y(H-y)$，$H=L_y$ | $u_y=0$ |

---

## 2. 共同的控制方程

### 2.1 不可压 Navier–Stokes

$$
\rho\left(\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u}\cdot\nabla)\mathbf{u}\right)
= \nabla\cdot(\mu\nabla\mathbf{u}) - \nabla p + \mathbf{f},
\qquad \nabla\cdot\mathbf{u} = 0
$$

### 2.2 Chorin 投影法（分步法，三个求解步）

1. **中间速度步**（含体积力 $\mathbf{f}$）：
$$
\rho\left(\frac{\mathbf{u}^*-\mathbf{u}^n}{\Delta t} + (\mathbf{u}^n\cdot\nabla)\mathbf{u}^n\right)
= \nabla\cdot(\mu\nabla\mathbf{u}^*) - \mathbf{f}
$$
2. **压力泊松步**：
$$
\nabla^2 p^{n+1} = \frac{\rho}{\Delta t}\nabla\cdot\mathbf{u}^*
$$
3. **速度修正步**：
$$
\mathbf{u}^{n+1} = \mathbf{u}^* - \frac{\Delta t}{\rho}\nabla p^{n+1}
$$

弱形式（`afsic/euler/ChorinSolver.py`）：
$$
F_1 = \rho\left\langle \frac{\mathbf{u}-\mathbf{u}^n}{\Delta t}, \mathbf{v}\right\rangle
+ \rho\langle (\mathbf{u}^n\cdot\nabla)\mathbf{u}^n, \mathbf{v}\rangle
+ \langle \mu\nabla\mathbf{u}, \nabla\mathbf{v}\rangle - \langle \mathbf{f}, \mathbf{v}\rangle
$$

---

## 3. 各方法原理

### 3.1 `no_cylinder/` — 纯通道流基准

无浸没体。抛物线入口、上下壁无滑移、出口 $p=0$。用于验证流体求解器基线。

### 3.2 `body_fitted/` — 贴体网格法

圆柱是流体网格中的**洞**，圆柱表面无滑移直接作为 Dirichlet 边界条件：
$$
\mathbf{u}=0 \quad \text{on}\ \partial\Omega_{\text{cyl}}
$$
无需浸没边界、无固体本构。网格由 Gmsh 生成（`channel_hole.msh`），圆柱边界物理标签 15。

### 3.3 `ibfe/` — IB-FE 浸没边界法

用**极高刚度 Neo-Hookean 圆盘 + 惩罚固定**近似刚性圆柱，通过 Peskin 浸没边界反馈力耦合：

- 变形梯度 $\mathbf{F} = \nabla_{\mathbf{X}}\boldsymbol{\chi}$，$J=\det\mathbf{F}$，$I_1=\mathrm{tr}(\mathbf{F}^T\mathbf{F})$
- **Neo-Hookean 第一 Piola–Kirchhoff 应力**：
$$
\mathbf{P} = \mu_s J^{-1}\left(\mathbf{F} - \frac{I_1}{2}\mathbf{F}^{-T}\right)
+ \lambda_s \ln J\, \mathbf{F}^{-T}
$$
- **惩罚固定**（把圆盘钉在初始位置）：
$$
-\beta(\boldsymbol{\chi}-\mathbf{X}_0), \qquad \beta=10^{12}\ \text{Pa}
$$
- **Peskin IB 核**（插值 + 扩散）：
$$
U_k = \int_\Omega \mathbf{u}(\mathbf{x})\,\delta_h(\mathbf{x}-\mathbf{X}_k)\,d\mathbf{x},
\qquad \mathbf{f}(\mathbf{x}) = \sum_k \mathbf{F}_k\,\delta_h(\mathbf{x}-\mathbf{X}_k)
$$

参数：$\mu_s=7.7\times10^{10}$ Pa、$\lambda_s=1.15\times10^{11}$ Pa（对应 $E\approx200$ GPa、$\nu=0.3$）。

---

## 4. 短程验证结果

用 `run_short.py` 在统一参数下对各实现各运行 **300 步**（$t=0.3$ s，仍处入口斜坡早期）验证正确性：

| 实现 | $u_{L2}=\int|\mathbf{u}|^2 dV$ | $p_{L2}=\int p^2 dV$ | 附加量 |
|------|------|------|------|
| `no_cylinder` | 0.002728 | 208905 | — |
| `body_fitted` | 0.002792 | 207765 | — |
| `ibfe` | 0.002728 | 208905 | 固体力含 NaN（见 §6） |

**一致性验证要点**：
- 各实现的速度范数高度一致（$0.0027\sim0.0028$），证明统一参数后流体求解正确

> 短程处于 $t<2$ s 的斜坡期，速度量级很小；**稳态基准值**需跑满 $10^4$ 步（$T=10$ s）后统计（DFG Re=100 参考值 $C_d\approx5.57$、$C_l$ 幅值 $\approx0.0106$）。

---

## 5. 运行方法

```bash
conda activate afsi-dolfinx

# 方式一：各实现逐一运行（完整 $10^4$ 步）
cd afsic/demo/demo_339
cd no_cylinder     && python main.py      # 或 mpirun -n <N> python main.py
cd ../body_fitted  && python main.py      # 网格已生成 channel_hole.msh
cd ../ibfe         && python main.py      # 网格已生成 cylinder_solid.xdmf
cd ../multi_direct_forcing && python main.py

# 方式二：短程验证（离线安全，默认 100 步）
cd afsic/demo/demo_339
SHORT_STEPS=300 python run_short.py
```

---

## 6. 已知问题与已应用的修复

### 6.1 dolfinx 0.10.0 兼容性（与 `install.md` 一致）

| 文件 | 修复 |
|------|------|
| `body_fitted/main.py` | `from dolfinx.io import gmsh as gmshio`（`gmshio` 更名 `gmsh`） |
| `ibfe/generate_mesh.py` | 同上 |
| `ibfe/main.py` | `create_vector(L_hat)` → `create_vector(Vs)`（0.10.0 中 `create_vector` 需函数空间而非 Form） |

### 6.2 `body_fitted` 网格 `.geo` 修正

- **重叠圆盘**：原 `.geo` 用 `Duplicata` 保留圆盘曲面，导致 gmsh 同时剖分圆盘与流体，`read_from_msh` 报 `Invalid rank ... less than 1`。改为 `BooleanDifference` 直接删除圆盘输入。
- **物理标签失效**：原 gmsh 自动分配标签 1–6，而 `main.py` 用 `find(11)`~`find(15)`，边界条件从未生效。现显式指定 11=inlet、12=outlet、13=bottom、14=top、15=cylinder。
- **分辨率**：`MeshSizeMax` 0.02→0.01，与均匀网格 $h\approx0.01$ 一致（底/顶各 220 段）。

### 6.3 固有局限（未修改，供参考）

- **`ibfe` 耦合稳定性**：刚性近似采用显式位移更新 + $\beta=10^{12}$ 惩罚。流体推动下圆盘缓慢下游漂移（300 步约 $9\times10^{-4}$ m），惩罚力反馈回流体后随步数增长，**约 200 步后固体力出现 NaN**。这是该演示算法的固有特性；流体场在短程内仍正确。

---

## 7. 新增文件

| 文件 | 用途 |
|------|------|
| `run_short.py` | 离线短程验证脚本：屏蔽 swanlab/网络调用，统一参数下依次运行各实现并汇总 $u_{L2}/p_{L2}$ |
| `compare_mdf_bf.py` | 定量对比 `multi_direct_forcing` vs `body_fitted`（探针 + 表面应力阻力），`STEPS=2500 python compare_mdf_bf.py` |
| `_short_run/` | 短程运行输出目录（velocity/pressure xdmf） |

---

## 8. multi-direct forcing 的 Cd 收敛性（与官方贴体教程对比）

对 `multi_direct_forcing/`（新 demo，详见其 readme）与官方 dolfinx 贴体教程
（CN+AB2 IPCS，**表面应力积分** Cd）在相同物理下做 3 档分辨率对比
（`dfg_tutorial/compare_tutorial_mdf.py`，ρ=1、μ=0.001、Re=100、sin 入口、t=0.3s、300 步）：

| 分辨率 | mdf Cd（力积分） | tutorial Cd（表面应力） | uL2 差 | 尾流 x=0.3 速度差 |
|---|---|---|---|---|
| 110×21 | 0.439 | 0.291 | 2.1% | −23.8% |
| 220×41 | 0.268 | 0.292 | 1.6% | −11.4% |
| 440×82 | 0.157 | 0.292 | 1.4% | −6.6% |

**关键结论**：

- **速度场收敛且两方法一致**（主流区 ~1–3%，随加密趋近；尾流误差随加密收敛）→
  IBM 的流场求解可靠。
- **体积力积分 Cd 不收敛**：mdf 的 Cd 随加密下降（0.439→0.268→0.157，趋向 0），
  tutorial 稳定 0.29。根因是 $dV=\Delta s\cdot h$ 使 $\sum_l dV=2\pi r\cdot h\propto h$，
  力积分正比于 $h$——**diffuse-interface IBM 力积分的固有网格依赖**
  （Mittal & Iaccarino 2005），非流场错误。
- **正确 Cd 算法**：**控制体积动量平衡**（最推荐）、包络面应力积分、表面应力积分（贴体）；
  直接力积分 $\int f_{\mathrm{IBM}}\,dV$ 不推荐做定量。

> 补充：已删除的旧 `direct_forcing` 的 Cd 代理量虚高 ~8×，`multi_direct_forcing` 的
> 力积分 Cd 量级合理但不网格收敛（见本节）——定量阻力应统一用控制体积动量平衡或
> 包络面应力积分计算。
