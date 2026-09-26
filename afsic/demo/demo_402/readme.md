# Turek FSI2 benchmark（固定圆柱 + 弹性尾巴）

Reference: https://kratosmultiphysics.github.io/Examples/fluid_structure_interaction/validation/fsi_turek_FSI2/

2D channel flow around a fixed cylinder with a flexible flag attached to its rear. This
demo runs the benchmark's physical parameters through this repo's own solver,
which is an **immersed boundary method** (fixed Cartesian fluid mesh + Lagrangian solid
mesh advected by interpolated fluid velocity, coupled via `IBMesh`/`IBInterpolation`) —
the same architecture as `demo_400` (turtle) and `demo_401` (sperm), not Kratos's
body-fitted ALE mesh.

## 实际参数（`configuration.py`，CGS 单位）

几何按 benchmark 的 SI 值换算成 cm（`turek.geo` 是 SI，`main.py` 读入后 ×100）：

| 量 | 值 |
|---|---|
| 通道 | `Lx × Ly = 220 × 41` cm |
| 圆柱 | 圆心 `(20, 20)` cm，直径 `D = 10` cm |
| 尾巴（弹性板） | 长 35 cm、厚 2 cm |
| 流体 | `rho = 1` g/cm³，`mu = 10` dyne·s/cm²（=1 Pa·s），`Um = 200` cm/s（2 m/s，余弦 2 s 斜坡） |
| 固体 | Neo-Hookean，`mu_s = 2e7`、`lambda_s = 8e7` dyne/cm²（E≈5.6 MPa, ν=0.4）；圆柱区用惩罚 `beta=1e6` 钉住 |
| 网格 / 时间 | `Nx × Ny = 220 × 41`（h = 1 cm），`dt = 5e-5` s，`T = 10` s（= 200000 步） |
| Re | `rho·Um·D/mu = 100` |

固体网格：`generate_mesh.py` 把 `turek.geo` 转成 `turek_mesh.xdmf/.h5`
（`MeshSizeMax=0.006` m → 595 节点 / 1017 三角形；速度场 P2、固体 P2）。

> 说明：本文件早先那段 "SI 单位 + Saint-Venant-Kirchhoff + gravity + cyl_stiffness_factor"
> 的描述与实际代码不符（代码是 CGS + Neo-Hookean + 惩罚固定，没有重力项）。

> 与文献中的改版 Turek–Hron 设置（*Local Divergence-Free Immersed Finite
> Element-Difference Method Using Composite B-Splines* 用的 L=6H、N=128、Δt=0.00164Δx、
> SVK、κ_s=5e4Δx/Δt²）的逐项参数对照，见
> `docs/turek-hron-bspline-paper-vs-demo402.md`。

## 已知局限（与官方 benchmark 相比）

- 只是**定性**复现，不追求与 published 位移/频率数值吻合。
- IB 方案没有独立的固体惯性项（经典无质量 Peskin IB），而真实 FSI2 依赖 `rho_s = 10 rho_f`
  的附加惯性；这里 `rho_s` 不参与（没有重力项）。
- 圆柱用"很硬的弹性区 + 惩罚"近似，不是严格刚体约束。
- **稳定性对 `dt` 敏感**——本算例的显式 IB 反馈在小 `dt` 下才稳，改动 `dt`/网格前先跑短程。

## 运行（本机环境）

```bash
export REPO=$(git rev-parse --show-toplevel); RUN=$REPO/.tools/afsi-run.sh
cd $REPO/afsic/demo/demo_402

# 1) 固体网格（首次，或改了几何后）
$RUN python -B -u generate_mesh.py

# 2) 直接用 main.py（默认 Chorin，完整 T=10 s 需要很久）
SOLVER=chorin $RUN python -B -u main.py
SOLVER=ipcs   $RUN python -B -u main.py        # IPCS（含 f 的符号约定处理）

# 3) 脱离网络跑 + 记录对比曲线（推荐）：run_compare.py
SOLVER=chorin T=0.5 FPS=100 $RUN python -B -u run_compare.py
SOLVER=ipcs   T=0.5 FPS=100 $RUN python -B -u run_compare.py
```

环境变量（`configuration.py` / `main.py` / `run_compare.py`）：

| 变量 | 默认 | 说明 |
|---|---|---|
| `SOLVER` | `chorin` | `chorin`（投影/分步法）或 `ipcs`（增量压力修正）|
| `T` / `DT` | `10.0` / `5e-5` | 物理时长 / 时间步（步数 = `T/DT`）|
| `NX` / `NY` | `220` / `41` | 流体网格（`turek.geo` 的固体网格与几何无关，不用重生成）|
| `UM` | `200.0` | 入口平均速度 [cm/s] |
| `FPS` | `100` | 仅 `run_compare.py`：输出频率（`TimeManager(..., fps)`），调大可加密输出 |
| `OUT` | `plot/compare_<solver>` | 仅 `run_compare.py`：输出目录 |

### dolfinx 0.10 兼容修正（本 demo 已修）

| 文件 | 修正 |
|---|---|
| `generate_mesh.py` | `from dolfinx.io import XDMFFile, gmshio` → `from dolfinx.io import gmsh as gmshio`（同 demo_339 README §6.1）|
| `main.py` | `b1 = create_vector(L_hat)` → `create_vector(Vs)`（0.10 需要函数空间）|

### 求解器切换（`SOLVER=`）

`main.py` 里 IPCS 与 Chorin 的动量方程中 `f` 的符号相反（Chorin 的 `F1` 是
`-inner(f,v)`，`rhs` 得到 `+f`；IPCS 是 `+dot(f,v)`，`rhs` 得到 `-f`），所以 IPCS 的浸没
边界力要乘 `force_scale = -1`；切换后两者在早期时刻的场量差 0.02 %（见下面结果）。

## 文件

| 文件 | 说明 |
|---|---|
| `configuration.py` | 参数（含 `T/DT/NX/NY/UM` 环境变量覆盖）|
| `generate_mesh.py` | `turek.geo` → `turek_mesh.xdmf/.h5` |
| `main.py` | 流体 + 固体 + IB 耦合 + 时间循环（`SOLVER` 选择求解器）|
| `run_compare.py` | **离线对比运行器**：屏蔽 swanlab/内网 API、固定输出目录、把 `swanlab_upload` 换成本地 `history.csv` 记录器 |
| `plot/compare_from_output.py` | 从 `velocity.h5`/`pressure.h5`/`solid.h5` 重建时间序列并对比两套求解器（不依赖 `history.csv`，更稳）|
| `plot/compare_solvers.py` | 直接对比两份 `history.csv`（同上的轻量版）|

输出：`plot/<OUT>/velocity.xdmf(.h5)`、`pressure.xdmf(.h5)`、`solid.xdmf(.h5)`
（每个输出时刻一个数据集），`history.csv`，以及对比脚本产出的
`plot/rebuild_*.csv/.png`。

## 求解器对比：Chorin vs IPCS（实测）

命令（两档网格，各跑两个求解器；`T`/`DT`/`NX`/`NY` 由环境变量覆盖）：

```bash
RUN=$REPO/.tools/afsi-run.sh
# 粗网格档（110x21, dt=1e-4, T=1.0 s, 10000 步）
for s in chorin ipcs; do
  SOLVER=$s NX=110 NY=21 DT=1e-4 T=1.0 FPS=100 \
    OUT=$PWD/plot/coarse_$s $RUN python -B -u run_compare.py
done
# 细网格档（220x41, dt=5e-5, T=0.5 s）
for s in chorin ipcs; do
  SOLVER=$s T=0.5 FPS=100 OUT=$PWD/plot/compare_$s $RUN python -B -u run_compare.py
done
# 对比（从 h5 重建时间序列，输出 csv + png）
$RUN python -B plot/compare_from_output.py plot/coarse_chorin plot/coarse_ipcs \
     "Chorin(110x21)" "IPCS(110x21)"
```

### 结论

1. **发散前两者一致**：t = 0.01 s 时 `max|u|/scale` = 1.913（Chorin）vs 1.912（IPCS），
   `u_L2/scale²` = 9399 vs 9404（差 0.05 %），尾尖位移差 0.2 %。说明 `SOLVER=` 切换时的
   浸没边界力符号约定（IPCS 取 `force_scale = -1`）是对的。
2. **IPCS 在前 0.1 s 内因 IB 处的局部速度尖峰而发散**，Chorin 稳定：

   | 网格 / dt | IPCS 首发散时刻 | 尖峰位置 | 峰值 max\|u\| | 之后 | Chorin |
   |---|---|---|---|---|---|
   | 110×21 / 1e-4 | t ≈ 0.04 s（`max\|u\|/scale` 2.06 → 2.94）| 圆柱下方 (32, 15.6) / (22, 14) | 3207 cm/s @ t=0.12（斜坡速度仅 1.8 cm/s）| 场冻结、固体解耦 | 稳定跑到 T = 1.0 s |
   | 220×41 / 5e-5 | t ≈ 0.03 s（1.98 → 3.05）| 圆柱下方 (18, 14) | 805 cm/s @ t=0.09 | 同上（t>0.10 无意义）| 稳定（记录到 t = 0.23 s 后停止）|

   尖峰增长率 ≈ ×3.2–3.4 / 0.01 s，即 **≈ 120 s⁻¹，且与网格、dt 无关**（两档速率一致），
   位置都在**圆柱下表面**（圆柱中心 (20,20)、半径 5，底部 y=15）——即惩罚力最大处。
3. **IPCS 发散之后的输出不可用**：`max|u|/scale` 恰好回到 1.500（= 入口抛物剖面峰值），
   `u_L2/scale²` 冻结在 11.6，`tip_dx`、`cyl_dx` 变成常数 → 固体耦合已经"死掉"，流场退化成
   入口剖面直接扫过通道。所以对比只能用到 t ≈ 0.1 s。
4. **Chorin 的表现（粗网格，T = 1.0 s）**：`max|u|/scale` 稳定在 1.9→2.3（拍动瞬态到 4.4），
   `u_L2/scale²` 从 9400 漂到 10500（+12 %，属正常发展），尾尖横向位移单调增长到
   0.038 cm 后转为大幅摆动（t=1.0 时 |Δy| ≈ 0.84 cm），圆柱漂移 ≤ 1.2e-2 cm
   （惩罚基本守住）。
5. **成本**（本机、单进程）：细网格 Chorin ≈ 0.3–0.5 s/步、IPCS ≈ 0.5–0.9 s/步（并发跑时）；
   粗网格两者都在 0.12–0.15 s/步量级。IPCS 每步多解一次额外的压力修正系统。

**成因假设（未定论）**：IPCS 的动量预测步把压力显式滞后使用（`-dot(p_, div v)`），
再用 `-(dt/rho)grad(phi)` 修正；圆柱处的惩罚体力很大时，`u*` 局部过冲 → 泊松右端
`(rho/dt)div(u*)` 变大 → 修正步压不住 → 自激。Chorin 的预测步不含压力、对流全显式、只有一次
投影，在这个算例上更鲁棒。若要进一步定位：减小 `beta`（需要给 `configuration.py` 加环境变量
覆盖）、或把圆柱改成强 Dirichlet 约束再跑 IPCS 对比。

> 注意：本结论**只针对这个算例/这套显式 IB 耦合**，不能推广成"IPCS 不行"——demo_423/424/426
> 里 IPCS 用在别的配置上是正常的。

## 出厂密度 220×41 跑满 ramp（T = 2 s）实测

之前**没有任何 220×41 的完整运行记录**（站点索引页把 demo_402 标为 "Configuration
only"；唯一的存档图是 88×17 降网格那次，而那张图是冻结的）。本次补上了：

```bash
SOLVER=chorin NX=220 NY=41 DT=5e-5 T=2.0 FPS=100 \
  OUT=$PWD/plot/chorin_220x41_T2 $RUN python -B -u run_compare.py
$RUN python -B plot/fig_T2_220x41.py        # 出图 fig_chorin_220x41_T2.png
```

结果（40000 步，6713 s ≈ 1.86 h，单步 ≈0.17 s；输出 `plot/chorin_220x41_T2/`）：

| 指标 | 值 | 判读 |
|---|---|---|
| `max\|u\|/scale` | 1.79 → **2.32** | 有界 ✓ 无发散 |
| `u_L2/scale²` | 9344 → **11662** | 有界（+25 %，随拍动发展）✓ |
| 尾尖 Δy | −0.042 → **+0.62 cm**（t≈1.0 s 后振荡，幅值随 ramp 增大）| 真拍动 ✓ |
| 尾尖 Δx | +0.013 → −0.032 cm | 单调发展后回摆 ✓ |
| 圆柱漂移 | `\|Δx\|max` = 7.3e-3 cm、`\|Δy\|max` ≈ 3.7e-4 cm | 惩罚守住 ✓ |
| 流动 | t = 2 s 时 `\|u\|` 已充满整条通道（峰值 ~460 cm/s）| ✓ |

**与 88×17 那次的本质区别**：88×17（h≈2.5 cm，尾巴厚 0.8 格）在 t≈0.6 s 把 tip 冻结在
60.16 cm、流场只停在入口附近；出厂密度下同一时段（图中橙色带）指标平滑通过，随后继续演化。
所以"220×41 = 出厂/published 密度"这条现在有了结果支撑，而 88×17 的图不能当作 402 的结论。

仍然只是**定性**结果（无固体惯性项、圆柱为惩罚近似），不要拿去和 published FSI2 的
位移/频率数值对比；另外 T = 10 s 的官方全长按此速率约需 **9–10 h**（200000 步）。

## Note

the density of and is the same
the solid is added with the same viscosity with fluid
