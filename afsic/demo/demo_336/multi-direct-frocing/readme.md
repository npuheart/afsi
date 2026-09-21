# Demo 336 — 方腔驱动圆盘（multi-direct forcing 版）

在 AFSI（FEniCSx/dolfinx）中，用与 `demo_421/main.py` **完全相同的求解器结构**
（DFIBMFoam multi-direct forcing 的 FEniCSx 移植）求解 demo_336 的经典算例
**方腔驱动圆盘**：1×1 方腔顶盖匀速滑动驱动流体，圆盘随方腔环流运动。

原始 `fsi_paralell.py` 用弹性固体（mu_s/lambda_s）模拟可变形的圆盘（IBM-FEM 耦合）；
本目录改用 **multi-direct forcing（迭代直接力法）** 的 **刚性圆盘**：
圆盘内部用 Peskin 4 点 $\delta$ 核标记填满（刚性体），逐标记直接力
$F_l=(U^d_l-U_l)/\Delta t$ 迭代累加进体积力场。

## 求解器（= demo_421/main.py 的算法）

预测步（AB2 对流 + 3/2-1/2 半隐式扩散）→ 迭代直接力（每步 `n_iter` 次，
$U^d$ 为圆盘期望速度）→ 压力泊松 $\nabla^2 p=\frac{2}{3\Delta t}\nabla\cdot U$ →
速度修正（L2 投影）。与固定圆柱版（demo_339）唯一的本质区别在 $U^d$：

- **`disk_motion = "free"`（默认，"驱动圆盘"）**：圆盘为**刚性体随流驱动**。
  每步先用 $U^* - 1.5\Delta t\nabla p^n$ 处的流体速度插值到边界环标记，
  求刚体速度
  $$\mathbf V_c=\langle \mathbf U_l\rangle,\qquad
  \omega=\Big\langle\frac{\mathbf r_l\times \mathbf U_l}{r^2}\Big\rangle,$$
  再令各标记期望速度为刚体速度
  $\mathbf U^d(\mathbf X)=\mathbf V_c+\omega\times(\mathbf X-\mathbf X_c)$，
  并更新圆心/转角、重设标记坐标、重调 `evaluate_current_points`
  （移动体机制同 demo_421 的鱼体）。
- **`disk_motion = "fixed"`**：圆盘固定，$U^d=0$，且每步把圆盘内部速度硬置零
  （`mask_interior=True`，demo_339 圆柱风格）。

流体域为**闭合方腔**（顶盖滑动 + 三壁无滑移），压力在角落固定一个 DOF。

## 文件

| 文件 | 说明 |
|------|------|
| `configuration.py` | 方腔/圆盘/IBM 参数，`STEPS`/`NX`/`NY`/`DISK_MOTION`/`MARKER_MODE` 环境变量覆盖 |
| `main.py` | AB2 分步求解器 + 刚体圆盘 multi-direct forcing 时间循环 |
| `output/` | `velocity.xdmf/.h5`、`pressure.xdmf/.h5`、`disk.xdmf/.h5`（固体位移场）、`forces.csv`、`disk_trace.csv`(free) |

## 固体（圆盘）输出

mdf 版圆盘是**刚体**，没有可变形固体网格，故不像原始 `fsi_paralell.py` 那样输出
`solid_force.xdmf`；这里在 `disk.xdmf` 中输出：**三角化参考圆盘网格（固定初始位姿）
+ 逐时间步刚体位移场 `u`**。

- 位移场公式：$\mathbf u(\mathbf X_{\text{ref}}) = (\mathbf X_c(t)-\mathbf X_c(0)) + (R(\theta)-I)(\mathbf X_{\text{ref}}-\mathbf X_c(0))$（刚体平动+转动）
- **ParaView 可视化**：打开 `disk.xdmf` 后应用滤镜 **Warp by vector**，选位移场 `u`，
  即可看到圆盘随流平移/旋转
- 为什么不是直接移动网格：dolfinx 0.10 的 XDMF 时间序列**只写一次几何**（各时间步用
  `xi:include` 引用首个 Grid 的 Geometry，不支持移动网格），故用位移场 + Warp 是标准做法
  （与原版输出 `solid_coords` 位移场同一思路）。
- 圆盘刚体运动学同时记录在 `disk_trace.csv`（圆心/转角/刚体速度），便于后处理。

## 圆盘是否有本构？

**没有。** mdf 版圆盘是**刚性体**（无限刚度），不是弹性固体：

- 期望速度 $\mathbf U^d=\mathbf V_c+\omega\times(\mathbf X-\mathbf X_c)$ 由刚体运动学给出，
  直接力 $F_l=(\mathbf U^d-\mathbf U_l)/\Delta t$ 只是**运动学约束力**，使标记处流体速度
  等于刚体速度——不涉及任何应力-应变关系；
- 对比原始 `fsi_paralell.py`：其圆盘是**弹性固体**，有本构（`L_hat = -inner(mu_s*(FF-inv(FF).T), grad(dVs))*dx`，
  圣维南-基尔霍夫/类 neo-Hookean，参数 `mu_s=0.1, lambda_s=10`），力由变形梯度 $F=\nabla \mathbf X$ 算出；
- 若需弹性圆盘（可变形 + 本构），应走原始 `fsi_paralell.py` 的 IB-FEM 路线（见「后续可扩展」）。

## 运行

```bash
conda activate afsi-dolfinx
cd afsic/demo/demo_336/multi-direct-frocing
python main.py                      # 默认 4000 步 (T=10.0 s)，free 模式
STEPS=200 python main.py            # 短程冒烟验证
DISK_MOTION=fixed python main.py    # 固定圆盘模式
MARKER_MODE=boundary python main.py # 仅边界环标记（更快，圆盘厚需 mask）
```

## 默认参数（与 demo_336 原始算例一致）

| 参数 | 值 | 说明 |
|------|-----|------|
| 方腔 | $1\times1$ | $N_x=N_y=128,\ h\approx7.8$ mm |
| 顶盖速度 $U_{\text{lid}}$ | 1.0 m/s | 沿 +x，恒定 |
| $\rho$ / $\mu$ | 1.0 / 0.01 | $\text{Re}=\rho U L/\mu=100$ |
| $\Delta t$ | 0.0025 s | CFL≈0.32 |
| 圆盘 | 圆心 $(0.6,0.5)$，$r=0.2$ | 与原始 circle_20 网格一致 |
| 标记 | disk 模式，边界环 $\Delta s\approx h/2$，内部间距 $0.5h$ | 共 ~8500 标记 |
| `n_iter` | 10 | DFIBMFoam 默认迭代数 |

## 验证结果（128×128，短程）

- **free 模式**（500 步 / t=1.25 s）：稳定无 NaN；圆盘被主涡环流带动，从
  $(0.6,0.5)$ 左移到 $(0.48,0.485)$，$V_c\approx(-0.12,0)$ m/s，逆时针旋转
  $\omega\to-0.18$ rad/s —— 符合方腔主涡（顺时针）对中部圆盘的驱动。
- **fixed 模式**：$C_x\approx-0.78$（圆盘静止、相对流速大），明显大于 free 模式
  $C_x\approx-0.03\sim-0.17$（圆盘随流、相对速度小）—— 物理一致。

## 全程结果（64×64，4000 步 / T=10 s，free 模式）

完整 10 s 运行（`NX=NY=64 STEPS=4000 OUT_INTERVAL=1`，无 NaN、无触壁）：

- **轨迹（圆心）**：$(0.600,0.500)\to(0.395,0.514)\to(0.276,0.750)
  \to(0.568,0.760)\to(0.541,0.518)\to(0.326,0.621)$（$t=0,2,4,6,8,10$ s）。
  圆盘被主涡带向左侧上升、贴顶盖下方横越（$t=6$ s 顶边距盖 ≈4 cm）、
  再沿右侧下沉、最后被回流重新卷起。
- **运动学范围**：$|\mathbf V_c|_{\max}=0.251$ m/s，$|\omega|_{\max}=0.295$ rad/s，
  总转角 $-1.055$ rad；圆心始终在 $x\in[0.273,0.663]$、$y\in[0.484,0.769]$ 内。
- **刚体位移场**：$\max|u_s|=0.509$；位移 = 平动 + 绕初始圆心转动
  （到质心的节点均方半径每次输出恒为 $0.1244$，即形状不变）；
  由 `disk.xdmf` 位移重构的质心与 `disk_trace.csv` 相差 $<5\times10^{-7}$。
- **流体**：$u_{L2}$ 由 0 增长到 $0.0529$；$|C_x|\le1.07$、$|C_y|\le2.04$
  （仅量级参考，见「已知局限」的力积分不收敛）。

> 注意：$64\times64$ 与 $128\times128$ 的圆盘轨迹一致（见 elastic 版 readme 的
> DF@64 ≈ DF@128 对比），故全程运行按 $64^2$ 完成；上表的 128×128 短程结果
> 只覆盖 $t\le1.25$ s。

## 已知局限（同 multi-direct forcing / afsic IBM / demo_421）

- **均匀笛卡尔网格且域必须从原点 (0,0) 出发**（afsic `IBKernel` 的 `base_node`
  用 $X/dh$ 未减域原点，见 demo_421 readme）；方腔 $[0,1]^2$ 正好满足。
- **单进程**（MPI 多进程需重写 `IBMesh` 映射）。
- **力积分定量性**：直接力积分 $\int f_{\text{IBM}}dV$ 因 $\Delta V_l=\Delta s\cdot h$
  而 $\propto h$，随网格加密不收敛（见 demo_339 §8）。`forces.csv` 里的
  $C_x,C_y,C_m$ 仅作**固定网格下的量级参考**；定量受力建议用控制体积动量平衡
  或包络面应力积分。
- **C++ `solid_to_fluid` 是"替换"非"累加"**：`IBMesh::assign_dofs` 用 `setitem()`
  覆盖目标函数。故 `main.py` 采用 demo_421 的修复——每次迭代先扩散到临时场
  `f_spread` 再 `f_ibm.x.array += f_spread.x.array`（Python 层累加）。直接往
  `f_ibm` 扩散只会留下最后一次迭代的力（圆盘近似"透明"）。
- **free 模式为显式刚体耦合**（一步更新），$\Delta t$ 过大时圆盘可能与壁面
  交互；`main.py` 内置了圆心到壁面 margin 的安全钳位（仅极限情况下触发）。

## 后续可扩展

- **圆盘-流体换热 / 旋转阻力矩对标**：输出 $C_m$ 可与经典"方腔圆柱旋转"基准对比
- **弹性圆盘**：把 rigid 标记改为 FEniCS 弹性固体 + IB-FEM 力耦合（即原始
  `fsi_paralell.py` 的路线）
- **定量受力**：接入控制体积动量平衡计算圆盘阻力/升力/力矩

## 详细算例研究（刚性盘）

同一方腔/圆盘（$r=0.2$，$64^2$，$T=10$ s，4000 步）的刚性盘版本：

| 运行 | 结果 |
|------|------|
| `rigid_free`（free） | 质心 $x\in[0.273,0.663]$、$y\in[0.484,0.769]$，路径长 1.399，**扫过 $-338^\circ$**（10 s 内差一点跑满一圈）；$u_{L2}=0.0529$ |
| `rigid_fixed`（fixed） | 圆盘不动（质心恒定），仅作对照 |

注意与弹性版的对比（见 elastic 版 readme 的同一研究）：**弹性盘转得更快**（$-587^\circ$
对 $-338^\circ$，即 10 s 内两圈对一圈），因为刚体盘被直接力约束在局部流速上、
相对主涡存在滑移；弹性盘随流更"贴"。能量上刚体盘拿走 21.2%，弹性轻软盘只拿 7.3%。

`OUTPUT_PATH` 覆盖已补上（原实现直接写死到 `output/`，会污染算例目录）；
逐步日志写入 `metrics.csv`（`t,cx,cy,theta,Vc_x,Vc_y,omega,u_L2,p_L2,u_max`），
输出间隔由 `OUT_INTERVAL` 控制（默认 40 步）。
