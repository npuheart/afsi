# demo_444 — IB2d 弹性圆环（Standard Rubberband）：AFSI 移植与对照

## 1. 问题描述

IB2d（Battista et al., MMAS 2018）的第一个算例 `Example_Standard_Rubberband/Rubberband_with_Springs`：
64 个拉格朗日点组成的椭圆弹性环（$x$ 半轴 0.2，$y$ 半轴 0.4，中心 $(0.5,0.5)$），
相邻点之间以零静长线性弹簧连接（$k=2.5\times10^4$），浸没在双周期单位方形域
$\Omega=[0,1]^2$ 的粘性流体中（$\rho=1$，$\mu=0.01$）。椭圆在弹簧张力作用下振荡并趋于圆形。
$N_x=N_y=32$，$\Delta t=10^{-3}$，$T=1.5$。

本 demo **直接读取 IB2d 原始输入文件**（`input2d`、`rubberband.vertex`、`rubberband.spring`），
用 AFSI 组件（`IBMesh` / `IBInterpolation` + 新流体求解器 `afsic.euler.PeskinRK2Solver`）重算，
并与 IB2d（MATLAB 源码，Octave 运行）逐帧对照。

本 demo 提供两条 AFSI 管线（`FLUID`）：

* `rk2`（默认）——周期 Taylor–Hood + Peskin (2002) 两阶段格式，与 IB2d 逐项对应（§2.1–2.3）；
* `chorin` | `ipcs` —— 闭合方腔 P2/P1 投影格式 + Peskin 4 点核 + 直接载荷（方法自
  `afsic/demo/demo_444` 移入，§2.4），用于研究 **grad-div 稳定化系数 γ（即对比研究中的
  “λ”，取 0 与 100）** 对 fiber 面积泄漏的影响（§3.3，共 4 组新结果）。

两条管线的结果与 IB2d 参考一并进入 **6 面板 × 6 曲线** 的形状对比图（`figures/compare_shapes.png`）。

## 2. 方法

### 2.1 IB2d 与 AFSI 的对应关系

| 环节 | IB2d (matIB2d) | AFSI（本 demo） |
|---|---|---|
| 流体离散 | $N_x\times N_y$ 同位网格中心差分 + FFT | Q2/Q1 Taylor–Hood，$N_x\times N_y$ 四边形单元 |
| 周期边界 | FFT | 周期约化 $P^\top A P$（不依赖 dolfinx_mpc） |
| 时间推进 | Peskin (2002) 两阶段：$\Delta t/2$ 隐式 Euler + $\Delta t$ Crank–Nicolson | `PeskinRK2Solver`（完全相同） |
| 不可压 | 每阶段 FFT 精确投影 | 每阶段整体 (monolithic) Stokes 求解，无分裂误差 |
| IB 网格 / 核函数 | IB2d 网格上 Peskin 4 点核 | `IBMesh(order=1)`：网格顶点 = IB2d 网格，同一 4 点核 |
| 固体 | 弹簧 $\mathbf F=k(\lvert\Delta\mathbf X\rvert-L)\,\Delta\mathbf X/\lvert\Delta\mathbf X\rvert$ | 一维 FE 曲线，能量 $\frac{k}{2h_0}(h_0\lvert\mathbf X_s\rvert-L)^2$（P1 下与弹簧公式逐点一致，误差 $10^{-14}$） |
| 拉格朗日权重 | $\mathbf F_k\,\mathrm ds$，$\mathrm ds=\min(L_x/2N_x,L_y/2N_y)$ | 相同 |
| 拉格朗日点更新 | $\mathbf X^{n+1/2}=\mathbf X^n+\frac{\Delta t}{2}\mathbf U^n(\mathbf X^n)$，$\mathbf X^{n+1}=\mathbf X^n+\Delta t\,\mathbf U^{n+1/2}(\mathbf X^{n+1/2})$ | 相同 |

### 2.2 流体求解器 `PeskinRK2Solver`

$$
\begin{aligned}
&\rho\frac{\mathbf u^{n+1/2}-\mathbf u^n}{\Delta t/2}+\rho\,\mathcal C(\mathbf u^n)=-\nabla p^{n+1/2}+\mu\Delta\mathbf u^{n+1/2}+\mathbf f^{n+1/2},\qquad \nabla\cdot\mathbf u^{n+1/2}=0,\\
&\rho\frac{\mathbf u^{n+1}-\mathbf u^n}{\Delta t}+\rho\,\mathcal C(\mathbf u^{n+1/2})=-\nabla p^{n+1/2}+\frac{\mu}{2}\Delta(\mathbf u^{n+1}+\mathbf u^n)+\mathbf f^{n+1/2},\qquad \nabla\cdot\mathbf u^{n+1}=0,
\end{aligned}
$$

其中 $\mathcal C(\mathbf w)=\tfrac12[(\mathbf w\cdot\nabla)\mathbf w+\nabla\cdot(\mathbf w\otimes\mathbf w)]$（IB2d 的斜对称形式）。
第二阶段乘 2 后与第一阶段左端矩阵相同，鞍点矩阵只组装、LU 分解一次（MUMPS）。
可选 grad-div 项 $\gamma(\nabla\cdot\mathbf u,\nabla\cdot\mathbf v)$（`GRAD_DIV`，默认 $\gamma=100$）。
压力在一点固定后平移为零均值（与 FFT 相同的规范）。

### 2.3 求解器验证（Taylor–Green 涡，`afsic/tests/test_peskin_rk2.py`）

| 网格 $n$ | 8 | 16 | 32 |
|---|---|---|---|
| 相对 $L^2$ 速度误差（$\Delta t=10^{-3}$, $T=0.1$） | $5.1\times10^{-2}$ | $2.9\times10^{-3}$ | $1.7\times10^{-4}$ |

时间方向：$\Delta t=0.04\to0.02$ 误差比 3.75（二阶）。

### 2.4 投影管线（`FLUID=chorin | ipcs`，方法自 `afsic/demo/demo_444` 移入）

| 环节 | 实现 |
|---|---|
| 流体离散 | P2/P1 于 $N_x/2\times N_y/2$ 四边形单元；速度节点格 = IB2d 网格（$(N_x{+}1)\times(N_y{+}1)$，间距 $1/N_x$） |
| 边界 | 闭合方腔：四壁无滑移 + $(0,0)$ 点压力基准（IB2d 为周期 FFT；此处为盒子保守性测试） |
| 时间推进 | Chorin（非增量）/ IPCS（增量压力修正）投影格式，显式耦合：求解 → 插值 $\mathbf u$ 到标记点 → $\mathbf X \mathrel{+}= \Delta t\,\mathbf u$ → 新位置重算弹簧力并扩散 |
| IB 耦合 | Peskin 4 点核（`IBInterpolation`）+ 直接载荷 $\mathbf b=V_h\,\mathbf f_{\rm spread}$（与 demo_402/424 一致的伴随格式） |
| grad-div 稳定化 | 预测步弱式增加一致的 $\gamma(\nabla\cdot\mathbf u,\nabla\cdot\mathbf v)$ 项（`GRAD_DIV` $\equiv$ 研究中的 λ；`ChorinSolver`/`IPCSSolver` 的 `grad_div` 参数，库内默认 0 = 原格式不变） |

弹簧力权重与 IB2d 相同（$\mathbf f_i = k\,\mathrm ds\,(\mathbf X_{i-1}+\mathbf X_{i+1}-2\mathbf X_i)$，
$\mathrm ds=1/64$；代码内含与 IB2d 公式的逐点自检，误差 $10^{-14}$）。

## 3. 结果（$N_x=32$）

![shapes](figures/compare_shapes.png)

![history](figures/compare_history.png)

| $t$ | $\max_k\lvert\mathbf X_k^{\rm AFSI}-\mathbf X_k^{\rm IB2d}\rvert$ | 面积 AFSI | 面积 IB2d | $e_p$ |
|---|---|---|---|---|
| 0.02 | $5.7\times10^{-3}$ | 0.2506 | 0.2514 | 0.084 |
| 0.10 | $8.1\times10^{-3}$ | 0.2494 | 0.2519 | 0.075 |
| 0.20 | $4.6\times10^{-3}$ | 0.2484 | 0.2512 | 0.096 |
| 0.50 | $3.7\times10^{-3}$ | 0.2458 | 0.2436 | 0.062 |
| 1.00 | $2.3\times10^{-2}$ | 0.2435 | 0.2173 | 0.180 |
| 1.50 | $3.8\times10^{-2}$ | 0.2423 | 0.1840 | 0.401 |

（初始面积 $\pi\cdot0.2\cdot0.4=0.2513$；网格间距 $h=1/32=3.1\times10^{-2}$。）

- **前 0.5 s（约 2.5 个振荡周期）**：两者轨迹几乎重合，拉格朗日点最大偏差 $\le 0.31h$，压力相对 $L^2$ 差 6–10%。
- **0.5 s 之后**：差异主要来自 IB2d 自身的体积泄漏。物理上环内流体不可压，面积应守恒；
  IB2d 在 $t=1.5$ 已丢失 26.7% 面积（同位中心差分 + 零静长弹簧持续张拉），AFSI 仅 3.5%。
- 速度场相对差 $e_u$ 在振荡转折点（速度接近零）被放大，见 `figures/compare_fields.png`。
- 图 `compare_shapes.png` / `compare_history.png` 中已叠加 §3.3 的 4 条投影曲线（共 6 条）。

### 3.1 网格加密（`make_rubberband.py`，保持带张力不变：$k\propto N_x^2$）

![refinement](figures/refinement.png)

| 网格 | 面积损失 AFSI | 面积损失 IB2d | $\max\lvert\Delta\mathbf X\rvert$, $t\le0.5$ |
|---|---|---|---|
| $N_x=32$ | 3.45% | 26.69% | $9.7\times10^{-3}$ |
| $N_x=64$ | 1.79% | 18.51% | $8.3\times10^{-3}$ |

AFSI 在 $N_x=32/64$ 下彼此接近；IB2d 的后期轨迹随加密仍明显移动，说明后期差异是 IB2d 的离散误差（泄漏）。

### 3.2 消融

| 变体 | $t=1.5$ 面积 | 结论 |
|---|---|---|
| 默认（`FLUID_MESH=full`, `GRAD_DIV=100`） | 0.2423 | 推荐 |
| `GRAD_DIV=0`（纯 Taylor–Hood） | 0.0002 | 环几乎完全塌缩：Q1 压力无法表示跨膜压力跳，残余力驱动穿膜流 |
| `FLUID_MESH=half`（$N_x/2$ 个 Q2 单元，速度节点 = IB2d 网格） | 0.2471 | 位置偏差更大（$t=1.5$ 时 $4.5\times10^{-2}$） |
| `GRAD_DIV` 取 $10^2$–$10^4$ | — | 结果几乎不变（$\gamma\gtrsim100$ 饱和） |

### 3.3 六条曲线对比与 γ（λ）扫描（4 组新结果）

![shapes6](figures/compare_shapes.png)

图 1 为 $t = 0.00,\ 0.04,\ 0.10,\ 0.20,\ 0.50,\ 1.50$ 六个时刻的环形状，每幅 6 条曲线：
IB2d 参考、AFSI rk2、fiber Chorin/IPCS × γ=0/100。

| 运行 | $t=1.5$ 面积 | $\max_k\lvert\Delta\mathbf X\rvert$（$t=1.5$） | 结论 |
|---|---|---|---|
| IB2d（参考） | 0.1840 | — | 自身泄漏 26.7% |
| AFSI rk2 | 0.2423 | $3.8\times10^{-2}$ | 面积守恒（仅泄漏 3.5%） |
| fiber Chorin, γ=0 | $3.8\times10^{-6}$ | $4.9\times10^{-1}$ | **塌缩**（$t\approx0.14$ s 面积掉到 $10^{-2}$ 以下） |
| fiber Chorin, γ=100 | 0.2008 | $1.5\times10^{-2}$ | **稳定**，全程贴合 IB2d |
| fiber IPCS, γ=0 | $3.8\times10^{-6}$ | $4.3\times10^{-1}$ | **塌缩**（同上） |
| fiber IPCS, γ=100 | 0.2037 | $1.6\times10^{-2}$ | **稳定** |

- γ=0 时两条投影曲线都在 ~0.14 s 内把环内面积泄漏殆尽——Q2/Q1（quad）上投影格式
  对跨膜压力跳无约束，残余力驱动穿膜流（与 `afsic/demo/demo_444` 的读法一致）；
- γ=100 的 grad-div 项把两条曲线都救回：$t\le1.5$ 内标记点最大偏差
  $1.5$–$1.7\times10^{-2}$（约半步网格间距 $h=1/32$），末态面积 0.200/0.204 与 IB2d 的 0.184 同量级；
- 两个 γ 端点上 Chorin 与 IPCS 行为几乎一致——该算例区分它们的是 γ，而不是格式本身。

`compare.py` 还会生成 `figures/compare_history.png`（面积、$\max|\Delta\mathbf X|$、$e_u$ 三条时间曲线）、
`figures/compare_fields.png`（rk2 vs IB2d 的场）与完整数据表 `figures/compare_table.csv`。

## 4. 运行

```bash
conda activate afsi-dolfinx          # dolfinx 0.10 + MUMPS；不需要 dolfinx_mpc
cd afsic/demo/ib2d/demo_444

# (1) IB2d 参考解：有 MATLAB 用 MATLAB，否则用 Octave（自动生成兼容补丁副本，不改 IB2d 源码）
./run_ib2d_reference.sh                               # -> ib2d_reference.npz（约 3.5 min, Octave）
MATLAB=/Applications/MATLAB_R2025b.app/bin/matlab ./run_ib2d_reference.sh

# (2) AFSI（rk2 管线）
python main.py                                        # 约 7 s -> plot/afsi_result.npz + XDMF
TFINAL=0.1 python main.py                             # 快速检查
GRAD_DIV=0 python main.py                             # 消融
FLUID_MESH=half python main.py

# (2b) 投影管线 + γ（λ）扫描（4 组新结果）
FLUID=chorin GRAD_DIV=0   python main.py              # ~3 s  -> plot/afsi_result_chorin_g0.npz（塌缩）
FLUID=chorin GRAD_DIV=100 python main.py              # ~40 s -> plot/afsi_result_chorin_g100.npz
FLUID=ipcs   GRAD_DIV=0   python main.py              # ~5 s  -> plot/afsi_result_ipcs_g0.npz（塌缩）
FLUID=ipcs   GRAD_DIV=100 python main.py              # ~12 s -> plot/afsi_result_ipcs_g100.npz

# (3) 对照（自动载入 plot/ 下所有存在的运行，6 面板 × 6 曲线）
python compare.py                                     # -> figures/compare_*.png, compare_table.csv

# (4) 加密研究
python make_rubberband.py /tmp/rb64 64 1e-3 1.5 20    # 生成 IB2d 输入（k 按 Nx^2 缩放）
HERE=$PWD; cp ib2d_run/main2d.m /tmp/rb64/ && (cd /tmp/rb64 && octave --eval "addpath('$HERE/ib2d_run/IBM_Blackbox_octave'); main2d")
python ib2d_reference.py /tmp/rb64/viz_IB2d /tmp/rb64/ref.npz
IB2D_EXAMPLE=/tmp/rb64 OUTPUT_PATH=/tmp/afsi64 python main.py
python refinement.py figures/refinement.png N32:plot/afsi_result.npz:ib2d_reference.npz N64:/tmp/afsi64/afsi_result.npz:/tmp/rb64/ref.npz
```

IB2d 源码在仓库根目录 `third_party/ib2d`（已加入 `.gitignore`；不存在时 `run_ib2d_reference.sh` 会自动 clone）。

## 5. 文件

| 文件 | 说明 |
|---|---|
| `main.py` | AFSI 计算：`FLUID=rk2`（一维弹簧曲线 + 周期 Taylor–Hood + Peskin 两阶段）或 `FLUID=chorin/ipcs`（P2/P1 投影 + 4 点核 + 直接载荷，含 `GRAD_DIV`=γ 扫描） |
| `ib2d_io.py` | IB2d `input2d` / `.vertex` / `.spring` 读取 |
| `run_ib2d_reference.sh` | 运行 IB2d 原版 MATLAB 代码（MATLAB 或 Octave）并转换输出 |
| `octave_compat.py` | 生成 Octave 兼容的 `IBM_Blackbox` 副本（只改 3 处 I/O：`ver('MATLAB')`、`textscan` 的 NaN 补齐） |
| `ib2d_reference.py` | IB2d VTK 输出 → `ib2d_reference.npz` |
| `ib2d_reference.npz` | Octave 8.4 生成的 IB2d 参考解（$N_x=32$） |
| `compare.py` / `refinement.py` | 对照图表（6 面板 × 6 曲线、历史、场）/ 加密图 |
| `make_rubberband.py` | `Rubberband.m` 的 Python 版，生成任意 $N_x$ 的 IB2d 输入 |

## 6. 已知限制

- 串行（afsic 的 IB 耦合在 rank 0 汇总）。
- 与 IB2d 的逐项对照针对 `rk2` 管线；`chorin`/`ipcs` 管线用的是闭合方腔（非周期）
  与一阶显式耦合（求解 → 移动 → 重扩散），误差量级（$e_u\sim10^0$）大于 rk2，
  适合看“泄漏/稳定化”而不是精细收敛。
- `IBInterpolation` 的核函数不跨周期边界回绕；拉格朗日点须距边界 $>2h$（本算例满足）。
  落在右/上边界节点上的扩散力由 `periodic_fold` 折回左/下镜像节点。
- 仅移植了 IB2d 的线性弹簧（`alpha=1`）；其他纤维模型（beam、target、muscle）待后续算例。
