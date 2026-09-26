# 在本机环境运行 demo_424 / demo_426

本文档只讲**当前这台机器上、用仓库自带的环境**怎么把这两个算例跑起来、跑完怎么看结果、
以及哪些参数组合已经被实测验证过（含踩过的坑）。两个 demo 自己的 `readme.md` 里的
`conda activate afsi-dolfinx` 在本机**不适用**（本机 PATH 里既没有 `conda` 也没有
`micromamba`，环境是仓库内的一份 micromamba 前缀，入口脚本见下）。

---

## 1. 环境与统一入口

本机环境事实（实测）：

| 项 | 值 |
|---|---|
| 解释器 | Python 3.12.13 |
| FEniCSx | `dolfinx 0.10.0` / `basix 0.10.0` |
| PETSc | `petsc4py 3.25.5` |
| 其他 | `numpy 2.5.3`、`meshio 5.3.5`、`pyvista 0.49.0`、`matplotlib 3.11.1` |
| AFSI | 已装进该环境（`afsic/src/afsic`，editable），`import afsic` 直接可用 |
| 环境前缀 | `<repo>/.conda/envs/afsi-dolfinx`（micromamba，无需 root） |

**统一入口是仓库根目录下的 `.tools/afsi-run.sh`**：它设置 `MAMBA_ROOT_PREFIX`、
`HOME=<repo>/.home`、`PIP_CACHE_DIR`、并关掉 HDF5 文件锁（本机 HDF5 advisory lock 不可靠，
中断后会留下 stale lock），然后 `micromamba run -p .conda/envs/afsi-dolfinx "$@"`。

```bash
export REPO=$(git rev-parse --show-toplevel)   # 在仓库里执行
RUN=$REPO/.tools/afsi-run.sh

# 自检：能打印 0.10.0 就说明环境可用
$RUN python -c "import dolfinx, afsic; print(dolfinx.__version__)"
```

不用这个 wrapper 时的等价写法：

```bash
export MAMBA_ROOT_PREFIX=$REPO/.conda/mamba
export HOME=$REPO/.home
export PIP_CACHE_DIR=$REPO/.conda/pipcache
export HDF5_USE_FILE_LOCKING=FALSE
$REPO/.tools/bin/micromamba run -p $REPO/.conda/envs/afsi-dolfinx python main.py
```

约定：

* **串行单进程**。`mpirun -n 2 ...` 能启动不报错，但 IB 耦合（`IBMesh`/`IBInterpolation`）
  与 `verify` 报告都按 rank 局部数据计算，两个 rank 会各打印一份不同的报告，结果是错的 —— 别用。
* 输出目录由**脚本所在目录**决定（`configuration.output_path()`），与 cwd 无关；同一配置
  重跑会**覆盖** `plot/<tag>/` 里的旧结果。要保留旧结果就先整份拷目录再跑：

  ```bash
  cp -r $REPO/afsic/demo/demo_426 /path/to/scratch/demo_426_try1
  ```
* `plot/` 输出目录会自动创建（dolfinx 的 XDMF 写入器会建父目录），不用手工 `mkdir`。
* 固体网格要**先生成**：`generate_mesh.py` 产出的文件名带参数（424：`CASE/NY/SOLID_DIV`；
  426：`N`/`PLATE_THICKNESS`），改了这些参数必须重新生成，否则 `main.py` 读不到网格。

---

## 2. demo_424（二维系绳主动脉：`open` / `closed`）

### 2.1 运行

```bash
cd $REPO/afsic/demo/demo_424

# 1) 生成固体（壁条；closed 再加封堵膜）网格
CASE=open NY=45 SOLID_DIV=4 $RUN python -B -u generate_mesh.py

# 2) 跑算例（默认 CASE=open NY=45 SOLID_DIV=2 SOLVER=chorin DT=2e-4 T_END=0.4）
CASE=open NY=45 SOLID_DIV=4 SOLVER=ipcs T_END=0.4 $RUN python -B -u main.py

# 最短自检（100 步，约 10 s）
CASE=open NY=45 SOLID_DIV=4 SOLVER=ipcs T_END=0.02 $RUN python -B -u main.py
```

### 2.2 可调环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `CASE` | `open` | `open`（通腔）或 `closed`（中段封堵，外形呈 "H"） |
| `NY` | `45` | 横向网格数，必须是 9 的倍数（45/90/180），`H=BOX_W/NY` |
| `SOLID_DIV` | `2` | 固体单元 = `H/SOLID_DIV`；`2` 与 `4` 精度相当（readme 的 S2 为 5.43 %，S4 实测 5.42 %）|
| `SOLVER` | `chorin` | `chorin` 或 `ipcs`；`ipcs` 走 `main.py` 内置的 traction 修正 |
| `DT` | `2e-4` | 时间步 |
| `T_END` | `0.4` | 物理时长，`NSTEPS=T_END/DT`（默认 2000 步） |
| `RAMP_T` | `0.05` | 入口压力线性上升时间 |
| `DP_MMHG` | `0.02`（`closed` 为 `0.2`） | 驱动压差 [mmHg] |
| `BETA` | `1e7` | 系绳（体积弹簧）刚度 [N/m³]，**不要随手调小**：调到 `1e4` 时 open 的 `u_max` 误差从 −8.6 % 变 +23 %（见 findings 文档 §6）|
| `DIAG=1` | — | 打印压力分布、`\|u\|` 极值位置等诊断 |
| `VELOCITY_BC` | `0` | `1` = 改用两端速度 Dirichlet 驱动（实验分支） |
| `GAP_DRAG` | `open` 为 `0`，`closed` 为 `1e6` | 外侧间隙的隐式阻尼（`closed` 用） |
| `GAP_DRAG_LENGTH` / `EDGE_DRAG` / `EDGE_BAND` | `-1` / `0` / `2H` | 阻尼只加在端部 / 壁面附近额外阻尼（实验分支） |
| `FLOW_DIAG_EVERY` | `NSTEPS/20` | `flow_history.csv` 的采样间隔 |

其余辅助脚本：`test_channel.py`（无固体对照，隔离开边界处理）、`test_ipcs.py`
（IPCS 压力边界对照）、`verify.py`（解析参考与误差指标）。

### 2.3 输出

`plot/<CASE>_NY<NY>[_S<SOLID_DIV>][_<solver>]/`：

```
verify.json        全部误差指标（机器可读）
flow_history.csv   Q_gap / Q_leak / max|u| 历史
velocity.xdmf(.h5) pressure.xdmf(.h5)
solid_displacement000000.pvtu, solid_displacement_p0_000000.vtu  (固体位移)
```

判读要点（`open`）：`u_relL2_x0.25/0.5/0.75`（腔道剖面相对 L2，目标 ≲5 %）、
`u_max_rel_err`（目标 −5 % 左右）、`wall_disp`（壁面漂移，目标 ≲0.2 格）、
`G_fit` vs `G_ideal`（压力梯度，应差 ≲0.1 %）。

### 2.4 已验证配方（实测）

`CASE=open NY=45 SOLID_DIV=4 SOLVER=ipcs T_END=0.4`，结果在
`plot/open_NY45_S4_ipcs/verify.json`：

| 指标 | 值 |
|---|---|
| `u_relL2_x0.5` | 5.42 % |
| `u_max_num` / 解析 | 0.8101 / 0.8474 m/s（−4.40 %）|
| `G_fit` / `G_ideal` | 26.3624 / 26.4005 Pa/m（−0.14 %）|
| `wall_disp` | 1.647e-4 m（≈0.165 格）|
| 耗时 | 225 s（2000 步）|

默认 `SOLVER=chorin NY=45 SOLID_DIV=2` 的 readme 数字是 5.43 % / −4.42 %，两者一致。

---

## 3. demo_426（斜通道 IB 基准，二维）

### 3.1 运行

```bash
cd $REPO/afsic/demo/demo_426

# 1) 生成两块斜板（文件名带 N 与板厚）
$RUN python -B -u generate_mesh.py

# 2) 推荐配方：用隐式 drag 带把板子按住（实测 5.3 % 通道误差）
SMOKE=1 SMOKE_STEPS=320 USE_IMPLICIT_DRAG=1 PLATE_DRAG=1e4 PLATE_DRAG_BAND=1.5 \
    $RUN python -B -u main.py

# 3) 基准全长跑（T_END=20 s）
T_END=20 USE_IMPLICIT_DRAG=1 $RUN python -B -u main.py

# 最短自检（50 步，约 20 s）
SMOKE=1 SMOKE_STEPS=50 USE_IMPLICIT_DRAG=1 $RUN python -B -u main.py
```

### 3.2 可调环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `N` | `32` | 网格，`DX=1/N`（网格数由盒子尺寸推出，不反过来） |
| `DT_FACTOR` | `0.2` | `DT=DT_FACTOR*DX`；`DT_OVERRIDE` 可直接给 `DT` |
| `T_END` | `2.0` | 物理时长（基准为 20 s） |
| `SMOKE` / `SMOKE_STEPS` | — / `50` | `SMOKE=1` 时按 `SMOKE_STEPS` 步跑，输出目录带 `_smoke` |
| `THETA_DEG` | `30` | 通道倾角 |
| `DP_DL` | `1.0` | 轴向压力梯度（驱动） |
| `X_MIN/X_MAX/Y_MIN/Y_MAX` | 扩展盒子 | 几何域，见 `configuration.py` 注释 |
| `SHIFT_Y` | 由 `PLATE_MARGIN` 推出 | benchmark 坐标系在本框架里的纵向平移 |
| `RAMP_T` | `0.1` | 入口解析剖面线性上升时间 |
| `SOLVER` | `ipcs` | `ipcs`（已验证）或 `chorin` |
| `USE_IMPLICIT_DRAG` | `1` | `1` = 用动量 LHS 里的隐式 drag 带按住板子（**推荐**）|
| `PLATE_DRAG` / `PLATE_DRAG_BAND` | `1e4` / `1.5` | drag 系数 [1/s] / 带宽（格） |
| `BETA` / `DAMP` | `1e6` / `1e3` | 系绳（显式 Peskin 路线）参数，见 §4.1/§4.2 |
| `SPREAD_NORM` | `none` | Lagrangian 力的缩放：`none`（默认，按 AFSI 原样，IB 力不缩放）/ `h2`（旧默认，等于乘 `DX²`，把 IB 力削弱 1024 倍）/ 数值 |
| `LAGRANGIAN_DIV` / `PLATE_THICKNESS` | `2` / `DX` | 固体网格沿板间距 = `DX/DIV`；板厚（向外长，不改通道内表面）|
| `MAP_AT_REFERENCE` | `0` | `1` = 核函数固定取参考构型位置 |
| `USE_BODY_FORCE` | `0` | `1` = 额外加体力 `-grad(p)`；**开 drag 带时被忽略，见 §4.3** |
| `IB_ITERATIONS` / `DIAG_IB` | `1` / — | 步内定点迭代次数 / 打印 `\|F_lag\|`、`\|f_fluid\|`、δ |
| `X_PROFILE` | `0.5` | 剖面采样位置（该采样线目前有 bug，见 §4.4）|

辅助脚本：`verify.py`（指标）、`tune_C.py` / `tune_C2.py`（β 标定）、`sweep_tether.py`、
`plot/make_beta_figures.py`、`plot/make_cgs_figures.py`、`plot/plot_kappa200_dt.py`。

### 3.3 输出

`plot/N<N>_<solver>[_smoke]/`：

```
verify.json          指标
history.csv          step,t,max|u|,plate_disp
velocity.xdmf(.h5)   pressure.xdmf(.h5)
solid_coords.xdmf(.h5)  solid_displacement.xdmf(.h5)
```

判读要点：**看 `err_L2_rel_channel`**（通道内相对 L2，目标 ≲5 %）、`plate_u_max`
（板面流体 |u|，应 ≪ u_max，≈0.005 u_max 算合格）、`plate_max_disp`（板漂移，应为 0 或
≪0.5 格）。`err_L2_rel_box`、`profile_relL2`、`u_max_xi` 目前**不可信**（§4.4）。

### 3.4 已验证配方（实测）

`SMOKE=1 SMOKE_STEPS=320 DT_FACTOR=0.2 USE_IMPLICIT_DRAG=1 PLATE_DRAG=1e4`
（N=32，320 步 ≈ 2 min）：

| 指标 | 值 |
|---|---|
| `err_L2_rel_channel` | **0.0533** |
| `plate_u_max` / `plate_u_mean` | 0.00526 / 0.00159 m/s（≈0.03 / 0.01 u_max）|
| `plate_max_disp` | 0（板子按构造固定）|
| `u_max_num` | 0.1649（解析 0.16667，−1.06 %）|

驱动方式（入口解析速度 BC vs 额外体力）对这个结果**没有影响**（两种设置结果 4 位有效数字
完全相同）；readme 里 6.76 % 的那次也是同一路线。

---

## 4. 已知坑（会改变你怎么跑、怎么读结果）

### 4.1 `SPREAD_NORM=h2` 是错的，已回退（默认改为 `none`）

commit `00d7d8c` 依据"实测对偶性比值 `<S*u,F>/<u,SF>` 按 h² 变化"加了 `SPREAD_NORM=h2`
并设为默认。那个 h² 实际上是**测量时漏掉了网格面积因子 `dx·dy`**（`dx=1/(2N)`）：

| 内积写法 | N=16 | N=32 | N=64 | N=128 |
|---|---|---|---|---|
| 节点和 `Σ u·(SF)` | 9.765625e-04 | 2.441406e-04 | 6.103516e-05 | 1.525879e-05 |
| 体积权 `Σ u·(SF)·dx·dy` | **1.000000** | **1.000000** | **1.000000** | **1.000000** |

readme 里那三个数正好等于 `dx·dy`；加上体积权后比值精确为 1（`afsic/tests/test_duality.py`
本来就是按体积权写的，一直通过）。所以扩散/插值算子本来就是严格伴随的，
`h2` 只是把 Lagrangian 力乘了 `DX²=9.77e-4`。

后果：板子几乎不受力 → 等于**没有通道壁**。当时提交的
`plot/N32_ipcs_smoke`（`κ=60` + `h2`）实测就是这个状态：板沿板切向漂 4 格、
**通道外**流体 `|u|` 达 0.37 u_max（本应静止）、通道中心线 `|u|=0.0893` vs 解析 0.1667、
通道内 L2 = 34.7 %。

**已按最小范围回退**（只回退这一处，其余提交保留）：
`configuration.py` 里 `SPREAD_NORM` 默认改成 `none`（`SPREAD_SCALE = 1.0`，实测确认），
并更正了该处注释与 demo_426 readme 中"归一化缺陷"那一节（标注 RETRACTED）。
`h2` 仍可作为旋钮显式使用，但不要再当默认。

* 走显式（系绳）路线：保持 `SPREAD_NORM=none`（现在就是默认）。
* 用默认的隐式 drag 带（§3.4）时 `SPREAD_SCALE` 根本不参与计算，不受影响。

### 4.2 即使 `SPREAD_NORM=none`，`dt=0.2dx` 下系绳路线也不稳定

静止持位实验（关掉驱动、把板向外偏 0.5 格、看位移是否收敛）：

| 配置 | 5 步 | 20 步 | 200 步 |
|---|---|---|---|
| 426, `BETA=1e6 SPREAD_NORM=none` | 3.3e-1 | 3.8e14 | 崩 |
| 426, `BETA=1e5 SPREAD_NORM=none` | 1.8e-2 | 6.8e-1 | 3.34 |
| 426, `BETA=1e6 SPREAD_NORM=h2` | 1.57e-2 | 1.57e-2 | 1.59e-2（冻住，无恢复力）|
| **424, `BETA=1e7`（自身 dt=2e-4）** | — | — | 0.5 格扰动衰减到 2.7e-4（**收敛**）|
| **424, `BETA=1e7`，dt 换成 426 的 6.25e-3** | 第 10 步 max\|u\|=2.4e22 | | |

控制量是显式罚力的每步增益 ∝ `β·dt²/ρ`：424 = 0.4（稳），426 在 `β=1e6` 时 = 39（炸）、
`β=1e5` 时 = 3.9（缓慢发散）。板厚改 2dx/4dx、Lagrangian 网格加密、`MAP_AT_REFERENCE=1`
都仍然发散 —— 所以这不是几何问题，而是**显式（冻结力）耦合的时间步预算**问题：
424 用 `dt=2e-4` 才站得住。结论：**426 里要"按住"静止板，用隐式 drag 带，不要用系绳。**

### 4.3 `USE_BODY_FORCE=1` 在开 drag 带时被静默忽略

`main.py` 里加体力那行位于 `if not cfg.USE_IMPLICIT_DRAG:` 块内，所以
`USE_IMPLICIT_DRAG=1` 时体力根本没加（驱动完全来自入口/出口的解析速度边界条件）。

### 4.4 Fig.23 剖面采样线漏了 `SHIFT_Y`

`main.py::profile_line()` 用的是 benchmark 坐标系、没加 `SHIFT_Y`，线过 `(0.433, 0.250)`，
而通道中心线应是 `(0.5, 1.055342)`；沿线 `xi ∈ [-1.2413, -0.0866]`，201 个采样点只有 85 个
落在通道内，其余都在跟"无界抛物线"解析解比（readme 自己也提醒过整盒指标有这个假象）。
所以 `verify.json` 里的 `profile_relL2`（0.74）、`profile_Linf`、`u_max_xi`（=+0.5774，
正好是壁面）都不可信；真正有意义的是 `err_L2_rel_channel`。

正确写法（把 `X_PROFILE=0.5` 处的中心线取对，沿 `(-sinθ, cosθ)` 走 `t`）：

```python
y_c = cfg.SHIFT_Y + ((X - cfg.SHIFT_X) * cfg.SIN_T) / cfg.COS_T
x = X - t * cfg.SIN_T
y = y_c + t * cfg.COS_T          # 沿该线 xi == t
```

---

## 5. 结果可视化

* **ParaView**：直接开 `velocity.xdmf` / `pressure.xdmf`（`*_smoke` 目录同样适用）。
* **脚本出图**：demo_424 见 `afsic/demo/demo_424/plot/PLOTTING_GUIDE.md`（含 headless OSMesa
  写法，本机 pyvista 0.49 已可用，离屏渲染实测能出 PNG）；demo_426 用
  `plot/make_beta_figures.py`、`plot/make_cgs_figures.py`、`plot/plot_kappa200_dt.py`。
* 读 XDMF 用 `meshio.xdmf.TimeSeriesReader`（VTK 的 `XdmfReader` 对当前 2-D XY 格式容易出错）。

---

## 6. 自检与耗时参考

```bash
# IB 算子伴随性（应为 9 passed）
cd $REPO/afsic && $RUN python -B tests/test_duality.py
```

| 运行 | 规模 | 实测耗时 |
|---|---|---|
| 424 `open NY=45 SOLID_DIV=4 ipcs T_END=0.4` | 2000 步 | ≈225 s |
| 424 `open NY=45 ... T_END=0.02` | 100 步 | ≈10 s |
| 426 `N=32 SMOKE_STEPS=320` | 320 步 | ≈2 min |
| 426 `N=32 SMOKE_STEPS=50` | 50 步 | ≈20 s |
| 426 `T_END=20`（`DT_FACTOR=0.2` → 3200 步） | 3200 步 | ≈20 min |
| 426 `T_END=20 DT_FACTOR=0.15`（readme 的基准全长，4267 步） | 4267 步 | readme 记录 2922 s |

详细的对照实验、脚本与原始数据在仓库根的 `.diag426/`（未跟踪的临时目录，可删）；
§4 各项结论的完整实测记录见 `docs/demo-426-ib-coupling-findings.md`。
