# demo_445 — IB2d 水母游动（Hoover Jellyfish）的 AFSI 复现

## 1. 问题描述

IB2d 的 `Example_Jellyfish_Swimming/Hoover_Jellyfish`（Hoover & Miller 2015,
J. Theor. Biol. 374: 13-25；Battista 的 IB2d 移植版）：二维"水母"截面——

* **钟形**：截断椭圆 $x=a\cos\theta,\ y=b\sin\theta$（$a=0.5$、$b=0.75$，下端截到
  $y=-0.25$），平移到 $(1.5,\ 2.0)$；顶点序 = 顶点 → 左臂 → 右臂；
* **弹簧**：轮廓链（$\kappa_{\rm spring}=10^7$）+ $n_{\rm musc}$ 根**肌肉弹簧**
  （收缩力 $F=10^5$）连接两臂下端；
* **非不变梁**：轮廓上每三点一根（$\kappa_{\rm beam}=2.5\times10^5$；文件存参考
  二阶差分 $\mathbf C$）；
* **flow blocker**：顶边一排目标点（$k=2.5\times10^6$，钉在初始位置）；
* 流体：$\rho=1000$、$\mu=6.667$、周期盒 $[0,3]\times[0,10]$；
* **驱动**：每步把肌肉静长设为 $L(t)=|\cos(\text{freq}\,\pi t)|$（freq = 2 ⇒
  $|\cos 2\pi t|$，每 0.5 s 一次收缩）——对应 IB2d 参考实现的 `update_Springs`。

原始设置是 $192\times640$ 网格、$\mathrm dt=10^{-5}$、$T=5$ s（50 万步）。
本 demo 用**精确 4× 粗化**的对应算例：$48\times160$ 网格、
$\mathrm ds=\mathrm dx/2=1/32$、$\mathrm dt=4\times10^{-5}$、$T=1$ s（2 次收缩脉冲）；
几何由 `make_jelly.py` 按 IB2d 生成器在粗分辨率下**重新生成**（结构元素数量随 ds
自动变化，与原版生成器逐位一致，见 §2.3）。

## 2. 方法

### 2.1 IB2d 与 AFSI 的对应关系

| 环节 | IB2d（pyIB2d） | AFSI（本 demo） |
|---|---|---|
| 流体离散 | $48\times160$ 同位网格中心差分 + FFT | Q2/Q1 Taylor–Hood，$48\times160$ 四边形单元（顶点格 = IB2d 网格），grad-div γ=100 |
| 周期边界 | FFT | 周期约化 $P^\top A P$ |
| 时间推进 | Peskin (2002) 两阶段：$\Delta t/2$ 隐式 Euler + $\Delta t$ Crank–Nicolson | `PeskinRK2Solver`（相同） |
| IB 网格 / 核 | Peskin 4 点核 | `IBMesh(order=1)` + `IBInterpolation`（同一 4 点核） |
| 固体 | 弹簧 + 非不变梁 + 目标点，肌肉静长每步更新 | 弹簧走 FE 能量式；梁/目标点为原式 numpy 力；肌肉静长每步更新 |
| 拉格朗日权重 | $\mathbf F\cdot\mathrm ds$ | 相同（$\mathrm ds=\min(L_x/2N_x,\ L_y/2N_y)$） |
| 拉格朗日点更新 | $\mathbf X^{n+1/2}=\mathbf X^n+\frac{\Delta t}{2}\mathbf U^n$；$\mathbf X^{n+1}=\mathbf X^n+\Delta t\,\mathbf U^{n+1/2}$ | 相同 |

### 2.2 三种力的实现

* **弹簧**：FE 能量式 $E=\sum_e \frac{k}{2h_0}\,(h_0|\mathbf X_s|-L_e)^2$，P1 下逐点
  等价于 IB2d 的 $k(|\Delta\mathbf X|-L)\,\widehat{\Delta\mathbf X}$（**任意弹簧图**
  均适用——水母的链 + 肌肉共 87 根）；肌肉单元的 $L_e$ 每步按
  $|\cos(\text{freq}\,\pi t)|$ 更新（`freq=2`，与参考实现一致）；
* **非不变梁**：$\mathbf F_{p_1}, \mathbf F_{p_3} \mathrel{-}= k_b(\Delta^2\mathbf X-\mathbf C)$，
  $\mathbf F_{p_2} \mathrel{+}= 2k_b(\Delta^2\mathbf X-\mathbf C)$（IB2d 原式）；
* **目标点**：$\mathbf F_i=k_t(\mathbf X_i^{\rm target}-\mathbf X_i)$（最小镜像），
  锚点 = 初始位置；
* 三者求和后统一 $\times\,\mathrm ds$、再经 4 点核扩散（IB2d 约定）。

### 2.3 验证与减分辨率设置

* **弹簧力校核**：扰动构型下 FE 弹簧力与 IB2d 公式相对误差 $2\times10^{-15}$；
* **几何生成**：`make_jelly.py` 在 $N_y=640$ 下生成的 `.vertex/.spring/.target/
  .nonInv_beam` 与原版文件逐位一致（最大差 1 ulp：415 点、39 肌肉、357 弹簧、317 梁、96 目标点）；
* **4× 粗化**：$48\times160$、$\mathrm ds=1/32$、$\mathrm dt=4\times10^{-5}$——
  dt 按 dx 线性缩放（IB2d 显式格式的弹簧稳定性限制，原版 $\mathrm dt=10^{-5}$ @ $\mathrm dx=1/64$）。
  结构变为 103 标记（79 钟 + 24 blocker）、87 弹簧（9 肌肉）、77 梁、24 目标点。

## 3. 结果

![shapes](figures/compare_shapes.png)

![history](figures/compare_history.png)

与 IB2d 参考（pyIB2d，本仓库 `third_party/ib2d/pyIB2d`）的对照（com_y = 钟形 79 个标记的质心 y，γ 为 grad-div 系数）：

> 参考解全程纯 Python（pyIB2d）；它与原版 MATLAB 的参考在本算例上是**同一解**
> （100 帧逐帧：标记点差 $\le 5\times10^{-11}$、速度差 $\le 5\times10^{-7}$、
> 压力差 $\le 0.05$ ≈ 相对 $10^{-4}$，即迭代/输出精度内），下述对照结论不变。

| $t$ (s) | IB2d com_y | AFSI γ=0 | 差 | AFSI γ=100 | 差 |
|---|---|---|---|---|---|
| 0.05 | 2.33079 | 2.33080 | $+10^{-5}$ | 2.33080 | $+10^{-5}$ |
| 0.10 | 2.33124 | 2.33130 | $+6\times10^{-5}$ | 2.33122 | $-2\times10^{-5}$ |
| 0.15 | 2.33926 | 2.33747 | $-0.0018$ | 2.33839 | $-0.0009$ |
| 0.20 | 2.36021 | 2.35290 | $-0.0073$ | 2.35784 | $-0.0024$ |
| 0.25 | 2.39356 | 2.37744 | $-0.0161$ | 2.38871 | $-0.0048$ |
| 0.30 | 2.44331 | 2.41251 | $-0.0308$ | 2.43480 | $-0.0085$ |

* 第一次收缩内：γ=100 全程偏差 ≤ 钟高的 1%（~0.01），γ=0 逐渐落后到 0.03——粗网格上
  P2/P1 投影的伪散度削弱了射流，grad-div 项把它压回 IB2d（FFT 精确投影）的行为，
  故默认取 γ=100（与 demo_444 一致）；
* **全程**（0–1 s，两次收缩脉冲）：γ=100 的标记点最大偏差
  $\overline{\max_k|\Delta\mathbf X|}=0.021$（最大 0.054、末帧 0.054），
  基本均在网格间距 $h=1/16=0.0625$ 以内；γ=0 为均值 0.041 / 最大 0.088。两个脉冲内
  γ=100 都更贴近参考；
* 脉冲结束后两者与 IB2d 的质心差都停在 ~0.05（约 1/4 个脉冲的相位差）——粗分辨率下
  FEM + P2/P1 投影与 IB2d 的同位有限差分 + FFT 精确投影之间的离散差异；
* 完整的每个 0.01 s 对照（dXmax / com_y / apex_y / 臂端距）见
  `figures/compare_table.csv`；涡量场与速度差见 `figures/compare_fields.png`。

（调试记录：肌肉驱动频率曾误写成 $|\cos(2\pi f t)|$（freq 前多乘了 $2\pi$），
导致 2× 频率的系统性相位偏差；修正为 IB2d 的 $|\cos(\text{freq}\,\pi t)|$ 后，
早期轨迹差从 ~0.04 降到 ~0.001。）

## 4. 运行

全程纯 Python：AFSI 侧用 `main.py`，参考侧用 `run_reference.py`（官方 pyIB2d 移植）；
不需要 MATLAB/Octave，也不依赖本地 IB2d 源码（首次运行自动 clone pyIB2d 到
`<repo>/third_party/ib2d`）。

```bash
conda activate afsi-dolfinx
cd afsic/demo/ib2d/demo_445

# (1) AFSI（γ=100 为默认；本机约 13 min）
python main.py                            # -> plot/afsi_result_g100.npz + XDMF
TFINAL=0.002 python main.py               # 冒烟
GRAD_DIV=0 python main.py                 # 纯投影（对照，见 §3）
UPDATE_SPRINGS=0 python main.py           # 无肌肉驱动（被动钟）

# (2) IB2d 参考（纯 Python，pyIB2d；本机约 6 min）
python run_reference.py                   # -> ib2d_reference.npz
python run_reference.py --tend 0.02       # 冒烟
# 首次运行会自动 git clone pyIB2d 到 <repo>/third_party/ib2d（无需 MATLAB/Octave）

# (3) 对照
python compare.py                         # -> figures/*.png + compare_table.csv
```

原分辨率（笔记本上不建议）：用 `python make_jelly.py /tmp/jelly_full 640 1e-5 5.0 5000`
重新生成几何（该目录同时含 `input2d`），AFSI 侧
`IB2D_EXAMPLE=/tmp/jelly_full python main.py`（约 50 万步）。

## 5. 文件

| 文件 | 说明 |
|---|---|
| `main.py` | AFSI 计算（rk2 管线 + 弹簧/梁/目标点 + 肌肉驱动，`GRAD_DIV` 选择 γ） |
| `make_jelly.py` | `Make_Jelly_Geometry.m` 的 Python 版 + `input2d` 写入（任意 $N_y$） |
| `ib2d_input/` | 减分辨率算例输入：`input2d`、`jelly.vertex/.spring/.target/.nonInv_beam` |
| `ib2d_io.py` | IB2d 输入读取（含 `.target` 与 `.nonInv_beam`） |
| `run_reference.py` | 参考解（纯 Python）：pyIB2d 运行 + VTK→npz；肌肉驱动 `update_Springs.py` 自动生成 |
| `ib2d_reference.py` | pyIB2d VTK 输出 → npz 转换（结构点 + u/p 场，朝向自适应） |
| `compare.py` | 对照图表与数据表（形状 / 历史 / 涡量场；γ=0 曲线自动叠加） |

## 6. 已知限制

- **减分辨率**（4× 粗化、$T$ 截到 1 s）：结构元素数随 ds 变化（肌肉 39 → 9 根，$k$
  按生成器规则不缩放），属定性复现而非定量收敛研究；
- 显式耦合（与 IB2d 相同）⇒ $\mathrm dt$ 受限；粗化后取 $\mathrm dt=4\times10^{-5}$；
- blocker 点在本实现中用一条**零刚度链**挂进结构 FE（dolfinx 会丢弃孤立点），
  其力严格为 0，与 IB2d 的纯目标点等价；
- 与 IB2d 的残余差异（γ=100 时 ~0.01 量级）主要来自：FEM 离散 vs 同位有限差分、
  P1 顶点网格的核插值细节、P2/P1 投影 vs FFT 精确投影。
