# demo_451 — Tracers in Impedance Pump（阻抗泵 + 示踪粒子，IB2d）

pure-python 参考：`third_party/ib2d/pyIB2d/Examples/Tracers_In_Impedance_Pump`
（已拷贝为 `ib2d_input/`，0-based，无需索引换算）。

## 物理设置

5×5 盒，$\mu=1,\ \rho=1$，64×64。柔性"心管"（156 个标记点：154 根管壁
弹簧 $k=10^7$ + 152 个不变梁 $k_b=7.5\times10^7$ + 4 个角点目标点）浸在
流体中，其 **11 根跨径"泵弹簧"**（$k=10^5$，``.spring`` 的第
163–173 行）静止长度被时间驱动：

$$\text{RL}(t) = d - \left|0.9\,d\,\sin(2\pi f t)\right|,\qquad f=10\ \text{Hz},\ d=1$$

（示例自带 `update_Springs.py`；pyIB2d 中即弹簧行 ``N-3+10 : N-2+20``，
$N$ 为标记点数）。管子被周期性地捏扁再回弹——这正是"阻抗泵"：无阀门，
靠管壁不同位置收缩造成的流体阻抗差产生**净单向输运**。

110 个 **示踪粒子**（tracers）随流体运动，不产生任何力：
$X_t^{n+1}=X_t^n+\Delta t\,U_h(X_t^n)$（driver 中紧随标记点、用同一 4 点核）。

## 数值实现要点

- **泵激活时点**：driver 每步先用步初时刻 $t_n$ 更新静止长度、再计算回归力
  （``update_Springs`` 在力计算前调用，`current_time = n·dt`）。
- **dt**：官方 $\Delta t=10^{-3}$；AFSI 的显式耦合在该步长下 ~30 步即失稳
  （NaN），$\Delta t=5\times10^{-4}$ 稳定 ⇒ AFSI 用 $5\times10^{-4}$、
  `PRINT_DUMP=10`，使输出帧间隔（5 ms）与参考（$10^{-3}\times5$）对齐。
- **pyIB2d tracer 移植 bug**：`IBM_Driver.py` 的 tracer 移动调用只传了
  11 个参数，而 `Supp.py` 的函数签名要 15 个（缺 `porous_Yes..F_Poro`）。
  `run_reference.py` 内做了 dispatcher 补丁（标记点调用的 15 参数直通），
  不改动 third_party 源码。

## 结果（T = 2 s，400 帧 @5 ms，AFSI dt=5e-4 vs 参考 dt=1e-3）

**结构（可靠对比量）**：

| 量 | 结果 |
|---|---|
| 标记点偏差 $\max_k\lVert X^{AFSI}-X^{IB2d}\rVert$ | 全程 $1.6\times10^{-2}\sim5.1\times10^{-2}$（0.2–0.65 h） |
| 管高（捏扁-回弹） | 相位一致；幅值差 ~0.5–3%（如 t=0.66: 1.107 vs 1.120；t=0.825: 0.989 vs 0.984） |
| 参考自身 dt=1e-3 vs 5e-4 | 结构差 ≤8e-3（0.1h）——参考结构已收敛 |

**示踪粒子的两个重要事实**：

1. **tracer 轨迹是拉格朗日混沌的**：把参考自己的 dt 从 1e-3 减半到
   5e-4，tracer 逐点位置在 t≈1.0 后发散到 4.5（全局尺度）——中点附近的
   拉伸-折叠动力学对微小扰动指数敏感。**逐点 tracer 轨迹不能作为对比
   基准**；有意义的量是分布统计（中位数 / 分位数）与云的整体输运。
2. **泵的整流（净输运）两码定量不同**：云团 x 中心在云被拉散前
   （t≲0.8）：

   | $t$ | AFSI com$_x$ | IB2d com$_x$（dt=1e-3） | 参考自身 dt 差 |
   |---|---|---|---|
   | 0.25 | 2.158 | 2.240 | 0.012 |
   | 0.50 | 2.612 | 2.920 | 0.003 |
   | 0.75 | 3.143 | 3.738 | 0.035 |

   即 AFSI 的整流流量比 IB2d 小 10–16%（参考自身 dt 收敛到 <1%，说明
   这不是数值噪声而是**两套离散格式的整流差异**——泵的净流量由粘性/惯性
   的细微平衡决定，Q2/Q1+变分投影与规则格子中心差分投影在粗网格上给出
   定量不同的整流效率，恰如此算例的设计意义"impedance pump is hard"）。
   截面周期平均流速进一步显示：左端（x=1.5）两码同号且差 ~20%，管中段
   （x=2.0–2.5）AFSI 明显偏小甚至回流。

**收敛性实验（判别"数值噪声"还是"格式差异"）**：

| 实验 | 结果 |
|---|---|
| 参考 dt=1e-3 vs 5e-4 | 结构差 ≤8e-3；云 com$_x$ **差 <1%**（如 t=0.8: 3.905 vs 3.853）→ 参考收敛 |
| AFSI dt=5e-4 vs 2.5e-4 | 结构差 3.4e-3（0.04h）；云 com$_x$ 差 1–2%（如 t=0.8: 3.257 vs 3.214）→ AFSI 亦收敛 |
| AFSI（任一 dt）vs 参考 | com$_x$ 偏差 2.6%→17%（t=0.2→0.8，随时间累积） |

⇒ 两套离散**各自收敛到不同的整流流量**（差 ~15–20%）：泵净流量由
粘性/惯性在泵周期内的微妙平衡决定，是两格式"数值粘性/色散"差异的
二阶放大——与 447 同类但温和得多（结构本身匹配到亚格子）。

**空间离散敏感性（128×128 实验）**：把 AFSI 网格加密一倍（结构 ds=0.051
不变，此时 ds/h=1.3，已偏离 IB 惯例 ds≈h/2）后，整流流量进一步变化
（云心 @t=0.4：AFSI64 2.41 / 参考 2.63 / AFSI128 **3.64**），管宽脉动
几乎翻倍（std 0.039 → 0.079，参考 0.044）。该配置本身并非"更精确解"
（标记-网格采样失配），但它表明**本算例的整流流量对 IB 力散布的离散
配置高度敏感**：64×64（ds/h=0.65，与 IB2d 官方配置一致）下两码的
15–20% 差异应理解为"粗网格+格式"的联合不确定带，而非某方更正确；
定量该泵的输运需要 ds、h、dt 的联合收敛研究。

结论：demo_451 是**离散敏感应力算例**——定性物理（泵送方向、管壁相位、
tracer 被推送）一致，结构在 ~0.5h 内可比，而净输运流量与 tracer 逐点
轨迹在此分辨率下不可定量对比（前者离散敏感、后者混沌）。

## 文件

- `main.py`          AFSI 运行（`TFINAL=0.02 python main.py` 冒烟；
                     `DT=5e-4 PRINT_DUMP=10 python main.py` 正式），
                     `plot/afsi_result_g<GRAD_DIV>.npz`（`X` 标记、`Xt` 示踪、
                     `u/p` 网格场）
- `run_reference.py` 纯 Python pyIB2d 参考（自带 lag/out_par 表；含 tracer
                     调用补丁 + `update_Springs.py` 拷入运行目录）
- `ib2d_io.py`       0-based 输入读取（本 demo 加了 `.tracer` 读取）
- `ib2d_reference.py` VTK→npz 转换（支持 `tracer.*.vtk`）
- `compare.py`       对比表 + `figures/`（形状、管宽/示踪位置历史、场图）

```bash
conda run -n afsi-dolfinx python run_reference.py
DT=5e-4 PRINT_DUMP=10 conda run -n afsi-dolfinx python main.py
conda run -n afsi-dolfinx python compare.py
```
