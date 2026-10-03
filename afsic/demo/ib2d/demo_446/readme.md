# demo_446 — IB2d "Wobbly Beam"（摆动梁）的 AFSI 复现

## 1. 问题描述

IB2d 的 `Examples/Wobbly_Beam`（pyIB2d 版，输入文件逐字复制自
`third_party/ib2d/pyIB2d/Examples/Wobbly_Beam`）：一根两端被钉住的弹性"拱形梁"
（从 (0.25, 0.5) 到 (0.75, 0.5) 的拱，最高点 y ≈ 0.625）放在 1×1 的周期盒黏性流体
中（μ = 0.1、ρ = 1），释放后梁上下摆动并不断向流体中释放涡量。

结构（0 基索引，直接读 `ib2d_input/`）：

* **不变梁（invariant beams）**：沿梁的相邻三点共 62 根"扭转弹簧"，
  $\kappa_{\rm beam}=7.5\times10^9$（IB2d 的叉积公式，逐字复刻；参考叉积 $C=0$）；
* **目标点**：梁两端（id 0 与 63），$k=2\times10^8$，钉在初始位置——梁两端固定。

流体/耦合：与其它 ib2d demo 完全相同的管线——周期 Q2/Q1 Taylor–Hood、
Peskin (2002) 两阶段格式（`PeskinRK2Solver`）、Peskin 4 点核、拉格朗日权重
$\mathrm ds=\min(L_x/2N_x, L_y/2N_y)$、`IBMesh(order=1)`（网格顶点 = IB2d 笛卡尔网格）。
结构力用 numpy 按 IB2d 原式在**半步位置** $X^{n+1/2}$ 上求值（恒定不变的零刚度 FE 链
只用来承载标记点自由度）：

$$
S=(\Delta x)_{r\to q}(\Delta y)_{q\to p}-(\Delta y)_{r\to q}(\Delta x)_{q\to p},\;
K=\kappa\,(S-C)
$$

$$
\begin{aligned}
p:&\quad f_x \mathrel{+}= K(y_r-y_q), & f_y &\mathrel{-}= K(x_r-x_q)\\
q:&\quad f_x \mathrel{+}= K[(y_q-y_p)+(y_r-y_q)], & f_y &\mathrel{-}= K[(x_r-x_q)+(x_q-x_p)]\\
r:&\quad f_x \mathrel{+}= K(y_q-y_p), & f_y &\mathrel{-}= K(x_q-x_p)
\end{aligned}
$$

与参考完全一致（力校核：扰动构型下与逐根循环差 $1.1\times10^{-16}$）。

## 2. 结果

![shapes](figures/compare_shapes.png)

![history](figures/compare_history.png)

* 首段（t ≤ 0.01）：梁形状几乎重合（标记点最大偏差 $3\times10^{-3}$）；
* 全程 1000 步：AFSI vs IB2d 标记点最大偏差 **均值 0.028、峰值 0.056**
  （网格间距 $h=1/32=0.031$，即约 0.9–1.8 $h$）；
* 中点在 t≈0.02–0.035 的下拍深度、后续回弹幅度基本一致，但摆动**相位**随时间缓慢漂移
  （末期 y_mid 差 ~0.04）——梁是一根刚度为 $7.5\times10^9$ 的"近刚性"振子，
  微小的离散差会累积成相位差；
* $\gamma$（grad-div）在此算例无益：$\gamma=100$ 时偏差略大（均值 0.030 / 峰值 0.064），
  因此默认 $\gamma=0$。

## 3. 运行

```bash
conda activate afsi-dolfinx
cd afsic/demo/ib2d/demo_446

# (1) AFSI（本机约 5 s）
python main.py                    # -> plot/afsi_result_g0.npz + XDMF
TFINAL=0.001 python main.py       # 冒烟
GRAD_DIV=100 python main.py       # 可选 grad-div（见上）

# (2) pyIB2d 参考（纯 Python，本机约 15 s；首次运行自动 clone pyIB2d）
python run_reference.py           # -> ib2d_reference.npz

# (3) 对照图
python compare.py                 # -> figures/*.png + compare_table.csv
```

## 4. 文件

| 文件 | 说明 |
|---|---|
| `main.py` | AFSI 计算（不变梁 + 目标点；numpy 原式力 + 零刚度 FE 链） |
| `run_reference.py` | 参考解（纯 Python，pyIB2d；直接使用示例的 0 基文件） |
| `ib2d_io.py` | IB2d 输入读取（pyIB2d 示例约定，0 基；`.beam`/`.target` 等） |
| `ib2d_reference.py` | pyIB2d VTK 输出 → npz（帧类型自适应朝向，见模块文档） |
| `ib2d_input/` | `input2d`、`BeamCurve.vertex/.beam/.target`（逐字复制） |
| `compare.py` | 形状/历史/场对照与 csv |

## 5. 已知限制

- 梁是近刚性高频振子，晚段存在 ~1 格的相位漂移（图二右），属两套离散格式的正常差异；
- 不做周期性边界跨越的梁三点最小镜像处理（IB2d 中该分支对居中的算例不触发）。
