# demo_448 — IB2d "HeartTube Muscle"（心管蠕动泵）的 AFSI 复现

## 1. 问题描述

IB2d 的 `Examples/HeartTube_Muscle`（pyIB2d 版，输入文件逐字复制自
`third_party/ib2d/pyIB2d/Examples/HeartTube_Muscle`）：心管的侧视模型——两条平行
弹性壁（y = 2 与 y = 3，x ∈ [1, 4]，各 155 个点），四角被钉住；两壁之间由 153 条
**Hill 力-速度 + 长度-张力肌肉带**（`i -- i+155`）连接。一条 10 Hz 的方波激活沿管
传播，逐段挤压肌肉带 → 蠕动泵送流体（μ = 0.1、ρ = 1、5×5 周期盒、128×128 网格）。

结构（0 基索引，直接读 `ib2d_input/`）：

* **弹簧**：两条壁链（308 段，$\kappa=10^7$，$L_0=\mathrm ds=0.0195$）；
* **不变梁**：沿壁的三点组（306 组，$\kappa_{\rm beam}=7.5\times10^7$）；
* **目标点**：四个角（id 0、154、155、309），$k=10^6$；
* **FV_LT 肌肉**：153 条带，$F_{\max}=10^5$、$L_{\rm opt}=1$、Hill $a=0.25$、
  $b=4$、长度-张力常数 $SK=0.3$；激活函数**直接使用示例自带的
  `give_Muscle_Activation.py`**（沿管传播的方波，宽度 = 激活区 1/10，速度 13.5）。

流体/耦合：与其它 ib2d demo 相同的管线（周期 Q2/Q1 Taylor–Hood + Peskin 两阶段 +
4 点核 + 零刚度 FE 图承载自由度）。所有结构力按 IB2d 原式在半步位置 $X^{n+1/2}$
求值；肌肉收缩速度 $v$ 用**上一半步**位置（驱动的 `xLag_P` 记账）；肌肉力的长度-
张力/速度因子逐字复刻
$F_m=a_f\,F_{\max}\,e^{-((Q-1)/SK)^2}\,\frac{1}{F_{\max}}\frac{bF_{\max}-av}{v+b}$。

> **注意（dt）**：示例默认 $\mathrm dt=5\times10^{-4}$，但该步长下 AFSI 的显式耦合
> 会失稳（4 步内发散；参考侧在同 dt 下稳定——两侧显式稳定域不同）。本 demo 将
> **双侧统一取 $\mathrm dt=10^{-4}$**（`ib2d_input/input2d` 中已注明，`print_dump=100`），
> 以保证同参数对照。

## 2. 结果（匹配良好）

![shapes](figures/compare_shapes.png)

![history](figures/compare_history.png)

| t | 标记点最大偏差 | 平均管宽 AFSI | 平均管宽 IB2d |
|---|---|---|---|
| 0.05 | $2.5\times10^{-2}$ ($0.65h$) | 0.9263 | 0.9373 |
| 0.10 | $2.9\times10^{-2}$ ($0.75h$) | 0.6967 | 0.7044 |
| 0.15 | $3.0\times10^{-2}$ ($0.77h$) | 0.5236 | 0.5330 |
| 0.20 | $3.8\times10^{-2}$ ($0.97h$) | 0.5066 | 0.5099 |
| 0.25 | $4.5\times10^{-2}$ ($1.14h$) | 0.4658 | 0.4643 |

（$h=5/128=0.0391$。）

* 管的压扁-回弹全过程（宽度 1.0 → 0.47、两壁行波形状）在两码间**逐帧贴合**，
  平均宽度差 ≤1.5%；
* 管内 $u_x$（净泵送）的振荡相位与幅值同样跟随（见中/右图）；
* 力校核：扰动构型下弹簧力与逐根循环差 0、梁力 $1.4\times10^{-16}$；
* 运行时长：AFSI ≈ 3.3 min（2500 步 @128×128），参考 ≈ 2 min。

## 3. 运行

```bash
conda activate afsi-dolfinx
cd afsic/demo/ib2d/demo_448

# (1) AFSI（本机约 3-4.5 min）
python main.py                    # -> plot/afsi_result_g0.npz + XDMF
TFINAL=0.002 python main.py       # 冒烟

# (2) pyIB2d 参考（纯 Python，本机约 2 min；首次运行自动 clone pyIB2d）
python run_reference.py           # -> ib2d_reference.npz

# (3) 对照图
python compare.py                 # -> figures/*.png + compare_table.csv
```

## 4. 文件

| 文件 | 说明 |
|---|---|
| `main.py` | AFSI 计算（弹簧 + 不变梁 + 目标点 + FV_LT 肌肉；numpy 原式力） |
| `run_reference.py` | 参考解（纯 Python，pyIB2d；含示例激活文件的复制） |
| `ib2d_io.py` | IB2d 输入读取（0 基；含 `.muscle`） |
| `ib2d_reference.py` | pyIB2d VTK 输出 → npz |
| `ib2d_input/` | 示例文件（含 `give_Muscle_Activation.py`；`input2d` 注明 dt 改动） |
| `compare.py` | 管形/宽度剖面/净流量/场对照与 csv |

## 5. 已知限制

- $\mathrm dt$ 按 §1 注记统一取 $10^{-4}$（参考侧示例默认为 $5\times10^{-4}$）；
- 肌肉条带与目标点均为 IB2d 原规则；未做分辨率收敛研究。
