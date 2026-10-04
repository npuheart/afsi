# demo_472 — IB2d `Example_Flow_Around_Cylinder`（长条域内的高 Re 射流 + 圆柱绕流）

IB2d 官方算例（`matIB2d/Examples/Example_Channel_Flow/Example_Flow_Around_Cylinder`）
的复现，沿用 demo_467/468 的 `arb_ext_force` 模板。

## 算例内容

* `1 x 0.25` 长条周期盒、`mu = 0.001`、`rho = 1`、`512 x 128`、`dt = 5e-5`；
  官方 `T = 1.25`（25000 步），**本 demo 默认 `T = 0.02`（400 步）+ `print_dump = 50`**
  以便两个代码都能在几分钟内跑完（`TFINAL`/`PRINT_DUMP` 可恢复官方值）。
* **结构**：2273 个标记点，**全部**为 `k = 1e5` 的**软**目标点（比 demo_467 软 100 倍），
  描出上下两条壁（`y = 0.025 / 0.225`，`x in [0.05, 0.949]`）以及条带内的一个小圆柱。
* **外力**：窗口覆盖整个条带 `x in [0.1, 0.16]`、`y in [0.035, 0.215]`，

  $$u_x^{\rm tar} = -u_{\max}\,5\tanh(t)\,(\tfrac{L_y}{2}+\tfrac{W}{2}-y)(\tfrac{L_y}{2}-\tfrac{W}{2}-y),$$

  `k = 1e4`、`u_max = 250`（注意剖面用 `Ly/2` 而非 `Lx/2`，即射流横跨 0.25 高的条带）。

## 结果

参考运行 ~60 s、AFSI 运行 ~80 s，各 8 帧（`t = 0 .. 0.0175`）。

| 指标（末帧） | 原样 `512 x 128`, `dt = 5e-5` | 加密 `256 x 64`, `dt = 2.5e-5`（`T=0.01`） |
|---|---|---|
| 标记点偏差（软钉死 `k=1e5`） | 3.83e-5 m = **2.0e-2 h** | 1.88e-5 m = **4.8e-3 h** |
| 施力窗口 `max\|u\|` AFSI / IB2d | 0.2107 / 0.2061（**2.2%**） | 0.1038 / 0.0981（5.8%） |
| 窗口内剖面（`x = 0.125`） | 逐点 ~1%（峰值 0.2074 / 0.2061） | — |
| 下游 `x = 0.5` 的射流峰值 | 0.0715 / 0.0908（**27%**） | — |
| 速度场相对 $L_2$ 误差（窄带外） | 2.49e-1 | 2.50e-1 |
| 流量（中列、窗口行）相对差 | 2.1e-1 | 2.3e-1 |
| 罚力公式核对 | ≤0.66（`\|f\|_{\max}` 1.5e3，**0.04%**） | ≤0.6 量级 |

**能验证什么**：驱动机制与软钉死结构——窗口内剖面逐点吻合到 1%、窗口 `max|u|` 2.2%、
罚力公式核对 0.04%、标记点（`k=1e5` 软钉死）在两种分辨率下都 ≤2e-2 h。

**不能验证什么**：本算例的射流是 **高雷诺数**工况（$Re = uL/\nu \approx 3125$，
单元 $Re \approx 25$），下游剪切层的卷起在两种离散下本来就不同（`x = 0.5` 峰值差 27%），
加密也不收敛（$L_2$ 2.49e-1 → 2.50e-1）。这与 demo_461/471 的结论一致：
**"驱动量"可比，"这个网格分辨不出来的流动结构"不可比**；要比较下游涡结构需要
解析尺度网格，超出本仓库的算力预算。

## 运行

```bash
conda activate afsi-dolfinx
TFINAL=0.02 PRINT_DUMP=50 python run_reference.py --tend 0.02 --print-dump 50   # ~60 s
TFINAL=0.02 PRINT_DUMP=50 python main.py                                        # ~80 s
python compare.py

# 加密交叉核对（compare.py 自动配对）
python run_reference.py --nx 256 --dt 2.5e-5 --tend 0.01 --print-dump 50 \
    --out ib2d_reference_dt5e-5.npz --run ib2d_run_dt5e-5
NX=256 DT=2.5e-5 TFINAL=0.01 PRINT_DUMP=50 OUTPUT_PATH=./plot_dt5e-5 python main.py
```
