# demo_475 — IB2d `Example_Brinkman_Channel`（射流注入 Brinkman 多孔区）

IB2d 官方算例（`matIB2d/Examples/Example_Channel_Flow/Example_Brinkman_Channel`）。
**新机制：Brinkman 阻力项** `F -= mu * Brink(x) * u^n`。

## 算例内容

* `1 x 0.25` 长条周期盒、`mu = 2`、`rho = 1000`、`512 x 128`、`dt = 5e-5`；
  官方 `T = 1.25`（25000 步），本 demo 默认 `T = 0.05`（1000 步，`print_dump = 200`）。
* **结构**：1844 个 `k = 5e6` 的钉死目标点 = 上下两条壁（`y = 0.025/0.225`，
  `x in [0.05, 0.949]`）。
* **Brinkman 场**：算例自带的 `channel.brinkman` 给出 `1/k = 0` 或 `1e5` 的**逐节点阻抗图**
  （首行 `k = 1e-5`）；被阻塞区占右侧 `x in [0.25, 0.90]`、整条高度（11322/65536 节点）。
* **外力**：左侧自由段 `x in [0.1, 0.16]`、`y in [0.03, 0.22]` 施加**归一化抛物射流**
  `u_x^{tar} = uMax tanh(4t) ((MP+w/2)-y)((MP-w/2)-y)/scale`
  （`k = 1e4`、`W = 0.2`、`uMax = 2.5`、`MP = 0.125`、`scale = -0.01`，中面峰值恰为
  `uMax tanh(4t)`）。射流把流体推入右侧多孔区——这正是算例主题。

## 移植要点

* Brinkman 项按 IB2d 原样实现：用 `u^n` 显式计算，在 IB 力、外力**之后**叠加到
  `solver.f`（Q1 节点乘积 → Q2 插值，与外力同一套机制）；两边速度场取的是同一步的 `u^n`，
  因此处理完全一致。
* `.brinkman` 文件由 `main.py` 直接读入（首行 `k`，其余为 `(Ny, Nx)` 矩阵）。

## 结果

参考运行 ~200 s、AFSI 运行 ~200 s，各 5 帧（`t = 0 .. 0.04`）。

| 指标（末帧 `t = 0.04`） | 结果 |
|---|---|
| 标记点偏差（1844 点钉死） | 1.15e-6 m = **5.9e-4 h**（本族最好） |
| 施力窗口 `max\|u\|` AFSI / IB2d | 0.01624 / 0.01633（**0.62%**） |
| 速度场相对 $L_2$ 误差（窄带外） | **1.56e-2** |
| 窄带外 `max\|du\|` | 5.9e-4 = 3.6e-2 `max\|u\|` |
| 流量（中列、窗口行）相对差 | **0.31%** |
| 罚力公式核对 | ≤4.6（`\|f\|_{\max}` 1.96e4，**0.02%**） |

流场被多孔区强烈抑制（`max|u| = 0.016`，而目标峰值 0.49）：罚力、Brinkman 阻力与
封闭段的反压达到平衡，两边给出**同一条**低速渗流解（窗口 0.6%、流量 0.3%、$L_2$ 1.6%）。
本例是 `arb_ext_force` 族里逐项吻合度最好的一个（也说明 Brinkman 项与外力项的叠加顺序正确）。

## 上游 bug（本 demo 的额外发现，共两个）

pyIB2d 的 Brinkman 路径**完全跑不起来**，两个 bug 都在 `run_reference.py` 里打了补丁：

1. `read_In_Brinkman_Info()` 把矩阵读进 `C` 却 `return (Brink, kBrink)` —— `Brink` 未定义 ⇒ `NameError`；
2. `IBM_Driver.main()` 在读入 Brinkman 后 `os.chdir('viz_IB2d')` **又执行一次**
   `os.chdir('viz_IB2d')` ⇒ 除非存在嵌套目录否则 `FileNotFoundError`
   （补丁：预先创建 `viz_IB2d/viz_IB2d/`）。

## 运行

```bash
conda activate afsi-dolfinx
python run_reference.py --tend 0.05 --print-dump 200        # ~200 s
TFINAL=0.05 PRINT_DUMP=200 python main.py                   # ~200 s
python compare.py
```
