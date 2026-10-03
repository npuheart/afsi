# demo_447 — IB2d "Rubberband with Damped Springs"（阻尼橡皮筋）的 AFSI 复现

## 1. 问题描述

IB2d 的 `Examples/Rubberband_with_Damped_Springs`（pyIB2d 版，输入文件逐字复制自
`third_party/ib2d/pyIB2d/Examples/Rubberband_with_Damped_Springs`）：一个由 64 个
标记点组成的封闭橡皮筋（初始为绕 (0.5, 0.5) 的椭圆环），相邻点用**阻尼弹簧**
（$\kappa=2.5\times10^4$、静长 $L_r=0$、阻尼 $b=5$）连成一圈
（`0-1-2-…-63-0`），放在 1×1 周期盒的黏性流体中（μ = 0.01、ρ = 1）。

$L_r=0$ 意味着**每个弹簧的平衡长度是零**：结构在力学上倾向于整体塌缩，
过程由流体阻力与相对微弱的材料阻尼调制。

结构力（IB2d `give_Me_Damped_Springs_Lagrangian_Force_Densities`，逐字复刻）：

$$
\mathbf s_i = \kappa\,(L_i-L_r)\,\hat{\mathbf d}_i - b\,\mathbf V_i^{\rm leader},
\qquad \mathbf V^{\rm leader}=\frac{X_h-X_h^{\rm prev}}{\mathrm dt}
$$

（$+s$ 加在 leader 点、$-s$ 加在 follower 点；$\mathbf V$ 用 leader 在
**上一半步**的位置——即驱动的 `xLag_P` 记账方式；含 IB2d 的最小镜像处理）。

流体/耦合：与其它 ib2d demo 相同的管线（周期 Q2/Q1 + Peskin 两阶段 + 4 点核 +
零刚度 FE 环承载自由度）。

## 2. 核心结论：grad-div 稳定项 γ 修正了"塌缩分岔"

**历史**：本 demo 最初用官方 $\gamma_{\rm grad\text{-}div}=0$ 运行，结果
AFSI 的环持续收缩至塌缩（面积 → $2\times10^{-4}$），与 IB2d 参考的"卡住"
形成定性分岔。原先把它记录为"两个离散格式的差异"。**后来在 demo_453
（同类收缩环）中发现这是粗网格 P2/P1 投影的伪散度"漏流"问题：引入
grad-div 稳定项 γ（=用户口径的 λ）后立刻修正。** 回到本 demo 做 γ 扫描：

| γ | 面积 @t=1.5 | dXmax @t=1.0 | dXmax @t=1.5 |
|---|---|---|---|
| 0（原配置） | 0.00023（全塌缩） | 0.222（7h） | 0.257（8h） |
| **2.5** | 0.2385 | **0.030（1h）** | 0.064（2h） |
| 10 | 0.2419 | 0.031 | 0.066 |
| 50 | 0.2425 | 0.030 | 0.065 |

⇒ **γ=2.5 后分岔消失**：t≤1.0 全程 dXmax ≤ 0.03（≈1h），
晚段（t=1.5）偏差缓增至 0.064（≈2h，AFSI 的塌缩进度略慢）。本 demo
的正式配置为 `GRAD_DIV=2.5`。

## 3. 结果（γ=2.5）

| t | dXmax | 面积 AFSI / IB2d |
|---|---|---|
| 0.010 | $2\times10^{-3}$ | 0.2508 / 0.2509 |
| 0.200 | $6.2\times10^{-3}$（0.20h） | 0.2487 / 0.2512 |
| 0.600 | $9.0\times10^{-3}$（0.29h） | 0.2451 / 0.2390 |
| 1.000 | $3.0\times10^{-2}$（0.96h） | 0.2420 / 0.2165 |
| 1.500 | $6.4\times10^{-2}$（2.0h） | 0.2385 / 0.1833 |

（晚段两码都以极慢速率继续收缩，AFSI 略慢——残余的"塌缩速率"小偏差，
量级 ~2h。）

## 4. 实现正确性的交叉验证（保留）

1. **力实现校核**：扰动构型下与逐根循环完全一致（误差 0）；
2. **AFSI 侧 dt 收敛**：$\mathrm dt=10^{-3}$ 与 $2.5\times10^{-4}$ 结果 4 位有效数字一致；
3. **参考侧 dt 收敛**：pyIB2d 在 $\mathrm dt=10^{-3}$ 与 $2.5\times10^{-4}$ 下同样一致；
4. **高黏准静态极限**：$\mu=50$ 时两码都几乎冻结、漂移同量级（$10^{-4}\sim10^{-5}$）；
5. **换静长实验**：$L_r$ 改成初始边长后，两码一致（形状振荡、面积 ~0.25）；
6. **阻尼项开关**（$b=0$ / $b=5$）与**换用普通弹簧**对参考的解几乎无影响。

（1–6 说明结构力/耦合实现无误；第 2、3 说明不是 dt 问题；剩下的分岔最终由
γ 扫描定位为**投影格式的伪散度**，γ=2.5 修复。）

![shapes](figures/compare_shapes.png)

![history](figures/compare_history.png)

## 5. 运行

```bash
conda activate afsi-dolfinx
cd afsic/demo/ib2d/demo_447

# (1) AFSI（正式：GRAD_DIV=2.5，本机约 25 s）
GRAD_DIV=2.5 python main.py       # -> plot/afsi_result_g2.5.npz + XDMF
TFINAL=0.01 python main.py        # 冒烟

# (2) pyIB2d 参考（纯 Python，本机约 30 s）
python run_reference.py           # -> ib2d_reference.npz

# (3) 对照图（默认读 g2.5；GRAD_DIV 可覆盖）
GRAD_DIV=2.5 python compare.py    # -> figures/*.png + compare_table.csv
```

## 6. 文件

| 文件 | 说明 |
|---|---|
| `main.py` | AFSI 计算（阻尼弹簧按 IB2d 原式；`xLag_P` 半步记账；`GRAD_DIV` 稳定项） |
| `run_reference.py` | 参考解（纯 Python，pyIB2d；damped_springs=1，其余关） |
| `ib2d_io.py` | IB2d 输入读取（0 基；`.d_spring` 等） |
| `ib2d_reference.py` | pyIB2d VTK 输出 → npz |
| `ib2d_input/` | `input2d`、`rubberband.vertex/.d_spring`（逐字复制） |
| `compare.py` | 形状/面积/偏差对照与 csv（默认 g2.5） |

## 7. 已知限制

- 晚段（t>1.0）AFSI 的塌缩进度仍略慢（~2h @t=1.5）；γ∈[2.5,50] 内几乎不变，
  属残余的格式差（同 demo_453 的模式），不再有定性分岔；
- 更细网格下的 γ 定标未做（γ ∝ h⁻² 量纲上应随加密变化）。
