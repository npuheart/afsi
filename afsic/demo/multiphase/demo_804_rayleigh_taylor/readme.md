# demo_804 — Rayleigh–Taylor 失稳（密度分层 + 重力）

重的流体（`rho_2`）压着轻的流体（`rho_1`），界面加一个单模扰动
`A_0 cos(2 pi x / L)`，方腔四壁无滑移。这是经典的 RT 配置，用来检验**变密度**通路：
`rho(phi)` 每步重新装配矩阵、浮力 `(rho(phi) - rho_1) g` 进动量方程。

## 解析参照

* 无黏线性增长率 `sigma = sqrt(A_t g k)`，`A_t = (rho_2 - rho_1)/(rho_2 + rho_1)`，
  `k = 2 pi / L`。
* 等黏度黏性修正：`sigma^2 + 2 nu k^2 sigma = A_t g k`，
  `nu = (mu/rho_1 + mu/rho_2)/2`。
* 本 demo 默认：`rho_2/rho_1 = 3`（`A_t = 0.5`）、`g = 1`、`L = 1`、`mu = 0.005`
  -> `k = 6.283`、无黏 `1.772`、黏性 `1.646`；`A_0 = 0.05`、`eps = 1.5h`、`M = 0.05`。

## 结果（`NX = 96`，`dt = 5e-4`，`Tfinal = 1.0`，2000 步，460 s）

| 量 | 测值 | 参照 |
|---|---|---|
| 界面位移振幅 `A_0 -> A(1.0)` | `0.05 -> 0.115`（2.3 倍） | 黏性理论 `0.05 e^{1.646}` = 0.26 |
| 增长率（对 `A > 1.05 A_0` 段整体拟合，`main.py` 输出） | **1.020** | 黏性 1.646 / 无黏 1.772 |
| 增长率（末段 `t in [0.75, 1.0]` 局部斜率） | **约 1.2** | 同上 |
| 比值 | 0.62（整体） / 0.73（末段） | - |
| 最大速度（末态） | 0.15 | - |

* 增长率在 `t <= 0.5` 期间从 0.03 单调爬升到 0.7（含启动瞬态），
  之后稳定在 **1.2 左右**，与黏性线性理论同量级（低 ~25~40%，取决于取哪一段）。
* 差异的合理来源：**扩散界面**把密度阶跃摊到 `4 eps = 6` 个格子上，
  等效的密度梯度被削弱；以及振幅有限（`A/R` 已到 0.23）后的非线性。
  另外界面的 Allen–Cahn 弛豫本身不携带 Korteweg 应力，会对早期增长有微弱抑制。

**能验证**：变密度 + 重力的**静力学**是精确的（见下），因此这里测到的
不稳定增长确实是"重上轻下"驱动的；增长率为正、随黏度降低而增大、
量级与线性理论一致。

**不能验证**：**定量**增长率（不追求 5% 以内）。要收敛到锐界面极限需要
`eps << h` 的锐界面（或改用守恒的 Cahn–Hilliard + Korteweg 应力）+ 更细网格，
本 demo 未做。

## 顺带验证：两层静水压力（同一套代码）

把重流体放**下面**（`rho = 2`，`y < 0.5`）且不加扰动，重力向下，
平衡时约化压力 `p` 满足 `dp/dy = (rho(phi) - rho_1) g`：

| `NX = 48`，`T = 0.5`，`mu = 0.05` | 测值 | 精确 |
|---|---|---|
| 重层 `dp/dy` | **-0.993** | -1.000 |
| 轻层 `dp/dy` | -0.014 | 0.000 |
| 界面形状 | 保持水平 | 水平 |
| `max\|u\|`（应静止） | 5.2e-4 | 0 |

```bash
conda activate afsi-dolfinx
cd afsic/demo/multiphase
python - <<'PY'
import sys, numpy as np; sys.path.insert(0,".")
from two_phase import TwoPhaseSolver
N=48; L=1.0; eps=L/N
s = TwoPhaseSolver((0,L,0,L), N, N, 1e-3, 1.0, 2.0, 0.05, 0.05, 0.0, eps,
                   mobility=0.02, gravity=(0.0,-1.0),
                   bc_spec=lambda x: np.zeros((2, x.shape[1])))
s.set_phase(lambda x: 0.5*(1-np.tanh((x[1]-0.5)/(2*np.sqrt(2)*eps))))
s.set_velocity(); s.run(0.5)
P = s.nodal(s.p_); y = np.linspace(0,L,N+1); g = np.gradient(P[:,N//2], y)
print("heavy layer dp/dy = %.4f (exact -1)" % g[3:N//2-3].mean())
print("light layer dp/dy = %.4f (exact  0)" % g[N//2+3:N-3].mean())
PY
```

## 运行

```bash
conda activate afsi-dolfinx
cd afsic/demo/multiphase/demo_804_rayleigh_taylor

python main.py                      # 64x64, T = 0.6
python main.py --nx 96 --tend 1.0 --plot    # 上表，~7~8 min
python main.py --sweep-mu --plot            # mu = 0.02/0.01/0.005
```

结果写到 `output/rayleigh_taylor.csv`（`--plot` 时另有 `output/rayleigh_taylor.png`：
振幅-时间对数图，虚线为黏性线性理论）。
