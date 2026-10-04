# demo_457 — User-Defined Fiber Model（IB2d 通用用户力接口）

参考算例：`pyIB2d/Examples/User_Defined_Fiber_Model`（0-based，输入文件逐字拷贝为
`ib2d_input/`，含示例自带的用户力模块 `give_Me_General_User_Defined_Force_Densities.py`）。

## 1. 物理设置

与 demo_444/455 相同的 64 点椭圆环（半轴 0.2×0.4，中心 (0.5, 0.5)），
$32\times32$、$\rho=1$、$\mu=0.01$、$\mathrm{dt}=10^{-3}$、$T=1.5$（1500 步）。
区别在于：结构力**不再来自 IB2d 内置模型**，而是走
**通用用户力接口**（`user_force_model = 1`）：

* 用户提供任意文本文件 `<name>.user_force`（首行 = 连接数，其后自由格式列）；
* IB2d 把该数组（`np.loadtxt(..., unpack=True)` 的**转置**）连同
  `ds, Nb, xLag, yLag, xLag_P, yLag_P, dt, current_time` 交给用户的力函数
  （pyIB2d 从运行目录 `import give_Me_General_User_Defined_Force_Densities`），
  函数返回 `(fx, fy)` 后按普通结构力铺开。
* 示例的用户模型恰是"零静长线性弹簧"每行 `(master, slave, k, RL)`，
  $k=2.5\times10^4$、$RL=0$ ⇒ 环在摆动中持续塌缩（即 demo_444 的
  Standard-Rubberband 物理）。

本 demo 按 `give_Me_General_User_Defined_Force_Densities.py` 的公式逐位复刻
（`main.py:user_forces`，与逐行循环校验差 $1.5\times10^{-16}$ 量级），因此验证的是
**接口通路**：AFSI 能像 IB2d 一样接受任意用户力。

## 2. 结果（32×32，dt=1e-3，T=1.5）

![shapes](figures/compare_shapes.png)

![history](figures/compare_history.png)

| $\gamma$ | $t=1.5$ 面积 | dXmax$/h$ |
|---:|---:|---:|
| 0 | 0.00024（几乎全塌缩） | 7.68 |
| **100（默认）** | **0.24226** | 1.21（末态；$t\le0.84$ 内 $\le0.45$） |

| $t$ | dXmax$/h$ | 面积 AFSI | 面积 IB2d |
|---:|---:|---:|---:|
| 0.12 | 0.174 | 0.24946 | 0.25110 |
| 0.36 | 0.129 | 0.24683 | 0.24805 |
| 0.84 | 0.449 | 0.24404 | 0.22693 |
| 1.20 | 1.008 | 0.24294 | 0.20445 |
| 1.50 | 1.210 | 0.24226 | 0.18396 |

* $\gamma=0$ 与 $\gamma=100$ 的分岔、以及末态数值（AFSI 0.2423 / IB2d 0.1840 /
  dXmax $3.8\times10^{-2}$）与 **demo_444 逐位一致**——demo_444 用 6 条曲线
  （IB2d + rk2 + fiber Chorin/IPCS×{0,100}）与 MATLAB/Octave 参考详细论证过：
  Q2/Q1 无法表示跨膜压力跳，$\gamma=0$ 时残余力驱动穿膜流把环"漏"成一点；
  $\gamma\gtrsim100$ 后饱和，AFSI 面积守恒（泄漏 3.5%）。
* 后期偏差主要来自**参考自身的泄漏**（IB2d 在 32² 上面积损失 26.7%）；demo_444 的
  加密研究（32²/64²）表明参考轨迹随加密仍明显移动，属其离散误差。
* 与 demo_444 的差别仅在于力的来源：本 demo 走 `.user_force` + 用户模块，
  验证通用接口后得到相同物理。

## 3. 运行

```bash
conda activate afsi-dolfinx
cd afsic/demo/ib2d/demo_457
python run_reference.py                 # pyIB2d 参考（1500 步，~10 s）
python main.py                          # AFSI，默认 GRAD_DIV=100（~9 s）
GRAD_DIV=0 python main.py               # 消融：环塌缩 -> plot/afsi_result_g0.npz
python compare.py                       # -> figures/ + compare_table.csv
```

环境变量：`IB2D_EXAMPLE`、`NX`/`NY`、`DT`、`TFINAL`、`PRINT_DUMP`、`GRAD_DIV`、`OUTPUT_PATH`。

## 4. 文件

- `main.py`（用户力 + 两阶段 rk2 管线）、`run_reference.py`（含用户力模块拷贝）
- `ib2d_io.py`（本 demo 新增 `read_user_force`）、`ib2d_reference.py`、`compare.py`
- `ib2d_input/`：`input2d`、`rubberband.vertex`、`rubberband.user_force`、
  `give_Me_General_User_Defined_Force_Densities.py`
