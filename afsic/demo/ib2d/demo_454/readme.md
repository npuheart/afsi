# demo_454 — Poroelastic Rubberband（孔弹性橡皮筋，IB2d）

pure-python 参考：`third_party/ib2d/pyIB2d/Examples/Poroelastic_Rubberband`
（已拷贝为 `ib2d_input/`，0-based，无需索引换算）。

## 物理设置

在 demo_453（Single Porous Rubberband）基础上的扩展：64 点闭合橡皮筋
（64 根线性弹簧 $k=10^7$、静长 $L_0=0$），**全部 64 个标记**同时开启：

1. **porous 滑移**（$\kappa=10^{-4}$，4 阶差分法向；与 453 相同的
   逐字复刻，见 demo_453 readme）；
2. **poroelastic**（`.poroelastic`，Brinkman 系数 $c=2.5\times10^5$）：
   每次移动时标记获得额外位移

   $$\Delta X_{\rm pe} = \Delta t\;\frac{F_{\rm spring}}{\mu\,c}$$

   其中 $F_{\rm spring}$ 是**仅弹簧**的力（不含其余结构力），在 $X_h$
   处求值；**半步移动**用的是上一步的 $F_{\rm spring}$（driver 的
   `F_Poro` 记账方式），全步移动用本步值。

物理效果：多孔带不仅"漏流"，还随局部弹性载荷成比例地**蠕变**穿过流体
（Brinkman 型阻力）。

## 验证与结果（γ=2.5，T=0.1，dt=1e-4，1000 步）

稳定项沿用 453/447 的定标 $\gamma_{\rm grad\text{-}div}=2.5$
（γ=0 时同类收缩环会因伪散度漏流塌缩过快甚至崩溃，见 demo_453）。

| $t$ | dXmax | 面积 AFSI / IB2d |
|---|---|---|
| 0.008 | 0.0139（0.44h） | 0.2121 / 0.2197 |
| 0.024 | 0.0121（0.39h） | 0.1483 / 0.1640 |
| 0.040 | 0.0237（0.76h） | 0.0988 / 0.1100 |
| 0.064 | **0.0053（0.17h）** | 0.0462 / 0.0453 |
| 0.099 | 0.0102（0.33h） | 0.0067 / 0.0043 |

**全程 dXmax 5.3e-3 ～ 2.4e-2（0.17–0.76h）**，面积轨迹几乎重合
（差在 t≈0.06 处交叉；末期环收缩到极小尺度，面积绝对值都很小）。
poroelastic 项使塌缩比 453 快（同参考一致）：末态面积 0.0067 vs 453 的
0.0375。

力核验（弹簧，扰动态 loop 对照）：机器精度；poroelastic 复刻按 driver
的 `please_Move_Lagrangian_Point_Positions` poroelastic 分支实现
（$1/(\mu c)\cdot F_{\rm Poro}\,\Delta t$ 加在对应点的移动上）。

## 文件

- `main.py`          AFSI 运行（正式：`GRAD_DIV=2.5 python main.py`；
                     冒烟 `TFINAL=0.005 ...`）；输出 `plot/afsi_result_g<γ>.npz`
- `run_reference.py` 纯 Python pyIB2d 参考（lag: springs+porous+poroelastic；
                     支持 `--src/--tend/--dt/--print-dump` 覆盖）
- `ib2d_io.py`       0-based 输入读取（本 demo 加了 `.poroelastic`）
- `ib2d_reference.py` VTK→npz 转换
- `compare.py`       对比表 + `figures/`（形状、面积历史、涡量场）

```bash
conda run -n afsi-dolfinx python run_reference.py
GRAD_DIV=2.5 conda run -n afsi-dolfinx python main.py
GRAD_DIV=2.5 conda run -n afsi-dolfinx python compare.py
```
