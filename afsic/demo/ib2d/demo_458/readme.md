# demo_458 — Thrips（单质点重力沉降，IB2d）

参考算例：`pyIB2d/Examples/Thrips`（0-based，输入文件逐字拷贝为 `ib2d_input/`）。

## 1. 物理设置

$0.5\times0.5$ 周期盒，$\rho=1$、$\mu=0.1$、$32\times32$、$\mathrm{dt}=10^{-4}$、
$T_{\rm final}=0.1$（1000 步）。**只有一个标记点**（$(0.25,0.41129)$），它本身
没有任何结构，通过一根"质量弹簧"（$k=10^3$）连到一个**幽灵质点**（$M=10^{-3}$ kg），
重力 $(0,-g)$、$g=9.80665$ 只作用在幽灵质点上：

$$
\mathbf F = k(\mathbf X_{\rm ghost}-\mathbf X_{\rm marker}),\qquad
M\frac{d\mathbf V_{\rm ghost}}{dt} = -\mathbf F + M\mathbf g
$$

标记点随流体运动并感受 $\mathbf F$（不锚定初始位置），于是"重力 → 幽灵下落 →
弹簧拖拽标记点 → 流体黏性阻力"构成一个弹簧-质量-阻尼振子：幽灵以
$\omega=\sqrt{k/M}=1000\ \mathrm{rad/s}$ 振荡，整个系统缓慢下沉。这是 IB2d 里
最小规模的质量点模型（二维"沉降颗粒"）。

本 demo 直接复刻 IB2d `Supp.py` 的 `please_Move_Massive_Boundary` /
`please_Update_Massive_Boundary_Velocity`（与 demo_450 相同的、经考证的记账方式）：

* 标记点先按流体做半步；幽灵按半步移动（并保存步初位置）；
* 在 $X^{n+1/2}$、$X_{\rm ghost}^{n+1/2}$ 处求 $F$ 并用 4 点核铺开；
* 两阶段流体求解；
* 标记点用 $U_h(X_h)$ 前进整步；幽灵退回到步初位置再用
  $v_{\rm half}=v-\tfrac{\mathrm{dt}}{2}(F/M-g)$ 前进整步，并更新
  $v^{n+1}=v-\mathrm{dt}(F/M-g)$。

**单粒子网格**：结构自由度用一个**DG0 "point" 单元**承载（Basix 在 point 单元上
只允许 0 阶 Lagrange），恰好 1 个（分块）自由度 = 1 个 IB 粒子。

## 2. 结果（T=0.1 s，1000 步）

![trajectory](figures/compare_trajectory.png)

| $t$ | $y$ AFSI | $y$ IB2d | $\Delta y$ | $\lvert\Delta X\rvert$ | 幽灵 $y$（AFSI 内部量）|
|---:|---:|---:|---:|---:|---:|
| 0.02 | 0.411289152 | 0.411289041 | +1.11e-7 | 1.11e-7 | 0.411283113 |
| 0.05 | 0.411286069 | 0.411285752 | +3.17e-7 | 3.17e-7 | 0.411285683 |
| 0.08 | 0.411282306 | 0.411281780 | +5.27e-7 | 5.27e-7 | 0.411272335 |
| 0.099 | 0.411279657 | 0.411278997 | +6.60e-7 | **6.60e-7** | 0.411271317 |

* 标记点总位移：AFSI $-1.067\times10^{-5}$ m vs IB2d $-1.133\times10^{-5}$ m
  （比值 0.942）；
* **标记点最大偏差 $6.6\times10^{-7}$ m $=4.2\times10^{-5}\,h$**（$h=1/32$），
  随时间近似线性累积；
* 幽灵质点的振荡幅度 $1.46\times10^{-5}$ m（AFSI 内部记录；pyIB2d 的
  `ib2d_reference.npz` 只输出标记点，幽灵是驱动内部状态，故不作对照）；
* 本算例是**小信号**测试：IB2d 原例 $T=0.1$ s 内标记点只移动 $\sim10^{-5}$ m
  （约 $3\times10^{-4}\,h$），所以这里比较的是绝对量级极小的轨迹，而不是可见的
  沉降位移；AFSI 与参考在该量级上一致。

## 3. 运行

```bash
conda activate afsi-dolfinx
cd afsic/demo/ib2d/demo_458
python run_reference.py          # pyIB2d 参考（1000 步，~10 s）-> ib2d_reference.npz
python main.py                   # AFSI（1000 步，~6 s）-> plot/afsi_result.npz
python compare.py                # -> figures/ + compare_table.csv
```

环境变量：`IB2D_EXAMPLE`、`NX`/`NY`、`DT`、`TFINAL`、`PRINT_DUMP`、`OUTPUT_PATH`；
`run_reference.py` 支持 `--tend/--out/--ib2d/--run/--gravity`。

## 4. 文件

- `main.py` / `run_reference.py` / `compare.py` / `readme.md`
- `ib2d_io.py`（含 `.mass` 读取）、`ib2d_reference.py`（VTK → npz）
- `ib2d_input/`：`input2d`、`thrips.vertex`、`thrips.mass`（逐字拷贝）
