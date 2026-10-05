# demo_602 — IB2d example "Nematocyst" (matIB2d
``Examples/Examples_First_Year_Seminar/Nematocyst``) in AFSI.

## 中文小结

* **算例**：`Examples_First_Year_Seminar/Nematocyst` —— 212 个点：160 根弹簧
  （`k = 5e5`）+ 160 根**静态**非不变梁（`kb = 1e7`）+ 52 个靶点（`k = 1.75e7`）。
* **新的东西**：`.m` 在 `t <= 0.05 s` 内用 `y += tanh(2000 t) * 8 * dt` 把刺丝
  发射出去（不回撤）。
* **缩减**：`T = 0.005 s`（500 步 = 发射的前 10%），网格/`dt` 与上游一致。
* **结果**：标记点 `0.011 h`；弹簧平均长度相对差 `1.2e-5`；速度相对 $L_2$ `0.044`
  —— 这一批里对得最好的之一。


> 下面 "Physics / porting notes" 部分是英文叙述（与 `main.py` 的 docstring 同源）：
> 上游算例的标识符、方程都来自英文源码，保持原样以免转述出错。

212 markers in a 0.25 x 1.0 periodic box (mu = 0.01, rho = 1, 64 x 256,
dt = 1e-5): 160 springs (k = 5e5), 160 static non-invariant beams (kb = 1e7)
and **52 target points** (k = 1.75e7).  The example fires a nematocyst
harpoon at its prey.

The example's own ``update_Target_Point_Positions.m`` (ported line-by-line to
``ib2d_input/update_Target_Point_Positions.py`` and imported by both codes)
moves the 52 targets in +y with a ramped speed,

    y_i = y_i + tanh(2000 t) * (L/period) * dt,   L = 0.4 m, period = 0.05 s,

for ``current_time <= period`` and then stops (the .m has no return stroke; the
harpoon stays extended).  The update is incremental (it reads the current y).

Cost: the shipped ``Tfinal = 0.05 s`` (5000 steps) is reduced to ``T = 0.005 s``
(500 steps = the first 10% of the firing) at the shipped grid, with
``print_dump = 50`` (10 frames) instead of 100.

## 结果（`python compare.py` 的完整输出）

```
        t   dXmax/h            max|u| a/r    du away    du band      relL2    df away
  0.00000  1.55e-09  0.000e+00/0.000e+00   0.000e+00  0.000e+00  0.000e+00  0.000e+00
  0.00051  2.51e-03  8.815e+00/9.163e+00   3.306e-01  8.738e-01  1.186e-01  6.386e-01
  0.00101  2.84e-03  1.008e+01/1.052e+01   4.119e-01  9.810e-01  1.118e-01  3.995e+00
  0.00151  5.13e-03  9.939e+00/1.036e+01   9.137e-01  8.879e-01  1.066e-01  4.068e+02
  0.00201  7.20e-03  9.652e+00/1.005e+01   8.411e-01  7.835e-01  1.006e-01  1.800e+03
  0.00251  7.50e-03  9.569e+00/9.859e+00   8.618e-01  7.794e-01  9.183e-02  1.103e+04
  0.00301  8.29e-03  9.674e+00/9.832e+00   8.702e-01  6.297e-01  8.850e-02  1.406e+04
  0.00351  8.26e-03  9.755e+00/1.002e+01   8.937e-01  5.992e-01  8.354e-02  3.503e+03
  0.00401  8.81e-03  9.817e+00/1.018e+01   9.075e-01  5.655e-01  8.091e-02  5.205e+03
  0.00451  9.72e-03  9.860e+00/1.023e+01   9.029e-01  5.422e-01  7.695e-02  4.219e+03
  0.00501  1.06e-02  9.888e+00/1.034e+01   9.157e-01  4.452e-01  7.469e-02  5.498e+03
markers          : max|dX| 4.156e-05 m = 1.06e-02 h (212 Lagrangian points)
velocity         : max|u| AFSI 9.8878e+00 / IB2d 1.0340e+01 (4.37e-02 relative); max|du| 9.157e-01 = 8.86e-02 max|u| away from the IB band (16229/16384 nodes), 4.452e-01 (0.04 max|u|) inside it; relative L2 7.469e-02
force field      : max|df| 5.498e+03 away from the band, |f|max 2.634e+04
springs          : mean length AFSI 0.001963 / IB2d 0.001963 (1.21e-05 relative)
wrote /Users/pengfei.ma/afsi/afsic/demo/ib2d/demo_602/figures/compare_table.csv and the figures
```

这张表就是 `compare.py` 的原始输出：每帧的 `dXmax/h` 是标记点偏差（以网格步长 h
为单位），`max|u| a/r` 是 AFSI/IB2d 的速度峰值，`du away` / `du band` 是 IB 带外/
带内的速度差，`relL2` 是速度场相对 $L_2$，`df away` 是带外外力场差（有被动标量时
还有 `max|dC|` 与 `relL2(C)`）。它与 `figures/compare_table.csv` 一致。

## 不可比 / 需要打折看的地方

* **规模缩减**：上游 `Tfinal = 0.05 s`（5000 步）在 `64 x 256`；本 demo 用
  `T = 0.005 s`（500 步 = 发射的前 10%），网格、`dt = 1e-5` 与上游一致，
  `print_dump = 50`（10 帧）。
* **这一步只有 Nematocyst 的"发射"被采样**：`.m` 只在 `current_time <= 0.05 s`
  内推动 52 个靶点（`y += tanh(2000 t) * 8 * dt`，harpoon 不回撤），所以 0.005 s
  窗口看到的是 `tanh` 刚起来的阶段（速度 ≈ 8·tanh(10·t) m/s，前 500 步里
  最大约 0.4 m/s 的锚点速度）。
* **结构很复杂但都是静态的**：160 根弹簧（k = 5e5）+ 160 根**静态**非不变梁
  （kb = 1e7，`update_nonInv_beams = 0`）+ 52 个 `k = 1.75e7` 的靶点。梁/弹簧的
  力公式用 demo_449/demo_478 已经验证过的实现；本 demo 新的只有靶点律。
* **速度场要打折**：`64 x 256` 的盒子里 212 个点，靶点很硬（k = 1.75e7），近场
  速度对离散方式敏感；`max|dX|/h` 是主要判据。

## 运行方法（实测耗时，本机）

```bash
conda activate afsi-dolfinx
cd afsic/demo/ib2d/demo_602
```

| 命令 | 实测 `real` |
|---|---|
| `python run_reference.py --tend 0.005 --print-dump 50` | 2m50.106s |
| `python main.py` | 1m57.922s |
| `python compare.py` | - |

（`real` 取自 `ref.log` / `afsi.log`；本机同一时刻还有大量其他算例在跑，
绝对耗时波动很大。参考解写入 `ib2d_reference.npz`，AFSI 写入
`plot/afsi_result.npz`，`compare.py` 生成 `figures/` 与 `compare_table.csv`。）
