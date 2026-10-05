# demo_597 — IB2d example "Pulsing Heart" (matIB2d ``Examples/Example_Pulsing_Heart/512x512``)

in AFSI.

6829 Lagrangian markers (a 2-d heart outline plus interior) in a 1 x 1 periodic
box (mu = 0.0125, rho = 1, 512 x 512 as shipped, dt = 1e-5); every marker is a
*target point* (k = 1.5e6) plus a set of structural springs (the shipped
``heart.spring`` is unused, ``springs = 0``).

The example's own ``update_Target_Point_Positions.m`` (ported line-by-line to
``ib2d_input/update_Target_Point_Positions.py`` and imported by both this port
and pyIB2d) contracts the heart between the two shapes stored in
``All_Positions.txt`` (columns ``x1, y1, x2, y2``) with the family's piecewise
polynomial blend and a 0.1 s cycle:

    X = A + g(t/tP1) (B - A)        (t <= tP1 = 0.05 s, contraction),
    X = B + g((t-tP1)/tP2) (A - B)  (expansion, tP2 = 0.05 s),

with ``g`` the parabola/cubic/inverted-parabola blend (knots 0.1 / 0.9) used
throughout the IB2d teaching examples.  Every marker moves.

Cost: 512 x 512 with 20000 steps is far out of reach, so the demo runs a
**reduced** ``NX = 64`` (64 x 64; a 64x cheaper step) and ``T = 0.005 s``
(500 steps, half a contraction) with ``print_dump = 50`` (10 frames) instead of
100.

## 结果（`python compare.py` 的完整输出）

```
        t   dXmax/h            max|u| a/r    du away    du band      relL2    df away
  0.00000  4.50e-10  0.000e+00/0.000e+00   0.000e+00  0.000e+00  0.000e+00  0.000e+00
  0.00051  1.53e-03  2.428e-01/6.429e-01   3.751e-01  5.135e-01  7.152e-01  9.397e-15
  0.00101  9.78e-03  5.855e-01/3.709e+00   2.733e+00  3.391e+00  8.834e-01  1.560e-14
  0.00151  2.35e-02  1.085e+00/8.629e+00   6.758e+00  8.194e+00  9.057e-01  1.956e-14
  0.00201  3.38e-02  1.712e+00/1.254e+01   1.026e+01  1.188e+01  8.935e-01  2.167e-14
  0.00251  3.73e-02  2.473e+00/1.479e+01   1.227e+01  1.332e+01  8.618e-01  2.310e-14
  0.00301  3.98e-02  3.378e+00/1.682e+01   1.414e+01  1.447e+01  8.324e-01  2.303e-14
  0.00351  4.51e-02  4.409e+00/2.048e+01   1.742e+01  1.783e+01  8.262e-01  2.169e-14
  0.00401  5.27e-02  5.559e+00/2.520e+01   2.183e+01  2.245e+01  8.277e-01  1.923e-14
  0.00451  5.94e-02  6.821e+00/2.890e+01   2.581e+01  2.645e+01  8.187e-01  1.584e-14
  0.00501  6.42e-02  8.186e+00/3.210e+01   2.882e+01  2.894e+01  8.005e-01  1.474e-14
markers          : max|dX| 1.004e-03 m = 6.42e-02 h (6829 Lagrangian points)
velocity         : max|u| AFSI 8.1858e+00 / IB2d 3.2104e+01 (7.45e-01 relative); max|du| 2.882e+01 = 8.98e-01 max|u| away from the IB band (3187/4096 nodes), 2.894e+01 (0.90 max|u|) inside it; relative L2 8.005e-01
force field      : max|df| 1.474e-14 away from the band, |f|max 4.468e+04
wrote /Users/pengfei.ma/afsi/afsic/demo/ib2d/demo_597/figures/compare_table.csv and the figures
```

这张表就是 `compare.py` 的原始输出：每帧的 `dXmax/h` 是标记点偏差（以网格步长 h
为单位），`max|u| a/r` 是 AFSI/IB2d 的速度峰值，`du away` / `du band` 是 IB 带外/
带内的速度差，`relL2` 是速度场相对 $L_2$，`df away` 是带外外力场差（有被动标量时
还有 `max|dC|` 与 `relL2(C)`）。它与 `figures/compare_table.csv` 一致。

## 不可比 / 需要打折看的地方

* **规模缩减**：上游 `512 x 512`、`Tfinal = 0.2 s`（20000 步）；本 demo 用
  `NX = 64`（`64 x 64`，每步便宜 64 倍）和 `T = 0.005 s`（500 步 = 半个收缩期，
  `tP1 = 0.05 s` 之半），`dt = 1e-5` 与上游一致，`print_dump = 50`（10 帧）。
* **可比的是锚点律 + 靶点耦合**：6829 个靶点的锚点由同一份移植模块给出（pyIB2d 从
  run 目录 import 同一文件），所以 `max|dX|/h` 直接度量"结构能否跟上锚点"，是两个
  代码在**同一套强靶点（k = 1.5e6）+ 同一个运动律**下的耦合差异。
* **下限分辨率**：心跳轮廓在 512² 上有 ~6829 个点（约每 6 个网格一个点），缩到
  64² 后 Lagrangian 点比网格点还密（点距 ≈ 0.012 vs h = 0.0156），属于"过采样"，
  因此 `max|dX|`/速度对这种粗网格是偏大的；这是**本 demo 主要要打折的地方**。
* 上游的 `.spring` 文件存在但 `springs = 0`，所以本算例没有弹簧，只有靶点。

## 运行方法（实测耗时，本机）

```bash
conda activate afsi-dolfinx
cd afsic/demo/ib2d/demo_597
```

| 命令 | 实测 `real` |
|---|---|
| `python run_reference.py --nx 64 --tend 0.005 --print-dump 50` | 1m50.387s |
| `NX=64 python main.py` | 4m8.290s |
| `python compare.py` | - |

（`real` 取自 `ref.log` / `afsi.log`；本机同一时刻还有大量其他算例在跑，
绝对耗时波动很大。参考解写入 `ib2d_reference.npz`，AFSI 写入
`plot/afsi_result.npz`，`compare.py` 生成 `figures/` 与 `compare_table.csv`。）
