# demo_600 — IB2d example "Tethered Jellyfish" (matIB2d

``Examples/Example_Jellyfish_Swimming/Tethered_Jellyfish/1200x800``) in AFSI.

The high-resolution member of the pair (demo_599 is 600 x 400): 693 Lagrangian
markers, all target points (k = 5e6) in a 3 x 2 periodic box, mu = 0.625,
rho = 1000, dt = 1e-4, shipped grid 1200 x 800.  The update law is the *same
file* as demo_599 (``update_Target_Point_Positions.m``); only the marker states
(``XY_2Pos.txt``, 693 rows) and the grid/viscosity differ, so the port is the
same module and the same physics as demo_599.

The ported law interpolates each bell arm between its two recorded states with
the Moving_Circle piecewise cubic over 0.25 s per half-stroke (see demo_599 for
the equations and for the upstream ``tRTilde`` copy-paste bug it inherits).

Cost: 1200 x 800 with Tfinal = 2 s (20000 steps) is out of reach, so the demo
runs a **reduced** ``NX = 150`` (150 x 100) and ``T = 0.05 s`` (500 steps) with
``print_dump = 50`` (10 frames) instead of 100.

## 结果（`python compare.py` 的完整输出）

```
        t   dXmax/h            max|u| a/r    du away    du band      relL2    df away
  0.00000  3.47e-09  0.000e+00/0.000e+00   0.000e+00  0.000e+00  0.000e+00  0.000e+00
  0.00510  8.80e-05  4.921e-03/5.070e-03   5.119e-04  3.341e-03  1.097e-01  2.512e-14
  0.01010  1.37e-03  4.131e-02/4.915e-02   9.061e-03  2.397e-02  2.982e-01  1.090e-13
  0.01510  4.83e-03  1.004e-01/1.460e-01   4.220e-02  7.669e-02  4.888e-01  1.815e-13
  0.02010  1.09e-02  1.692e-01/2.835e-01   1.055e-01  1.771e-01  5.996e-01  2.878e-13
  0.02510  1.85e-02  2.866e-01/4.625e-01   1.921e-01  3.053e-01  6.247e-01  4.691e-13
  0.03010  2.61e-02  4.372e-01/6.979e-01   2.919e-01  4.423e-01  6.092e-01  7.607e-13
  0.03510  3.26e-02  6.200e-01/9.477e-01   3.957e-01  5.648e-01  5.828e-01  1.257e-12
  0.04010  3.82e-02  8.159e-01/1.175e+00   5.010e-01  6.538e-01  5.578e-01  1.926e-12
  0.04510  4.20e-02  9.970e-01/1.332e+00   5.967e-01  7.569e-01  5.342e-01  3.190e-12
markers          : max|dX| 8.392e-04 m = 4.20e-02 h (693 Lagrangian points)
velocity         : max|u| AFSI 9.9703e-01 / IB2d 1.3323e+00 (2.52e-01 relative); max|du| 5.967e-01 = 4.48e-01 max|u| away from the IB band (14705/15000 nodes), 7.569e-01 (0.57 max|u|) inside it; relative L2 5.342e-01
force field      : max|df| 3.190e-12 away from the band, |f|max 2.380e+05
wrote /Users/pengfei.ma/afsi/afsic/demo/ib2d/demo_600/figures/compare_table.csv and the figures
```

这张表就是 `compare.py` 的原始输出：每帧的 `dXmax/h` 是标记点偏差（以网格步长 h
为单位），`max|u| a/r` 是 AFSI/IB2d 的速度峰值，`du away` / `du band` 是 IB 带外/
带内的速度差，`relL2` 是速度场相对 $L_2$，`df away` 是带外外力场差（有被动标量时
还有 `max|dC|` 与 `relL2(C)`）。它与 `figures/compare_table.csv` 一致。

## 不可比 / 需要打折看的地方

* **规模缩减**：上游 `1200 x 800`、`Tfinal = 2 s`；本 demo 用 `NX = 150`
  （`150 x 100`）和 `T = 0.05 s`（500 步），`dt = 1e-4`、`print_dump = 50`。
* 这是 demo_599（600x400）的高分辨率孪生兄弟：**更新律文件完全相同**（同一份
  `update_Target_Point_Positions.m`），只有 `XY_2Pos.txt`（693 行 vs 349 行）、
  `mu`（0.625 vs 1.25）和上游网格不同。本 demo 的移植模块与 demo_599 是**同一份
  代码**（同样的 `tRTilde` 上游 bug 被保留）。
* **要打折的地方**：与 demo_599 相同 —— 上游 1200x800 缩到 150x100 后薄壁
  （点距 ~0.012 m）比 h = 0.02 m 细，壁面附近的速度场不可信；`max|dX|` 与锚点律
  才是判据。

## 运行方法（实测耗时，本机）

```bash
conda activate afsi-dolfinx
cd afsic/demo/ib2d/demo_600
```

| 命令 | 实测 `real` |
|---|---|
| `python run_reference.py --nx 150 --tend 0.05 --print-dump 50` | 3m39.450s |
| `NX=150 python main.py` | 4m3.217s |
| `python compare.py` | - |

（`real` 取自 `ref.log` / `afsi.log`；本机同一时刻还有大量其他算例在跑，
绝对耗时波动很大。参考解写入 `ib2d_reference.npz`，AFSI 写入
`plot/afsi_result.npz`，`compare.py` 生成 `figures/` 与 `compare_table.csv`。）
