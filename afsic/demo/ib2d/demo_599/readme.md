# demo_599 — IB2d example "Tethered Jellyfish" (matIB2d

``Examples/Example_Jellyfish_Swimming/Tethered_Jellyfish/600x400``) in AFSI.

349 Lagrangian markers, all target points (k = 5e6), in a 3 x 2 periodic box
with mu = 1.25, rho = 1000; the shipped grid is 600 x 400 and dt = 1e-4.  The
first marker is the tether (row 1 of ``XY_2Pos.txt``, which has no header), the
next 174 the right arm and the last 174 the left arm.

The example's own ``update_Target_Point_Positions.m`` (ported line-by-line to
``ib2d_input/update_Target_Point_Positions.py`` and imported by both codes)
interpolates each arm between its two recorded states (columns ``x1, x2, y1,
y2``) with the piecewise cubic of the Moving_Circle example, each arm
contracting/expanding over a 0.5 s cycle:

    X = X1 + g(t/tR1) (X2 - X1)      (first half period),
    X = X2 + g((t-tR1)/tR2) (X1 - X2) (second half),

with ``g`` the same a/b/c cubic.  Upstream bug reproduced faithfully: the LEFT
arm's middle cubic piece tests ``tRTilde`` (the RIGHT arm's normalised time)
instead of ``tLTilde`` -- a copy-paste slip in the .m; the port keeps it so
AFSI and pyIB2d remain identical.

Cost: 600 x 400 with Tfinal = 2 s (20000 steps) is out of reach, so the demo
runs a **reduced** ``NX = 150`` (150 x 100) and ``T = 0.05 s`` (500 steps, a
tenth of a stroke) with ``print_dump = 50`` (10 frames) instead of 100.

## 结果（`python compare.py` 的完整输出）

```
        t   dXmax/h            max|u| a/r    du away    du band      relL2    df away
  0.00000  3.26e-09  0.000e+00/0.000e+00   0.000e+00  0.000e+00  0.000e+00  0.000e+00
  0.00510  4.74e-05  2.849e-03/2.873e-03   2.571e-04  1.867e-03  7.658e-02  1.544e-14
  0.01010  9.39e-04  2.811e-02/3.360e-02   4.336e-03  1.704e-02  1.613e-01  9.084e-14
  0.01510  3.97e-03  8.994e-02/1.150e-01   2.195e-02  4.521e-02  2.798e-01  2.040e-13
  0.02010  8.92e-03  1.691e-01/2.438e-01   6.339e-02  1.130e-01  3.884e-01  3.357e-13
  0.02510  1.67e-02  2.615e-01/4.065e-01   1.319e-01  2.198e-01  4.684e-01  5.351e-13
  0.03010  2.62e-02  4.080e-01/6.335e-01   2.249e-01  3.527e-01  5.110e-01  8.840e-13
  0.03510  3.65e-02  5.879e-01/9.080e-01   3.362e-01  4.966e-01  5.177e-01  1.474e-12
  0.04010  4.50e-02  7.845e-01/1.176e+00   4.513e-01  6.141e-01  5.022e-01  2.289e-12
  0.04510  4.84e-02  9.638e-01/1.355e+00   5.416e-01  7.022e-01  4.750e-01  3.435e-12
markers          : max|dX| 9.673e-04 m = 4.84e-02 h (349 Lagrangian points)
velocity         : max|u| AFSI 9.6381e-01 / IB2d 1.3553e+00 (2.89e-01 relative); max|du| 5.416e-01 = 4.00e-01 max|u| away from the IB band (14707/15000 nodes), 7.022e-01 (0.52 max|u|) inside it; relative L2 4.750e-01
force field      : max|df| 3.435e-12 away from the band, |f|max 2.358e+05
wrote /Users/pengfei.ma/afsi/afsic/demo/ib2d/demo_599/figures/compare_table.csv and the figures
```

这张表就是 `compare.py` 的原始输出：每帧的 `dXmax/h` 是标记点偏差（以网格步长 h
为单位），`max|u| a/r` 是 AFSI/IB2d 的速度峰值，`du away` / `du band` 是 IB 带外/
带内的速度差，`relL2` 是速度场相对 $L_2$，`df away` 是带外外力场差（有被动标量时
还有 `max|dC|` 与 `relL2(C)`）。它与 `figures/compare_table.csv` 一致。

## 不可比 / 需要打折看的地方

* **规模缩减**：上游 `600 x 400`、`Tfinal = 2 s`（20000 步）；本 demo 用
  `NX = 150`（`150 x 100`，每步便宜 16 倍）和 `T = 0.05 s`（500 步 = 一次
  0.25 s 半冲程的 1/5），`dt = 1e-4` 与上游一致，`print_dump = 50`。
* **上游 bug 被原样保留**：`.m` 的**左臂**中段三次多项式判据写成
  `elseif tRTilde <= p2`（用了右臂的归一化时间），是复制粘贴错误。移植保留它
  （因此 AFSI 与 pyIB2d 完全一致），但**这不是"正确"的律** —— 如果要和
  MATLAB 端的 *预期*行为对比，需要先修上游。
* **参数很硬**：`mu = 1.25`、`rho = 1000`、`k = 5e6`、`dt = 1e-4`，`tether` 让
  结构固定在盒子里；缩到 150x100 后钟体的薄壁（点距约 0.017 m）仍比 h = 0.02 m
  略细，所以**壁面附近的速度场要打折**；标记点偏差与锚点律是可比的。

## 运行方法（实测耗时，本机）

```bash
conda activate afsi-dolfinx
cd afsic/demo/ib2d/demo_599
```

| 命令 | 实测 `real` |
|---|---|
| `python run_reference.py --nx 150 --tend 0.05 --print-dump 50` | 3m38.489s |
| `NX=150 python main.py` | 4m38.836s |
| `python compare.py` | - |

（`real` 取自 `ref.log` / `afsi.log`；本机同一时刻还有大量其他算例在跑，
绝对耗时波动很大。参考解写入 `ib2d_reference.npz`，AFSI 写入
`plot/afsi_result.npz`，`compare.py` 生成 `figures/` 与 `compare_table.csv`。）
