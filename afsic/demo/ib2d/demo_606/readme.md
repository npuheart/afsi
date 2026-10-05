# demo_606 — IB2d example "Swimmer" (matIB2d

``Examples/Examples_First_Year_Seminar/Swimmer``) in AFSI.

184 markers in an 8 x 8 periodic box (mu = 10, rho = 1000, 512 x 512 as shipped,
dt = 2.5e-5): 183 springs (k = 9.6e9) and 182 **non-invariant beams**
(kb = 2.0e12).  The swimmer's body is a flagellum whose target shape is
prescribed through the beams' reference curvatures.

The example's own ``update_nonInv_Beams.m`` (ported line-by-line to
``ib2d_input/update_nonInv_Beams.py``) interpolates the body between the two
phases recorded in ``swimmer.phases`` (columns ``x1, y1, y2``; 184 rows, no
header) with the family's piecewise polynomial blend -- ``a s^2``, then a
cubic, then ``-b (s-1)^2 + 1`` (down-stroke and up-stroke of 0.25 s each) --
and writes the second differences of the interpolated centreline as the beam
curvatures: the y-curvature ``Cy`` actually changes, while ``xP2 = xP1`` makes
``Cx`` constant (upstream quirk, kept verbatim).  The .m assigns those vectors
to ``beams_info[:, 4:6]`` *without* using the beam connectivity, relying on the
file's ordering (checked: the 182 beams span consecutive point triples).

The same pyIB2d driver bug as in demo_604 applies (``beams_info`` -- here also
the scalar 0, since ``beams = 0`` -- is passed to ``update_nonInv_Beams``), so
``run_reference.py`` patches that call.

Cost: 512 x 512 with ``Tfinal = 3 s`` is out of reach, so the demo runs a
**reduced** ``NX = 64`` (64 x 64) and ``T = 0.01 s`` (400 steps = 4% of the
0.5 s stroke) with ``print_dump = 40`` (10 frames) instead of 800.

## 结果（`python compare.py` 的完整输出）

```
        t   dXmax/h            max|u| a/r    du away    du band      relL2    df away
  0.00000  5.35e-10  0.000e+00/0.000e+00   0.000e+00  0.000e+00  0.000e+00  0.000e+00
  0.00103  6.46e-06  9.306e-03/6.154e-03   7.244e-04  3.568e-03  3.807e-01  2.033e-26
  0.00202  1.79e-05  1.217e-02/1.041e-02   1.263e-03  4.751e-03  3.695e-01  1.082e-24
  0.00303  2.55e-05  1.405e-02/1.192e-02   1.474e-03  5.454e-03  3.574e-01  9.721e-24
  0.00402  2.99e-05  1.314e-02/1.192e-02   1.256e-03  6.444e-03  3.511e-01  1.643e-22
  0.00502  3.54e-05  1.252e-02/1.113e-02   1.093e-03  8.744e-03  3.705e-01  4.938e-22
  0.00602  5.06e-05  1.426e-02/1.168e-02   1.433e-03  8.124e-03  3.978e-01  1.256e-20
  0.00702  7.15e-05  2.229e-02/1.407e-02   1.772e-03  1.038e-02  3.764e-01  5.853e-20
  0.00803  9.24e-05  2.977e-02/2.085e-02   2.373e-03  1.056e-02  3.277e-01  1.616e-19
  0.00903  1.18e-04  3.595e-02/2.807e-02   3.386e-03  1.181e-02  3.020e-01  6.144e-19
markers          : max|dX| 1.477e-05 m = 1.18e-04 h (184 Lagrangian points)
velocity         : max|u| AFSI 3.5951e-02 / IB2d 2.8067e-02 (2.81e-01 relative); max|du| 3.386e-03 = 1.21e-01 max|u| away from the IB band (4001/4096 nodes), 1.181e-02 (0.42 max|u|) inside it; relative L2 3.020e-01
force field      : max|df| 6.144e-19 away from the band, |f|max 1.604e+04
springs          : mean length AFSI 0.007812 / IB2d 0.007812 (1.22e-05 relative)
wrote /Users/pengfei.ma/afsi/afsic/demo/ib2d/demo_606/figures/compare_table.csv and the figures
```

这张表就是 `compare.py` 的原始输出：每帧的 `dXmax/h` 是标记点偏差（以网格步长 h
为单位），`max|u| a/r` 是 AFSI/IB2d 的速度峰值，`du away` / `du band` 是 IB 带外/
带内的速度差，`relL2` 是速度场相对 $L_2$，`df away` 是带外外力场差（有被动标量时
还有 `max|dC|` 与 `relL2(C)`）。它与 `figures/compare_table.csv` 一致。

## 不可比 / 需要打折看的地方

* **规模缩减**：上游 `512 x 512`、`Tfinal = 3 s`（120000 步）；本 demo 用
  `NX = 64`（`64 x 64`）和 `T = 0.01 s`（400 步 = 0.5 s 冲程的 4%），
  `dt = 2.5e-5` 与上游一致，`print_dump = 40`（10 帧）。
* **上游 bug 被保留**：`.m` 把插值中心线的二阶差分**直接写进**
  `beams_info(:,5:6)`，完全不用梁的连接关系，依赖 `swimmer.nonInv_beam` 的点是
  连续排列的（已核对：182 根梁正好连着 (s, s+1, s+2)）；另外 `xP2 = xP1` 使
  `Cx` 恒定不变（x 方向的曲率永远等于初值）。
* **pyIB2d 驱动 bug**：同 demo_604（`beams = 0` ⇒ 传给 `update_nonInv_Beams` 的
  是标量 0）；`run_reference.py` 已打最小补丁。
* **要打折的地方**：512² 缩到 64² 后，8 m 长的游泳者（184 个点，点距 0.044 m）
  在 h = 0.125 m 的网格上只有 ~0.35 个点/网格 —— 游泳者几乎"看不见"，速度场与
  推力完全不可信。本 demo 的判据是**曲率律本身**（两边同一份模块）与
  `max|dX|/h`。

## 运行方法（实测耗时，本机）

```bash
conda activate afsi-dolfinx
cd afsic/demo/ib2d/demo_606
```

| 命令 | 实测 `real` |
|---|---|
| `python run_reference.py --nx 64 --tend 0.01 --print-dump 40` | 1m17.747s |
| `NX=64 python main.py` | 1m44.512s |
| `python compare.py` | - |

（`real` 取自 `ref.log` / `afsi.log`；本机同一时刻还有大量其他算例在跑，
绝对耗时波动很大。参考解写入 `ib2d_reference.npz`，AFSI 写入
`plot/afsi_result.npz`，`compare.py` 生成 `figures/` 与 `compare_table.csv`。）
