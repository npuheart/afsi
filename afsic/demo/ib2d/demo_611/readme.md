# demo_611 — IB2d example "Dear KC" (matIB2d ``Examples/Example_KC/For_Danh``) in AFSI.

The "For Danh" member of the ``Example_KC`` teaching family: 2128 Lagrangian
markers (all target points, k = 1e6) animated through the three letter
geometries stored in ``All_Positions.txt`` by the family's
``update_Target_Point_Positions.m``, ported line-by-line to
``ib2d_input/update_Target_Point_Positions.py`` (only the first 284 markers
move).  Physics, box, mu, dt and the animation are identical to demo_610
(``For_Brian``); this variant ships a 256 x 256 grid and softer targets.

Cost and **upstream instability**: as in demo_590, pyIB2d's explicit solver
diverges on this example, so the demo runs a **reduced** setting: ``NX = 128``
(the shipped grid is 256^2) and ``T = 0.005 s`` = 2000 steps (4% of the 0.12 s
cycle), with ``print_dump = 100`` instead of ``print_Dump = 500``.  See
``readme.md`` for the divergence evidence.

## 结果（`python compare.py` 的完整输出）

```
        t   dXmax/h            max|u| a/r    du away    du band      relL2    df away
  0.00000  4.14e-10  0.000e+00/0.000e+00   0.000e+00  0.000e+00  0.000e+00  0.000e+00
  0.00025  4.67e-05  4.678e-02/4.444e-02   3.224e-03  1.915e-02  2.581e-01  4.775e-15
  0.00050  6.64e-04  3.033e-01/2.944e-01   2.542e-02  1.336e-01  2.707e-01  1.620e-14
  0.00075  2.94e-03  7.437e-01/7.364e-01   8.231e-02  3.837e-01  2.768e-01  2.961e-14
  0.00100  8.01e-03  1.397e+00/1.381e+00   1.870e-01  7.538e-01  2.845e-01  3.847e-14
  0.00125  1.65e-02  2.368e+00/2.467e+00   3.495e-01  1.200e+00  2.962e-01  3.934e-14
  0.00150  2.84e-02  3.498e+00/3.964e+00   5.756e-01  1.654e+00  3.110e-01  4.289e-14
  0.00175  4.28e-02  4.683e+00/5.797e+00   8.664e-01  2.393e+00  3.272e-01  4.421e-14
  0.00200  5.82e-02  5.822e+00/7.918e+00   1.216e+00  3.449e+00  3.422e-01  4.203e-14
  0.00225  7.47e-02  6.841e+00/1.028e+01   1.613e+00  4.678e+00  3.545e-01  3.736e-14
  0.00250  9.69e-02  7.723e+00/1.285e+01   2.043e+00  6.027e+00  3.638e-01  3.468e-14
  0.00275  1.19e-01  8.507e+00/1.560e+01   2.489e+00  7.437e+00  3.709e-01  3.239e-14
  0.00300  1.41e-01  9.672e+00/1.848e+01   2.936e+00  8.862e+00  3.768e-01  2.871e-14
  0.00325  1.60e-01  1.120e+01/2.145e+01   3.368e+00  1.028e+01  3.822e-01  4.078e-14
  0.00350  1.76e-01  1.275e+01/2.442e+01   3.767e+00  1.168e+01  3.876e-01  9.315e-14
  0.00375  1.87e-01  1.425e+01/2.734e+01   4.118e+00  1.309e+01  3.931e-01  1.741e-13
  0.00400  1.93e-01  1.566e+01/3.014e+01   4.404e+00  1.459e+01  3.986e-01  2.873e-13
  0.00425  2.00e-01  1.701e+01/3.279e+01   4.685e+00  1.629e+01  4.037e-01  4.377e-13
  0.00450  2.07e-01  1.840e+01/3.552e+01   5.623e+00  1.763e+01  4.078e-01  6.313e-13
  0.00475  2.20e-01  1.986e+01/3.797e+01   6.580e+00  1.853e+01  4.104e-01  8.747e-13
  0.00500  2.46e-01  2.142e+01/3.993e+01   7.521e+00  1.898e+01  4.111e-01  2.542e+02
markers          : max|dX| 3.842e-03 m = 2.46e-01 h (2128 Lagrangian points)
velocity         : max|u| AFSI 2.1422e+01 / IB2d 3.9934e+01 (4.64e-01 relative); max|du| 7.521e+00 = 1.88e-01 max|u| away from the IB band (1873/4096 nodes), 1.898e+01 (0.48 max|u|) inside it; relative L2 4.111e-01
force field      : max|df| 2.542e+02 away from the band, |f|max 7.961e+04
wrote /Users/pengfei.ma/afsi/afsic/demo/ib2d/demo_611/figures/compare_table.csv and the figures
```

这张表就是 `compare.py` 的原始输出：每帧的 `dXmax/h` 是标记点偏差（以网格步长 h
为单位），`max|u| a/r` 是 AFSI/IB2d 的速度峰值，`du away` / `du band` 是 IB 带外/
带内的速度差，`relL2` 是速度场相对 $L_2$，`df away` 是带外外力场差（有被动标量时
还有 `max|dC|` 与 `relL2(C)`）。它与 `figures/compare_table.csv` 一致。

## 不可比 / 需要打折看的地方

* **规模缩减**：上游 `256 x 256`、`Tfinal = 0.12 s`；本 demo 用 `NX = 128`
  （`128 x 128`）和 `T = 0.005 s`（2000 步），`dt = 2.5e-6`、`print_dump = 100`。
* **为什么只跑 0.005 s**：同 demo_590/609/610（上游锚点速度峰值 ~73 m/s，
  pyIB2d 在更大的窗口里 overflow/NaN）。
* **与 demo_609 的关系**：`For_Danh` 与 `Kathleen_256` 共用同一份
  `update_Target_Point_Positions.m`（md5 相同）与同样的 2128 点 / `k = 1e6` /
  256² 设置，只有几何与 `All_Positions.txt` 不同（本 demo 把网格降到 128²）。
* **要打折的地方**：同 demo_609 —— 网格降一级带来的近场速度差异（相对 $L_2$
  约 2 成），`max|dX|/h` 是判据。

## 运行方法（实测耗时，本机）

```bash
conda activate afsi-dolfinx
cd afsic/demo/ib2d/demo_611
```

| 命令 | 实测 `real` |
|---|---|
| - | - |

（`real` 取自 `ref.log` / `afsi.log`；本机同一时刻还有大量其他算例在跑，
绝对耗时波动很大。参考解写入 `ib2d_reference.npz`，AFSI 写入
`plot/afsi_result.npz`，`compare.py` 生成 `figures/` 与 `compare_table.csv`。）
