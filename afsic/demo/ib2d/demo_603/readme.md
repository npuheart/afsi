# demo_603 — IB2d example "Beehive" (matIB2d
``Examples/Examples_First_Year_Seminar/Beehive``) in AFSI.

## 中文小结

* **算例**：`Examples_First_Year_Seminar/Beehive` —— 104 个靶点（`k = 1e7`）的
  "蜂巢花瓣" 旋转，外加盒内被动标量。
* **新的东西**：与 demo_596 同一份旋转律（改的是 `hive.vertex`），再叠加大
  demo_456/476 已验证的被动标量算子。
* **缩减**：`T = 0.005 s`（500 步 = 0.5%），网格/`dt` 与上游一致。
* **结果**：标记点 `6.1e-3 h`；浓度相对 $L_2$ `2.2%`、峰值相对差 `6.2e-5`、
  总量差 `0.2%` —— 标量部分很好；速度相对 $L_2$ `0.40` 要打折。


> 下面 "Physics / porting notes" 部分是英文叙述（与 `main.py` 的 docstring 同源）：
> 上游算例的标识符、方程都来自英文源码，保持原样以免转述出错。

104 Lagrangian markers -- two hive pairs, four petals of 26 markers -- all
target points (k = 1e7) in a 1 x 1 periodic box (mu = 0.01, rho = 1, 64 x 64,
dt = 1e-5), plus a passive concentration scalar seeded inside the hive
(``hive.concentration``; the diffusive/advective update is the split explicit
operator shared with demo_456/476/478).

The petal motion is the *same law as* ``Examples_First_Year_Seminar/Corals``
(demo_596): each quarter of the structure rotates about its base point by a
triangular-wave angle (periods ``pi dt`` and ``pi dt/2`` folded into
``[0, 2 pi/3]``), with the reference geometry read from ``hive.vertex``.
The example's own ``update_Target_Point_Positions.m`` is ported line-by-line to
``ib2d_input/update_Target_Point_Positions.py`` and imported by both codes.

Cost: the shipped ``Tfinal = 1 s`` (100000 steps) is reduced to ``T = 0.005 s``
(500 steps) at the shipped grid, with ``print_dump = 50`` (10 frames) instead
of the example's 2000.

## 结果（`python compare.py` 的完整输出）

```
        t   dXmax/h            max|u| a/r    du away    du band      relL2    df away    max|dC|   relL2(C)
  0.00000  8.88e-15  0.000e+00/0.000e+00   0.000e+00  0.000e+00  0.000e+00  0.000e+00  4.460e-12  3.792e-12
  0.00051  5.49e-03  1.657e+00/2.104e+00   1.073e+00  1.572e+00  6.433e-01  1.086e-13  6.806e-03  3.462e-03
  0.00101  8.14e-03  1.557e+00/1.292e+00   3.408e-01  1.146e+00  3.201e-01  7.214e-14  2.618e-02  8.658e-03
  0.00151  7.09e-03  9.575e-01/1.083e+00   4.242e-01  1.121e+00  4.370e-01  4.132e-14  2.262e-02  9.027e-03
  0.00201  8.35e-03  9.275e-01/1.835e+00   7.128e-01  9.196e-01  4.626e-01  6.935e-14  3.681e-02  1.408e-02
  0.00251  3.81e-03  1.003e+00/1.376e+00   5.860e-01  1.479e+00  5.646e-01  2.498e-14  3.800e-02  1.481e-02
  0.00301  4.73e-03  9.444e-01/1.714e+00   8.797e-01  1.395e+00  4.495e-01  3.711e-14  4.398e-02  1.661e-02
  0.00351  6.17e-03  1.006e+00/1.207e+00   5.494e-01  9.273e-01  6.593e-01  2.292e-14  5.356e-02  1.886e-02
  0.00401  6.88e-03  1.124e+00/1.575e+00   5.378e-01  1.028e+00  3.555e-01  7.115e-15  5.012e-02  1.857e-02
  0.00451  5.55e-03  1.055e+00/1.272e+00   3.317e-01  9.297e-01  2.261e-01  1.500e-14  6.292e-02  2.308e-02
  0.00501  6.12e-03  1.039e+00/1.286e+00   5.637e-01  6.630e-01  4.047e-01  1.649e-14  5.924e-02  2.191e-02
markers          : max|dX| 9.560e-05 m = 6.12e-03 h (104 Lagrangian points)
velocity         : max|u| AFSI 1.0395e+00 / IB2d 1.2862e+00 (1.92e-01 relative); max|du| 5.637e-01 = 4.38e-01 max|u| away from the IB band (3756/4096 nodes), 6.630e-01 (0.52 max|u|) inside it; relative L2 4.047e-01
force field      : max|df| 1.649e-14 away from the band, |f|max 1.563e+04
concentration    : max|dC| 5.924e-02, relative L2 2.191e-02; peak AFSI 0.978666 / IB2d 0.978606 (6.15e-05 relative); total AFSI 91.22769918 / IB2d 91.04641131
                   max|dC| 5.924e-02 away from the band, 4.964e-02 inside it
wrote /Users/pengfei.ma/afsi/afsic/demo/ib2d/demo_603/figures/compare_table.csv and the figures
```

这张表就是 `compare.py` 的原始输出：每帧的 `dXmax/h` 是标记点偏差（以网格步长 h
为单位），`max|u| a/r` 是 AFSI/IB2d 的速度峰值，`du away` / `du band` 是 IB 带外/
带内的速度差，`relL2` 是速度场相对 $L_2$，`df away` 是带外外力场差（有被动标量时
还有 `max|dC|` 与 `relL2(C)`）。它与 `figures/compare_table.csv` 一致。

## 不可比 / 需要打折看的地方

* **规模缩减**：上游 `Tfinal = 1 s`（100000 步）；本 demo 用 `T = 0.005 s`
  （500 步 = 0.5%），网格与时间步保持上游（`64 x 64`、`dt = 1e-5`）。
* **好的部分**：标记点 `6.1e-3 h`（104 点）、浓度峰值相对差 `6.2e-5`、浓度总量
  相对差 `0.2%`、浓度场相对 $L_2$ `2.2%` —— 被动标量的算子（demo_476 已验证到
  舍入）与旋转运动都对上了。
* **速度场要打折**：相对 $L_2$ `0.40`，IB 带内 `max|du| = 0.66` m/s。上游的靶点
  罚力/展布与 AFSI 的一致力密度插值在近场本来就不同（与 demo_460/461/594 同量级）。
* **旋转角序列**：与 demo_596 同一份律（`mod(n, L)` 的 1-based 索引，`mod == 0`
  时上游会报错，本窗口不可达）。
* **上游 textscan 错位**：`hive.vertex` 首行是点数，MATLAB `textscan('%f %f')` 会
  错位配对；本移植按行读（意图），AFSI 与 pyIB2d 共用同一模块，互相对齐，但
  **与 MATLAB 原样运行的结果不同**。

## 运行方法（实测耗时，本机）

```bash
conda activate afsi-dolfinx
cd afsic/demo/ib2d/demo_603
```

| 命令 | 实测 `real` |
|---|---|
| `python run_reference.py --tend 0.005 --print-dump 50` | 0m36.376s |
| `python main.py` | 0m48.989s |
| `python compare.py` | - |

（`real` 取自 `ref.log` / `afsi.log`；本机同一时刻还有大量其他算例在跑，
绝对耗时波动很大。参考解写入 `ib2d_reference.npz`，AFSI 写入
`plot/afsi_result.npz`，`compare.py` 生成 `figures/` 与 `compare_table.csv`。）
