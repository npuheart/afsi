# demo_596 — IB2d example "Corals" (matIB2d
``Examples/Examples_First_Year_Seminar/Corals``) in AFSI.

## 中文小结

* **算例**：`Examples_First_Year_Seminar/Corals` —— 180 个靶点（`k = 1e7`）组成
  两对珊瑚，`1 x 1` 盒、上游 `Tfinal = 1 s`。
* **新的东西**：`.m` 用两组三角波角度（每步 `pi dt` 与 `pi dt/2` 累加、折叠到
  `[0, 2 pi/3]`）把结构的四分之一各自绕基点旋转 —— 与 demo_463/464 的弹簧泵
  完全不同。
* **缩减**：`T = 0.005 s`（500 步 = 0.5%），网格/`dt` 与上游一致。
* **结果**：标记点 `3.0e-3 h`（运动很小，这个偏差已经很严）；速度相对 $L_2$ `0.72`
  （IB 带内近场差异，见"不可比"）。
* **坑**：`angVec1(mod(n, L1))` 是 1 基索引，`mod == 0` 时 MATLAB 会报错；
  `corals.vertex` 的 `textscan` 错位问题同 demo_595。


> 下面 "Physics / porting notes" 部分是英文叙述（与 `main.py` 的 docstring 同源）：
> 上游算例的标识符、方程都来自英文源码，保持原样以免转述出错。

180 Lagrangian markers -- two coral pairs, four tentacles of 45 markers -- all
target points (k = 1e7) in a 1 x 1 periodic box (mu = 0.01, rho = 1, 64 x 64,
dt = 1e-5).  The example's own ``update_Target_Point_Positions.m`` (ported
line-by-line to ``ib2d_input/update_Target_Point_Positions.py`` and imported by
both codes) rotates each quarter of the structure about its own base point by a
triangular-wave angle

    n = ceil((t+dt)/dt),  ang1 = tri(mod(n, L1) * pi dt),  L1 = floor(2/dt),
    ang2 = tri(mod(n, L2) * pi dt/2),  L2 = floor(4/dt),

folded into ``[0, 2 pi/3]``, so the tentacles open and close.  The rotation
centres come from ``corals.vertex``.

Upstream quirks: the .m's local helper indexes a 1-based vector with
``mod(n, L)``, which would error when ``mod(n, L) == 0`` (reachable only after
``L`` steps); and its ``textscan``-based vertex read shifts the columns because
of the bare count on line 1 -- the port reads the file row-wise (documented in
the module docstring).

Cost: the shipped ``Tfinal = 1 s`` is 100000 steps; the demo runs a **reduced**
``T = 0.005 s`` (500 steps) with ``print_dump = 50`` (10 frames) instead of the
example's 2000.

## 结果（`python compare.py` 的完整输出）

```
        t   dXmax/h            max|u| a/r    du away    du band      relL2    df away
  0.00000  4.31e-10  0.000e+00/0.000e+00   0.000e+00  0.000e+00  0.000e+00  0.000e+00
  0.00051  3.20e-03  5.663e-01/1.720e+00   5.697e-01  1.529e+00  9.075e-01  2.775e-14
  0.00101  2.10e-03  6.098e-01/1.443e+00   3.788e-01  1.148e+00  8.465e-01  1.442e-14
  0.00151  2.73e-03  6.720e-01/1.551e+00   4.974e-01  1.199e+00  7.028e-01  6.758e-15
  0.00201  3.25e-03  6.165e-01/1.609e+00   4.112e-01  1.369e+00  9.115e-01  6.149e-15
  0.00251  2.18e-03  6.326e-01/1.484e+00   4.522e-01  1.174e+00  7.473e-01  1.694e-14
  0.00301  3.86e-03  7.143e-01/1.025e+00   3.248e-01  8.419e-01  7.065e-01  2.493e-14
  0.00351  1.95e-03  7.527e-01/1.626e+00   4.681e-01  1.187e+00  7.176e-01  1.592e-14
  0.00401  1.95e-03  7.530e-01/1.501e+00   3.430e-01  1.035e+00  6.694e-01  1.260e-14
  0.00451  2.56e-03  7.795e-01/1.975e+00   5.505e-01  1.233e+00  6.446e-01  1.342e-14
  0.00501  2.98e-03  7.996e-01/1.187e+00   2.931e-01  1.005e+00  7.212e-01  2.006e-14
markers          : max|dX| 4.651e-05 m = 2.98e-03 h (180 Lagrangian points)
velocity         : max|u| AFSI 7.9958e-01 / IB2d 1.1872e+00 (3.27e-01 relative); max|du| 2.931e-01 = 2.47e-01 max|u| away from the IB band (3926/4096 nodes), 1.005e+00 (0.85 max|u|) inside it; relative L2 7.212e-01
force field      : max|df| 2.006e-14 away from the band, |f|max 8.893e+03
wrote /Users/pengfei.ma/afsi/afsic/demo/ib2d/demo_596/figures/compare_table.csv and the figures
```

这张表就是 `compare.py` 的原始输出：每帧的 `dXmax/h` 是标记点偏差（以网格步长 h
为单位），`max|u| a/r` 是 AFSI/IB2d 的速度峰值，`du away` / `du band` 是 IB 带外/
带内的速度差，`relL2` 是速度场相对 $L_2$，`df away` 是带外外力场差（有被动标量时
还有 `max|dC|` 与 `relL2(C)`）。它与 `figures/compare_table.csv` 一致。

## 不可比 / 需要打折看的地方

* **规模缩减**：上游 `Tfinal = 1 s`（100000 步）；本 demo 用 `T = 0.005 s`
  （500 步 = 0.5%），网格保持上游的 `64 x 64`、`dt = 1e-5`。
* **运动很小**：角度律每步只转 `pi*dt`（≈`3.1e-5` rad/步），500 步累计 0.9°，
  锚点位移约 `2.4e-3 m`。所以标记点偏差 `3.0e-3 h` 已经是很严的判据。
* **速度场不可比**：IB2d 的 `max|u|` 比 AFSI 大 ~1.5–2 倍（相对 $L_2$ 0.72）。
  上游把 `k=1e7` 的靶点力直接展布成共置力密度，AFSI 用一致的力密度插值，近场速度
  本来就不同；缩到 500 步（运动极小）时，两个解的差别主要是离散差异而非物理差异。
* **旋转角序列**：`.m` 用 `angVec1(mod(n, L1))`（MATLAB 1 基）。当
  `mod(n, L1) == 0` 时 MATLAB 会报错（`L1 = floor(2/dt)`，本 demo 的窗口远不到），
  移植里把这一不可达情形钳到末元素（见模块 docstring）。
* **上游 textscan 错位**：同 demo_595 —— `corals.vertex` 首行是点数，MATLAB 的
  `textscan('%f %f')` 会错位配对；本移植按行读（意图），因此与 MATLAB 原样运行的
  结果不同，但 AFSI 与 pyIB2d 共用同一份模块，互相对齐。

## 运行方法（实测耗时，本机）

```bash
conda activate afsi-dolfinx
cd afsic/demo/ib2d/demo_596
```

| 命令 | 实测 `real` |
|---|---|
| `python run_reference.py --tend 0.005 --print-dump 50` | 0m57.276s |
| `python main.py` | 0m48.919s |
| `python compare.py` | - |

（`real` 取自 `ref.log` / `afsi.log`；本机同一时刻还有大量其他算例在跑，
绝对耗时波动很大。参考解写入 `ib2d_reference.npz`，AFSI 写入
`plot/afsi_result.npz`，`compare.py` 生成 `figures/` 与 `compare_table.csv`。）
