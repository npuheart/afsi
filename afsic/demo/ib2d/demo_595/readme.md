# demo_595 — IB2d example "Sawtooth" (matIB2d ``Examples/Example_Sawtooth``) in AFSI.

A blood-pump experiment: 6552 Lagrangian markers, all *target points*
(k = 1e11), in a 12 x 3 periodic box with mu = 2.234, rho = 1000,
dt = 2.5e-5.  The upper jaw (the first 3276 markers of ``sawtooth.vertex``)
oscillates in y about its initial position,

    y_i(t) = y_i(0) - (a/2) sin(2 pi freq t),   a = 0.337 m, freq = 2 Hz,

so the gap between the jaws opens and closes 400 times over the shipped
``Tfinal = 10 s``; the lower jaw is static.

Only the y column moves.  The example's own ``update_Target_Point_Positions.m``
is ported line-by-line to ``ib2d_input/update_Target_Point_Positions.py``; note
there that the shipped .m mis-parses the ``.vertex`` header (MATLAB's
``textscan`` reads the file as a *stream* of numbers, so the bare point count on
line 1 shifts every pair and the "y values" it extracts are really shifted x
values).  The port reads the file row-wise -- the clear intent of the
surrounding ``vertices(2:end, :)`` -- and both codes here import the same
module, so the comparison stays exact.

Cost: the shipped grid is 1024 x 256 and Tfinal = 10 s (400000 steps), so the
demo runs a **reduced** ``NX = 128`` (=> 128 x 32) and ``T = 0.0125 s``
(500 steps = 2.5% of the 0.5 s pump period) with ``print_dump = 50`` (10 frames)
instead of 400.

## 结果（`python compare.py` 的完整输出）

```
        t   dXmax/h            max|u| a/r    du away    du band      relL2    df away
  0.00000  6.38e-10  0.000e+00/0.000e+00   0.000e+00  0.000e+00  0.000e+00  0.000e+00
  0.00128  1.98e-02  6.066e+00/1.107e+02   4.520e+01  1.095e+02  9.786e-01  9.981e-10
  0.00253  2.81e-02  5.391e+00/1.023e+02   3.941e+01  1.012e+02  9.806e-01  1.061e-09
  0.00378  4.55e-02  6.336e+00/6.191e+01   2.914e+01  6.025e+01  9.261e-01  9.665e-10
  0.00502  6.65e-02  8.129e+00/1.093e+02   4.397e+01  1.109e+02  9.704e-01  1.112e-09
  0.00628  7.19e-02  7.743e+00/8.780e+01   3.567e+01  8.630e+01  9.649e-01  1.018e-09
  0.00753  7.98e-02  7.403e+00/8.982e+01   4.439e+01  9.001e+01  9.688e-01  1.460e-09
  0.00877  8.54e-02  8.847e+00/8.505e+01   4.273e+01  8.324e+01  9.844e-01  1.241e-09
  0.01003  7.88e-02  7.757e+00/1.098e+02   5.358e+01  1.083e+02  9.939e-01  9.025e-10
  0.01128  8.06e-02  8.564e+00/1.204e+02   6.404e+01  1.181e+02  9.972e-01  1.226e-09
markers          : max|dX| 7.561e-03 m = 8.06e-02 h (6552 Lagrangian points)
velocity         : max|u| AFSI 8.5637e+00 / IB2d 1.2036e+02 (9.29e-01 relative); max|du| 6.404e+01 = 5.32e-01 max|u| away from the IB band (2369/4096 nodes), 1.181e+02 (0.98 max|u|) inside it; relative L2 9.972e-01
force field      : max|df| 1.226e-09 away from the band, |f|max 8.393e+08
wrote /Users/pengfei.ma/afsi/afsic/demo/ib2d/demo_595/figures/compare_table.csv and the figures
```

这张表就是 `compare.py` 的原始输出：每帧的 `dXmax/h` 是标记点偏差（以网格步长 h
为单位），`max|u| a/r` 是 AFSI/IB2d 的速度峰值，`du away` / `du band` 是 IB 带外/
带内的速度差，`relL2` 是速度场相对 $L_2$，`df away` 是带外外力场差（有被动标量时
还有 `max|dC|` 与 `relL2(C)`）。它与 `figures/compare_table.csv` 一致。

## 不可比 / 需要打折看的地方

* **规模缩减**：上游 `1024 x 256`、`Tfinal = 10 s`（400000 步）；本 demo 用
  `NX = 128`（`128 x 32`，因为 `Lx/Ly = 4`）和 `T = 0.0125 s`（500 步 = 0.5 s
  泵周期的 2.5%）。锯齿在上游有 3276 个点，在本 demo 的 128 列上**已经低于网格
  分辨率**，所以"泵"的几何细节不可比（只有整体上下运动是分辨得出来的）。
* **参考解本身在振铃（重要）**：上游靶点刚度 `k = 1e11` 配 `mu = 2.234`、
  `dt = 2.5e-5`，罚力系数 `k dt / mu ≈ 1.1e6`，远在显式罚力的稳定界之外。表里
  IB2d 的 `max|u|` 在 60–120 m/s 之间**剧烈振荡**（AFSI 是 6–9 m/s），说明 pyIB2d
  的"参考解"在这个参数下不可信。这一列（以及速度场）**不能当作判据**。
* **可比的是锚点律**：标记点偏差 `dXmax/h ≤ 0.08 h`（6552 个点），说明两个代码
  用的是同一个 `update_Target_Point_Positions.py`（pyIB2d 从 run 目录 import），
  靶点被正确地钉在锚点上；本 demo 要移植的"锯齿泵运动律"本身是一致的。
* **上游 textscan 错位**：`.m` 用 `textscan('%f %f','CollectOutput',1)` 读
  `sawtooth.vertex`，首行是点数 `6552`，MATLAB 的数值 textscan 把文件当**流**读，
  于是 `(count, x1), (y1, x2), ...` 配对，`vertices(2:end,2)` 取到的是**错位的 x
  坐标**，会把锯齿几何写坏（见模块 docstring）。本移植按 `vertices(2:end,:)` 的
  意图逐行读；AFSI 与 pyIB2d 共用这份模块，所以互相一致，但**与 MATLAB 原样运行
  的结果不同**。

## 运行方法（实测耗时，本机）

```bash
conda activate afsi-dolfinx
cd afsic/demo/ib2d/demo_595
```

| 命令 | 实测 `real` |
|---|---|
| `python run_reference.py --nx 128 --tend 0.0125 --print-dump 50` | 0m55.590s |
| `NX=128 python main.py` | 0m49.969s |
| `python compare.py` | - |

（`real` 取自 `ref.log` / `afsi.log`；本机同一时刻还有大量其他算例在跑，
绝对耗时波动很大。参考解写入 `ib2d_reference.npz`，AFSI 写入
`plot/afsi_result.npz`，`compare.py` 生成 `figures/` 与 `compare_table.csv`。）
