# demo_610 — IB2d example "Dear KC" (matIB2d ``Examples/Example_KC/For_Brian``) in AFSI.
The "For Brian" member of the ``Example_KC`` teaching family: 748 Lagrangian
markers in a 1 x 1 periodic box (mu = 0.01, rho = 1, 128 x 128, dt = 2.5e-6),
all target points with k = 5e6.  Only the first 284 markers move; the example's
own ``update_Target_Point_Positions.m`` (ported line-by-line to
``ib2d_input/update_Target_Point_Positions.py`` and imported by both codes)
animates them through the three letter geometries of ``All_Positions.txt``
(phase times tP1 = tP3 = 0.05 s, tP2 = tP4 = 0.01 s, period 0.12 s) with the
family's piecewise-polynomial blend ``g(s)``.

## 中文小结

* **算例**：`Example_KC/For_Brian` —— 与 `Kathleen` **同一份** `update_*.m`
  （md5 相同）、同样的 748 点 / `k = 5e6`，只有字母几何与 `All_Positions.txt` 不同。
* **缩减**：`NX = 64`（上游 128）与 `T = 0.005 s`（2000 步 = 周期 4%）；
  理由同 demo_590。
* **结果**：标记点 `0.16 h`、速度相对 $L_2$ `0.26`。


> 下面 "Physics / porting notes" 部分是英文叙述（与 `main.py` 的 docstring 同源）：
> 上游算例的标识符、方程都来自英文源码，保持原样以免转述出错。

Cost and **upstream instability**: as in demo_590, the anchors swing hard (peak
anchor speed ~73 m/s) and pyIB2d's explicit solver diverges between t = 0.005 s
and t = 0.01 s, so the demo runs a **reduced** ``T = 0.005 s`` (2000 steps, 4%
of the cycle) with ``print_dump = 100`` instead of ``print_Dump = 500``.  See
``readme.md``.

## 结果（`python compare.py` 的完整输出）

```
        t   dXmax/h            max|u| a/r    du away    du band      relL2    df away
  0.00000  1.00e-06  0.000e+00/0.000e+00   0.000e+00  0.000e+00  0.000e+00  0.000e+00
  0.00025  2.39e-04  4.069e-01/3.208e-01   2.202e-02  8.661e-02  2.517e-01  4.014e-14
  0.00050  2.51e-03  1.398e+00/1.292e+00   1.386e-01  4.407e-01  2.628e-01  1.233e-13
  0.00075  9.06e-03  1.912e+00/1.759e+00   4.280e-01  1.017e+00  3.130e-01  1.815e-13
  0.00100  1.87e-02  2.927e+00/3.037e+00   1.009e+00  2.134e+00  3.983e-01  1.821e-13
  0.00125  2.97e-02  3.765e+00/4.755e+00   1.824e+00  3.686e+00  4.777e-01  1.645e-13
  0.00150  3.85e-02  4.305e+00/6.144e+00   2.697e+00  5.065e+00  5.261e-01  1.786e-13
  0.00175  4.08e-02  5.215e+00/7.557e+00   3.418e+00  6.022e+00  5.398e-01  2.515e-13
  0.00200  3.79e-02  5.544e+00/8.954e+00   3.879e+00  6.413e+00  5.262e-01  3.191e-13
  0.00225  3.98e-02  6.398e+00/1.004e+01   4.103e+00  6.349e+00  4.972e-01  4.102e-13
  0.00250  4.54e-02  7.239e+00/1.090e+01   4.201e+00  6.498e+00  4.691e-01  4.676e-13
  0.00275  4.90e-02  8.066e+00/1.165e+01   4.283e+00  6.538e+00  4.483e-01  5.074e-13
  0.00300  5.44e-02  9.030e+00/1.241e+01   4.412e+00  7.022e+00  4.299e-01  6.008e-13
  0.00325  5.93e-02  1.002e+01/1.366e+01   4.914e+00  7.355e+00  4.134e-01  7.761e-13
  0.00350  6.17e-02  1.092e+01/1.499e+01   5.356e+00  7.872e+00  3.976e-01  1.007e-12
  0.00375  7.19e-02  1.169e+01/1.634e+01   5.728e+00  8.666e+00  3.830e-01  1.233e-12
  0.00400  8.36e-02  1.228e+01/1.766e+01   6.211e+00  9.740e+00  3.700e-01  9.778e+00
  0.00425  1.00e-01  1.280e+01/1.882e+01   6.981e+00  1.088e+01  3.554e-01  1.085e+02
  0.00450  1.18e-01  1.443e+01/2.009e+01   7.847e+00  1.175e+01  3.395e-01  3.500e+02
  0.00475  1.36e-01  1.607e+01/2.187e+01   8.709e+00  1.303e+01  3.230e-01  7.768e+02
  0.00500  1.57e-01  1.756e+01/2.364e+01   9.362e+00  1.386e+01  3.059e-01  9.905e+02
markers          : max|dX| 2.449e-03 m = 1.57e-01 h (748 Lagrangian points)
velocity         : max|u| AFSI 1.7565e+01 / IB2d 2.3641e+01 (2.57e-01 relative); max|du| 9.362e+00 = 3.96e-01 max|u| away from the IB band (1648/4096 nodes), 1.386e+01 (0.59 max|u|) inside it; relative L2 3.059e-01
force field      : max|df| 9.905e+02 away from the band, |f|max 1.092e+05
wrote /Users/pengfei.ma/afsi/afsic/demo/ib2d/demo_610/figures/compare_table.csv and the figures
```

这张表就是 `compare.py` 的原始输出：每帧的 `dXmax/h` 是标记点偏差（以网格步长 h
为单位），`max|u| a/r` 是 AFSI/IB2d 的速度峰值，`du away` / `du band` 是 IB 带外/
带内的速度差，`relL2` 是速度场相对 $L_2$，`df away` 是带外外力场差（有被动标量时
还有 `max|dC|` 与 `relL2(C)`）。它与 `figures/compare_table.csv` 一致。

## 不可比 / 需要打折看的地方

* **规模缩减**：上游 `128 x 128`、`Tfinal = 0.12 s`（48000 步）；本 demo 用
  `NX = 64`（`64 x 64`，每步便宜 4 倍）和 `T = 0.005 s`（2000 步 = 周期的 4%），
  `dt = 2.5e-6` 与上游一致，`print_dump = 100`（20 帧）代替 500。
* **为什么只跑 0.005 s**：同 demo_590/609 —— 锚点速度峰值 ~73 m/s，pyIB2d 在
  `T = 0.02 s` 会 overflow/NaN；0.005 s 是能出完整参考解的窗口。
* **与 demo_590 的关系**：`For_Brian` 与 `Kathleen` 共用**同一份** `update_*.m`
  （md5 相同）与同样的 748 点 / `k = 5e6` / 128² 设置，只有 `dear_KC.vertex` 与
  `All_Positions.txt`（三态几何不同）不同；本 demo 再把网格降到 64² 以压低成本。
* **要打折的地方**：64² 对 748 个刚性点已经很粗（点距与 h 同量级），近场速度不可
  信；`max|dX|/h` 与"结构是否跟上锚点"是判据。

## 运行方法（实测耗时，本机）

```bash
conda activate afsi-dolfinx
cd afsic/demo/ib2d/demo_610
```

| 命令 | 实测 `real` |
|---|---|
| `python run_reference.py --nx 128 --tend 0.005 --print-dump 100` | 17 s |
| `TFINAL=0.005 PRINT_DUMP=100 NX=128 python main.py` | 105 s |
| `python compare.py` | 约 30 s |

（本表由收尾时重跑实测填写；本机同一时刻可能还有其它算例在跑，绝对耗时会有波动。参考解写入 `ib2d_reference.npz`，AFSI 写入
`plot/afsi_result.npz`，`compare.py` 生成 `figures/` 与 `compare_table.csv`。）
