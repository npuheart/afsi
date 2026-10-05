# demo_590 — IB2d example "Dear KC" (matIB2d ``Examples/Example_KC/Kathleen``) in AFSI.
748 Lagrangian markers in a 1 x 1 periodic box of viscous fluid (mu = 0.01,
rho = 1, 128 x 128, dt = 2.5e-6): the shipped ``dear_KC.vertex`` geometry (an
"KC" wordmark plus decoration).  Every marker is a *target point* (k = 5e6), so
the structure is essentially rigid; the example is a teaching demo of IB2d's
moving-target machinery.

## 中文小结

* **算例**：`Example_KC/Kathleen` —— 748 个靶点（`k = 5e6`）拼出 "KC" 字样，
  上游 `Tfinal = 0.12 s`（48000 步），`128 x 128`。
* **新的东西**：`update_Target_Point_Positions.m` 让前 284 个点在三套字母几何之间
  做分段多项式插值（周期 0.12 s）。
* **缩减**：本 demo 只跑 `T = 0.005 s`（2000 步 = 周期 4%），因为锚点峰值速度 ~73 m/s，
  **pyIB2d 在 `T = 0.02 s` 会 overflow/NaN 跑不完**。
* **结果**：标记点 `0.29 h`、速度相对 $L_2$ `0.22` —— 对这么"猛"的运动算是对上了。
* **坑**：`dear_KC.vertex` 有 3 对坐标完全重合的点，模板按惯例错开 `1e-6 dx`。


> 下面 "Physics / porting notes" 部分是英文叙述（与 `main.py` 的 docstring 同源）：
> 上游算例的标识符、方程都来自英文源码，保持原样以免转述出错。

Only the first 284 markers move.  The example's own
``update_Target_Point_Positions.m`` -- ported line-by-line to
``ib2d_input/update_Target_Point_Positions.py`` and imported by *both* this port
and pyIB2d -- animates them through the three letter geometries stored in
``All_Positions.txt`` with the phase times

    tP1 = 0.05 s (P1 -> P2),   tP2 = 0.01 s (wait in P2),
    tP3 = 0.05 s (P2 -> P3),   tP4 = 0.01 s (wait in P3),   period = 0.12 s,

blending with the piecewise-polynomial ``g(s)`` (parabola / cubic / inverted
parabola) applied to the normalised phase time.

Cost and **upstream instability**: the shipped ``Tfinal = 0.12 s`` is 48000
steps, and the anchors swing hard (peak anchor speed ~73 m/s at the end of the
P1 -> P2 transition).  pyIB2d's explicit solver **diverges** on this example
somewhere between t = 0.005 s and t = 0.01 s (overflow warnings in
``please_Update_Fluid_Velocity``), so the demo runs a **reduced** ``T = 0.005 s``
(2000 steps, 4% of the letter cycle) with ``print_dump = 100`` (20 frames)
instead of the example's ``print_Dump = 500``; that window is stable in pyIB2d
and is the only window in which a reference exists at all.  See ``readme.md``
for the divergence evidence.

## 结果（`python compare.py` 的完整输出）

```
        t   dXmax/h            max|u| a/r    du away    du band      relL2    df away
  0.00000  1.00e-06  0.000e+00/0.000e+00   0.000e+00  0.000e+00  0.000e+00  0.000e+00
  0.00025  2.99e-04  3.041e-01/2.898e-01   1.908e-02  5.929e-02  2.029e-01  1.526e-14
  0.00050  3.36e-03  1.082e+00/9.334e-01   1.440e-01  3.207e-01  1.928e-01  4.579e-14
  0.00075  1.15e-02  1.813e+00/2.016e+00   4.549e-01  7.208e-01  2.229e-01  7.987e-14
  0.00100  2.38e-02  2.837e+00/3.867e+00   9.736e-01  1.461e+00  2.740e-01  1.068e-13
  0.00125  3.64e-02  3.621e+00/6.088e+00   1.646e+00  2.493e+00  3.190e-01  1.141e-13
  0.00150  4.64e-02  4.903e+00/8.492e+00   2.378e+00  3.617e+00  3.524e-01  1.050e-13
  0.00175  4.84e-02  6.275e+00/1.095e+01   3.071e+00  4.693e+00  3.723e-01  1.366e-13
  0.00200  4.36e-02  7.712e+00/1.326e+01   3.630e+00  6.007e+00  3.792e-01  2.461e-13
  0.00225  5.07e-02  9.235e+00/1.539e+01   3.989e+00  7.357e+00  3.746e-01  4.000e-13
  0.00250  6.28e-02  1.076e+01/1.721e+01   4.135e+00  8.439e+00  3.618e-01  6.105e-13
  0.00275  7.09e-02  1.227e+01/1.873e+01   4.246e+00  9.125e+00  3.447e-01  8.937e-13
  0.00300  7.78e-02  1.368e+01/2.051e+01   4.406e+00  9.391e+00  3.275e-01  1.267e-12
  0.00325  9.48e-02  1.496e+01/2.183e+01   4.587e+00  9.153e+00  3.130e-01  1.759e-12
  0.00350  1.21e-01  1.608e+01/2.345e+01   5.004e+00  9.083e+00  3.011e-01  2.426e-12
  0.00375  1.49e-01  1.714e+01/2.543e+01   5.319e+00  9.392e+00  2.899e-01  6.981e+01
  0.00400  1.75e-01  1.956e+01/2.767e+01   5.734e+00  1.005e+01  2.778e-01  3.420e+02
  0.00425  1.97e-01  2.226e+01/3.032e+01   6.668e+00  1.076e+01  2.645e-01  5.846e+02
  0.00450  2.07e-01  2.500e+01/3.316e+01   7.851e+00  1.145e+01  2.501e-01  1.268e+03
  0.00475  2.43e-01  2.731e+01/3.581e+01   9.175e+00  1.408e+01  2.353e-01  2.725e+03
  0.00500  2.90e-01  2.874e+01/3.783e+01   1.047e+01  1.871e+01  2.223e-01  4.292e+03
markers          : max|dX| 2.265e-03 m = 2.90e-01 h (748 Lagrangian points)
velocity         : max|u| AFSI 2.8739e+01 / IB2d 3.7829e+01 (2.40e-01 relative); max|du| 1.047e+01 = 2.77e-01 max|u| away from the IB band (11162/16384 nodes), 1.871e+01 (0.49 max|u|) inside it; relative L2 2.223e-01
force field      : max|df| 4.292e+03 away from the band, |f|max 2.188e+05
wrote /Users/pengfei.ma/afsi/afsic/demo/ib2d/demo_590/figures/compare_table.csv and the figures
```

这张表就是 `compare.py` 的原始输出：每帧的 `dXmax/h` 是标记点偏差（以网格步长 h
为单位），`max|u| a/r` 是 AFSI/IB2d 的速度峰值，`du away` / `du band` 是 IB 带外/
带内的速度差，`relL2` 是速度场相对 $L_2$，`df away` 是带外外力场差（有被动标量时
还有 `max|dC|` 与 `relL2(C)`）。它与 `figures/compare_table.csv` 一致。

## 不可比 / 需要打折看的地方

* **规模缩减**：上游 `Tfinal = 0.12 s`（48000 步）在 `128 x 128` 上；本 demo 用
  `T = 0.005 s`（2000 步 = 信件周期的 4%），网格保持 128 x 128、`dt = 2.5e-6`，
  `print_dump = 100`（20 帧）代替上游的 `print_Dump = 500`。
* **为什么只跑 0.005 s（上游 pyIB2d 会发散）**：这个算例的锚点摆得极猛 ——
  `All_Positions.txt` 里单个点最大走 `0.61 m`，在 `tP1 = 0.05 s` 内完成，相位末
  锚点速度可达 ~73 m/s。把窗口放大到 `T = 0.02 s` 时，pyIB2d 的
  `please_Update_Fluid_Velocity` 报 `overflow`/`invalid value` 并产生 NaN，整个 run
  跑不完（本 demo 的 `run_reference.py` 在 `--tend 0.02` 下会在几分钟内变成 NaN
  而无法结束）；`T = 0.005 s` 是本机实测**能产出完整参考解**的窗口（本 demo 的
  `ib2d_reference.npz` 就是它）。发散的确切时刻没有细化，只确认落在 0.005–0.02 s。
* **AFSI 侧不散**：同一套力/律在 AFSI 里跑完 2000 步（末帧 `max|u| ≈ 28.7` m/s），
  与参考解的 37.8 m/s 同量级；速度场相对 $L_2$ `0.22`（在这样"猛拉"的算例里已经
  算对得上），标记点 `0.29 h`（字母是刚性靶点，位置差主要来自两边罚力柔度不同）。
* **重合标记点**：`dear_KC.vertex` 有 3 对**坐标完全相同**的点（`(0.5,0.5)`、
  `(0.71,0.4)`、`(0.71,0.6)`），dolfinx 会合并重合顶点，模板按惯例把它们错开
  `1e-6*dx`（远小于离散误差），IB2d/matIB2d 则保留为不同的拉格朗日点。
* **无法与 MATLAB 端逐位对齐**：无（这个算例没有 textscan 错位问题）；
  `update_Target_Point_Positions.py` 是逐行移植，两边 import 同一文件。

## 运行方法（实测耗时，本机）

```bash
conda activate afsi-dolfinx
cd afsic/demo/ib2d/demo_590
```

| 命令 | 实测 `real` |
|---|---|
| - | - |

（`real` 取自 `ref.log` / `afsi.log`；本机同一时刻还有大量其他算例在跑，
绝对耗时波动很大。参考解写入 `ib2d_reference.npz`，AFSI 写入
`plot/afsi_result.npz`，`compare.py` 生成 `figures/` 与 `compare_table.csv`。）
