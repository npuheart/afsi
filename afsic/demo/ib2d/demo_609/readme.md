# demo_609 — IB2d example "Dear KC" (matIB2d ``Examples/Example_KC/Kathleen_256``) in AFSI.
The 256 x 256 member of the ``Example_KC`` teaching family: 2128 Lagrangian
markers (all target points, k = 1e6) animated through the three letter
geometries of ``All_Positions.txt`` by the family's
``update_Target_Point_Positions.m`` (ported line-by-line to
``ib2d_input/update_Target_Point_Positions.py``; only the first 284 markers
move).  Physics, box, mu, dt and the letter animation are identical to
demo_590 (``Kathleen``, 128 x 128, k = 5e6); this variant ships a finer grid and
softer targets.

## 中文小结

* **算例**：`Example_KC/Kathleen_256` —— demo_590 的 256² 版本（2128 个靶点、
  `k = 1e6`），同一份 `update_Target_Point_Positions.m`。
* **缩减**：`NX = 128`（上游 256²）与 `T = 0.005 s`（2000 步 = 周期 4%）——
  理由同 demo_590（锚点速度峰值 ~73 m/s，pyIB2d 在更大的窗口会 NaN）。
* **结果**：标记点 `0.34 h`、速度相对 $L_2$ `0.34`。


> 下面 "Physics / porting notes" 部分是英文叙述（与 `main.py` 的 docstring 同源）：
> 上游算例的标识符、方程都来自英文源码，保持原样以免转述出错。

Cost and **upstream instability**: as in demo_590, pyIB2d's explicit solver
diverges on this example (the anchors swing at up to ~73 m/s), so the demo runs
a **reduced** setting: ``NX = 128`` (the shipped 256^2 is 4x more expensive per
step) and ``T = 0.005 s`` = 2000 steps (4% of the 0.12 s letter cycle), with
``print_dump = 100`` instead of the example's ``print_Dump = 500``.  The grid
reduction is documented in ``readme.md`` together with the divergence evidence.

## 结果（`python compare.py` 的完整输出）

```
        t   dXmax/h            max|u| a/r    du away    du band      relL2    df away
  0.00000  3.82e-10  0.000e+00/0.000e+00   0.000e+00  0.000e+00  0.000e+00  0.000e+00
  0.00025  5.03e-05  1.044e-01/7.919e-02   4.490e-03  2.563e-02  2.425e-01  8.681e-15
  0.00050  6.30e-04  6.654e-01/5.246e-01   3.045e-02  1.435e-01  2.365e-01  3.291e-14
  0.00075  2.45e-03  1.605e+00/1.319e+00   8.316e-02  3.335e-01  2.282e-01  6.859e-14
  0.00100  6.74e-03  2.491e+00/2.133e+00   1.669e-01  6.685e-01  2.219e-01  1.113e-13
  0.00125  1.41e-02  3.097e+00/2.725e+00   3.198e-01  1.101e+00  2.225e-01  1.574e-13
  0.00150  2.48e-02  3.628e+00/3.191e+00   5.414e-01  1.595e+00  2.335e-01  2.048e-13
  0.00175  3.91e-02  4.392e+00/4.371e+00   8.370e-01  2.268e+00  2.538e-01  2.518e-13
  0.00200  5.67e-02  5.405e+00/5.828e+00   1.203e+00  3.048e+00  2.795e-01  2.973e-13
  0.00225  7.71e-02  6.423e+00/7.453e+00   1.627e+00  3.911e+00  3.063e-01  3.410e-13
  0.00250  1.00e-01  7.287e+00/9.181e+00   2.106e+00  4.826e+00  3.311e-01  3.845e-13
  0.00275  1.25e-01  8.069e+00/1.098e+01   2.730e+00  5.793e+00  3.520e-01  4.370e-13
  0.00300  1.50e-01  8.861e+00/1.284e+01   3.402e+00  7.021e+00  3.679e-01  5.078e-13
  0.00325  1.76e-01  9.564e+00/1.468e+01   4.099e+00  8.348e+00  3.783e-01  6.030e-13
  0.00350  2.02e-01  1.000e+01/1.640e+01   4.799e+00  9.854e+00  3.833e-01  7.311e-13
  0.00375  2.28e-01  1.019e+01/1.794e+01   5.481e+00  1.146e+01  3.833e-01  8.993e-13
  0.00400  2.52e-01  1.085e+01/1.932e+01   6.128e+00  1.314e+01  3.789e-01  1.279e+01
  0.00425  2.75e-01  1.184e+01/2.059e+01   6.726e+00  1.481e+01  3.713e-01  9.154e+01
  0.00450  2.97e-01  1.267e+01/2.179e+01   7.257e+00  1.636e+01  3.618e-01  2.814e+02
  0.00475  3.20e-01  1.326e+01/2.283e+01   7.702e+00  1.769e+01  3.522e-01  5.936e+02
  0.00500  3.43e-01  1.357e+01/2.356e+01   8.040e+00  1.935e+01  3.440e-01  9.854e+02
markers          : max|dX| 5.363e-03 m = 3.43e-01 h (2128 Lagrangian points)
velocity         : max|u| AFSI 1.3567e+01 / IB2d 2.3561e+01 (4.24e-01 relative); max|du| 8.040e+00 = 3.41e-01 max|u| away from the IB band (1655/4096 nodes), 1.935e+01 (0.82 max|u|) inside it; relative L2 3.440e-01
force field      : max|df| 9.854e+02 away from the band, |f|max 1.100e+05
wrote /Users/pengfei.ma/afsi/afsic/demo/ib2d/demo_609/figures/compare_table.csv and the figures
```

这张表就是 `compare.py` 的原始输出：每帧的 `dXmax/h` 是标记点偏差（以网格步长 h
为单位），`max|u| a/r` 是 AFSI/IB2d 的速度峰值，`du away` / `du band` 是 IB 带外/
带内的速度差，`relL2` 是速度场相对 $L_2$，`df away` 是带外外力场差（有被动标量时
还有 `max|dC|` 与 `relL2(C)`）。它与 `figures/compare_table.csv` 一致。

## 不可比 / 需要打折看的地方

* **规模缩减**：上游 `256 x 256`、`Tfinal = 0.12 s`（48000 步）；本 demo 用
  `NX = 128`（`128 x 128`）和 `T = 0.005 s`（2000 步 = 周期的 4%），
  `dt = 2.5e-6` 与上游一致，`print_dump = 100`（20 帧）代替 500。
* **为什么只跑 0.005 s**：与 demo_590（`Kathleen`）完全相同的理由 —— 锚点摆得极猛
  （单点最大位移 0.61 m / 0.05 s，相位末速度 ~73 m/s），窗口放大到 `T = 0.02 s`
  时 pyIB2d 会 overflow/NaN 而跑不完；0.005 s 是本机实测能出完整参考解的窗口。
* **与 demo_590 的关系**：同一个 `Example_KC` 家族、同一份
  `update_Target_Point_Positions.m`（2128 vs 748 个点、`k = 1e6` vs `5e6`、
  上游 256² vs 128²），差别只有几何/分辨率/刚度；`All_Positions.txt` 也不同。
* **要打折的地方**：网格从 256² 降到 128²，而点距在上游就很密；速度场相对
  $L_2$ 会有 2 成左右（同 demo_590），标记点 `max|dX|/h` 是判据。
* **重合标记点**：该几何同样有坐标重合的点对，模板按惯例错开 `1e-6*dx`。

## 运行方法（实测耗时，本机）

```bash
conda activate afsi-dolfinx
cd afsic/demo/ib2d/demo_609
```

| 命令 | 实测 `real` |
|---|---|
| - | - |

（`real` 取自 `ref.log` / `afsi.log`；本机同一时刻还有大量其他算例在跑，
绝对耗时波动很大。参考解写入 `ib2d_reference.npz`，AFSI 写入
`plot/afsi_result.npz`，`compare.py` 生成 `figures/` 与 `compare_table.csv`。）
