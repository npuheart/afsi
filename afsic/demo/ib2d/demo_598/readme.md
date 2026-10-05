# demo_598 — IB2d example "Insect Flight" (matIB2d

``Examples/Example_Insect_Flight/Insect_1024x1024``) in AFSI.

205 Lagrangian markers (a single wing cross-section) in a 1 x 1 periodic box
(``mu = 0.187``, ``rho = 10``, 1024 x 1024 as shipped, dt = 2.5e-5); all markers
are target points with k = 5e7.

The example's own ``update_Target_Point_Positions.m`` (ported line-by-line to
``ib2d_input/update_Target_Point_Positions.py``) strokes the wing to the right,

    x_i = x_i + tanh(200 t) * (L/period) * dt,   L = 0.6 m, period = 0.025 s,

for ``current_time <= period`` (24 m/s after the tanh ramp) and then *stops*:
the .m has no return stroke, so the wing translates 0.6 m and stays there.  Only
x moves.

Cost: 1024 x 1024 is out of reach, so the demo runs a **reduced** ``NX = 64``
(64 x 64) and keeps the shipped ``T = 0.025 s`` (1000 steps = one full stroke)
with ``print_dump = 100`` (10 frames) instead of 40.

## 结果（`python compare.py` 的完整输出）

```
        t   dXmax/h            max|u| a/r    du away    du band      relL2    df away
  0.00000  4.52e-10  0.000e+00/0.000e+00   0.000e+00  0.000e+00  0.000e+00  0.000e+00
  0.00253  8.08e-03  1.758e+01/1.755e+01   1.865e+00  3.318e+00  1.102e-01  2.862e+03
  0.00502  2.28e-02  2.806e+01/2.641e+01   5.166e+00  6.722e+00  1.382e-01  3.948e+05
  0.00753  5.36e-02  3.006e+01/3.174e+01   1.114e+01  8.952e+00  1.749e-01  5.603e+05
  0.01003  9.46e-02  3.012e+01/3.469e+01   1.300e+01  7.325e+00  2.043e-01  1.285e+06
  0.01253  1.27e-01  3.300e+01/3.686e+01   1.947e+01  2.615e+00  2.155e-01  1.335e+06
  0.01503  1.08e-01  3.472e+01/3.552e+01   2.080e+01  2.018e+00  2.489e-01  1.378e+06
  0.01753  1.02e-01  3.701e+01/3.770e+01   2.242e+01  1.924e+00  2.549e-01  8.329e+05
  0.02003  9.56e-02  3.949e+01/4.202e+01   1.978e+01  1.975e+00  2.611e-01  7.601e+05
  0.02252  1.26e-01  4.115e+01/4.526e+01   2.068e+01  1.733e+00  2.586e-01  9.713e+05
markers          : max|dX| 1.967e-03 m = 1.26e-01 h (205 Lagrangian points)
velocity         : max|u| AFSI 4.1146e+01 / IB2d 4.5260e+01 (9.09e-02 relative); max|du| 2.068e+01 = 4.57e-01 max|u| away from the IB band (4035/4096 nodes), 1.733e+00 (0.04 max|u|) inside it; relative L2 2.586e-01
force field      : max|df| 9.713e+05 away from the band, |f|max 1.012e+06
wrote /Users/pengfei.ma/afsi/afsic/demo/ib2d/demo_598/figures/compare_table.csv and the figures
```

这张表就是 `compare.py` 的原始输出：每帧的 `dXmax/h` 是标记点偏差（以网格步长 h
为单位），`max|u| a/r` 是 AFSI/IB2d 的速度峰值，`du away` / `du band` 是 IB 带外/
带内的速度差，`relL2` 是速度场相对 $L_2$，`df away` 是带外外力场差（有被动标量时
还有 `max|dC|` 与 `relL2(C)`）。它与 `figures/compare_table.csv` 一致。

## 不可比 / 需要打折看的地方

* **规模缩减**：上游 `1024 x 1024`、`Tfinal = 0.05 s`；本 demo 用 `NX = 64`
  （`64 x 64`）并保留 `T = 0.025 s`（1000 步 = 一个完整冲程），`print_dump = 100`。
  1024² 是 64² 的 256 倍，本机负载下无法承受。
* **物理上本来就很"硬"**：翅膀只有 205 个点，却要在 `0.025 s` 内平移 0.6 m
  （24 m/s），`k = 5e7` 的靶点在 `mu = 0.187`、`rho = 10` 的流体里属于强罚力；
  64² 的 h = 0.0156 m 与 0.6 m 的行程相比很粗，**近场速度不可信**。
* **可比的部分**：`.m` 是**增量**律（`xPts += tanh(200 t) * 24 * dt`），所以只要
  两边都用同一份模块，"第 k 步的锚点"完全一样；`max|dX|/h` 度量的是结构跟随，
  是这里唯一稳的判据。上游没有回程（`current_time > period` 后 `.m` 什么都不做），
  翅膀停在右边 —— 移植保持原样。

## 运行方法（实测耗时，本机）

```bash
conda activate afsi-dolfinx
cd afsic/demo/ib2d/demo_598
```

| 命令 | 实测 `real` |
|---|---|
| `python run_reference.py --nx 64 --tend 0.025 --print-dump 100` | 0m54.216s |
| `NX=64 python main.py` | 2m1.552s |
| `python compare.py` | - |

（`real` 取自 `ref.log` / `afsi.log`；本机同一时刻还有大量其他算例在跑，
绝对耗时波动很大。参考解写入 `ib2d_reference.npz`，AFSI 写入
`plot/afsi_result.npz`，`compare.py` 生成 `figures/` 与 `compare_table.csv`。）
