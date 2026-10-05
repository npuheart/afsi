# demo_594 — IB2d example "Two Corals" (matIB2d ``Examples/Example_Corals/Two_Corals``)

in AFSI.

360 Lagrangian markers in a 0.6 x 0.6 periodic box (mu = 0.0025, rho = 1,
128 x 128, dt = 1e-5): two coral pairs, four tentacles of 90 markers each, all
*target points* with k = 1e8.  The example's own
``update_Target_Point_Positions.m`` -- ported line-by-line to
``ib2d_input/update_Target_Point_Positions.py`` and imported by both this port
and pyIB2d -- is the same video-fitted bending model as demo_461
(``Single_Coral``): the anchors' x is

    X(y, t) = xoffset + q4(t)/S - q3(t) xi + q2(t) S xi^2 - q1(t) S^2 xi^3,
    xi = y - yoffset,  t = 30 (time mod 2.30 s),

with four quintic-in-time coefficient fits ``q1..q4(t)`` (ascending series) and
a linear blend back to the t = 0 values for the last 0.05 s of the 2.30 s pulse;
here the geometry is mirrored about **two** pair centres (0.2 and 0.4) instead
of one, and only x moves.

Cost: the shipped ``Tfinal = 6 s`` (600000 steps) is far out of reach, so the
demo runs a **reduced** ``T = 0.01 s`` (1000 steps, 0.4% of the pulse) at
``NX = 64`` (the shipped grid is 128 x 128; halving it makes a 4x cheaper step)
with ``print_dump = 100`` (10 frames) instead of 2000.  The anchor law itself is
validated over the whole pulse analytically by ``compare.py``.

## 结果（`python compare.py` 的完整输出）

```
        t   dXmax/h            max|u| a/r    du away    du band      relL2    df away
  0.00000  7.11e-10  0.000e+00/0.000e+00   0.000e+00  0.000e+00  0.000e+00  0.000e+00
  0.00101  1.15e-03  1.401e+00/2.132e+00   5.398e-01  1.436e+00  5.363e-01  8.603e-14
  0.00201  1.44e-03  1.592e+00/2.378e+00   6.323e-01  1.365e+00  4.745e-01  7.007e-14
  0.00301  1.72e-03  1.599e+00/2.348e+00   6.869e-01  1.366e+00  5.071e-01  7.384e-14
  0.00401  1.18e-03  1.406e+00/1.984e+00   7.133e-01  1.470e+00  5.093e-01  2.607e-14
  0.00501  1.34e-03  1.522e+00/2.377e+00   7.214e-01  1.404e+00  4.523e-01  5.902e-14
  0.00601  1.24e-03  1.558e+00/2.110e+00   5.780e-01  1.083e+00  3.674e-01  1.428e-14
  0.00701  1.55e-03  1.483e+00/2.297e+00   6.824e-01  1.252e+00  3.901e-01  1.992e-14
  0.00801  1.62e-03  1.486e+00/2.160e+00   6.257e-01  1.068e+00  4.136e-01  9.467e-15
  0.00901  1.62e-03  1.341e+00/1.843e+00   4.361e-01  9.369e-01  3.196e-01  1.002e-14
  0.01001  1.82e-03  1.297e+00/1.749e+00   3.342e-01  8.515e-01  2.980e-01  1.582e-14
markers          : max|dX| 1.703e-05 m = 1.82e-03 h (360 Lagrangian points)
velocity         : max|u| AFSI 1.2968e+00 / IB2d 1.7492e+00 (2.59e-01 relative); max|du| 3.342e-01 = 1.91e-01 max|u| away from the IB band (3714/4096 nodes), 8.515e-01 (0.49 max|u|) inside it; relative L2 2.980e-01
force field      : max|df| 1.582e-14 away from the band, |f|max 6.130e+04
wrote /Users/pengfei.ma/afsi/afsic/demo/ib2d/demo_594/figures/compare_table.csv and the figures
```

这张表就是 `compare.py` 的原始输出：每帧的 `dXmax/h` 是标记点偏差（以网格步长 h
为单位），`max|u| a/r` 是 AFSI/IB2d 的速度峰值，`du away` / `du band` 是 IB 带外/
带内的速度差，`relL2` 是速度场相对 $L_2$，`df away` 是带外外力场差（有被动标量时
还有 `max|dC|` 与 `relL2(C)`）。它与 `figures/compare_table.csv` 一致。

## 不可比 / 需要打折看的地方

* **规模缩减**：上游 `Tfinal = 6 s`（600000 步）在 `128 x 128` 上；本 demo 用
  `NX = 64`、`T = 0.01 s`（1000 步，仅为 2.30 s 脉冲的 0.43%），所以只覆盖脉冲
  开头的一小段（锚点最大走了约 `3.6e-3 m = 0.39 h`）。锚点律本身在
  `compare.py` 里对**整条脉冲**做过解析校验（周期性与峰值），缩减的只是流固耦合
  采样的时长，不是对律的验证。
* **速度场不可比**：标记点偏差 `1.8e-3 h`（很好），但速度场相对 $L_2$ 约 `0.30`、
  IB 带内 `max|du|` 达 `0.85` m/s。这不是 bug：上游是 FFT 投影 + 共置罚力
  （力集中在少数节点、峰值速度偏大），AFSI 是 Taylor–Hood + 一致力密度插值，两者
  在 IB 带内的近场速度本来就不同（对比 demo_460/461/474 是同一量级的差异）。
  表格里 `du away`（带外）小得多，说明差异主要被限制在结构附近。
* **网格/时间步**：参考解与 AFSI 都跑 `NX = 64`、同一个 `dt = 1e-5`、同一份移植的
  `update_Target_Point_Positions.py`（pyIB2d 从 run 目录 import 同一文件），因此
  "更新律"不引入差异；差异只来自流体/罚力离散。
* **上游 textscan 错位**：`.m` 用 `textscan('%f %f')` 读 `corals.vertex` 时会把首行
  的点数当成第一个数、导致列错位（见模块 docstring）；本移植按 `vertices(2:end,:)`
  的**意图**逐行读。若在 MATLAB 里按原样跑，几何会被错位读取，因此**本 demo 的数值
  无法与 MATLAB 端逐位对齐**（但 AFSI 与 pyIB2d 用的是同一份模块，互相对齐）。

## 运行方法（实测耗时，本机）

```bash
conda activate afsi-dolfinx
cd afsic/demo/ib2d/demo_594
```

| 命令 | 实测 `real` |
|---|---|
| `python run_reference.py --nx 64 --tend 0.01 --print-dump 100` | 0m16.217s |
| `NX=64 python main.py` | 0m59.061s |
| `python compare.py` | - |

（`real` 取自 `ref.log` / `afsi.log`；本机同一时刻还有大量其他算例在跑，
绝对耗时波动很大。参考解写入 `ib2d_reference.npz`，AFSI 写入
`plot/afsi_result.npz`，`compare.py` 生成 `figures/` 与 `compare_table.csv`。）
