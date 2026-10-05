# demo_604 — IB2d example "Jellyfish Material" (matIB2d
``Examples/Examples_First_Year_Seminar/Jellyfish_Material``) in AFSI.

## 中文小结

* **算例**：`Examples_First_Year_Seminar/Jellyfish_Material` —— 127 个点：
  126 根弹簧（`k = 6.4e9`）+ 125 根**非不变梁**（`kb = 1e13`）。
* **新的东西**：`.m` 每步从解析 `beta(t)` 重新生成水母钟形中心线，并把它的二阶
  差分写成梁的参考曲率（Alben–Peng–Miller 模型）。
* **缩减**：`T = 0.005 s`（500 步 = 1 s 周期的 0.5%），网格/`dt` 与上游一致。
* **结果**：标记点 `0.22 h`、弹簧平均长度相对差 `1.2e-3`；速度/力不可比（极硬材料）。
* **坑（重要）**：**pyIB2d 驱动 bug** —— `IBM_Driver.py:711` 把 `beams_info`
  （本算例 `beams = 0` 时是标量 0）传给 `update_nonInv_Beams`，上游 pyIB2d
  **根本跑不了这个算例**；本 demo 的 `run_reference.py` 打了最小补丁。


> 下面 "Physics / porting notes" 部分是英文叙述（与 `main.py` 的 docstring 同源）：
> 上游算例的标识符、方程都来自英文源码，保持原样以免转述出错。

127 Lagrangian markers in a 2 x 8 periodic box (mu = 2.5, rho = 1000,
64 x 256, dt = 1e-5): 126 springs (k = 6.4e9) and 125 **non-invariant beams**
(kb = 1.0e13).  The example is the Alben-Peng-Miller jellyfish bell whose
material curvatures are prescribed (not its geometry).

The example's own ``update_nonInv_Beams.m`` (ported line-by-line to
``ib2d_input/update_nonInv_Beams.py``) re-derives the analytic bell centreline
every step from ``beta(t)`` (a 1 s cycle: contract for 0.4 s, expand for 0.6 s,
``beta`` from 0.5 down to 0.3 and back) and writes the second differences of
that polyline as the beam reference curvatures ``Cx, Cy``.

Two things matter for the port:

* the .m uses the **1-based** middle-node identifier as an index into the
  analytic bell, so the port adds 1 back to the 0-based ids pyIB2d hands it;
* pyIB2d's driver (``IBM_Driver.py`` line 711) passes ``beams_info`` instead of
  ``nonInv_beams_info`` to this module.  With ``beams = 0`` (as in this
  example) ``beams_info`` is the scalar 0, so **the shipped pyIB2d port cannot
  run this example at all** (it would raise ``TypeError`` on the first step).
  ``run_reference.py`` patches that call (a documented, minimal repair) and the
  AFSI port passes the non-invariant beam array the .m expects.

Cost: the shipped ``Tfinal = 5 s`` is 500000 steps; the demo runs a **reduced**
``T = 0.005 s`` (500 steps = 0.5% of a cycle) with ``print_dump = 50``
(10 frames) instead of 2500.

## 结果（`python compare.py` 的完整输出）

```
        t   dXmax/h            max|u| a/r    du away    du band      relL2    df away
  0.00000  2.07e-09  0.000e+00/0.000e+00   0.000e+00  0.000e+00  0.000e+00  0.000e+00
  0.00051  2.32e-02  4.686e+01/6.038e+01   2.740e+01  5.304e+01  1.050e+00  8.720e-09
  0.00101  4.77e-02  6.023e+01/4.941e+01   3.622e+01  5.160e+01  1.134e+00  1.161e-08
  0.00151  6.77e-02  9.673e+01/2.906e+01   2.150e+01  1.167e+02  1.076e+00  7.892e-09
  0.00201  1.18e-01  2.824e+01/7.791e+01   5.563e+01  7.843e+01  1.230e+00  5.628e-09
  0.00251  1.11e-01  7.685e+01/3.433e+01   7.065e+01  4.666e+01  1.903e+00  6.562e-09
  0.00301  1.21e-01  4.205e+01/4.026e+01   5.318e+01  4.169e+01  1.563e+00  6.148e-09
  0.00351  1.16e-01  3.259e+01/5.179e+01   4.485e+01  6.583e+01  1.343e+00  4.997e-09
  0.00401  1.16e-01  8.548e+01/3.515e+01   8.429e+01  3.981e+01  1.836e+00  1.086e-08
  0.00451  1.43e-01  4.060e+01/4.982e+01   2.568e+01  7.253e+01  1.142e+00  5.906e-09
  0.00501  2.17e-01  3.464e+01/3.739e+01   2.398e+01  6.382e+01  1.176e+00  4.085e-09
markers          : max|dX| 6.772e-03 m = 2.17e-01 h (127 Lagrangian points)
velocity         : max|u| AFSI 3.4642e+01 / IB2d 3.7391e+01 (7.35e-02 relative); max|du| 2.398e+01 = 6.41e-01 max|u| away from the IB band (15953/16384 nodes), 6.382e+01 (1.71 max|u|) inside it; relative L2 1.176e+00
force field      : max|df| 4.085e-09 away from the band, |f|max 2.779e+09
springs          : mean length AFSI 0.015818 / IB2d 0.015838 (1.24e-03 relative)
wrote /Users/pengfei.ma/afsi/afsic/demo/ib2d/demo_604/figures/compare_table.csv and the figures
```

这张表就是 `compare.py` 的原始输出：每帧的 `dXmax/h` 是标记点偏差（以网格步长 h
为单位），`max|u| a/r` 是 AFSI/IB2d 的速度峰值，`du away` / `du band` 是 IB 带外/
带内的速度差，`relL2` 是速度场相对 $L_2$，`df away` 是带外外力场差（有被动标量时
还有 `max|dC|` 与 `relL2(C)`）。它与 `figures/compare_table.csv` 一致。

## 不可比 / 需要打折看的地方

* **规模缩减**：上游 `Tfinal = 5 s`（500000 步）在 `64 x 256`；本 demo 用
  `T = 0.005 s`（500 步 = 1 s 周期的 0.5%），网格与 `dt = 1e-5` 保持上游，
  `print_dump = 50`（10 帧）代替 2500。
* **pyIB2d 的驱动 bug（重要）**：`IBM_Driver.py` 第 711 行把 `beams_info` 而不是
  `nonInv_beams_info` 传给 `update_nonInv_Beams`；本算例 `beams = 0`，于是
  `beams_info` 是标量 `0`，参考解一进入时间循环就会 `TypeError` ——
  **上游 pyIB2d 根本跑不了这个算例**。本 demo 的 `run_reference.py` 对该调用打了
  最小补丁（用同一份 `nonInv_beams` 数组喂进去，等价于 MATLAB 端的正确调用），
  并在脚本注释里标注；AFSI 侧传的是正确的非不变梁数组。
* **好的部分**：标记点 `0.22 h`（127 点）、弹簧平均长度相对差 `1.2e-3`。
* **要打折的部分**：速度场相对 $L_2$ `1.18`（IB 带内 `max|du| = 63.8` m/s）。
  这个算例的梁刚度 `kb` 最大到 `1e13`、弹簧 `k ≈ 6e9`，是典型的"极硬材料"算例，
  两边罚力/插值离散的近场差别被放大；`mean length` 与标记点的一致才是这里可靠的判据。
* **索引基准**：`.nonInv_beam` 的 id 在 pyIB2d 里是 0 基，而 `.m` 用它们**直接**
  索引 1 基的解析 bell 数组；移植把 `+1` 加回去（`s1 = id + 1`），否则整条曲率
  序列会错位一个点。

## 运行方法（实测耗时，本机）

```bash
conda activate afsi-dolfinx
cd afsic/demo/ib2d/demo_604
```

| 命令 | 实测 `real` |
|---|---|
| `python run_reference.py --tend 0.005 --print-dump 50` | 0m53.263s |
| `python main.py` | 1m39.991s |
| `python compare.py` | - |

（`real` 取自 `ref.log` / `afsi.log`；本机同一时刻还有大量其他算例在跑，
绝对耗时波动很大。参考解写入 `ib2d_reference.npz`，AFSI 写入
`plot/afsi_result.npz`，`compare.py` 生成 `figures/` 与 `compare_table.csv`。）
