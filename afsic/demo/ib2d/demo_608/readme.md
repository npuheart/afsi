# demo_608 — IB2d example "Jellyfish Update Beam Curvature" (matIB2d

``Examples/Example_Jellyfish_Swimming/Failed_Attempts/Jellyfish_Update_Beam_Curavture``)
in AFSI.

83 markers in a 1 x 1 periodic box (mu = 0.1, rho = 1, 128 x 128, dt = 5e-4):
82 springs (k = 1e3) and 81 **invariant beams** (kb = 1e10).  The upstream
directory is one of the ``Failed_Attempts`` of the jellyfish family: instead of
prescribing the bell geometry, it drives the beams' reference **curvatures**
directly (no target points at all).

The example's own ``update_Beams.m`` (ported line-by-line to
``ib2d_input/update_Beams.py``) hardcodes a two-state curvature table (81 rows,
one per beam: contracted and expanded) and linearly interpolates it over a
0.1 s cycle (contraction over the first 0.3 period, expansion over the rest).
The .m wraps the assignment in a pointless ``for i = 1:length(...)`` loop that
rewrites the whole column each iteration; the port performs the vectorised
assignment once (documented in the module docstring).

Cost: the shipped ``Tfinal = 1 s`` (2000 steps) is run as a **reduced**
``T = 0.25 s`` (500 steps = 2.5 cycles) at the shipped 128 x 128 grid, with
``print_dump = 50`` (10 frames) instead of 20.

## 结果（`python compare.py` 的完整输出）

```
        t   dXmax/h            max|u| a/r    du away    du band      relL2    df away
  0.00000  8.56e-10  0.000e+00/0.000e+00   0.000e+00  0.000e+00  0.000e+00  0.000e+00
  0.02550  1.98e-02  6.601e-01/6.535e-01   9.907e-03  1.738e-02  1.582e-02  2.219e-14
  0.05050  4.89e-02  4.489e-01/4.361e-01   2.457e-02  2.486e-02  5.375e-02  3.018e+00
  0.07550  5.67e-02  4.151e-01/4.080e-01   7.912e-03  7.555e-03  2.574e-02  1.944e+00
  0.10050  3.41e-02  3.249e-01/3.112e-01   1.376e-02  8.859e-03  4.291e-02  7.071e-01
  0.12550  4.11e-02  6.096e-01/6.062e-01   9.521e-03  1.633e-02  1.499e-02  4.517e-02
  0.15050  6.56e-02  4.641e-01/4.555e-01   2.330e-02  2.245e-02  4.934e-02  9.685e+00
  0.17550  6.83e-02  4.231e-01/4.152e-01   8.442e-03  7.795e-03  2.356e-02  3.481e+00
  0.20050  4.69e-02  3.386e-01/3.261e-01   1.279e-02  8.582e-03  3.784e-02  2.197e+01
  0.22550  5.39e-02  6.041e-01/6.024e-01   1.246e-02  1.605e-02  1.908e-02  9.137e-01
markers          : max|dX| 4.214e-04 m = 5.39e-02 h (83 Lagrangian points)
velocity         : max|u| AFSI 6.0405e-01 / IB2d 6.0243e-01 (2.69e-03 relative); max|du| 1.246e-02 = 2.07e-02 max|u| away from the IB band (16105/16384 nodes), 1.605e-02 (0.03 max|u|) inside it; relative L2 1.908e-02
force field      : max|df| 9.137e-01 away from the band, |f|max 9.246e+02
springs          : mean length AFSI 0.003886 / IB2d 0.003885 (2.27e-04 relative)
wrote /Users/pengfei.ma/afsi/afsic/demo/ib2d/demo_608/figures/compare_table.csv and the figures
```

这张表就是 `compare.py` 的原始输出：每帧的 `dXmax/h` 是标记点偏差（以网格步长 h
为单位），`max|u| a/r` 是 AFSI/IB2d 的速度峰值，`du away` / `du band` 是 IB 带外/
带内的速度差，`relL2` 是速度场相对 $L_2$，`df away` 是带外外力场差（有被动标量时
还有 `max|dC|` 与 `relL2(C)`）。它与 `figures/compare_table.csv` 一致。

## 不可比 / 需要打折看的地方

* **规模缩减**：上游 `Tfinal = 1 s`（2000 步）在 `128 x 128`；本 demo 用
  `T = 0.25 s`（500 步 = 2.5 个 0.1 s 周期），网格、`dt = 5e-4` 与上游一致，
  `print_dump = 50`（10 帧）代替 20（上游会写 101 帧，npz 会很大）。
* **上游是"Failed_Attempts"**：这个算例没有靶点，直接给 81 根**不变梁**指定参考
  曲率（`.beam` 的 `C` 列），也就是"把钟体的形状直接以曲率形式喂进去"。曲率表是
  `.m` 里硬编码的 81 x 2 个数字（收缩/扩张两个状态），本移植把这 81 行逐字搬进
  `update_Beams.py`（脚本解析，避免手抄）。
* **上游无意义循环**：`.m` 把整列的赋值包在 `for i = 1:length(beams_info(:,5))`
  里，循环体每次都重写整列 —— 移植去掉这个空转循环（等效，已在模块 docstring
  说明）。
* **要打折的地方**：`kb = 1e10` 与 `mu = 0.1` 是极硬的组合，`dt = 5e-4` 下
  `kb*dt/mu = 5e4`，两边近场速度都会"振铃"；判据是曲率律本身（两边同一份模块）
  与 `max|dX|/h`、结构整体形变。

## 运行方法（实测耗时，本机）

```bash
conda activate afsi-dolfinx
cd afsic/demo/ib2d/demo_608
```

| 命令 | 实测 `real` |
|---|---|
| `python run_reference.py --tend 0.25 --print-dump 50` | 3m36.287s |
| `python main.py` | 2m37.301s |
| `python compare.py` | - |

（`real` 取自 `ref.log` / `afsi.log`；本机同一时刻还有大量其他算例在跑，
绝对耗时波动很大。参考解写入 `ib2d_reference.npz`，AFSI 写入
`plot/afsi_result.npz`，`compare.py` 生成 `figures/` 与 `compare_table.csv`。）
