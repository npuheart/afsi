# demo_607 — IB2d example "Swimmer" (matIB2d

``Examples/Examples_Education/Interpolation/Swimmer``) in AFSI.

440 markers in an 8 x 4 periodic box (mu = 10, rho = 1000, 512 x 256 as shipped,
dt = 2.5e-5): 183 springs (k = 9.6e9), 182 **non-invariant beams**
(kb = 2.0e12) and 256 pinned target points (k = 1e6) -- the education variant
adds "head" markers to the First-Year-Seminar swimmer (demo_606).

The example's own ``update_nonInv_Beams.m`` (ported line-by-line to
``ib2d_input/update_nonInv_Beams.py``) interpolates the two phases of
``swimmer.phases`` with a *piecewise cubic* (knots 0.10 / 0.90, down-stroke and
up-stroke of 0.25 s, no rest) and writes the second difference of the
interpolated **y** centreline as the beams' reference y-curvature; the x lines
and the ``Cx`` assignment are commented out in the .m, so column 5 keeps its
file value.

The same pyIB2d driver bug as in demo_604 applies (``beams_info`` -- the scalar
0, since ``beams = 0`` -- is passed to ``update_nonInv_Beams``), so
``run_reference.py`` patches that call.

Cost: 512 x 256 with ``Tfinal = 3 s`` is out of reach, so the demo runs a
**reduced** ``NX = 64`` (64 x 32) and ``T = 0.01 s`` (400 steps) with
``print_dump = 40`` (10 frames) instead of 800.

## 结果（`python compare.py` 的完整输出）

```
        t   dXmax/h            max|u| a/r    du away    du band      relL2    df away
  0.00000  5.35e-10  0.000e+00/0.000e+00   0.000e+00  0.000e+00  0.000e+00  0.000e+00
  0.00103  6.47e-06  9.303e-03/6.148e-03   3.554e-03  3.787e-05  5.823e-01  2.594e+03
  0.00202  1.78e-05  1.215e-02/1.038e-02   4.759e-03  7.734e-05  4.841e-01  6.178e+03
  0.00303  2.47e-05  1.387e-02/1.182e-02   5.488e-03  1.188e-04  4.703e-01  7.035e+03
  0.00402  2.88e-05  1.309e-02/1.254e-02   6.600e-03  1.575e-04  5.109e-01  7.597e+03
  0.00502  3.12e-05  1.245e-02/1.147e-02   8.905e-03  1.870e-04  5.409e-01  1.107e+04
  0.00602  3.80e-05  1.424e-02/1.147e-02   7.385e-03  2.040e-04  6.433e-01  8.275e+03
  0.00702  4.78e-05  1.894e-02/1.139e-02   9.577e-03  2.158e-04  7.519e-01  6.457e+03
  0.00803  5.09e-05  1.939e-02/1.199e-02   1.138e-02  2.425e-04  8.075e-01  9.598e+03
  0.00903  5.64e-05  1.611e-02/1.386e-02   1.214e-02  2.997e-04  8.335e-01  1.014e+04
markers          : max|dX| 7.056e-06 m = 5.64e-05 h (440 Lagrangian points)
velocity         : max|u| AFSI 1.6106e-02 / IB2d 1.3860e-02 (1.62e-01 relative); max|du| 1.214e-02 = 8.76e-01 max|u| away from the IB band (1952/2048 nodes), 2.997e-04 (0.02 max|u|) inside it; relative L2 8.335e-01
force field      : max|df| 1.014e+04 away from the band, |f|max 8.977e+03
springs          : mean length AFSI 0.007812 / IB2d 0.007813 (5.13e-06 relative)
wrote /Users/pengfei.ma/afsi/afsic/demo/ib2d/demo_607/figures/compare_table.csv and the figures
```

这张表就是 `compare.py` 的原始输出：每帧的 `dXmax/h` 是标记点偏差（以网格步长 h
为单位），`max|u| a/r` 是 AFSI/IB2d 的速度峰值，`du away` / `du band` 是 IB 带外/
带内的速度差，`relL2` 是速度场相对 $L_2$，`df away` 是带外外力场差（有被动标量时
还有 `max|dC|` 与 `relL2(C)`）。它与 `figures/compare_table.csv` 一致。

## 不可比 / 需要打折看的地方

* **规模缩减**：上游 `512 x 256`、`Tfinal = 3 s`；本 demo 用 `NX = 64`
  （`64 x 32`）和 `T = 0.01 s`（400 步），`dt = 2.5e-5`、`print_dump = 40`。
* **与 demo_606 是同一族的两个不同实现**：本算例（Education/Interpolation）用
  **分段三次**（knots 0.10 / 0.90）只插值 **y**，并且 `.m` 把 x 曲率那一行注释掉
  了（`Cx` 保持文件里的初值）；First-Year-Seminar 版本（demo_606）用的是
  a/b/c 三个多项式族且同时改 `Cx`、`Cy`。两者不要混淆。
* **pyIB2d 驱动 bug**：同 demo_604/606（`beams = 0` ⇒ 标量 0）；补丁已包含。
* **要打折的地方**：与 demo_606 相同 —— 8 x 4 的盒子里 440 个点（含 256 个钉住的
  "头"靶点），缩到 64x32 后 h = 0.125 m，结构本身的厚度小于一个网格，速度/推力
  不可信；判据是曲率律与 `max|dX|/h`。

## 运行方法（实测耗时，本机）

```bash
conda activate afsi-dolfinx
cd afsic/demo/ib2d/demo_607
```

| 命令 | 实测 `real` |
|---|---|
| `python run_reference.py --nx 64 --tend 0.01 --print-dump 40` | 4m3.388s |
| `NX=64 python main.py` | 1m39.584s |
| `python compare.py` | - |

（`real` 取自 `ref.log` / `afsi.log`；本机同一时刻还有大量其他算例在跑，
绝对耗时波动很大。参考解写入 `ib2d_reference.npz`，AFSI 写入
`plot/afsi_result.npz`，`compare.py` 生成 `figures/` 与 `compare_table.csv`。）
