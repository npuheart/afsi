# demo_601 — IB2d example "Beating Heart" (matIB2d

``Examples/Examples_Education/Interpolation/Beating_Heart``) in AFSI.

771 markers in a 1 x 1 periodic box (mu = 30, rho = 1000, 64 x 64, dt = 2e-5):
770 linear springs (k = 1e3) that give the heart its elasticity plus **771
target points** (k = 2e10) whose anchors interpolate between the two recorded
heart shapes ``State_A.pts`` / ``State_B.pts``.

The example's own ``update_Target_Point_Positions.m`` (ported line-by-line to
``ib2d_input/update_Target_Point_Positions.py`` and imported by both codes)
beats with a 0.22 s cycle (0.1 s contraction, 0.02 s rest, 0.1 s expansion),

    X = A + g(t/t1) (B - A)              (contraction),
    X = B + g((t-t1-tR)/t2) (A - B)      (expansion),

where ``g`` is the piecewise cubic of the Moving_Circle education example
(knots 0.25 / 0.925).  During the 0.02 s rest window the .m re-targets nothing,
so the anchors hold their last values.

Note: ``port_example.py`` missed ``State_A.pts``/``State_B.pts`` (they are not
matched by the ``<string_name>.*`` rule), so they were copied by hand -- the
demo is self-contained.

Cost: the shipped ``Tfinal = 0.22 s`` (11000 steps) is reduced to ``T = 0.01 s``
(500 steps, half a contraction) at the shipped 64 x 64 grid, with
``print_dump = 50`` (10 frames) instead of 125.

## 结果（`python compare.py` 的完整输出）

```
        t   dXmax/h            max|u| a/r    du away    du band      relL2    df away
  0.00000  4.34e-10  0.000e+00/0.000e+00   0.000e+00  0.000e+00  0.000e+00  0.000e+00
  0.00102  2.38e-05  3.411e-03/1.604e-02   1.123e-02  1.708e-02  8.549e-01  3.075e-14
  0.00202  9.11e-05  2.073e-02/8.659e-02   5.344e-02  9.227e-02  8.116e-01  1.065e-13
  0.00302  2.10e-04  5.921e-02/2.135e-01   1.241e-01  2.168e-01  7.700e-01  2.263e-13
  0.00402  3.91e-04  1.237e-01/3.982e-01   2.205e-01  3.868e-01  7.321e-01  3.923e-13
  0.00502  6.47e-04  2.178e-01/6.443e-01   3.423e-01  6.034e-01  7.003e-01  6.084e-13
  0.00602  9.74e-04  3.437e-01/9.532e-01   4.895e-01  8.652e-01  6.737e-01  8.754e-13
  0.00702  1.39e-03  5.031e-01/1.326e+00   6.616e-01  1.172e+00  6.512e-01  1.192e-12
  0.00802  1.91e-03  6.970e-01/1.763e+00   8.577e-01  1.522e+00  6.320e-01  1.556e-12
  0.00902  2.52e-03  9.261e-01/2.267e+00   1.078e+00  1.916e+00  6.157e-01  1.964e-12
  0.01002  3.24e-03  1.191e+00/2.838e+00   1.321e+00  2.353e+00  6.016e-01  2.409e-12
markers          : max|dX| 5.060e-05 m = 3.24e-03 h (771 Lagrangian points)
velocity         : max|u| AFSI 1.1906e+00 / IB2d 2.8381e+00 (5.80e-01 relative); max|du| 1.321e+00 = 4.65e-01 max|u| away from the IB band (3550/4096 nodes), 2.353e+00 (0.83 max|u|) inside it; relative L2 6.016e-01
force field      : max|df| 2.409e-12 away from the band, |f|max 3.710e+06
springs          : mean length AFSI 0.001697 / IB2d 0.001697 (1.20e-05 relative)
wrote /Users/pengfei.ma/afsi/afsic/demo/ib2d/demo_601/figures/compare_table.csv and the figures
```

这张表就是 `compare.py` 的原始输出：每帧的 `dXmax/h` 是标记点偏差（以网格步长 h
为单位），`max|u| a/r` 是 AFSI/IB2d 的速度峰值，`du away` / `du band` 是 IB 带外/
带内的速度差，`relL2` 是速度场相对 $L_2$，`df away` 是带外外力场差（有被动标量时
还有 `max|dC|` 与 `relL2(C)`）。它与 `figures/compare_table.csv` 一致。

## 不可比 / 需要打折看的地方

* **规模缩减**：上游 `Tfinal = 0.22 s`（11000 步）在 `64 x 64`；本 demo 用
  `T = 0.01 s`（500 步 = 半个收缩期），`dt = 2e-5`、`64 x 64` 与上游一致，
  `print_dump = 50`（10 帧）代替 125。
* **好的部分**：标记点 `3.2e-3 h`，并且**弹簧平均长度** AFSI/IB2d 相对差
  `1.2e-5`（`0.001697` vs `0.001697`）—— 说明 770 根弹簧 + 771 个 `k = 2e10` 的
  强靶点这一组合在两个代码里给出的结构状态几乎相同，插值律（`State_A/B.pts`
  的三次分段插值）被正确复现。
* **速度场要打折**：相对 $L_2$ `0.60`、IB 带内 `max|du| = 2.35` m/s。上游用
  FFT 投影 + 共置罚力（刚度 2e10），AFSI 用 Taylor–Hood + 一致力密度插值；
  在这种"靠极硬靶点拖住结构"的算例里，近场速度本来就对离散方式很敏感。
* **上游缺文件**：`port_example.py` 不会拷贝 `State_A.pts` / `State_B.pts`
  （它们不匹配 `<string_name>.*`），本 demo 已手工拷入，并且把 `run_reference.py`
  的辅助文件后缀扩到 `.pts`，所以 demo 自洽、可单独重跑。
* **静止窗口**：`.m` 在 `t1 < t < t1+tR`（0.02 s）没有分支，锚点保持上一个被写入的
  值 —— 移植保持原样。

## 运行方法（实测耗时，本机）

```bash
conda activate afsi-dolfinx
cd afsic/demo/ib2d/demo_601
```

| 命令 | 实测 `real` |
|---|---|
| `python run_reference.py --tend 0.01 --print-dump 50` | 0m30.605s |
| `python main.py` | 0m56.260s |
| `python compare.py` | - |

（`real` 取自 `ref.log` / `afsi.log`；本机同一时刻还有大量其他算例在跑，
绝对耗时波动很大。参考解写入 `ib2d_reference.npz`，AFSI 写入
`plot/afsi_result.npz`，`compare.py` 生成 `figures/` 与 `compare_table.csv`。）
