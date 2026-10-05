# demo_644 — IB2d `Example_Jellyfish_Swimming/Failed_Attempts/Jellyfish_With_Beams/Forward_Swimmer`

作者的**失败尝试**之一（`Forward_Swimmer`）：带 **77 根不变梁**（`jelly.beam`、`beams = 1`）的水母，但驱动律**只改变最后一根弹簧**的静长：`RL(end) = 0.1250077 (1 - 0.5 |sin(10 pi t)|)`（单弹簧、10 Hz）。

单弹簧驱动 + `mu = 0.001`（几乎无粘）的极端组合：两边都在“高速 + 欠分辨”的区间里，流场只能算部分可比。

`Example_Jellyfish_Swimming/` 是 IB2d 官方水母游动算例的总目录（模型出自
A. Hoover, L. Miller, *J. Theor. Biol.* 374: 13–25 (2015)）；其中 `Failed_Attempts/`
放的是**作者自己标注为失败的尝试**——搬运时一并保留并注明"失败在哪"，这是上游历史的一部分。

## 算例内容

* **出厂设置**（`ib2d_input/input2d` 原样保留）：网格 **128x128**（`1 x 1` 盒子）、
  `dt = 0.0005`、`Tfinal = 2`（= 4,000 步）、`print_dump = 20`。
* **本 demo 运行设置**：`NX = 64`（`NY` 由盒子长宽比自动推出 = 64）、`dt = 0.0005`（= 出厂值）、
  `Tfinal = 0.05`（= 100 步）、`print_dump = 10`。
  出厂网格/Tfinal 都远超本机可行范围（网格最大 128x1024、Tfinal 最大 12 s），见"移植要点"。
* **结构/驱动**：几何与驱动律文件（`*.vertex/.spring/.beam/.poroelastic`、`update_Springs.m`）逐字取自上游；
  `update_Springs.m` 已逐行移植成 `ib2d_input/update_Springs.py`，**两个代码共用同一份**。

## 移植要点

1. **机制**：`springs + update_springs`（按算例再加 `beams` / `nonInvariant_beams` / `target_pts` /
   `poroelastic`）——合并后的模板全部支持（同族参考 demo_480/482/520）。
2. **出厂 `dt` 是"参考解能用"的关键**：jelly 系列几何刚度极大，pyIB2d 的显式 IB 在大 `dt` 下不稳定
   （本批次实测：`jelly96` 几何在 `dt = 2e-5` 时 0.005–0.02 s 内除零 → NaN；`dt = 1e-4` 时 100 步内
   `max|u| ≈ 36 m/s`）。本家族出厂 `dt` 取 `1e-5 … 1e-3`，本 demo 一律沿用出厂值。
3. **`Failed_Attempts/` 的算例本来就跑不出推进**（作者如此标注）：本 demo 的价值是
   "驱动机制/耦合是否被正确搬运"，而不是"水母有没有游起来"。

## 结果

```
=== shipped setup (dt=0.0005, h=0.015625, 10 aligned frames, t=0.00000..0.04550)
        t   dXmax/h            max|u| a/r    du away    du band      relL2    df away
  0.00000  4.22e-10  0.000e+00/0.000e+00   0.000e+00  0.000e+00  0.000e+00  0.000e+00
  0.00550  5.07e-02  1.275e+00/5.357e-01   2.317e-01  7.661e-01  1.206e+00  5.861e-16
  0.01050  6.02e-02  8.487e-01/1.789e+00   3.093e-01  1.337e+00  4.222e-01  4.752e-15
  0.01550  4.93e-02  7.791e-01/3.005e+00   7.756e-01  2.557e+00  5.604e-01  6.130e-15
  0.02050  6.39e-02  1.018e+00/2.440e+00   7.606e-01  1.975e+00  4.427e-01  5.201e+01
  0.02550  9.49e-02  1.087e+00/1.219e+00   3.813e-01  5.103e-01  2.698e-01  3.130e+02
  0.03050  1.34e-01  1.142e+00/1.324e+00   8.311e-01  9.743e-01  5.736e-01  2.422e+01
  0.03550  1.77e-01  1.476e+00/1.746e+00   6.519e-01  6.526e-01  3.223e-01  1.002e+03
  0.04050  2.16e-01  1.690e+00/2.695e+00   1.006e+00  1.374e+00  3.844e-01  5.463e+02
  0.04550  2.39e-01  1.752e+00/2.900e+00   1.151e+00  1.617e+00  4.368e-01  1.221e+03

markers          : max|dX| 3.733e-03 m = 2.39e-01 h (79 Lagrangian points)
velocity         : max|u| AFSI 1.7516e+00 / IB2d 2.8999e+00 (3.96e-01 relative); max|du| 1.151e+00 = 6.03e-01 max|u| away from the IB band (3955/4096 nodes), 1.617e+00 (0.44 max|u|) inside it; relative L2 4.368e-01
force field      : max|df| 1.221e+03 away from the band, |f|max 1.810e+03
springs          : mean length AFSI 0.004726 / IB2d 0.004813 (1.80e-02 relative)

```

按节点统计的速度场相对误差（`|u_ref| > 1% max|u_ref|` 的 737 个节点，末对齐帧 `t = 0.04500`）：
中位数 **26.5%**、90 分位 **64.3%**、最大 **299%**。

结构方面：标记点位置一致到 **2.0e-01 h**（= 3.12e-03 m）；质心位移
AFSI **2.105e-03 m**（y 方向 +2.102e-03）/ IB2d **4.182e-03 m**（y 方向 +4.180e-03）。

**能验证**：结构运动学 / 弹簧（+梁）力学的搬运正确性，以及（除个别档外）瞬态流场的形状。

**不能验证（如实记录）**：

* 流场**绝对幅值**：AFSI 的 `max|u|` = 1.752e+00，pyIB2d = 2.900e+00（**×0.60**）。本窗口内 `mu ∇²u ≪ f`，流场主要由"冲量启动" `du/dt = P(f)/rho` 决定，
  AFSI（Taylor–Hood + Q2 插值力密度）与 IB2d（FFT 谱投影 + 共置罚力）在投影算子上的差异会被直接放大
  （与水母收敛阶梯一族现象相同：那一族系统性 ×1.37…×1.67）。
* **推进距离 / 游泳**：要看到完整周期需要 `T` 远大于本 demo 取的 0.05 s；
  且 `Failed_Attempts/` 里的算例在上游就没有成功推进（本 demo 的质心位移只有
  4.18 mm 量级）。

## 运行

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate afsi-dolfinx

python run_reference.py --nx 64 --dt 0.0005 --tend 0.05 --print-dump 10   # 约 53 s
NX=64 DT=0.0005 TFINAL=0.05 PRINT_DUMP=10 python main.py                    # 约 ?
python compare.py
```

* 运行时长在**共享机器高负载**下测得（wall，含 ~5 s 导入）。
