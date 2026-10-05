# demo_642 — IB2d `Example_Jellyfish_Swimming/Failed_Attempts/Jellyfish_Spring_RL_Interpolation`

作者的**失败尝试**之一：`Lx = Ly = 1` 的小方盒里，把所有静长在两个“相位”之间**线性插值**（`RL = RL1 + 5 t (RL2 - RL1)`，相位差 0.2 s）——想用“直接改静长”来驱动水母。

**上游自己的不一致（本次实测）**：`jelly.spring` 里写的静长与插值表的“相位 1”并不相等（尾部 41 根：`max|file - RL1| = 0.124`；前 106 根：`max|file - ds1| = 0.146`），所以第 1 步就会把静长整体跳变一次，第一个帧 `max|u|` 立刻冲到 14 m/s——这大概正是它“失败”的原因之一。另外该几何的 `.spring` 只有 4 列（没有 `alpha`），两个代码都按线性弹簧（`alpha = 1`）处理。

`Example_Jellyfish_Swimming/` 是 IB2d 官方水母游动算例的总目录（模型出自
A. Hoover, L. Miller, *J. Theor. Biol.* 374: 13–25 (2015)）；其中 `Failed_Attempts/`
放的是**作者自己标注为失败的尝试**——搬运时一并保留并注明"失败在哪"，这是上游历史的一部分。

## 算例内容

* **出厂设置**（`ib2d_input/input2d` 原样保留）：网格 **128x128**（`1 x 1` 盒子）、
  `dt = 0.00025`、`Tfinal = 0.2`（= 800 步）、`print_dump = 50`。
* **本 demo 运行设置**：`NX = 64`（`NY` 由盒子长宽比自动推出 = 64）、`dt = 0.00025`（= 出厂值）、
  `Tfinal = 0.05`（= 200 步）、`print_dump = 20`。
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
=== shipped setup (dt=0.00025, h=0.015625, 10 aligned frames, t=0.00000..0.04525)
        t   dXmax/h            max|u| a/r    du away    du band      relL2    df away
  0.00000  4.28e-10  0.000e+00/0.000e+00   0.000e+00  0.000e+00  0.000e+00  0.000e+00
  0.00525  4.95e-01  6.924e+00/1.391e+01   6.318e+00  1.675e+01  9.701e-01  4.773e+02
  0.01025  4.83e-01  1.779e+00/2.168e+00   6.883e-01  1.543e+00  3.544e-01  4.514e+02
  0.01525  3.47e-01  2.179e+00/2.173e+00   3.755e-01  8.788e-01  2.070e-01  4.669e+02
  0.02025  2.60e-01  2.278e+00/2.494e+00   5.273e-01  7.154e-01  1.878e-01  4.532e+02
  0.02525  2.82e-01  2.095e+00/2.229e+00   4.292e-01  4.188e-01  1.713e-01  3.465e+02
  0.03025  2.86e-01  1.842e+00/1.934e+00   3.301e-01  3.812e-01  1.529e-01  2.987e+02
  0.03525  3.05e-01  1.509e+00/1.567e+00   2.463e-01  2.664e-01  1.768e-01  3.615e+02
  0.04025  3.32e-01  1.265e+00/1.199e+00   3.453e-01  2.521e-01  2.306e-01  6.524e+02
  0.04525  3.54e-01  1.077e+00/1.012e+00   3.886e-01  2.717e-01  3.000e-01  7.622e+02

markers          : max|dX| 5.526e-03 m = 3.54e-01 h (83 Lagrangian points)
velocity         : max|u| AFSI 1.0766e+00 / IB2d 1.0124e+00 (6.34e-02 relative); max|du| 3.886e-01 = 6.16e-01 max|u| away from the IB band (3944/4096 nodes), 2.717e-01 (0.73 max|u|) inside it; relative L2 3.000e-01
force field      : max|df| 7.622e+02 away from the band, |f|max 3.676e+03
springs          : mean length AFSI 0.034489 / IB2d 0.035497 (2.84e-02 relative)

```

按节点统计的速度场相对误差（`|u_ref| > 1% max|u_ref|` 的 3931 个节点，末对齐帧 `t = 0.04500`）：
中位数 **30.0%**、90 分位 **40.1%**、最大 **795%**。

结构方面：标记点位置一致到 **3.4e-01 h**（= 5.30e-03 m）；质心位移
AFSI **1.069e-03 m**（y 方向 +1.069e-03）/ IB2d **1.128e-03 m**（y 方向 +1.128e-03）。

**能验证**：结构运动学 / 弹簧（+梁）力学的搬运正确性，以及（除个别档外）瞬态流场的形状。

**不能验证（如实记录）**：

* 流场**绝对幅值**：AFSI 的 `max|u|` = 1.077e+00，pyIB2d = 1.012e+00（**×1.06**）。本窗口内 `mu ∇²u ≪ f`，流场主要由"冲量启动" `du/dt = P(f)/rho` 决定，
  AFSI（Taylor–Hood + Q2 插值力密度）与 IB2d（FFT 谱投影 + 共置罚力）在投影算子上的差异会被直接放大
  （与水母收敛阶梯一族现象相同：那一族系统性 ×1.37…×1.67）。
* **推进距离 / 游泳**：要看到完整周期需要 `T` 远大于本 demo 取的 0.05 s；
  且 `Failed_Attempts/` 里的算例在上游就没有成功推进（本 demo 的质心位移只有
  1.13 mm 量级）。

## 运行

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate afsi-dolfinx

python run_reference.py --nx 64 --dt 0.00025 --tend 0.05 --print-dump 20   # 约 250 s
NX=64 DT=0.00025 TFINAL=0.05 PRINT_DUMP=20 python main.py                    # 约 130 s
python compare.py
```

* 运行时长在**共享机器高负载**下测得（wall，含 ~5 s 导入）。
