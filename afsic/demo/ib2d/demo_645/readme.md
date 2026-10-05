# demo_645 — IB2d `Example_Jellyfish_Swimming/Failed_Attempts/Jellyfish_With_Beams/Backwards_Swimmer_for_LOLz`

作者的**失败尝试**之一（文件夹名就叫 `Backwards_Swimmer_for_LOLz`）：带 **105 根不变梁**，但驱动律**只改最后一根弹簧**：`RL(end) = 0.08649829 |cos(10 pi t)|`（20 Hz）。

`dt = 1e-3`（本批次最大的时间步）下两个代码仍稳定，且弹簧平均长度一致到 3.7e-4 相对误差。

`Example_Jellyfish_Swimming/` 是 IB2d 官方水母游动算例的总目录（模型出自
A. Hoover, L. Miller, *J. Theor. Biol.* 374: 13–25 (2015)）；其中 `Failed_Attempts/`
放的是**作者自己标注为失败的尝试**——搬运时一并保留并注明"失败在哪"，这是上游历史的一部分。

## 算例内容

* **出厂设置**（`ib2d_input/input2d` 原样保留）：网格 **128x128**（`1 x 1` 盒子）、
  `dt = 0.001`、`Tfinal = 2`（= 2,000 步）、`print_dump = 10`。
* **本 demo 运行设置**：`NX = 64`（`NY` 由盒子长宽比自动推出 = 64）、`dt = 0.001`（= 出厂值）、
  `Tfinal = 0.05`（= 50 步）、`print_dump = 5`。
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
=== shipped setup (dt=0.001, h=0.015625, 10 aligned frames, t=0.00000..0.04600)
        t   dXmax/h            max|u| a/r    du away    du band      relL2    df away
  0.00000  4.22e-10  0.000e+00/0.000e+00   0.000e+00  0.000e+00  0.000e+00  0.000e+00
  0.00600  5.58e-04  6.934e-02/7.753e-02   4.015e-03  8.200e-03  4.294e-02  2.738e-15
  0.01100  3.03e-03  2.246e-01/2.449e-01   1.228e-02  2.040e-02  3.458e-02  1.124e-14
  0.01600  7.30e-03  4.114e-01/4.404e-01   2.329e-02  3.483e-02  3.456e-02  3.223e-14
  0.02100  1.30e-02  5.965e-01/6.285e-01   3.815e-02  5.424e-02  3.621e-02  8.136e-14
  0.02600  2.00e-02  7.571e-01/7.844e-01   5.891e-02  7.677e-02  3.750e-02  1.833e-13
  0.03100  2.89e-02  9.562e-01/9.978e-01   8.669e-02  1.007e-01  3.835e-02  3.711e-13
  0.03600  3.98e-02  1.233e+00/1.316e+00   1.153e-01  1.208e-01  3.920e-02  9.727e+00
  0.04100  5.28e-02  1.518e+00/1.640e+00   1.349e-01  1.297e-01  3.958e-02  3.129e+01
  0.04600  6.89e-02  1.844e+00/1.958e+00   1.951e-01  1.353e-01  3.887e-02  1.474e+02

markers          : max|dX| 1.077e-03 m = 6.89e-02 h (107 Lagrangian points)
velocity         : max|u| AFSI 1.8445e+00 / IB2d 1.9583e+00 (5.81e-02 relative); max|du| 1.951e-01 = 9.00e-01 max|u| away from the IB band (3926/4096 nodes), 1.353e-01 (0.93 max|u|) inside it; relative L2 3.887e-02
force field      : max|df| 1.474e+02 away from the band, |f|max 1.497e+04
springs          : mean length AFSI 0.004681 / IB2d 0.004682 (3.67e-04 relative)

```

按节点统计的速度场相对误差（`|u_ref| > 1% max|u_ref|` 的 4049 个节点，末对齐帧 `t = 0.04500`）：
中位数 **3.4%**、90 分位 **7.3%**、最大 **70%**。

结构方面：标记点位置一致到 **6.4e-02 h**（= 9.94e-04 m）；质心位移
AFSI **2.057e-03 m**（y 方向 +2.057e-03）/ IB2d **1.581e-03 m**（y 方向 +1.581e-03）。

**能验证**：结构运动学 / 弹簧（+梁）力学的搬运正确性，以及（除个别档外）瞬态流场的形状。

**不能验证（如实记录）**：

* 流场**绝对幅值**：AFSI 的 `max|u|` = 1.844e+00，pyIB2d = 1.958e+00（**×0.94**）。本窗口内 `mu ∇²u ≪ f`，流场主要由"冲量启动" `du/dt = P(f)/rho` 决定，
  AFSI（Taylor–Hood + Q2 插值力密度）与 IB2d（FFT 谱投影 + 共置罚力）在投影算子上的差异会被直接放大
  （与水母收敛阶梯一族现象相同：那一族系统性 ×1.37…×1.67）。
* **推进距离 / 游泳**：要看到完整周期需要 `T` 远大于本 demo 取的 0.05 s；
  且 `Failed_Attempts/` 里的算例在上游就没有成功推进（本 demo 的质心位移只有
  1.58 mm 量级）。

## 运行

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate afsi-dolfinx

python run_reference.py --nx 64 --dt 0.001 --tend 0.05 --print-dump 5   # 约 34 s
NX=64 DT=0.001 TFINAL=0.05 PRINT_DUMP=5 python main.py                    # 约 18 s
python compare.py
```

* 运行时长在**共享机器高负载**下测得（wall，含 ~5 s 导入）。
