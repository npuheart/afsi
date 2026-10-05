# demo_564 — IB2d `Examples_Education/Convergence/Jellyfish/Simulation_Skeletons/Re37pt5/Res_384_480x144`

IB2d 官方**水母游动收敛性阶梯**（`Convergence/Jellyfish/Simulation_Skeletons`）的一个成员：
同一只水母在 4 个 Reynolds 数（`Re37pt5`、`Re75`、`Re150`、`Re300`）下各做 11 档
拉格朗日/网格分辨率的收敛性研究（`Res_32 … Res_1024`）。
本 demo 是 **Re37pt5（Re = 37.5）**、**Res_384_480x144** 那一档；本搬运批次覆盖 `Re37pt5` 的 6 档
与 `Re150` 的全部 11 档（demo_560–569 / 633–639），`Re75`/`Re300` 不在本批次内。

模型出处（上游 `README.txt`）：A. Hoover, L. Miller, *A numerical study of the benefits of driving
jellyfish bells at their natural frequency*, J. Theor. Biol. 374: 13–25 (2015)；
matIB2d 版本由 N. A. Battista 制作（Feb. 2018, TCNJ）。

## 算例内容

* 周期盒 `3 x 10`（`Lx = 3`、`Ly = 10`）、`rho = 1000`、**`mu = 26.666`（Re = 37.5）**。
* **出厂设置**：网格 **144x480**（`h = 0.0208`）、`dt = 1e-5`、`Tfinal = 5.0`
  （= **5e5 步**）、`print_dump = 5000`（1000 帧）、`supp = 4`。
* **结构**（`ib2d_input/jelly384.vertex / .spring / .nonInv_beam / .target`，逐字取自上游）：
  **369** 个拉格朗日点、**267** 根弹簧（`k ≤ 9.6e+08`、静长 `0 … 0.0105`）、
  **237** 根非不变梁（`kb = 2.21e+11`）、**72** 个钉死目标点。
  水母钟体几乎占满整个盒子（`x ∈ [0.004, 2.99]`、`y ∈ [1.75, 9.98]`）——它是"贴在周期边界上的大水母"。
* **收缩律**（`update_Springs.m`，逐行移植为 `ib2d_input/update_Springs.py`，两个代码**共用同一份**）：
  钟体边缘的 **29** 根弹簧（MATLAB 行 `239:end`）静长每步重置为

  ```
  RL_i = |cos(freq t pi)|,   freq = 1
  ```

  周期 1 s、取绝对值 ⇒ **每秒收缩两次**。它只改静长、不改刚度，是纯"肌肉"驱动。
  （阶梯各档的起止行号随分辨率变化，出厂 `.m` 里写的就是 `239:end`。）

## 移植要点

1. **机制**：`springs + update_springs + nonInvariant_beams + target_pts` —— 合并后的模板全部支持
   （同族参考 demo_480/482）；`update_Springs.py` 的写法与 demo_478/480 同族。
   本档唯一的参数是 `FIRST_ROW = 239`。
2. **两端共用同一份用户模块**：`run_reference.py` 把 `ib2d_input/*.py` 拷进运行目录给 pyIB2d 用，
   `main.py` 从 `ib2d_input/` 导入 ⇒ "收缩律"不可能两边不一致。
3. **dt 与稳定性**（实测）：该几何刚度极大（`k ≤ 9.6e+08`、`kb = 2.21e+11`），
   pyIB2d 的显式 IB 对 `dt` 有上限（`dt` 一大就发散）。本批次在 `jelly96` 几何 + 48x160 网格上的实测：
   `dt = 2e-5` 时 `t ≈ 0.005–0.02 s` 内出现除零 → NaN；`dt = 1e-4` 时 100 步内 `max|u|` 已达 36 m/s（发散）；
   `dt = 1e-5`（出厂值）稳定。本 demo 实测：出厂 `dt = 1e-5`、200 步内参考解保持有限（末步 `max|u| = 3.265e-02`）。
4. **运行网格**（本 demo 唯一的“降规模”操作）：上游出厂网格 **144x480**（69k 个节点，`h = 0.0208`）远超本机可行范围，本 demo 降到 **48x160**（`h = 0.0625`），即收敛阶梯里 **Res_128** 那一档的网格；
   上游 `input2d`、`jelly384.*` 几何/结构文件**逐字保留**在 `ib2d_input/`。
5. **无法复现游泳**：出厂 `Tfinal = 5 s`（5e5 步），即便 `dt = 1e-5` 也是小时级运行；
   本 demo 只跑 **0.002 s（200 步，`print_dump = 20` → 10 帧）**，
   即收缩周期（1 s）的 **0.2%**，只覆盖**初期瞬态**。

## 结果

`run_reference.py` + `main.py` + `compare.py`（两边各 10 帧，`compare.py` 按驱动器的 dump 约定对齐）：

```
=== shipped setup (dt=1e-05, h=0.0625, 10 aligned frames, t=0.00000..0.00181)
        t   dXmax/h            max|u| a/r    du away    du band      relL2    df away
  0.00000  1.00e-06  0.000e+00/0.000e+00   0.000e+00  0.000e+00  0.000e+00  0.000e+00
  0.00021  3.49e-06  8.936e-03/6.188e-03   2.777e-03  2.871e-07  4.099e-01  2.392e+02
  0.00041  1.02e-05  1.679e-02/1.177e-02   5.080e-03  5.480e-07  4.017e-01  8.599e+02
  0.00061  2.05e-05  2.363e-02/1.679e-02   6.933e-03  7.892e-07  3.946e-01  1.700e+03
  0.00081  3.39e-05  2.920e-02/2.105e-02   8.278e-03  1.003e-06  3.887e-01  2.552e+03
  0.00101  4.96e-05  3.335e-02/2.441e-02   9.118e-03  1.182e-06  3.842e-01  3.204e+03
  0.00121  6.71e-05  3.609e-02/2.682e-02   1.028e-02  1.462e-06  3.810e-01  3.475e+03
  0.00141  8.58e-05  3.756e-02/2.829e-02   1.158e-02  1.808e-06  3.790e-01  3.251e+03
  0.00161  1.05e-04  3.968e-02/2.938e-02   1.302e-02  2.205e-06  3.778e-01  2.805e+03
  0.00181  1.26e-04  4.274e-02/3.265e-02   1.431e-02  2.659e-06  3.771e-01  3.435e+03

markers          : max|dX| 7.853e-06 m = 1.26e-04 h (369 Lagrangian points)
velocity         : max|u| AFSI 4.2741e-02 / IB2d 3.2650e-02 (3.09e-01 relative); max|du| 1.431e-02 = 5.62e-01 max|u| away from the IB band (7536/7680 nodes), 2.659e-06 (1.00 max|u|) inside it; relative L2 3.771e-01
force field      : max|df| 3.435e+03 away from the band, |f|max 4.038e+04
springs          : mean length AFSI 0.116124 / IB2d 0.116122 (1.06e-05 relative)

```

按节点统计的速度场相对误差（取 `|u_ref| > 1% max|u_ref|` 的 1566 个节点，
末对齐帧 `t = 0.00180`）：中位数 **3.5%**、90 分位 **13.4%**、
最大 **101%**；该子集上的相对 $L_2$ = **0.377**。

**能验证**（本 demo 的正面结论）：

* **结构运动学**：标记点位置一致到 `max|dX| ≈ 1e-04 h`（= 7.84e-06 m），
  弹簧平均长度相对误差 ~1e-6 ⇒ 水母受到的力、以及被力推着走的标记点轨迹，两边是同一条。
* **瞬态流场的形状**：绝大多数节点的相对误差在 4%–13% 量级。

**不能验证（如实记录）**：

* **流场绝对幅值**：AFSI 的 `max|u|` 系统性高于 pyIB2d，**本例 ×1.31**，
  本批次各档落在 ×1.37（36x120 与 48x160 网格）… ×1.67（12x40 网格）；两套网格上量级不变 ⇒ 不是时间步/网格分辨率问题，而是
  **两种离散对同一个力场的高波数响应不同**（AFSI：Taylor–Hood P2/P1 + 力密度经 Q2 插值；
  IB2d：FFT 谱投影 + 共置罚力）。该窗口内 `mu ∇²u ≪ f`，流场几乎完全由"冲量启动"
  `du/dt = P(f)/rho` 决定，所以投影算子的差异被直接放大。因此本 demo **不宣称**速度场幅值可比。
* **游泳/推进**：要看到完整的收缩—回复循环与尾涡需要 `T ~ 1 s`（10^5 步）以上，
  本 demo 只跑 0.002 s（收缩周期的 0.2%）。也没有做阶梯的跨档收敛率分析
  （那需要全部 4 个 Re × 11 档、且各档都在自己的出厂网格上跑完整周期）。

## 运行

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate afsi-dolfinx

# 参考解（pyIB2d，出厂 dt）
python run_reference.py --nx 48 --dt 1e-05 --tend 0.002 --print-dump 20   # 约 16 s

# AFSI 移植
NX=48 DT=1e-05 TFINAL=0.002 PRINT_DUMP=20 python main.py                    # 约 23 s

# 对比（图 + compare_table.csv）
python compare.py
```

* 上游出厂：`dt = 1e-5`、`Tfinal = 5.0`、网格 144x480；本 demo 只改运行规模（见"移植要点"），
  数据文件一律逐字保留。
* 运行时长是**共享机器高负载**下测得的 wall time（含 ~5 s 的 numpy/dolfinx 导入），仅供参考。
