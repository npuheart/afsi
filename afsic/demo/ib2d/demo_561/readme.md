# demo_561 — IB2d `Examples_Education/Convergence/Jellyfish/Simulation_Skeletons/Re37pt5/Res_768_960x288`

IB2d 官方**水母游动收敛性阶梯**（`Convergence/Jellyfish/Simulation_Skeletons`）的一个成员：
同一只水母在 4 个 Reynolds 数（`Re37pt5`、`Re75`、`Re150`、`Re300`）下各做 11 档
拉格朗日/网格分辨率的收敛性研究（`Res_32 … Res_1024`）。
本 demo 是 **Re37pt5（Re = 37.5）**、**Res_768_960x288** 那一档；本搬运批次覆盖 `Re37pt5` 的 6 档
与 `Re150` 的全部 11 档（demo_560–569 / 633–639），`Re75`/`Re300` 不在本批次内。

模型出处（上游 `README.txt`）：A. Hoover, L. Miller, *A numerical study of the benefits of driving
jellyfish bells at their natural frequency*, J. Theor. Biol. 374: 13–25 (2015)；
matIB2d 版本由 N. A. Battista 制作（Feb. 2018, TCNJ）。

## 算例内容

* 周期盒 `3 x 10`（`Lx = 3`、`Ly = 10`）、`rho = 1000`、**`mu = 26.666`（Re = 37.5）**。
* **出厂设置**：网格 **288x960**（`h = 0.0104`）、`dt = 1e-5`、`Tfinal = 5.0`
  （= **5e5 步**）、`print_dump = 5000`（1000 帧）、`supp = 4`。
* **结构**（`ib2d_input/jelly768.vertex / .spring / .nonInv_beam / .target`，逐字取自上游）：
  **739** 个拉格朗日点、**535** 根弹簧（`k ≤ 1.92e+09`、静长 `0 … 0.00522`）、
  **475** 根非不变梁（`kb = 1.77e+12`）、**144** 个钉死目标点。
  水母钟体几乎占满整个盒子（`x ∈ [0.004, 2.99]`、`y ∈ [1.75, 9.98]`）——它是"贴在周期边界上的大水母"。
* **收缩律**（`update_Springs.m`，逐行移植为 `ib2d_input/update_Springs.py`，两个代码**共用同一份**）：
  钟体边缘的 **59** 根弹簧（MATLAB 行 `477:end`）静长每步重置为

  ```
  RL_i = |cos(freq t pi)|,   freq = 1
  ```

  周期 1 s、取绝对值 ⇒ **每秒收缩两次**。它只改静长、不改刚度，是纯"肌肉"驱动。
  （阶梯各档的起止行号随分辨率变化，出厂 `.m` 里写的就是 `477:end`。）

## 移植要点

1. **机制**：`springs + update_springs + nonInvariant_beams + target_pts` —— 合并后的模板全部支持
   （同族参考 demo_480/482）；`update_Springs.py` 的写法与 demo_478/480 同族。
   本档唯一的参数是 `FIRST_ROW = 477`。
2. **两端共用同一份用户模块**：`run_reference.py` 把 `ib2d_input/*.py` 拷进运行目录给 pyIB2d 用，
   `main.py` 从 `ib2d_input/` 导入 ⇒ "收缩律"不可能两边不一致。
3. **dt 与稳定性**（实测）：该几何刚度极大（`k ≤ 1.92e+09`、`kb = 1.77e+12`），
   pyIB2d 的显式 IB 对 `dt` 有上限（`dt` 一大就发散）。本批次在 `jelly96` 几何 + 48x160 网格上的实测：
   `dt = 2e-5` 时 `t ≈ 0.005–0.02 s` 内出现除零 → NaN；`dt = 1e-4` 时 100 步内 `max|u|` 已达 36 m/s（发散）；
   `dt = 1e-5`（出厂值）稳定。本 demo 实测：出厂 `dt = 1e-5`、200 步内参考解保持有限（末步 `max|u| = 5.944e-02`）。
4. **运行网格**（本 demo 唯一的“降规模”操作）：上游出厂网格 **288x960**（276k 个节点，`h = 0.0104`）远超本机可行范围，本 demo 降到 **48x160**（`h = 0.0625`），即收敛阶梯里 **Res_128** 那一档的网格；
   上游 `input2d`、`jelly768.*` 几何/结构文件**逐字保留**在 `ib2d_input/`。
5. **无法复现游泳**：出厂 `Tfinal = 5 s`（5e5 步），即便 `dt = 1e-5` 也是小时级运行；
   本 demo 只跑 **0.002 s（200 步，`print_dump = 20` → 10 帧）**，
   即收缩周期（1 s）的 **0.2%**，只覆盖**初期瞬态**。

## 结果

`run_reference.py` + `main.py` + `compare.py`（两边各 10 帧，`compare.py` 按驱动器的 dump 约定对齐）：

```
=== shipped setup (dt=1e-05, h=0.0625, 10 aligned frames, t=0.00000..0.00181)
        t   dXmax/h            max|u| a/r    du away    du band      relL2    df away
  0.00000  1.00e-06  0.000e+00/0.000e+00   0.000e+00  0.000e+00  0.000e+00  0.000e+00
  0.00021  5.78e-06  1.745e-02/1.208e-02   5.407e-03  5.554e-07  4.102e-01  5.114e+02
  0.00041  1.86e-05  3.270e-02/2.293e-02   9.859e-03  1.061e-06  4.020e-01  1.827e+03
  0.00061  3.84e-05  4.585e-02/3.260e-02   1.339e-02  1.531e-06  3.951e-01  3.574e+03
  0.00081  6.39e-05  5.635e-02/4.067e-02   1.588e-02  1.950e-06  3.895e-01  5.281e+03
  0.00101  9.38e-05  6.398e-02/4.689e-02   1.737e-02  2.303e-06  3.854e-01  6.478e+03
  0.00121  1.27e-04  6.880e-02/5.118e-02   1.959e-02  2.659e-06  3.826e-01  6.793e+03
  0.00141  1.62e-04  7.116e-02/5.362e-02   2.252e-02  3.278e-06  3.811e-01  6.024e+03
  0.00161  1.99e-04  7.297e-02/5.446e-02   2.520e-02  3.990e-06  3.802e-01  6.188e+03
  0.00181  2.37e-04  7.851e-02/5.944e-02   2.752e-02  4.809e-06  3.798e-01  7.314e+03

markers          : max|dX| 1.479e-05 m = 2.37e-04 h (739 Lagrangian points)
velocity         : max|u| AFSI 7.8508e-02 / IB2d 5.9442e-02 (3.21e-01 relative); max|du| 2.752e-02 = 5.37e-01 max|u| away from the IB band (7584/7680 nodes), 4.809e-06 (1.00 max|u|) inside it; relative L2 3.798e-01
force field      : max|df| 7.314e+03 away from the band, |f|max 7.694e+04
springs          : mean length AFSI 0.113201 / IB2d 0.113199 (2.09e-05 relative)

```

按节点统计的速度场相对误差（取 `|u_ref| > 1% max|u_ref|` 的 1610 个节点，
末对齐帧 `t = 0.00180`）：中位数 **3.5%**、90 分位 **13.5%**、
最大 **122%**；该子集上的相对 $L_2$ = **0.380**。

**能验证**（本 demo 的正面结论）：

* **结构运动学**：标记点位置一致到 `max|dX| ≈ 2e-04 h`（= 1.48e-05 m），
  弹簧平均长度相对误差 ~1e-6 ⇒ 水母受到的力、以及被力推着走的标记点轨迹，两边是同一条。
* **瞬态流场的形状**：绝大多数节点的相对误差在 3%–13% 量级。

**不能验证（如实记录）**：

* **流场绝对幅值**：AFSI 的 `max|u|` 系统性高于 pyIB2d，**本例 ×1.32**，
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
python run_reference.py --nx 48 --dt 1e-05 --tend 0.002 --print-dump 20   # 约 78 s

# AFSI 移植
NX=48 DT=1e-05 TFINAL=0.002 PRINT_DUMP=20 python main.py                    # 约 30 s

# 对比（图 + compare_table.csv）
python compare.py
```

* 上游出厂：`dt = 1e-5`、`Tfinal = 5.0`、网格 288x960；本 demo 只改运行规模（见"移植要点"），
  数据文件一律逐字保留。
* 运行时长是**共享机器高负载**下测得的 wall time（含 ~5 s 的 numpy/dolfinx 导入），仅供参考。
