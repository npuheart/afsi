# demo_566 — IB2d `Examples_Education/Convergence/Jellyfish/Simulation_Skeletons/Re150/Res_192_240x72`

IB2d 官方**水母游动收敛性阶梯**（`Convergence/Jellyfish/Simulation_Skeletons`）的一个成员：
同一只水母在 4 个 Reynolds 数（`Re37pt5`、`Re75`、`Re150`、`Re300`）下各做 11 档
拉格朗日/网格分辨率的收敛性研究（`Res_32 … Res_1024`）。
本 demo 是 **Re150（Re = 150）**、**Res_192_240x72** 那一档；本搬运批次覆盖 `Re37pt5` 的 6 档
与 `Re150` 的全部 11 档（demo_560–569 / 633–639），`Re75`/`Re300` 不在本批次内。

模型出处（上游 `README.txt`）：A. Hoover, L. Miller, *A numerical study of the benefits of driving
jellyfish bells at their natural frequency*, J. Theor. Biol. 374: 13–25 (2015)；
matIB2d 版本由 N. A. Battista 制作（Feb. 2018, TCNJ）。

## 算例内容

* 周期盒 `3 x 10`（`Lx = 3`、`Ly = 10`）、`rho = 1000`、**`mu = 6.667`（Re = 150）**。
* **出厂设置**：网格 **72x240**（`h = 0.0417`）、`dt = 1e-5`、`Tfinal = 5.0`
  （= **5e5 步**）、`print_dump = 5000`（1000 帧）、`supp = 4`。
* **结构**（`ib2d_input/jelly192.vertex / .spring / .nonInv_beam / .target`，逐字取自上游）：
  **183** 个拉格朗日点、**132** 根弹簧（`k ≤ 4.8e+08`、静长 `0 … 0.021`）、
  **117** 根非不变梁（`kb = 2.76e+10`）、**36** 个钉死目标点。
  水母钟体几乎占满整个盒子（`x ∈ [0.004, 2.99]`、`y ∈ [1.75, 9.98]`）——它是"贴在周期边界上的大水母"。
* **收缩律**（`update_Springs.m`，逐行移植为 `ib2d_input/update_Springs.py`，两个代码**共用同一份**）：
  钟体边缘的 **14** 根弹簧（MATLAB 行 `119:end`）静长每步重置为

  ```
  RL_i = |cos(freq t pi)|,   freq = 1
  ```

  周期 1 s、取绝对值 ⇒ **每秒收缩两次**。它只改静长、不改刚度，是纯"肌肉"驱动。
  （阶梯各档的起止行号随分辨率变化，出厂 `.m` 里写的就是 `119:end`。）

## 移植要点

1. **机制**：`springs + update_springs + nonInvariant_beams + target_pts` —— 合并后的模板全部支持
   （同族参考 demo_480/482）；`update_Springs.py` 的写法与 demo_478/480 同族。
   本档唯一的参数是 `FIRST_ROW = 119`。
2. **两端共用同一份用户模块**：`run_reference.py` 把 `ib2d_input/*.py` 拷进运行目录给 pyIB2d 用，
   `main.py` 从 `ib2d_input/` 导入 ⇒ "收缩律"不可能两边不一致。
3. **dt 与稳定性**（实测）：该几何刚度极大（`k ≤ 4.8e+08`、`kb = 2.76e+10`），
   pyIB2d 的显式 IB 对 `dt` 有上限（`dt` 一大就发散）。本批次在 `jelly96` 几何 + 48x160 网格上的实测：
   `dt = 2e-5` 时 `t ≈ 0.005–0.02 s` 内出现除零 → NaN；`dt = 1e-4` 时 100 步内 `max|u|` 已达 36 m/s（发散）；
   `dt = 1e-5`（出厂值）稳定。本 demo 实测：出厂 `dt = 1e-5`、200 步内参考解保持有限（末步 `max|u| = 1.636e-02`）。
4. **运行网格**（本 demo 唯一的“降规模”操作）：上游出厂网格 **72x240**（17k 个节点，`h = 0.0417`）远超本机可行范围，本 demo 降到 **48x160**（`h = 0.0625`），即收敛阶梯里 **Res_128** 那一档的网格；
   上游 `input2d`、`jelly192.*` 几何/结构文件**逐字保留**在 `ib2d_input/`。
5. **无法复现游泳**：出厂 `Tfinal = 5 s`（5e5 步），即便 `dt = 1e-5` 也是小时级运行；
   本 demo 只跑 **0.002 s（200 步，`print_dump = 20` → 10 帧）**，
   即收缩周期（1 s）的 **0.2%**，只覆盖**初期瞬态**。

## 结果

`run_reference.py` + `main.py` + `compare.py`（两边各 10 帧，`compare.py` 按驱动器的 dump 约定对齐）：

```
=== shipped setup (dt=1e-05, h=0.0625, 10 aligned frames, t=0.00000..0.00181)
        t   dXmax/h            max|u| a/r    du away    du band      relL2    df away
  0.00000  1.00e-06  0.000e+00/0.000e+00   0.000e+00  0.000e+00  0.000e+00  0.000e+00
  0.00021  2.22e-06  4.447e-03/3.058e-03   1.399e-03  1.413e-07  4.160e-01  1.011e+02
  0.00041  5.60e-06  8.460e-03/5.845e-03   2.637e-03  2.741e-07  4.146e-01  3.749e+02
  0.00061  1.10e-05  1.209e-02/8.399e-03   3.724e-03  4.040e-07  4.139e-01  7.700e+02
  0.00081  1.82e-05  1.520e-02/1.063e-02   4.618e-03  5.292e-07  4.141e-01  1.214e+03
  0.00101  2.70e-05  1.770e-02/1.248e-02   5.301e-03  6.478e-07  4.153e-01  1.623e+03
  0.00121  3.72e-05  1.957e-02/1.390e-02   5.909e-03  7.572e-07  4.174e-01  1.917e+03
  0.00141  4.85e-05  2.082e-02/1.489e-02   6.738e-03  8.544e-07  4.204e-01  2.031e+03
  0.00161  6.07e-05  2.152e-02/1.548e-02   7.493e-03  9.360e-07  4.240e-01  1.926e+03
  0.00181  7.38e-05  2.347e-02/1.636e-02   8.161e-03  1.035e-06  4.279e-01  1.593e+03

markers          : max|dX| 4.610e-06 m = 7.38e-05 h (183 Lagrangian points)
velocity         : max|u| AFSI 2.3466e-02 / IB2d 1.6359e-02 (4.34e-01 relative); max|du| 8.161e-03 = 5.01e-01 max|u| away from the IB band (7536/7680 nodes), 1.035e-06 (1.00 max|u|) inside it; relative L2 4.279e-01
force field      : max|df| 1.593e+03 away from the band, |f|max 2.037e+04
springs          : mean length AFSI 0.123004 / IB2d 0.123004 (5.55e-06 relative)

```

按节点统计的速度场相对误差（取 `|u_ref| > 1% max|u_ref|` 的 1529 个节点，
末对齐帧 `t = 0.00180`）：中位数 **3.7%**、90 分位 **15.0%**、
最大 **126%**；该子集上的相对 $L_2$ = **0.428**。

**能验证**（本 demo 的正面结论）：

* **结构运动学**：标记点位置一致到 `max|dX| ≈ 7e-05 h`（= 4.60e-06 m），
  弹簧平均长度相对误差 ~1e-6 ⇒ 水母受到的力、以及被力推着走的标记点轨迹，两边是同一条。
* **瞬态流场的形状**：绝大多数节点的相对误差在 4%–15% 量级。

**不能验证（如实记录）**：

* **流场绝对幅值**：AFSI 的 `max|u|` 系统性高于 pyIB2d，**本例 ×1.43**，
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
python run_reference.py --nx 48 --dt 1e-05 --tend 0.002 --print-dump 20   # 约 30 s

# AFSI 移植
NX=48 DT=1e-05 TFINAL=0.002 PRINT_DUMP=20 python main.py                    # 约 35 s

# 对比（图 + compare_table.csv）
python compare.py
```

* 上游出厂：`dt = 1e-5`、`Tfinal = 5.0`、网格 72x240；本 demo 只改运行规模（见"移植要点"），
  数据文件一律逐字保留。
* 运行时长是**共享机器高负载**下测得的 wall time（含 ~5 s 的 numpy/dolfinx 导入），仅供参考。
