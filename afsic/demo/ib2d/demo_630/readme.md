# demo_630 — IB2d `Example_Coagulation/Example_Testing_Thru_Boundaries/Works_on_64x16`

上游算例路径：`matIB2d/Examples/Example_Coagulation/Example_Testing_Thru_Boundaries/Works_on_64x16`。

## 算例内容

* 结构：**3 个细胞（每个 34 个拉格朗日点）** + 活塞（14 个靶点，`k=1e7`，`update_target=1`）；
  116 个拉格朗日点、166 根弹簧、102 根不变梁。
* 流体：周期盒 `1 x 0.25`、`mu = 0.01`、`rho = 1`、64 x 16、
  `dt = 5e-5`，`supp = 4`。
* `coagulation = 0（上游）；本 demo 用 COAGULATION=1 打开`。

* 这是这一族里**最便宜**的算例（116 个标记点、64x16 网格），所以
  它同时承担**机制演示**和**稳定性探测**两个角色：上游 `coagulation = 0`，
  本 demo 用 `COAGULATION=1` 打开；细胞 0/2 初始中心距 0.0583 < 阈值 0.075，
  第一步成键。
* `mu = 0.01`（其余算例是 0.05）、`dt = 5e-5`。
* 这个算例在**上游自己的设置下**（`MAX_SPEED=15`）两个实现都会在
  `t ≈ 0.011`（参考）附近发散——目录名叫 "Works_on_64x16" 但它在 pyIB2d 里并不
  work，见下文"稳定性"。

## 机制（IB2d `coagulation` 模型）

上游没有把它做成"浓度/反应项"——**它和被动标量（`concentration`）完全无关**。
`coagulation` 是一个**拉格朗日颗粒聚集模型**：多个"细胞"（圆形结构）在流场中被
搅动，当两个细胞**中心**距离小于阈值时，在二者**最近的拉格朗日点**之间生成一根
"键"（bond）弹簧；键在受力超过断裂阈值时断开。

`<name>.coagulation` 文件（一列数字，行号见下）：

| 行 | 含义 |
|---|---|
| 1 | `start_id`：细胞块在 `.vertex` 中的起始拉格朗日编号（本算例为 1，即**细胞排在 .vertex 最前面**） |
| 2 | `threshold_radius`：成键半径（两细胞**中心**距离） |
| 3 | `bond_strength`：键弹簧刚度 `k` |
| 4 | `fracture_force`：断裂阈值 |
| 5.. | 每个细胞的拉格朗日点数（细胞依次连续存放） |

每个时间步（在**半步位置** `X^{n+1/2}` 上，和弹簧/梁/靶点力同一次装配、同一次
`delta` 展布）：

1. 细胞中心 = 该细胞第 `lag1` 个点与**直径对面**第 `lag1 + n/2` 个点的中点
   （`please_Find_Cell_Centers`）；
2. 对所有**尚未成键**的细胞对，若中心距 `r < threshold_radius`，新增一根键
   （`please_Find_Distances_Between_Cells`）；
3. 键连接的点 = 细胞 *i* 上离细胞 *j* 中心最近的点，与细胞 *j* 上离细胞 *i* 中心
   最近的点；键的**静长** `L_r` = 成键瞬间这两点的距离
   （`please_Find_Closest_Lag_Pts_To_Cell_Centers`）；
4. 键是线性的（`alpha = 1`）弹簧力密度，`dx/dy` 走 IB2d 的最小镜像折叠：

   ```
   sF = 0.5 (alpha+1) k (L - L_r)^alpha * d_hat = k (L - L_r) d_hat
   ```

   作用在 master 点上为 `+sF`，slave 点上为 `-sF`；
5. **上一时间步之前**生成的键，若 `sF_x^2 + sF_y^2 >= fracture_force` 则删除
   （`please_Remove_Bonds`）；本步新成的键当步不检查。

AFSI 侧的实现放在 `ib2d_input/coagulation_model.py`，在 `main.py` 的
`spread_forces_at_half()` 里加到 `Fm`（和弹簧/梁/靶点力一起乘 `ds` 后一次性展布）；
pyIB2d 侧没有这个能力，所以 `run_reference.py` 用一个 wrapper 把同一份
`coagulation_model` 接到 pyIB2d 的
`please_Find_Lagrangian_Forces_On_Eulerian_grid`（MATLAB 驱动也是在这一步把
`fx_coag` 加进 `fx` 再展布）——**展布用 pyIB2d 自己的
`give_Me_Delta_Function_Approximations_For_Force_Calc`**。

## 本移植的取舍（与上游的差异）

* **AFSI 侧每步把标记点坐标折回 `[0, L)`**（模板沿用的做法；pyIB2d 与 MATLAB 都
  **不折**——MATLAB 的 `mod(xLag,Lx)` 只在 `porous_Yes` 分支里，pyIB2d 连那里都
  只对 porous 点做）。对**所有周期核**（delta 展布、速度插值、最小镜像弹簧/梁/靶点）
  两者完全等价；只有**细胞中心**（不做最小镜像的裸坐标）在细胞跨过周期边界时会不同。
  本 demo 对比的时间窗内没有标记点越过 `x = 1`（`max|x| ≈ 0.9996`），因此两种约定
  给出相同结果；这也是"跨周期边界成键"这个上游缺陷在数据里暂时没触发的部分。
* **`ib2d_io._read_rows` 读结构文件的全部行**（不再用首行计数截断），以匹配
  MATLAB `textscan` 和 pyIB2d `np.loadtxt(skiprows=1)` 的行为；有两个算例的
  `.beam` 首行计数是过期的（见各 demo 说明）。
* **活塞行程速度 `MAX_SPEED` 可用环境变量覆盖**（默认 15 = 上游值）。本族算例
  在上游值时会在很短时间内发散（见"稳定性"），需要减小时才用这个开关；本批 6 个
  demo **都用了上游值 15**。

## 上游的坑 / bug（本移植中遇到并处理）

1. **MATLAB 的 `give_Me_Coagulation_Force_Densities.m` 是残缺的**：文件在
   `please_Remove_Bonds` 中间就结束了（最后一行是
   `N = length(aggregate_list);`，没有 `return`、没有右括号）。也就是说
   **MATLAB 一旦发生断键就会报错**，而本例的断裂阈值很小（起效阈值
   `|f| >= sqrt(fracture_force)`），断键一定会发生。本移植按"删除断键所在行"的
   显然意图把它补全。
2. **pyIB2d 根本没有 coagulation 支持**：`IBM_Driver.py` 从不 import
   `give_Me_Coagulation_Force_Densities.py`，那份模块是死代码，而且**多处写错**：
   `please_Find_Cell_Centers` 用浮点下标 `lag1_ind + 0.5*n` 索引 numpy 数组
   （必然 `IndexError`）；最近点搜索读 `connects[i,1]`、`connects[i,2]`（该数组只有
   2 列，越界）；"是否已连键"的判断写成 `if np.where(checkVec == j):`
   （非空 tuple 恒为真）——于是**第一根键之后再也连不上新键**；
   `aggregate_list != (0,)` 作用在 `int` 占位符上；`please_Remove_Bonds` 用
   `N/5`（浮点）reshape，返回值名字 `remove_bonds` 与 `remove_Bonds` 不一致。
3. **断裂判据量纲不一致**：MATLAB/pyIBd 比较的是 `sF_x^2 + sF_y^2`（力的平方）与
   `fracture_force`。本移植**原样复现**这个上游行为（`fracture=10` 实际是
   `|f| >= sqrt(10) ≈ 3.16`）。
4. **中心距/最近点搜索不做周期最小镜像**：只有键**弹簧**的 `dx/dy` 折叠；细胞
   中心、中心距、最近点搜索都是裸欧氏距离。因此**跨周期边界的细胞中心会算错、
   跨界成键永远不发生**——而 `Example_Testing_Thru_Boundaries` 这一族名字正是
   在测这件事。
5. **`update_Target_Point_Positions.m`（活塞/搅拌）**：`period=0.05`、`rest=0.01`、
   `max_Speed=15`，一次 0.11 s 的循环把活塞（靶点）右推 0.477 m 再推回；偏移量是
   `15 sin(pi t/period) dt`（按步长积分的**速度**），`period < t <= period+rest`
   的静止窗口内不施加偏移（`elseif` 分支）。
6. **算例目录内部不自洽**：`Channel_Channel.m` 里写死 `Nx=Ny=512, Lx=Ly=1`，而
   `input2d` 是 `512 x 128, 1 x 0.25`；`please_Compute_External_Forcing.m`
   （本目录唯一自带的强制外力模块，是一个 `y∈[0.41,0.59]`、`uMax=250` 的抛物线射流）
   与 `test_Parabola.m` 都是给 1x1 盒子写的，在 1x0.25 盒子里窗口在域外——
   而且 `input2d` 里 `arb_ext_force = 0`，这个模块根本没被用上。`input2d_2` 与
   `input2d` 物理设置完全相同（只多了 `save_uVec/save_fMag/save_hier = 1`）。
7. **两个实现都会在 t 很小的时候 NaN**：活塞把细胞挤到**两个标记点重合**
   （`L=0`）时，弹簧力密度和键力密度都写成 `... / L`，`0/0 = NaN` 直接污染全场。
   这不是 coagulation 的问题：把 coagulation 关掉，同一算例在同一时刻照样 NaN
   （见下文"稳定性"）。


## 稳定性：活塞一动就发散（上游设置本身的问题）

* **参考解**（pyIB2d + coagulation，`MAX_SPEED = 15`）：`t = 0.010` 还正常，
  在 `t = 0.010` 与 `0.0125` 之间 NaN（把 T 加到 0.015 的探测运行里，`t = 0.0125`
  这一帧的 `u` 和 `X` 同时是 NaN）。所以本 demo 对比到 `t = 0.010025`。
* **AFSI**：同一设置跑到 `t = 0.015` 时 `max|u| ≈ 79`（也已在发散边缘，末帧键已断裂）。
* **不是 coagulation 的问题**：把 `coagulation` 关掉后，参考解在**同一步**发散
  （`--coagulation off` 的对照运行）。
* **是活塞的问题**：把活塞冻结（`update_target = 0`）后，同一算例稳定跑到
  `t = 0.05`（1000 步）且第 1 根键始终存在。
* 机理：活塞靶点以 15 m/s 推进，贴体罚力把细胞挤到**两个标记点重合**，而弹簧/
  键力密度都写成 `... / L` ⇒ `0/0 = NaN` 立刻污染全场。

因为上游 `Tfinal = 1.25`（`dt = 2.5e-5` 时 5 万步）在本机上不可行，我们按
"减小 T 到发散之前"的方式运行，并把实测发散时刻记在下面。这是**上游设置本身**的
问题，不是移植造成的。


## 不能比的东西（诚实声明）

* **没有可运行的"上游原始实现"**：MATLAB 文件残缺、pyIB2d 模块是坏的死代码。
  本 demo 的参考解 = **pyIB2d 的流体求解器 + 本仓库按 MATLAB 语义重写的
  coagulation 模型**（两侧共用 `ib2d_input/coagulation_model.py`）。所以它验证的是
  **AFSI 侧的耦合是否正确**（力的位置、`ds` 缩放、半步求值、时间推进、展布算子），
  **不是**一条独立上游轨迹。
* 键模型的正确性由 `coagulation_model.analytic_self_test()` 独立验证（两个合成细胞
  的解析力，`compare.py` 每次都会跑）：
  形成的键、静长 = 初始最近距离、`|f| = k|L - L_r|`、两端等大反向。
* 发散之后（见上表 NaN 行）**没有任何可比性**。
* 网格/`ds`/`print_dump` 若被改动，两侧都改；未改动的算例保持上游值。

## 本 demo 的结果

运行设置（上游 `Tfinal = 1.25` = 5 万步在本机不可行，按"发散之前"缩减）：

| 项 | 值 |
|---|---|
| `T`（秒） | 0.010025 |
| 步数 | 201 |
| `print_dump` | 25 |
| coagulation | `COAGULATION=1`（打开） |
| `MAX_SPEED`（活塞行程速度） | 15（上游值） |

对比输出（`python compare.py`，逐帧对齐后由脚本原样摘录）：

```
coagulation model self-test: PASS  {'bonds': 1, 'RL': 0.09999999999999987, 'force_max': 0.0, 'k_dL': 12.50000000000001, 'k_abs_L_minus_RL': 12.500000000000046, 'sum_f': 0.0}
upstream coagulation flag  : on   (3 cells, threshold 0.075, bond_k 1000, fracture 100)
note: 10 AFSI / 9 reference frames, 9 aligned

=== coagulation comparison (h = 0.015625, 9 aligned frames, t = 0.00000 .. 0.01000)
       t   dXmax/h          umax a/r   max|du|   bonds a/r  shared     dLmax          f_coag a/r      dXa  relL2(f)  pairs a/r
 0.00000  4.03e-10 0.000e+00/0.000e+00  0.00e+00     0/0           0  0.00e+00 0.000e+00/0.000e+00  0.0e+00  0.00e+00    1/1   
 0.00125  2.27e-02 2.041e+00/1.846e+00  9.92e-01     1/1           1  5.61e-05 3.547e-02/1.747e-02  0.0e+00  5.93e-01    1/1   
 0.00250  6.42e-02 4.414e+00/4.260e+00  2.25e+00     1/1           1  1.66e-04 1.015e-01/5.775e-02  0.0e+00  7.65e-01    1/1   
 0.00375  1.02e-01 6.117e+00/6.415e+00  2.79e+00     1/1           1  3.23e-04 2.385e-01/7.931e-02  0.0e+00  5.76e-01    1/1   
 0.00500  1.38e-01 7.278e+00/8.638e+00  3.34e+00     1/1           1  6.20e-04 5.825e-01/3.399e-02  0.0e+00  7.17e-01    1/1   
 0.00625  2.02e-01 8.467e+00/1.060e+01  5.13e+00     1/1           1  1.22e-03 1.277e+00/6.587e-02  0.0e+00  7.25e-01    1/1   
 0.00750  3.83e-01 1.054e+01/1.089e+01  7.96e+00     1/1           1  1.92e-03 2.356e+00/4.040e-01  0.0e+00  8.81e-01    1/1   
 0.00875  6.11e-01 1.261e+01/1.312e+01  1.07e+01     1/1           1  3.02e-03 4.563e+00/1.501e+00  0.0e+00  9.77e-01    1/1   
 0.01000  9.00e-01 1.527e+01/1.438e+01  1.44e+01     1/1           1  3.47e-03 7.193e+00/3.684e+00  0.0e+00  1.17e+00    1/1   

markers      : max|dX| 1.407e-02 m = 9.00e-01 h
velocity     : max|u| AFSI 1.5273e+01 / IB2d 1.4382e+01  (6.20e-02 rel), max|du| 1.440e+01
Eulerian f   : relative L2 1.169e+00
bonds        : AFSI 1 / IB2d 1, shared 1, differing 0, max|dL| 3.473e-03
coag force   : max|f_coag| AFSI 7.1928e+00 / IB2d 3.6841e+00
targets      : max|dXa| 0.000e+00 (same .m in both codes)
geometry chk : cell pairs below threshold AFSI 1 / IB2d 1 (recomputed from the saved markers; a bonded pair that is still below threshold is consistent)

wrote /Users/pengfei.ma/afsi/afsic/demo/ib2d/demo_630/figures/compare_table.csv, compare_markers.png, compare_history.png, compare_fields.png
=== demo_630 done Mon Oct  5 12:27:14 BST 2026
```

## 运行

```bash
conda activate afsi-dolfinx
python run_reference.py --tend 0.010025 --print-dump 25 --coagulation on   # pyIB2d + coagulation_model, 实测 ~55 s
COAGULATION=1 TFINAL=0.010025 PRINT_DUMP=25 python main.py   # AFSI, 实测 ~4 s
COAGULATION=1 python compare.py   # 实测 ~6 s
```

（`coagulation_model.analytic_self_test()` 由 `compare.py` 每次运行，用于独立
验证键模型；`figures/compare_table.csv` 是同一张表的机器可读版本。）
