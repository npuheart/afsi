# demo_683 — IB2d `Example_Concentration_Dynamics/Corals_64`（脉动珊瑚 + 受限源项）

IB2d 官方算例（`matIB2d/Examples/Example_Concentration_Dynamics/Corals_64`）。
浓度源项家族里机制最复杂的一个：**珊瑚水螅体（polyp）按自带系数表脉动**，
其 59 个轮廓点被罚力（`k = 8e5`）钉到每步重建的目标形状上，
同时以**受限速率**向流体释放标量（`source = 2`）。共享的源项机制、上游缺口、
验证恒等式见 **demo_680 的 readme**。

## 上游路径

* 数据：`third_party/ib2d/matIB2d/Examples/Example_Concentration_Dynamics/Corals_64/`
  （`coral.vertex/target/concentration/geo_connect`、`cval.mat`、
  `total_coeffs.mat`、`coral_coeff_30.mat`、`input2d`、
  `update_Target_Point_Positions.m` → 已逐行移植为同名 `.py`；
  `main2d.m`、`Coral_Geometry.m`、`give_me_source_model.m` 未随 demo 提供，
  后两者是几何生成器/上游死文件）。
* 算法源码：`matIB2d/IBM_Blackbox/please_Find_Source_For_Concentration.m`、
  `please_Update_Adv_Diff_Concentration_Source.m`；`pyIB2d/IBM_Blackbox/`。

## 算例内容

* `2 x 5` 周期盒、`mu = 0.01`、`rho = 1`、官方 `64 x 160`（`dx = dy = 0.03125`）、
  `dt = 2.5e-4`、`supp = 4`；官方 `Tfinal = 19`（76000 步）→ 本 demo 默认
  **`T = 0.5`（2000 步）**。`input2d` **没有 `print_dump` 行**，两边都按
  `num_steps // 20 = 100` 取（21 帧）。
* **结构**：59 个标记点（27 + 5 桥 + 27）由 `target_pts = 1, update_target = 1`
  驱动：目标形状每步由 `cval.mat` 的三次系数重建（`total_meanL`、`total_offset`
  来自 `total_coeffs.mat`，5 点"柄"`XbStem/YbStem` 也存在 `cval.mat` 里），
  标记点被 `k = 8e5` 的罚力钉上去。**官方网格不可减**：
  `update_Target_Point_Positions.m` 里 `Nx = 64`、`ds = dx/2` 是硬编码的，
  减 `NX` 会改变 `s` 的点数（27 → 其它），与 59 点的 `coral.vertex` 不匹配。
* **脉动周期 ≈ 2 s**（由系数表算出：`t = 0` 时触手平铺 `y_max ≈ 0.065`、
  `t ≈ 0.25–0.5` 时竖直展开到 `y_max ≈ 0.36`、`t ≈ 1.25` 又缩回、`t ≈ 2.25` 再次展开）。
  本 demo 的 `T = 0.5` 正好覆盖"展开"这一段最强驱动过程。
* **浓度**：`kDiff = 8.7546e-4`、`advection = 0`、**`source = 2`（受限）**、
  `k_source = 0.0215`、`c_inf = 1`，`C0 ≡ 0`（`coral.concentration` 无 `kDiff` 头行）。

## 移植要点

1. **`update_Target_Point_Positions.py`（逐行移植）**：`s = (0:ds:total_meanL)/total_meanL`、
   三次多项式、`L_ten` 弧长归一化、`flip` 对称拼接、`xoffset/yoffset`。
   用 `scipy.io.loadmat` 读三个 `.mat`。
   * 上游怪癖 1：`XbStem`/`YbStem` **不是**在本模块里算的，而是靠
     `load('cval.mat')` 带进来的（`Coral_Geometry.m` 存进去的 5 点柄）；
   * 上游怪癖 2：`coral_coeff_30.mat`（`cs1`/`cs2`）与 `total_coeffs.mat` 里的
     `total_qxcoeffs/qycoeffs/typechange/...` 都没被用到，移植保留 `load` 以对齐行为；
   * 性能：MATLAB 每步都 `load` 三个 `.mat`（4.3 MB）；移植改成**一次加载 + 缓存**
     （数据只读），否则每步重读会主导运行时间，已在模块 docstring 注明；
   * 索引：`indx = ceil(t/dt) + 1`（1 起始）→ `cval.mat` 有 75601 行，
     `T < 19` 时不会越界（本例 `T = 0.5` ⇒ `indx ≤ 2001`）。
   * 移植后用**上游生成的首帧**校验：`t = 0` 时重建出的 59 点与 `coral.vertex`
     **逐位相同**（`max|Δ| = 0`），说明几何重建（含 `flip`、柄、偏移）完全正确。
2. **源项**：与 demo_680/681 完全同一份实现（参考侧补丁 + AFSI 侧独立实现 +
   非分裂 upwind + 每步守恒自检）。日志：
   `step 0: Σ Fs dxdy = 0.0198203125 = Σ fs ds（rel 0）`
   ——正好等于 `k_source·ds·Nb = 0.0215 × 0.015625 × 59`（`C0 ≡ 0` ⇒ `CL = 0` ⇒ `fs = k`）；
   末步 `0.01980926640257757 vs ...758（rel 1.8e-16）`。
3. **`*.concentration` 无 `kDiff` 头行**：`ib2d_io` 容忍 + 回退 `input2d` 的 `kDiff`，
   参考侧运行目录里补头行（同 demo_680）。
4. **`.mat` 文件要进 pyIB2d 的运行目录**：上游 `update_Target_Point_Positions.m`
   用裸 `load('cval.mat')`，所以 `run_reference.py` 的拷贝列表里加了 `*.mat`。
5. `.geo_connect` 未启用（pyIB2d 无读取器），两边都用 `fsds = fs·ds`；
   WENO（`advection = 1`）不涉及本算例。

## 结果

参考运行 ~9 s、AFSI 运行 ~80 s（各 2000 步、21 帧，`t = 0 .. 0.50025`）。

| 指标（末帧 `t = 0.50025`） | 结果 |
|---|---|
| 标记点偏差（59 点，被罚力钉在脉动目标上） | 2.26e-4 m = **7.2e-3 h** |
| 目标跟踪 `max\|X_tgt − X_a\|`（AFSI 自检打印） | 2.0e-4 m |
| `max\|u\|` AFSI / IB2d | 1.7038 / 1.7159（**7.0e-3** 相对） |
| 速度场相对 $L_2$（窄带外） | 8.05e-2 |
| 窄带外 `max\|du\|` | 3.06e-1 = 8.2e-1 `max\|u\|` |
| 浓度相对 $L_2$ | **9.94e-2** |
| 浓度 `max\|ΔC\|` | 2.12e-2（峰值 0.119 → 18% 局部差） |
| 峰值 `max C` AFSI / IB2d | 0.11892 / 0.10957（**8.5e-2** 相对） |
| 总量 `∑C` AFSI / IB2d | 9.553 / 9.481（相对差 7.6e-3） |
| 源项撒点守恒（step 0 / 末步） | `Σ Fs dxdy = Σ fs ds = 0.0198203125`（rel 0）/ `0.01980926640257757`（rel 1.8e-16） |
| 几何重建自检（离线） | `t = 0` 的重建轮廓与 `coral.vertex` **逐位相同** |
| 早期帧 `t = 0.05025` | `relL2(u)` 9.93e-2、`relL2(C)` 4.52e-2、标记点 2.9e-3 h |

**能验证**：
* **脉动目标轨迹 + 罚力跟踪**：59 点标记在 2000 步里偏差仅 **7.2e-3 h**，
  且 `t = 0` 的重建几何与上游 `coral.vertex` 逐位相同——说明
  `update_Target_Point_Positions.py`（含 `load`/缓存、`indx`、`flip`、柄拼接）正确；
* **源项撒点/插值**：`Σ_grid Fs dxdy = Σ_i fs_i ds` 精确成立（step 0 为解析值
  `0.0215·ds·59`，rel 0；末步 rel 1.8e-16），与 demo_680 的 δ 函数对拍
  （`max|ΔFs| = 3.6e-14`）一起覆盖了单位化与 stencil；
* **受限源项的定性行为**：`∑fs ds` 随边界饱和略微下降（0.0198203 → 0.0198093），
  两边一致；
* 浓度场整体演化：末帧 `relL2(C) = 9.9e-2`、总量守恒到 7.6e-3，
  首个输出帧 `relL2(C) = 4.5e-2`。

**不能验证 / 已知限制**：

1. **`T = 19`（76000 步）未跑满**，本 demo 用 `T = 0.5`（2000 步，覆盖一次
   "触手展开"的主驱动段）。原因：AFSI 在 `64 x 160` 上约 40 ms/步，
   跑满需要 ~50 分钟；参考侧也要 ~25 分钟。`input2d` 没有 `print_dump`，
   两边都按 `num_steps//20` 出帧，所以帧对齐不是问题，纯粹是时长。
2. **速度场只做定性比较**：本例的流动**完全靠被罚力驱动的快速变形边界**产生
   （无外力、无来流），两种代码在"强、快速变化的体积力"下差异被放大
   （末帧 `max|u|` 差 7.0e-3，但窄带外 `max|du| = 0.82 max|u|`、
   `relL2(u) = 8.1e-2`）。这不影响被动标量的量级，但 `relL2(C) = 9.9e-2`
   里有相当一部分是这个速度差异的脚印。`GRAD_DIV` 试过 1.0，反而更差
   （`max|u|` 偏低 19% vs 7%），故保持 0。
3. **源项公式本身**没有第三方核对（两侧同源一份 MATLAB；本机无 MATLAB/Octave），
   只能验证实现/网格传递与守恒恒等式。
4. `.geo_connect` 梯形法弧长未启用（上游 MATLAB 若存在该文件会启用）。
5. 上游 `source > 0` 走**非分裂** upwind，故 `relL2(C)` 不能与 demo_476/477 直接比较。
6. WENO（`advection = 1`）两侧均未实现；本例 `advection = 0` 不受影响。
7. **网格不可减**（`Nx = 64` 硬编码在用户模块里），故只能减 `T`。

## 运行

```bash
conda activate afsi-dolfinx
python run_reference.py --tend 0.5          # pyIB2d + 源项/目标模块，~9 s
TFINAL=0.5 python main.py                   # AFSI，~80 s
python compare.py                           # figures/ + compare_table.csv
```
