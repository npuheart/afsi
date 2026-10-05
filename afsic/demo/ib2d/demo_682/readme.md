# demo_682 — IB2d `Example_Concentration_Dynamics/Rubberband_Sink_64`（橡皮筋 + 汇项/反应模型）

IB2d 官方算例（`matIB2d/Examples/Example_Concentration_Dynamics/Rubberband_Sink_64`）。
浓度源项家族的第三个成员，也是**唯一使用 `source = 3`（反应/汇）**的一个：
同一根收缩橡皮筋把周围的标量"吸走"。共享机制的推导、验证恒等式与上游缺口见
**demo_680 的 readme**。

## 上游路径

* 数据：`third_party/ib2d/matIB2d/Examples/Example_Concentration_Dynamics/Rubberband_Sink_64/`
  （`rubberband.vertex/spring/concentration/geo_connect`、`input2d`；
  `main2d.m`、`Rubberband.m`、`give_me_source_model.m` 未随 demo 提供——后者是上游死文件）。
* 算法源码：`matIB2d/IBM_Blackbox/please_Find_Source_For_Concentration.m`、
  `please_Update_Adv_Diff_Concentration_Source.m`；
  `pyIB2d/IBM_Blackbox/`（参考驱动 + `Supp.py`）。

## 算例内容

* `1 x 1` 周期盒、`mu = 0.01`、`rho = 1`、`Nx = Ny = 64`、`dt = 3.125e-5`、`supp = 4`；
  官方 `Tfinal = 2`（64000 步）→ 本 demo 默认 **`T = 0.1`（3200 步）**，官方
  `print_dump = 320` 保留（11 帧）。
* **结构**：与 demo_680 相同（249 点闭合橡皮筋、249 根 `k = 1e5`、`L0 = 0` 的收缩弹簧）。
* **浓度**：`kDiff = 0.01`、`advection = 0`（一阶 upwind）、
  **`source = 3`（汇/反应）**、`k_source = −2.0`、`c_inf = 0.1`（模型 3 不使用）、
  `C0 ≡ 1`（**全场均匀**；`rubberband.concentration` 无 `kDiff` 头行）。
  源密度 `fs = k_source · CL = −2 CL`：边界处浓度被抽走，
  于是均匀场先在橡皮筋附近形成"凹陷"，凹陷再被扩散/平流输运。
  末帧总量 `∑C ≈ 3492`（初值 4096），即约 15% 被吸收。

## 移植要点

与 demo_680 完全一致（同一份 `main.py` / `run_reference.py` /
`ib2d_input/please_Find_Source_For_Concentration.py`，模型常数来自 `input2d`）：

1. 源项是 pyIB2d 完全缺失的机制（`source`/`k_source`/`c_inf` 被静默忽略）；
   参考侧把 matIB2d 的两个 MATLAB 文件逐行移植后接进 pyIB2d 驱动，
   AFSI 侧用自己的 IB 插值/撒点机器独立实现同一公式；
2. 上游 `source > 0` 时实际走**非分裂** upwind 格式
   （`C += dt (kΔC − u·∇C + Fs)`），照抄；
3. `*.concentration` 无 `kDiff` 头行（`ib2d_io` 容忍 + 回退 `input2d` 的 `kDiff`）；
4. `.geo_connect` 未启用（两边都用常数 `ds = Lx/(2Nx)`）；
5. 每步校验 `Σ_grid Fs·dx·dy = Σ_i fs_i·ds`（4 点核的分区单位性）。
   本例 `fs = k·CL`，所以这条恒等式**同时**校验了插值（`CL`）与撒点（`Fs`）的归一化：
   * `step 0`：`C0 ≡ 1` ⇒ `CL ≡ 1` ⇒ `Σ fs ds = k·ds·Nb = −2 × 0.0078125 × 249
     = −3.890625`，日志正好打印 `−3.890625 vs −3.890625`（rel 0）；
   * 末步：橡皮筋附近已被抽空（`CL` 均值 ≈ 0.25），
     `−0.9571308571145184 vs −0.9571308571145183（rel 1.2e-16）`。
   两侧逐位一致。

## 结果

参考运行 ~15 s、AFSI 运行 ~51 s（各 3200 步、11 帧，`t = 0 .. 0.10003`）。

| 指标（末帧 `t = 0.10003`） | 结果 |
|---|---|
| 标记点偏差 | 6.811e-3 m = **4.36e-1 h**（与 demo_680 同源，收缩折叠区敏感） |
| `max\|u\|` AFSI / IB2d | 1.8014 / 1.8169（**8.5e-3** 相对） |
| 速度场相对 $L_2$（窄带外） | 1.27e-1 |
| 窄带外 `max\|du\|` | 6.04e-1 = 6.7e-1 `max\|u\|` |
| 弹簧平均长度 AFSI / IB2d | 0.007204 / 0.007333（1.8e-2 相对） |
| 浓度相对 $L_2$ | **1.97e-2**（全场 O(1)，比 680/681 好得多） |
| 浓度 `max\|ΔC\|` | 6.55e-2（峰值 1.000 → 6.5%，出现在橡皮筋附近的凹陷里） |
| 峰值 `max C` AFSI / IB2d | 1.000000 / 1.000000（8.9e-15 相对） |
| 总量 `∑C` AFSI / IB2d | 3492.325 / 3492.050（相对差 7.9e-5） |
| 源项撒点守恒（step 0 / 末步） | `Σ Fs dxdy = Σ fs ds = −3.890625`（rel 0）/ `−0.9571308571145184`（rel 1.2e-16） |
| 早期帧 `t = 0.01003` | `relL2(C)` 1.50e-3、`max\|ΔC\|` 1.46e-2（峰值 1 → 1.5%） |

**能验证**：源项（`fs = k CL`）的插值 + 撒点归一化——`Σ Fs dxdy = Σ fs ds` 精确成立，
且两侧独立实现逐位一致；全场 O(1) 的浓度演化 `relL2(C) = 2.0e-2`、
总量守恒到 7.9e-5；结构收缩的宏观量（`max|u|` 8.5e-3 相对）。

**不能验证 / 已知限制**（与 demo_680 相同，摘要）：

1. `T = 2`（64000 步）未跑满：`L0 = 0` 的橡皮筋收缩折叠，`t ≳ 0.02` 后
   速度场进入敏感阶段（末帧窄带外 `max|du| = 0.67 max|u|`、`relL2(u) = 0.127`、
   标记点 0.44h）；与源项无关。
2. **源项公式本身**没有第三方核对（两侧同源一份 MATLAB；本机无 MATLAB/Octave），
   只能验证实现/网格传递与守恒恒等式。
3. `c_inf = 0.1` 对模型 3 是**无用量**（上游沿用了受限模型的注释），已按原样保留。
4. `.geo_connect` 梯形法弧长未启用。
5. 上游 `source > 0` 走**非分裂** upwind，故 `relL2(C)` 不能与 demo_476/477 直接比较。
6. WENO（`advection = 1`）两侧均未实现；本例 `advection = 0` 不受影响。

## 运行

```bash
conda activate afsi-dolfinx
python run_reference.py --tend 0.1          # pyIB2d + 源项补丁，~15 s
TFINAL=0.1 python main.py                   # AFSI，~51 s
python compare.py                           # figures/ + compare_table.csv
```
