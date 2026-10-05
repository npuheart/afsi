# demo_681 — IB2d `Example_Concentration_Dynamics/Rubberband_Limited_64`（橡皮筋 + 受限源项）

IB2d 官方算例（`matIB2d/Examples/Example_Concentration_Dynamics/Rubberband_Limited_64`）。
浓度源项家族的第二个成员：与 demo_680 **同一根橡皮筋**，但源模型换成
`source = 2`（受限/饱和模型）。共享机制的推导、验证恒等式与上游缺口见
**demo_680 的 readme**，这里只列出本例特有的设置与结果。

## 上游路径

* 数据：`third_party/ib2d/matIB2d/Examples/Example_Concentration_Dynamics/Rubberband_Limited_64/`
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
  **`source = 2`（受限）**、`k_source = 0.1`、`c_inf = 1.0`，
  `C0 ≡ 0`（`rubberband.concentration` 无 `kDiff` 头行）。
  源密度 `fs = k_source (c_inf − CL)`：边界饱和后源自动关闭，
  因此 `∑fs ds` 随时间下降（日志里 `step 0` 为 0.1755，末步更小）。

## 移植要点

与 demo_680 完全一致（同一份 `main.py` / `run_reference.py` /
`ib2d_input/please_Find_Source_For_Concentration.py`，只有模型常数来自 `input2d`）：

1. 源项是 pyIB2d 完全缺失的机制（`source`/`k_source`/`c_inf` 被静默忽略），
   参考侧把 matIB2d 的两个 MATLAB 文件逐行移植后**接进 pyIB2d 驱动**，
   AFSI 侧用**自己的 IB 插值/撒点机器**独立实现同一公式；
2. 采用上游在 `source > 0` 时实际执行的**非分裂** upwind 格式
   （`C += dt (kΔC − u·∇C + Fs)`），其 `give_Necessary_Derivative` 会填满倒数第二行/列
   （与 pyIB2d 独立版本的 off-by-one 不同）；
3. `*.concentration` 无 `kDiff` 头行 → `ib2d_io` 容忍、`main.py` 回退 `input2d` 的
   `kDiff`、参考侧运行目录里补头行；
4. `.geo_connect` 未启用（两边都用常数 `ds = Lx/(2Nx)`）；
5. 每步校验 `Σ_grid Fs·dx·dy = Σ_i fs_i·ds`（4 点核的分区单位性）。
   运行日志：`step 0: 0.19453125 vs 0.19453125（rel 1.4e-16）`（= `k·ds·Nb`）、
   末步 `0.175537475461383 vs 0.175537475461383（rel 1.6e-16）`——
   两侧同步下降，说明"受限源随边界饱和而关闭"这一定性行为被两边一致复现。

## 结果

参考运行 ~20 s、AFSI 运行 ~74 s（各 3200 步、11 帧，`t = 0 .. 0.10003`）。

| 指标（末帧 `t = 0.10003`） | 结果 |
|---|---|
| 标记点偏差 | 6.811e-3 m = **4.36e-1 h**（与 demo_680 同源，收缩折叠区敏感） |
| `max\|u\|` AFSI / IB2d | 1.8014 / 1.8169（**8.5e-3** 相对） |
| 速度场相对 $L_2$（窄带外） | 1.27e-1 |
| 窄带外 `max\|du\|` | 6.04e-1 = 6.7e-1 `max\|u\|` |
| 弹簧平均长度 AFSI / IB2d | 0.007204 / 0.007333（1.8e-2 相对） |
| 浓度相对 $L_2$ | **7.60e-2** |
| 浓度 `max\|ΔC\|` | 1.14e-2（峰值 0.116 → 约 10%） |
| 峰值 `max C` AFSI / IB2d | 0.11637 / 0.11292（**3.1e-2** 相对） |
| 总量 `∑C` AFSI / IB2d | 70.391 / 69.900（相对差 7.0e-3） |
| 源项撒点守恒（step 0 / 末步） | `Σ Fs dxdy = Σ fs ds = 0.19453125`（rel 1.4e-16）/ `0.175537475461383`（rel 1.6e-16） |
| 早期帧 `t = 0.01003` | `relL2(C)` 1.71e-2、`max\|ΔC\|` 9.6e-4（峰值 0.031 → 3%） |

**能验证**：源项撒点/插值恒等式（分区单位性与 `CL(1) = 1`，与 demo_680 同一份代码，
该 demo 里已与 pyIB2d 的 δ 函数实现直接对拍到 5.2e-15）；受限源项使 `∑fs ds` 随
边界饱和而下降（两侧一致到 `sum(C)` 相对 %）；浓度场早期（源还未饱和、
流动还弱）`relL2(C) = 1.7e-2`，以及结构收缩的宏观量（`max|u|` 8.5e-3 相对）。

**不能验证 / 已知限制**（与 demo_680 相同，摘要）：

1. `T = 2`（64000 步）未跑满：`L0 = 0` 的橡皮筋会**收缩折叠**，
   `t ≳ 0.02` 后标记点挤成一团，1% 级速度差被放大（末帧窄带外
   `max|du| = 0.67 max|u|`、`relL2(u) = 0.127`、标记点 0.44h）；
   这与源项无关（源项只加在被动标量上）。
2. **源项公式本身**没有第三方核对：两侧实现同源于一份 MATLAB 源码，
   本机无 MATLAB/Octave，因此只能验证"实现 + 网格传递"以及守恒恒等式；
   与上游 MATLAB 的逐行对照见 `ib2d_input/please_Find_Source_For_Concentration.py`。
3. `.geo_connect` 梯形法弧长未启用（上游 MATLAB 若存在该文件会启用）。
4. 上游 `source > 0` 走**非分裂** upwind，故 `relL2(C)` 不能与 demo_476/477
   （分裂 upwind，已验证 1e-11）直接比较。
5. WENO（`advection = 1`）两侧均未实现；本例 `advection = 0` 不受影响。

## 运行

```bash
conda activate afsi-dolfinx
python run_reference.py --tend 0.1          # pyIB2d + 源项补丁，~20 s
TFINAL=0.1 python main.py                   # AFSI，~74 s
python compare.py                           # figures/ + compare_table.csv
```
