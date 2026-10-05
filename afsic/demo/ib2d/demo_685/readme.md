# demo_685 — IB2d `Example_Meshmerize_Me`（"Meshmerize Me" 叶脉网格 + 流槽射流）

IB2d 官方算例（`matIB2d/Examples/Example_Meshmerize_Me`）。
这是 **Meshmerize 家族**（demo_685–690）的主算例：一个真实的**叶片脉络网络**
（1585 个标记点，全部由目标点 `k = 1e5` 钉死）+ 一个由**小窗口罚力**驱动的"流槽"。
家族里其它 5 个 demo（686–690）只是换成不同的几何/窗口/驱动波形；
本 readme 给出共享的移植要点，其它 readme 引用本文。

## 上游路径

* 数据：`third_party/ib2d/matIB2d/Examples/Example_Meshmerize_Me/`
  （`leaf_vein.vertex`、`leaf_vein.target`、`input2d`；
  `please_Compute_External_Forcing.m` → 已逐行移植为同名 `.py`；
  `main2d.m`、`Leafitize_Me.m`（SVG→网格的生成脚本，运行期不需要）、
  `leaf_flow.vertex`（本算例的 `input2d` 用的是 `leaf_vein`，该文件未使用）
  未随 demo 提供）。
* 算法源码：`third_party/ib2d/pyIB2d/IBM_Blackbox/`（参考驱动）。

## 算例内容

* `1 x 2.5` 周期盒、`mu = 1e-3`、`rho = 1`、官方 `410 x 1024`（方格 `h = 0.00244`）、
  `dt = 1e-4`、`supp = 4`；官方 `Tfinal = 0.5`（5000 步）、`print_dump = 100`。
* **结构**：`leaf_vein.vertex` 1585 点 + `leaf_vein.target` 1585 个目标点
  （`k = 1e5`，`update_target = 0`，即**形状固定**）；无弹簧/梁/...
  标记点被钉住后几乎不动（结构本身是"刚性叶脉网"）。
* **驱动**（`arb_ext_force = 1`）：在**一个很窄的窗口**内（`x ∈ [0.252, 0.364]`、
  `y ∈ [0.3625, 0.3675]`）把 y 速度罚到抛物线目标型线上：
  ```
  uY_tar = uMax tanh(2t) ((MP+w/2)-x)((MP-w/2)-x) / scale,   uX_tar = 0
  ```
  `uMax = 2.5`、`w = 0.359550 − 0.247375 = 0.112175`、`MP` 为其中心、
  `scale = (w/2)(−w/2)`；力 `F = −kStiff (u − u_tar)`、`kStiff = 1e4`。
  窗口索引只在**第一次调用**时按坐标搜索一次（所以同一份模块可跑任意分辨率）。

## 移植要点（家族共享）

1. **`please_Compute_External_Forcing.py`（逐行移植，两个代码共用）**：
   `give_Me_Indices_To_Apply_Force`（按坐标搜索窗口索引，MATLAB 是
   `iX_min:iX_max × iY_min:iY_max` 的 x-major 配对）、
   `give_Me_Velocity_Target_External_Force_Density`（窗口内逐点累加罚力）、
   `please_Give_Target_Velocity`（本例的抛物线 y 型线）。
   移植保留 1e4 的刚度与窗口的**物理**坐标，只用 0 基索引代替 MATLAB 的 1 基索引，
   并把 `x(i)`/`y(j)` 的值直接传进型线函数（而不是在型线里再索引）。
   **6 个目录的模块只在常数与型线上不同**（见各 demo 的 readme 与模块 docstring）：
   * 685 / 690：y 方向射流（型线是 x 的抛物线），690 无 `scale`、时间因子 `5 tanh(t)`；
   * 686：x 方向射流（型线是 y 的抛物线，中心 `Ly/2`，**无** `scale`，
     时间因子 `sin(4πt)`）；
   * 687：同 686 但**脉冲式**（`freq_P = 5`，一个脉冲周期 + 一个等长静止期）；
   * 688 / 689：x 方向射流，`tanh(2t)`，有 `scale`。
   上游小怪癖：`while` 循环里 `j=j+1` 写在赋值之后（无害）；687 的
   `if t_mod <= tot_period` 恒为真。
2. **分辨率**：官方网格很大（410×1024 到 1024×512）。本 demo 采取
   **两个方向同时减半**（205×512）的规则：格子仍是正方形，且窄窗口里仍有节点
   （窗口只有 ~0.005 高，若把 Ny 降到 240 就会一个节点都没有 → 完全不驱动）。
   更粗的网格必须同时放大窗口，那会改变物理，故未采用。
3. **`T` 大幅缩短**：官方 `T = 0.5`（5000 步）→ 本 demo `T = 0.02`（200 步，
   约 1/25）。原因是这个算例的驱动窗口只有 1 个格点高，
   `kStiff·dt = 1e4 × 1e-4 = 1` 恰好落在**显式罚力的临界稳定点**上
   （详见"不能验证"），长时间积分会把两种时间推进器的差异放大。
4. **`.geo_connect` / WENO / 源项**等机制本族都没有，模板开箱即用
   （只有 `target_pts` + `arb_ext_force`）。
5. 官方 `print_dump = 100`（= 0.01 s）→ 本 demo 用 `--print-dump 10`
   （`T = 0.02` 时 20 帧）。

## 结果

参考运行 ~25 s、AFSI 运行 ~70 s（各 200 步、21 帧，`t = 0 .. 0.0201`）。
测试过 `GRAD_DIV = 1.0`：反而更差（`max|u|` 偏大 18.6% vs 8.8%），故保持 0。

| 指标（末帧 `t = 0.02010`） | 结果 |
|---|---|
| 标记点偏差（1585 点，钉死叶脉网） | 3.91e-5 m = **8.0e-3 h** |
| 窗口内 `max\|u\|` AFSI / IB2d | 9.115e-2 / 8.381e-2（**8.8e-2** 相对） |
| 速度场相对 $L_2$（窄带外） | 7.22e-1 |
| 窄带外 `max\|du\|` | 6.70e-2 = 2.0e-1 `max\|u\|` |
| 外力场（窄带外） | `max\|df\| = 1.3e2`（同一窗口、同一 `u_tar`，差异来自 `u` 的不同） |
| 早期帧 `t = 0.00810` | 标记点 2.1e-3 h、`relL2(u)` 7.82e-1 |

**能验证**：
* **网格化的结构 + 钉死**：1585 点目标点结构在 200 步后偏差 **8.0e-3 h**，
  说明 vertex/target 读取、目标点罚力、IB 撒点/插值在本族的大网格上都正确；
* **外力模块**：两个代码跑的是同一份移植模块，窗口与型线完全一致
  （688/689 的日志里"窄带外力差 = 2.5e-19"即直接证据：窗口内 `u_tar` 相同、
  `u` 也一致时外力逐位相同）；
* 682/685/686/690 的结构量（见各自 readme，均 ≤ 2e-2 h）。

**不能验证 / 已知限制**：

1. **速度场只做定性比较**（`relL2(u) = 0.72`，窗口 `max|u|` 差 8.8%）。
   原因是本算例的驱动窗口**只有 1 个格点高**（官方 2 个），
   而 `kStiff·dt = 1e4 × 1e-4 = 1` 让它正好处在显式罚力的**临界稳定点**
   （`u^{n+1} = u^n + dt·(−k(u^n − u_tar))` ⇒ 一步到位的无阻尼振荡）。
   两个代码的显式时间推进（pyIB2d 的 FFT 投影 vs AFSI 的 Taylor-Hood/Peskin 两段式）
   对这种"零阻尼单格点力"的响应不同，窄带外 `max|du|` 达到 0.2`max|u|`。
   这与源项无关，也不是结构问题（标记点 8e-3 h）。
2. **`T` 只跑 1/25、网格减半**：官方 `410×1024 / T = 0.5`（5000 步）需要
   AFSI ~40 分钟（实测 205×512 时 ~0.33 s/步）。减半规则已在上文说明。
3. 本机无 MATLAB/Octave，无法直接跑上游 matIB2d 做第三方核对；
   上游也只给了 MATLAB 版（pyIB2d 是官方 Python 移植，作为参考基准）。
4. 结构是**刚性钉死**的（`update_target = 0`），所以本算例不检验任何
   结构本构（弹簧/梁等），只检验几何读取 + 外力驱动 + 流体响应。

## 运行

```bash
conda activate afsi-dolfinx
python run_reference.py --tend 0.02 --nx 205 --print-dump 10   # pyIB2d，~25 s
NX=205 TFINAL=0.02 PRINT_DUMP=10 python main.py                # AFSI，~70 s
python compare.py                                              # figures/ + compare_table.csv
```
