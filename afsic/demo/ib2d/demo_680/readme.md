# demo_680 — IB2d `Example_Concentration_Dynamics/Rubberband_Constant_64`（橡皮筋 + 恒定源项浓度）

IB2d 官方算例（`matIB2d/Examples/Example_Concentration_Dynamics/Rubberband_Constant_64`）。
本 demo 是**浓度源项（source/sink）家族**的第一个成员：新机制是"浸入边界以恒定速率向
流体释放标量"（`source = 1`，`fs = k_source`），并把它与已被验证到 1e-11 的浓度
平流-扩散算子（demo_476/477）组合起来。

## 上游路径

* 数据：`third_party/ib2d/matIB2d/Examples/Example_Concentration_Dynamics/Rubberband_Constant_64/`
  （`rubberband.vertex/spring/concentration/geo_connect`、`input2d`；
  `main2d.m`、`Rubberband.m`、`give_me_source_model.m` 未随 demo 提供，见下）。
* 算法源码：`third_party/ib2d/matIB2d/IBM_Blackbox/please_Find_Source_For_Concentration.m`、
  `please_Update_Adv_Diff_Concentration_Source.m`（源项）；
  `third_party/ib2d/pyIB2d/IBM_Blackbox/`（参考驱动，`Supp.py` 的 δ 函数与 `DD`）。

## 算例内容

* `1 x 1` 周期盒、`mu = 0.01`、`rho = 1`、`Nx = Ny = 64`（`h = 1/64`）、
  `dt = 3.125e-5`、`supp = 4`；官方 `Tfinal = 2`（64000 步）→ **本 demo 默认
  `T = 0.1`（3200 步）**，官方 `print_dump = 320` 保留（10 帧 + 末帧）。
* **结构**：249 个顶点连成一条闭合"橡皮筋"（半径 0.2、圆心 `(0.7, 0.5)`），
  249 根弹簧 `k = 1e5`、**静止长度 `L0 = 0`**（IB2d 的"收缩弹簧"约定），
  `alpha = 1`（线性，文件只有 4 列，α 缺省 1）。`ds = Lx/(2Nx) = 0.0078125`。
  `update_springs = target_pts = ... = 0`，只有弹簧：橡皮筋在自身上收缩并把流体卷起来。
* **浓度**：`kDiff = 0.01`（`input2d`）、`advection = 0`（一阶 upwind）、
  **`source = 1`（恒定源）**、`k_source = 0.1`、`c_inf = 1.0`（模型 1 不用）。
  初始浓度场 `C0 ≡ 0`（`rubberband.concentration` 是 64×64 全零、**没有 kDiff 头行**）。
  浓度是纯被动的（不反馈到动量方程），所以它只检验标量算子 + 源项。

## 移植要点

1. **源项是 pyIB2d 完全缺失的机制**（上游缺口）：
   `IBM_Driver.py` 只调用 `Supp.please_Update_Adv_Diff_Concentration(C,dt,dx,dy,U,V,kDiff)`，
   `read_In_Concentration_Info` 也只读 `kDiff` + 场，
   **`source` / `k_source` / `c_inf` 三行被静默忽略**。若不补，参考解会退化成
   普通平流-扩散，而本例 `C0 ≡ 0` 会**永远恒为 0**，本 demo 就毫无意义。
   因此本 demo 把 matIB2d 的两个 MATLAB 文件逐行移植：
   * 参考侧：`ib2d_input/please_Find_Source_For_Concentration.py`
     （用 pyIB2d 自己的 `Supp.give_1D_NonZero_Delta_Indices` /
     `Supp.give_Delta_Kernel` / `Supp.DD` 实现 δ 插值、源密度与撒点），
     `run_reference.py` 把它**接进 pyIB2d 驱动**（替换
     `Driver.please_Update_Adv_Diff_Concentration`，并记录标量更新时刻的标记点位置）；
   * AFSI 侧：`main.py` 用 **AFSI 自己的 IB 机器**（`IBInterpolation.fluid_to_solid`
     插值 C 到标记点、`solid_to_fluid` 撒 `fs*ds`）独立实现同一公式。
   两边因此共享的是**公式**（同源于 MATLAB），而**网格传递**是两套独立代码
   （afsic 的 C++ IBMesh ↔ pyIB2d 的 Supp δ 函数），这正是本 demo 真正交叉验证的部分。
2. **源项时间推进用 matIB2d 的"非分裂"格式**（`perform_Upwind_source`）：
   `C^{n+1} = C^n + dt ( kΔC^n − u^{n+1}·∇C^n + Fs )`，
   而不是 demo_476/477 那种分裂 upwind。原因是上游在 `source > 0` 时**只走这条分支**
   （`please_Update_Adv_Diff_Concentration_Source.m`，`Adv_flag = 0`）。
   顺带发现一个真实差异：该文件里的 `give_Necessary_Derivative` 内循环是
   `for j = 2:lenX-1`，**会填满倒数第二行/列**；而 pyIB2d 的独立版本
   `Supp.give_Necessary_Derivative` 循环 `range(1, lenX-2)`，把该行留成 0
   （demo_477 的 readme 记录过这个"照抄"）。本 demo 忠实采用**源项文件里的版本**，
   所以它和 `source = 0` 的算例不逐位可比。
3. **`*.concentration` 无头行**：本文件第一行就是网格数据。pyIB2d 的
   `read_In_Concentration_Info` 会把第一行当 `kDiff` → **开箱即崩**
   （`ValueError: could not convert string to float`）；MATLAB 版则把第一行数据当
   `kDiff` 并少读一行。处理（不修改上游）：`ib2d_io.read_concentration` 同时接受
   "有/无头行"，无头时 `k = None`，`main.py` 回退到 `input2d` 的 `kDiff = 0.01`；
   `run_reference.py` 在运行目录里补上头行使 pyIB2d 能跑。两边初值相同。
4. **上游死文件**：`give_me_source_model.m`（"恒定源" `fs = 0.1`）**没有被任何驱动引用**
   （matIB2d 的源模型是 `please_Find_Source_For_Concentration.m` 里**硬编码**的 1/2/3 分支），
   本 demo 不移植它，并把它从 `ib2d_input/` 里删掉。
5. **`.geo_connect` 未生效**：matIB2d 若有 `*.geo_connect` 会用梯形法真实弧长
   `fsds = 0.5 fs Σ 邻居距离`；pyIB2d 没有 `.geo_connect` 读取器，故两边都用
   Peskin 常数间距 `fsds = fs * ds`（`ds = Lx/(2Nx)`）。这是与上游 MATLAB 的一处
   **有意差异**（与其它 demo 的 ds 约定一致）。
6. **不实现 WENO**：本族里有算例 `advection = 1`（3 阶 WENO）。`WENO_3O.m` 在
   pyIB2d 里没有对应实现，AFSI 也没有；本 demo（`advection = 0`）不受影响，
   相关算例见 demo_684 的说明。
7. **可复现的守恒自检**：每步用 4 点核撒点后校验
   `Σ_grid Fs·dx·dy = Σ_i fs_i·ds`（4 点核的分区单位性，理论上精确）。
   `step 0` 与末步各打印一次。本例（恒定源）两次都给出
   `0.19453125 = 0.1 × 0.0078125 × 249`（相对误差 1.4e-16 / 0）。
   另外单独核对过：把单位浓度场插值到标记点得 `CL ≡ 1`（1e-16）。

## 结果

参考运行 ~6 s、AFSI 运行 ~57 s（各 3200 步、11 帧，`t = 0 .. 0.10003`）。
官方设置 `T = 2`（64000 步）未跑满，原因与差异见"不能验证"。

| 指标（末帧 `t = 0.10003`） | 结果 |
|---|---|
| 标记点偏差 | 6.81e-3 m = **4.36e-1 h**（249 点，橡皮筋正在收缩/折叠） |
| `max\|u\|` AFSI / IB2d | 1.8014 / 1.8169（**8.5e-3** 相对） |
| 速度场相对 $L_2$（窄带外） | 1.27e-1 |
| 窄带外 `max\|du\|` | 6.04e-1 = 6.7e-1 `max\|u\|` |
| 弹簧平均长度 AFSI / IB2d | 0.007204 / 0.007333（1.8e-2 相对） |
| 浓度相对 $L_2$ | **7.75e-2** |
| 浓度 `max\|ΔC\|` | 1.26e-2（峰值 ≈ 0.126 → 约 10%） |
| 峰值 `max C` AFSI / IB2d | 0.12588 / 0.12201（**3.2e-2** 相对） |
| 总量 `∑C` AFSI / IB2d | 74.957 / 74.376（相对差 7.8e-3） |
| 源项撒点守恒（step 0 / 末步） | `Σ Fs dxdy = Σ fs ds = 0.19453125`（相对 1.4e-16 / 0） |

**能验证**：
* **源项撒点与插值**：`Σ Fs dxdy = Σ fs ds` 精确成立（分区单位性，相对误差 0）；
  单位浓度场插值回标记点 `CL ≡ 1`；把两套独立实现（afsic C++ IBMesh ↔
  pyIB2d Supp）在**同一构型、同一 `fs*ds`** 下直接对拍，网格场 `max|ΔFs| = 3.6e-14`、
  相对 $L_2$ = 5.2e-15；——即 4 点核、`supp²` 配对索引与 `ds` 归一化完全一致。
* **浓度场演化**：早期（`t = 0.01003`，收缩刚开始）`relL2(C) = 1.7e-2`、
  `max|ΔC| = 9.8e-4`（峰值 0.033 → 3%）、峰值差 9.9e-3 相对；
  到 `t = 0.1` 退化到 7.75e-2 / 10%，与速度场退化同步（见下）。
* 结构收缩的宏观量（`max|u|` 8.5e-3 相对、弹簧平均长度 1.8e-2）。

**不能验证 / 已知限制**：

1. **`T = 2` 未跑满**（本 demo 默认 `T = 0.1`，3200/64000 步）。原因是橡皮筋按
   `L0 = 0` **收缩折叠**：`max|u|` 在 `t ~ 0.02` 升到 ~3，之后标记点挤成一团，
   结构-流体耦合进入强非线性/敏感阶段，两种代码的 1% 级速度差会被指数式放大
   （`t = 0.1` 时窄带外 `max|du| = 0.67 max|u|`、`relL2(u) = 0.127`，
   标记点偏差 0.44h）。这不是源项的问题（源项只加在被动标量上），
   而是这一族"自由收缩弹簧 + 罚力耦合"的固有敏感度。
2. **源项公式本身**：两侧的 AFSI/参考实现都来自同一份 MATLAB 源码，
   所以比较的是**实现/网格传递**，不是公式。公式的正确性只能靠
   （1）上面的守恒/分区单位性恒等式，（2）与上游 MATLAB 的逐行对照（本文件已列全部公式）。
   本机没有 MATLAB/Octave，无法直接跑上游 matIB2d 做第三方核对。
3. **`.geo_connect` 梯形法弧长**未启用（见移植要点 5），与上游 MATLAB 在此点上不同；
   与其它 demo 的常数 `ds` 约定一致。
4. `advection = 0` 时的**分裂/非分裂**差异：上游 `source > 0` 走非分裂分支，
   本 demo 完全照抄；所以本族的 `relL2(C)` 不能与 demo_476/477（分裂、已验证 1e-11）
   的数值直接比较。
5. **WENO（`advection = 1`）**：pyIB2d 与 AFSI 都没有 3 阶 WENO，本 demo 用不到。

## 运行

```bash
conda activate afsi-dolfinx
python run_reference.py --tend 0.1          # pyIB2d + 源项补丁，~6 s
TFINAL=0.1 python main.py                   # AFSI，~57 s
python compare.py                           # figures/ + compare_table.csv
```

（`--nx N` / `NX=N` 可加密网格，但对本族 64×64 意义不大；
`--print-dump N` / `PRINT_DUMP=N` 改变输出帧数，官方值为 320。）
