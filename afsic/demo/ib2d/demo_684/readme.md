# demo_684 — IB2d `Example_Concentration_Dynamics/Moving_Cylinders_64`（平移双圆柱 + 汇项）

IB2d 官方算例（`matIB2d/Examples/Example_Concentration_Dynamics/Moving_Cylinders_64`）。
浓度源项家族的最后一个成员，也是唯一同时用到 **`nonInvariant_beams`（非不变梁）+
`target_pts`（指定轨迹的驱动）+ `source = 3`（汇）** 的算例：
两个近似刚性的圆柱横向扫过一条竖直浓度带，同时把带体"吸掉"。
源项机制、上游缺口与验证恒等式见 **demo_680 的 readme**，
目标轨迹驱动的写法见 **demo_683 的 readme**。

## 上游路径

* 数据：`third_party/ib2d/matIB2d/Examples/Example_Concentration_Dynamics/Moving_Cylinders_64/`
  （`cylinders.vertex/spring/nonInv_beam/target/concentration`、
  `State_A.pts`、`State_B.pts`、`input2d`、`update_Target_Point_Positions.m`
  → 已逐行移植为同名 `.py`；`main2d.m`、`Make_Cylinder_Geometry.m`（几何生成器）、
  `interp_Function_Coeffs.m`（系数设计脚本）、`give_me_source_model.m`（上游死文件）
  未随 demo 提供）。
* 算法源码：`matIB2d/IBM_Blackbox/please_Find_Source_For_Concentration.m`、
  `please_Update_Adv_Diff_Concentration_Source.m`；`pyIB2d/IBM_Blackbox/`。

## 算例内容

* `1 x 1` 周期盒、`mu = 48`、`rho = 1000`（`nu = 0.048`）、`64 x 64`、
  `supp = 4`、`ds = 0.0078125`。
* **结构**：80 个标记点 = 两个 40 点圆环（`c = 0.07/0.93, y = 0.16/0.84`，
  半径 0.04）；120 根弹簧 `k = 1e6`（静止长度 `L0 ≈ 7.8e-3`）、
  80 根**非不变梁** `kb = 1e9`（`C = p1 + p3 − 2p2`，保持圆环刚度）、
  80 个目标点 `k = 5e6`。
* **驱动**：目标点按 `update_Target_Point_Positions.m` 的分段三次缓动在
  `State_A`（`x = 0.07`）与 `State_B`（`x = 0.97`）之间往返：
  `t1 = 0.5 s` 单程、`rem(t, 1 s)` 取模、`p1 = 0.05` 加速段 + 巡航段 + 减速段
  （系数 `a0..c3` 是 12×12 Hermite 系统的解，由设计脚本
  `interp_Function_Coeffs.m` 求出后硬编码在模块里）。`cylinders.vertex`
  与 `State_A.pts` **逐位相同**（已核对），所以初始构型就是状态 A。
* **浓度**：`kDiff = 7.2e-5`、`advection = 1`（3 阶 WENO，见下）、
  **`source = 3`（汇/反应）**、`k_source = −1`、`c_inf = 1`（模型 3 不用），
  `C0` 是 `input2d` 配套的竖直浓度带（非零，峰值 ≈ 0.998，
  `cylinders.concentration` 无 `kDiff` 头行）。

## 移植要点

1. **`update_Target_Point_Positions.py`**：`read_In_State`（`State_*.pts` **无表头**，
   两列 x/y，80 点）、`rem` 取模、分段三次 `g(tTilde)`、`A + g(B−A)` /
   `B + g(A−B)`。上游每一步都重读两个 `.pts`（且两个分支各读一次），
   移植改为一次加载 + 缓存（数据只读）。
   `interp_Function_Coeffs.m` **不是**运行期模块（它只解 12×12 系统并画图，
   没有任何驱动调用它），故不移植，系数按源码硬编码。
2. **源项 + 汇项**：与 demo_680–683 共用同一份
   `ib2d_input/please_Find_Source_For_Concentration.py`（参考侧补丁）
   与 `main.py` 里的 AFSI 侧实现；上游 `source > 0` 走**非分裂** upwind。
3. **新增发现并修复的移植陷阱（本 demo 独有）**：
   `IBInterpolation.solid_to_fluid` 撒点是在**非周期**的 Q1 网格上做的，
   当某个标记点的 4 点 stencil 跨过 `x = Lx` / `y = Ly` 时会落到"镜像"节点上。
   力场在模板里靠 `solver.periodic_fold` 折回，而**源场一开始没有折**，
   于是圆柱扫到 `x ≳ 0.9688`（`2h` 邻域）时源项会**丢质量**
   （末步守恒自检从 1e-16 变成 4.9e-4；逐点实测最差单点丢 0.89%）。
   已修：`source_field` 在 `solid_to_fluid` 之后同样调用 `solver.periodic_fold`，
   之后再用"跳过镜像节点"的掩码取回 `(Ny, Nx)` 网格。
   （demo_680–683 的标记点都不靠近边界，所以数值不变。）
4. **`.pts` 文件要进 pyIB2d 运行目录**（裸 `read_In_State`），
   `run_reference.py` 的拷贝列表里加了 `*.pts`。
5. **WENO 偏差**：`input2d` 写了 `advection = 1`（3 阶 WENO），但 `WENO_3O.m`
   在 pyIB2d / AFSI 里**都没有**实现，两边都跑一阶 upwind（本 demo 明确打印警告）。
   与上游 MATLAB 的差异是有意的、且两侧一致。
6. **上游怪癖：`source` 键重复**。`input2d` 的 `Concentration_Info` 里
   先有 `source = 1`（旧注释 `0/1`）、后有 `source = 3`（新注释 `0/1/2/3`）。
   pyIB2d 完全不读它；本移植的解析器取**最后一次**（=3，与官方
   `Example_Concentration_Dynamics/README` 对该算例"absorbing"的描述一致）；
   MATLAB 的 `please_Initialize_Simulation` 用
   `Con_Input{find(strcmp(...,'source')),2}`，遇到重复键会取到 2 个值
   （很可能直接报错）——上游这份 `input2d` 本身有歧义，已在 readme 记录。
7. **`.geo_connect`** 未启用（pyIB2d 无读取器），两侧都用 `fsds = fs·ds`。
8. 每步校验 `Σ_grid Fs·dx·dy = Σ_i fs_i·ds`：`step 0` 与末步都打印
   （见下表，末步正好检验了上面的周期折回修复）。

## 结果

参考运行 ~10 s、AFSI 运行 ~30 s（各 2000 步、21 帧，`t = 0 .. 0.50025`）。
本 demo 采用 `dt = 2.5e-4`（官方 `1.25e-5` 的 20 倍）、`T = 0.5`
（官方 2.000001，即两个完整往返；本 demo 覆盖一个完整 A→B 行程，`g(1) = 0.997`）、
`print_dump = 100`（官方 800）。

| 指标（末帧 `t = 0.50025`） | 结果 |
|---|---|
| 标记点偏差（80 点，被目标点拖动的圆柱） | 3.02e-4 m = **1.93e-2 h** |
| `max\|u\|` AFSI / IB2d | 2.9333 / 2.9195（**4.7e-3** 相对） |
| 速度场相对 $L_2$（窄带外） | 3.38e-2 |
| 窄带外 `max\|du\|` | 2.75e-1 = 9.1e-1 `max\|u\|` |
| 弹簧平均长度 AFSI / IB2d | 0.018472 / 0.018500（1.5e-3 相对） |
| 浓度相对 $L_2$ | **3.76e-2** |
| 浓度 `max\|ΔC\|` | 4.99e-2（峰值 0.64 → 7.8%） |
| 峰值 `max C` AFSI / IB2d | 0.64469 / 0.64138（**5.2e-3** 相对） |
| 总量 `∑C` AFSI / IB2d | 504.65 / 503.34（相对差 2.6e-3） |
| 源项撒点守恒（step 0 / 末步） | `Σ Fs dxdy = Σ fs ds = −3.032749506448138e-05`（rel 1.1e-16）/ `−0.009498358012098855`（rel 0） |
| 中期帧 `t = 0.25025`（圆柱到达 `g = 0.5`） | 标记点偏差 ≈ 2.8e-2 h、`relL2(u)` ≈ 2.9e-2、`relL2(C)` ≈ 3.9e-2 |

**能验证**：
* **指定轨迹驱动的结构**：80 个标记在 2000 步后只偏离参考 **1.9e-2 h**，
  弹簧平均长度差 1.5e-3（说明 `update_Target_Point_Positions.py` 的
  分段三次缓动 + `State_*.pts` 读取正确，且 `cylinders.vertex ≡ State_A`）；
* **汇项（`fs = k·CL`）与网格传递**：`Σ Fs dxdy = Σ fs ds` 在 `step 0` 与
  **末步**都精确成立（1.1e-16 / 0），前者等于解析值
  `k_source·Σ_i CL_i·ds`；这一条正是"周期折回"修复的回归测试
  （修复前末步 rel = 4.9e-4）；
* **浓度带被两圆柱扫掠/吸收的演化**：末帧 `relL2(C) = 3.8e-2`、
  峰值差 5.2e-3、总量守恒到 2.6e-3；
* 流场（`max|u|` 4.7e-3 相对、`relL2(u) = 3.4e-2`）——比 demo_683 好，
  因为这里流动由"给定轨迹的刚性边界"驱动，而非自由变形边界。

**不能验证 / 已知限制**：

1. **不是官方设置**：`dt` 放大 20 倍、`T` 取 0.5（官方 2.000001）、
   `print_dump` 改小。`dt` 放大是两侧**一致**的（`nu·dt/dx² = 0.049`、
   平流 CFL ≈ 0.046，都远低于稳定限），但时间离散误差比官方大；
   官方 `T` 需要 16 万步（AFSI ≈ 40 分钟），超出本任务的单次运行预算。
2. **WENO 未实现**：官方指定 3 阶 WENO 平流，pyIB2d 与 AFSI 都只有
   一阶 upwind；两侧一致但不等于上游格式。`relL2(C) = 3.8e-2` 里包含
   这个格式差异（以及速度场差异）。
3. **源项公式本身**没有第三方核对（两侧同源一份 MATLAB；本机无 MATLAB/Octave）。
4. **`.geo_connect` 梯形法弧长**未启用。
5. **上游 `input2d` 的 `source` 重复键歧义**（见移植要点 6）：
   本 demo 取 3（汇），若上游按"第一个"取 1（恒定源）则源项形式不同；
   无法在没有 MATLAB 的情况下判定上游实际行为。
6. `nonInvariant_beams` 的非不变梁力与 pyIB2d 的 `update_nonInv_Beams` 调用
   怪癖（`update_nonInv_beams = 0`，本例不触发更新分支）见模板注释与
   demo_573 系列 readme。

## 运行

```bash
conda activate afsi-dolfinx
python run_reference.py --tend 0.5 --dt 2.5e-4 --print-dump 100   # ~10 s
TFINAL=0.5 DT=2.5e-4 PRINT_DUMP=100 python main.py                # ~30 s
python compare.py                                                 # figures/ + compare_table.csv
```
