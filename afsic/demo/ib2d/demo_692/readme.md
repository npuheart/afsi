# demo_692 — IB2d `Examples_First_Year_Seminar/Corals`（四臂珊瑚摆动 + 目标点驱动）

IB2d 官方算例（`matIB2d/Examples/Examples_First_Year_Seminar/Corals`）。
四个"珊瑚臂"（180 个标记点）被 `update_target` 模块**刚性地绕各自内角摆动**，
同时被 `k = 1e7` 的目标点钉在摆动轨迹上——摆动带动流体。

## 上游路径

* 数据：`third_party/ib2d/matIB2d/Examples/Examples_First_Year_Seminar/Corals/`
  （`corals.vertex`（180 点 + 表头）、`corals.target`（180 点，`k = 1e7`）、
  `input2d`；`update_Target_Point_Positions.m` → 已逐行移植为同名 `.py`；
  `main2d.m`、`Coral_Geometry.m`（几何生成器）、`Make_Your_Plates_and_Input_Files.m`、
  `test_Rotations.m`、`coral2d_left/right_1024.vertex`（另一套几何）未随 demo 提供）。
* 算法源码：`third_party/ib2d/pyIB2d/IBM_Blackbox/`。

## 算例内容

* `1 x 1` 周期盒、`mu = 0.01`、`rho = 1`、官方 `64 x 64`、`dt = 1e-5`、
  `supp = 4`；官方 `Tfinal = 1.0`（100000 步）、`print_dump = 2000`。
  本 demo 用**官方网格与官方 `dt`**，只把 `T` 缩到 **0.05**（5000 步，
  `print_dump = 250`）。
* **结构**：180 点 = 四段各 45 点的"臂"；模块把每段绕其内角旋转
  `±ang1`/`±ang2` 后写入目标点列的 x/y。角度来自上游的两个"角度发生器"：
  ```
  n = ceil((t+dt)/dt);  angVec1 = πdt : πdt : 2π;  angInput1 = angVec1(mod(n, len)) (1 基)
  ang = mod(angInput, 2π/3);  若 ang > π/3:  ang = 2π/3 − ang     （三角波）
  ```
  即幅度 `π/3`、周期 `4/3 s` 的三角波，第二对臂用一半的角速度。

## 移植要点

1. **`update_Target_Point_Positions.py`（逐行移植）**：`give_Me_Oscillatory_Angle`
   （注意 MATLAB 的 `angVec(n1)` 是 **1 基**，`n1 = 0` 时会越界——上游只有
   `n < 2/dt` 时才不触发，本 demo 的 `t ≤ 0.05` 远小于该界限，移植里加了断言把这个
   前提写明）、`rotate_Geometry`、`read_File_In`（**跳过表头**，与 IB2d 其它读取器一致）、
   四段拼接写回 `targets[IDs, 1:2]`。顶点文件在 MATLAB 里每步重读，移植改为缓存。
2. **不能放大 `dt`**：先试过 `dt = 1e-4`（10 倍）→ **两个代码都立刻 NaN**
   （目标点 `k·dt = 1e7 × 1e-4 = 1000`，罚力远超显式稳定限）。因此本 demo
   保留官方 `dt = 1e-5`（`k·dt = 100`，上游本身就在"能跑但不稳"的边缘），
   改用较短的 `T = 0.05`。
3. **网格不可减**：模块用 `end/4` 索引顶点文件（要求 180 点、四等分），
   减 `NX` 不改变点数但会让 64 点的摆动解析变粗；官方本来就是 64²，
   故保持官方网格，只缩 `T`。
4. 与 demo_683（另一个珊瑚算例）的区别：683 是**多项式形状插值**（`cval.mat`），
   这里是**刚体旋转**；两者都验证（683 的结论不同，见其 readme）。

## 结果

参考运行 ~9 s、AFSI 运行 ~69 s（各 5000 步、20 帧，`t = 0 .. 0.04751`）。

| 指标（末帧 `t = 0.04751`） | 结果 |
|---|---|
| 标记点偏差（180 点，摆动臂被 `k = 1e7` 钉住） | 5.50e-5 m = **3.5e-3 h** |
| 初始构型自检 `max\|X_target − X_anchor\|`（AFSI 打印，`t = 0`） | **1.11e-16 m**（轨迹在 `t = 0` 与顶点文件逐位一致） |
| `max\|u\|` AFSI / IB2d | 9.548e-1 / 1.228（**2.2e-1** 相对） |
| 速度场相对 $L_2$（窄带外） | 3.48e-1 |
| 窄带外 `max\|du\|` | 4.07e-1 = 6.7e-1 `max\|u\|` |
| 外力场（窄带外） | `max\|df\| = 1.0e2`（`\|f\|max = 3.4e3`） |
| 早期帧 `t = 0.01001` | 标记点 2.5e-3 h、`max\|u\|` 1.011/1.457（31%）、`relL2` 0.464 |

**能验证**：
* **摆动目标轨迹 + 罚力跟踪**：`t = 0` 时目标点与顶点文件一致到 **1.11e-16**，
  5000 步后 180 点偏差仅 **3.5e-3 h**（与 demo_683 的 7.2e-3 h 同量级）——
  说明 `update_Target_Point_Positions.py`
  的角度发生器（含 1 基索引/三角波折叠）、`rotate_Geometry`、四段拼接、
  顶点文件表头处理都正确；
* 流场同量级（`max|u|` 22%）——本例是"`k·dt = 100` 的硬罚力"驱动，
  两套时间推进的差异可见，但结构本身（本算例的主要内容）逐位级吻合。

**不能验证 / 已知限制**：

1. **`T` 只跑 1/20**（5000/100000 步）：官方 1 s 需要 AFSI ~23 分钟（实测 ~13.8 ms/步）。
   三角波周期 `4/3 s` 只覆盖了 3.75%。
2. **速度场只做半定量比较**（`relL2 = 0.35`，幅值差 22%）：
   目标点刚度 `k = 1e7` 使 `k·dt = 100 ≫ 1`（上游如此），
   罚力的显式积分处于"能跑但不稳"的边缘，两套投影/时间推进的响应因此不同；
   结构偏差仍只有 3.5e-3 h。
3. **不能放大 `dt`**：实测 10 倍 `dt` 会让**两个**代码立刻 NaN，故未做加速。
4. 本机无 MATLAB/Octave，无法跑上游 matIB2d 做第三方核对。

## 运行

```bash
conda activate afsi-dolfinx
python run_reference.py --tend 0.05 --print-dump 250   # pyIB2d，~9 s
TFINAL=0.05 PRINT_DUMP=250 python main.py              # AFSI，~69 s
python compare.py                                      # figures/ + compare_table.csv
```
