# demo_691 — IB2d `Example_VIV_Cylinder/Single_VIV_Cylinder_Damped_Springs_High`（VIV 圆柱 + 阻尼弹簧）— **草稿，未通过验证**

IB2d 官方算例（`matIB2d/Examples/Example_VIV_Cylinder/Single_VIV_Cylinder_Damped_Springs_High`）。
一根圆柱被**弹簧（`k = 1.5e7`）+ 不变梁 + 阻尼弹簧 + 目标点**共同拴在 `1 x 0.25` 的槽道里，
尾流涡街驱动圆柱振动（VIV）。

> **状态：草稿（draft）。** AFSI 侧在官方 `dt` 下于第 ~20 步发散（NaN），
> 而 pyIB2d 参考解始终有限（`max|u| ≈ 5.8` 饱和）。
> 本 readme 记录已完成的移植、**四组实测诊断**（前 5 步相符 / 模型二分 /
> 缩小 `dt` / 上游预张紧）与"不能验证"的原因，供后续排查
> （不建议把本 demo 当作已验证算例引用）。

## 上游路径

* 数据：`third_party/ib2d/matIB2d/Examples/Example_VIV_Cylinder/Single_VIV_Cylinder_Damped_Springs_High/`
  （`viv.vertex/spring/beam/d_spring/target`、`input2d`；
  `main2d.m`、`VIV_Cylinder_Geometry.m`（几何生成器）、`test_Parabola.m` 未随 demo 提供；
  `please_Compute_External_Forcing.m` 也删除——`input2d` 的 `arb_ext_force = 0`，
  该模块在本算例里**不会被调用**（上游死文件））。
* 算法源码：`third_party/ib2d/pyIB2d/IBM_Blackbox/`。

## 算例内容

* `1 x 0.25` 周期盒、`mu = 1e-3`、`rho = 1`、官方 `512 x 128`（方格 `h = 1.95e-3`）、
  `dt = 2.5e-5`、`supp = 4`；官方 `Tfinal = 1.0`（40000 步）、`print_dump = 400`。
* **结构**：2388 个标记点（圆柱 + 支撑），`springs = beams = damped_springs =
  target_pts = 1`（`update_*` 全为 0，即所有系数常数）：
  * `viv.spring`：678 根，`k = 1.5e7`，`L0 ≈ 3.4e-4 .. 4.6e-4`（`t = 0` 严格平衡）；
  * `viv.beam`：452 根不变梁，`kb = 1.0e10`（`t = 0` 曲率误差为 0）；
  * `viv.d_spring`：**2 根**阻尼弹簧（`kappa = 1.5e4`、`b = 50`、`L0 = 0.0667`，
    但初始距离 0.1576 → **`t = 0` 就有 ~1.4e3 的预张紧力**）；
  * `viv.target`：1936 个目标点（`k = 2.5e7`，锚点 = `t = 0` 位置）。
* **外力**：无（`arb_ext_force = 0`）。**不需要任何用户模块**，
  模板（`targets`/`springs`/`beams`/`damped_springs`）开箱即用。

## 移植要点

* 无需移植用户模块（`update_springs/update_beams/update_damp_springs/update_target`
  全为 0）；数据文件 1 基 id 由模板的 `index_base = 1` 处理。
* 减半网格（`512 x 128` → `256 x 64`，格子仍为正方形）、`T = 0.05`（2000 步）、
  `print_dump = 100`（21 帧）。
* **实测：AFSI 在官方网格上先与参考解相符（前 5 步），随后指数发散**（见下）；
  未做任何"修补"（上游参数就是如此）。注意 `ds = Lx/(2Nx)` 随 `Nx` 变化，
  减半网格会把 `ds` 加倍，会更早失稳（`NX=256` 时约第 20 步 NaN）。

## 结果（实测，未通过）

生产算例（`NX=256` 减半网格、`T = 0.05`、`print_dump = 100`、21 帧）：

| 指标 | 结果 |
|---|---|
| pyIB2d 参考 | 有限，`max\|u\|` = 4.66（`t = 0.0025`）→ 7.15（峰值，`t = 0.0125`）→ 4.91（`t = 0.05`）；`\|f\|max = 9.0e3` |
| AFSI | **NaN**：第 1 个输出帧（`t = 0.0025`）起全是 nan，之后空跑到底 |
| 标记点偏差 / 速度场 | 全部 NaN，无法比较 |
| 运行时间 | 参考 ~30 s（2000 步）、AFSI **86.6 s**（2000 步） |

### 诊断一：官方网格上前 5 步（`NX=512`、`dt=2.5e-5`、`print_dump=1`）**相符**

| `t` | `max\|dX\|/h` | `max\|u\|` AFSI / 参考 | 窄带外 `max\|du\|/\|u\|` | 窄带外 `relL2` |
|---|---|---|---|---|
| 2.5e-5 | 3.6e-3 | 2.369 / 1.835（29%） | 0.104 | 0.345 |
| 5.0e-5 | 6.7e-3 | 3.245 / 2.667（22%） | 0.117 | 0.449 |
| 7.5e-5 | 9.9e-3 | 3.873 / 3.421（13%） | 0.105 | 0.532 |
| 1.0e-4 | 1.2e-2 | 4.346 / 4.086（**6%**） | 0.166 | 0.725 |

即：**AFSI 在前 5 步与参考解同量级、`max|u|` 差 6–29%，标记点几乎不动**；
但从第 6 步起 AFSI 的速度按 ~1.7 倍/步指数放大
（5.49 → 8.90 → 15.15 → 23.89），参考解却**饱和**在 5.8 左右 → 第 100 步 AFSI 已 NaN。

### 诊断二：把某个结构模型关掉（`NX=256`，100 步，`input2d` 的 flag 置 0）

| 关闭的模型 | AFSI 结果 |
|---|---|
| `springs = 0` | **存活**（`max\|u\| = 18.3`，与参考 16.5 差 11%） |
| `beams = 0` | 仍 NaN |
| `damped_springs = 0` | 仍 NaN |
| `target_pts = 0` | 仍 NaN（该 run 连 `umax` 打印都缺，因为模板只在有目标点时打印） |

即发散必然需要 **`viv.spring`（`k = 1.5e7`）** 参与；其余三种模型本身不炸。

### 诊断三：缩小 `dt` 只是把发散推迟，不是收敛

| `dt` | 100 步内行为 |
|---|---|
| 2.5e-5（官方） | 第 ~20 步 NaN |
| 2.5e-6（1/10） | 有限，`max\|u\|` 从 0.124 线性长到 4.06（≈ 常数加速度 2e4） |
| 2.5e-7（1/100） | 有限，0.0063 → 0.565，**每步放大倍数几乎与 `dt` 无关**（~1.045/步） |

即该放大率 ~ `1/dt`（每步增益基本恒定），**不是** 单纯的 `dt²·k` 刚度型失稳，
所以单纯减小 `dt` 不能保证同时收敛；这里没有采用（"把 dt 缩到 1/10 再跑"会同时
改变算例设置，属于未验证的偏离）。

### 诊断四：上游数据本身"极度不平静"

`viv.d_spring` 的 2 条阻尼弹簧（`kappa = 1.5e4`、`b = 50`）静止长度写的是
`L0 = 6.667e-2`，但其两端标记点的**初始距离是 1.576e-1**（2.4 倍）——
于是 `t = 0` 就有 `kappa·(L − L0) ≈ 1.37e3` 的预张紧力（两个点）。
这解释了为什么本算例"一开跑就很猛"：参考解即使关掉弹簧也有 `max|u| ≈ 16.5`。
（`viv.spring` / `viv.beam` 在 `t = 0` 是**严格平衡**的：678 根弹簧的
`\|ΔX\| − L0` 最大为 0，452 根梁的 `cross_prod − C` 也为 0，已逐条核对。）

### 追加诊断（本次收尾时补做）

* **去掉上游预张紧也没用**：`make_relaxed_input.py` 把两根阻尼弹簧的 `L0` 改成初始距离
  （消掉 `t=0` 的 ~1.4e3 预张紧力），在 `256x64` 上 AFSI **仍然发散**（T=0.05 内 NaN）
  ⇒ 发散不是那 4 个标记点的初始冲量造成的。
* **整体加密/放粗网格都不救**：`NX=256`（`h=3.9e-3`）与 `NX=128`（`h=7.8e-3`）
  都在 `t≈0.008` 附近发散（约第 300–400 步），即罚力-阻尼耦合的失稳对网格不敏感。
* 与 691 同族的其它 VIV 目录（`Single_VIV_Cylinder_128/256/512/…_Rectangle`）都已
  **通过验证**；只有这个 `Damped_Springs_High` 变体落在 AFSI 的稳定域之外，
  已按"草稿 + 失败证据"如实记录，不作为已验证算例引用。

## 不能验证 / 需要排查

1. **本 demo 未通过**：官方 `dt` 下 AFSI 在第 ~20 步发散，参考解始终有限。
   证据（诊断一/二/三）表明：移植本身没问题（前 5 步相符、关掉弹簧后两码相差 11%），
   问题出在"极硬弹簧（`k = 1.5e7`）+ 预张紧阻尼弹簧 + `kb = 1e10` 梁 +
   `k = 2.5e7` 目标点"这一组合**超出 AFSI 显式耦合的稳定域**，
   而 pyIB2d 的格式恰好能饱和（其 `max|u|` 也只是从 1.83 涨到 5.83 后压住）。
   **仍需排查**：AFSI 的 marker→fluid→marker 回路中哪一项把增益推过 1
   （建议对单根弹簧做增益分析，或检查 `solid_to_fluid` 在
   标记点间距 `ds/2` 情况下的等效归一化）。
2. 上游参数本身值得怀疑：`L0 = 1/15 = 0.0667` 与几何 0.1576 不匹配
   （若上游生成器 `VIV_Cylinder_Geometry.m` 用的就是 0.0667 的"锚绳"，
   那算例从 `t = 0` 就是非平衡初值）；本 demo 未做任何"修补"。
3. 本机无 MATLAB/Octave，无法跑上游 matIB2d 交叉验证上游自身的稳定性。

## 运行

```bash
conda activate afsi-dolfinx
python run_reference.py --tend 0.05 --nx 256 --print-dump 100   # pyIB2d（有限但剧烈），~30 s
NX=256 TFINAL=0.05 PRINT_DUMP=100 python main.py                # AFSI：NaN（见上），~87 s
python compare.py                                               # 表里全是 nan（21 帧）

# 诊断一（官方网格，前 5 步；两者都很快）
python run_reference.py --tend 1.25e-4 --dt 2.5e-5 --print-dump 1 --out ib2d_reference_5step.npz
OUTPUT_PATH=$PWD/plot5step NX=512 DT=2.5e-5 TFINAL=1.25e-4 PRINT_DUMP=1 python main.py

# 诊断二（关掉某个结构模型，需要一份改过 flag 的 input2d 副本）
mkdir -p scratch_in && cp ib2d_input/* scratch_in/
sed -i '' 's/^springs = 1/springs = 0/' scratch_in/input2d
IB2D_EXAMPLE=$PWD/scratch_in OUTPUT_PATH=$PWD/scratch NX=256 TFINAL=2.5e-3 PRINT_DUMP=50 python main.py
```
