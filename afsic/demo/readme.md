# afsic Demo 算例集

本目录为 afsic 的演示算例集合，统一采用**浸没边界流固耦合（IB-FSI）**架构：
固定笛卡尔流体网格（Chorin 投影求解 NS）+ 拉格朗日固体网格（Neo-Hookean / FRH /
SVK 等本构），通过 `IBMesh` / `IBInterpolation`（3D 用 `IBMesh3D` / `IBInterpolation3D`）
双向耦合（流体速度插值到固体、固体力插值回流体）。

## 二维算例

| Demo | 名称 | 说明 |
|---|---|---|
| `demo_336` | 方腔驱动圆盘 | 方腔驱动流中的圆盘随流（IB 与直接强迫多算法对比） |
| `demo_339` | 圆柱绕流四算例 | 1-无圆柱 / 2-贴体网格 / 3-IBFE / 4-multi-direct forcing 四种方法对比 |
| `demo_340` | 二维理想瓣膜（FRH） | 各向异性纤维增强（45°/60°/75°）双瓣膜，生理正弦入口，含位移对比图 |
| `demo_341` | 3D 方腔驱动圆球 | 3D 方腔中圆球随流；t=1 s 沿三条中线插值对比（纯 NS vs FSI，多网格密度） |
| `demo_343` | 圆盘随流通过二维理想瓣膜 | 上/下瓣膜（下瓣膜更硬 10×）+ 两个软圆盘随流；`CIRCLE=1`（有圆盘）/`CIRCLE=0`（无圆盘对照），中心线对比图 |
| `demo_400` | 2D 乌龟 FSI | 头尾固定、周期性压力驱动（follower pressure 沿脊柱方向）、四肢随流摆动 |
| `demo_421` | 鱼游动 | DFIBMFoam `CircularFishSwimming` 的 FEniCSx 移植（鱼体几何 + 运动学） |
| `demo_423` | 浸没各向异性圆环静态平衡 | 方腔内不可压缩流体 + 周向纤维增强圆环；解析压力解验证，含收敛误差输出 |

## 三维算例

| Demo | 名称 | 说明 |
|---|---|---|
| `demo_337` | 理想左心室舒张/收缩 | IB-FE，被动 Neo-Hookean 心室 + 3D Chorin 流体，生理心室内压加载 |
| `demo_401` | 3D 精子 IB-FSI | 精子固体几何与网格生成（球头 + 三段鞭毛，物理面分组），IB-FSI 前处理 |
| `demo_402` | Turek FSI2 基准 | 2D 通道圆柱 + 柔性旗（SVK），Re=100，1 m/s 抛物线入口（定性复现） |
| `demo_403` | 3D 横流弹性板 | Beam in Cross Flow（Tuković 2018 §4.5），Re=40，对称半域，底部固定板 |
| `demo_405` | 3D 血管壁 FSI | 血管壁（楔形/四面体网格）+ 瓣膜，正弦入口来流，IB 3D 耦合 |

> `demo_422` 目前为空目录（占位）。

## IB2d 移植算例（`ib2d/`）

把 [IB2d](https://github.com/nickabattista/ib2d)（Battista 等的浸没边界教学/研究代码）
的**全部官方算例**移植成自包含 demo：`ib2d_input/` 保留上游数据文件逐字不变（外加把该算例
自己的 `*.m` 用户模块逐行移植成同名 `.py`），`main.py` 是 AFSI 侧实现，`run_reference.py`
用 pyIB2d（同一份 `IBM_Blackbox`）跑参考解并转成 `.npz`，`compare.py` 做**帧对齐**的逐点数值
对比（并出图 + CSV）。参考与 AFSI 通常**共用同一个被移植的用户模块**（`update_Springs.py`、
`please_Compute_External_Forcing.py`、`coagulation_model.py` …），因此对比只反映
求解器与 IB 机制的差别。

覆盖情况：`matIB2d/Examples` 的 183 个含 `input2d` 的目录全部有对应 demo
（部分目录是同一几何的**雷诺数/分辨率阶梯**，各自成 demo 但统一用缩减网格运行，
并在各自 readme 里写明"官方设置 vs 本 demo 设置"）。共 **180 个 demo：`demo_444`–`demo_703`**（其中 3 个标注为 ⚠️ 草稿：`demo_675` Planets 是**上游本身损坏**、`demo_691` 与 `demo_693` 是 AFSI 在官方设置下超出稳定域的变体，三者的 readme 都给出了可复现的失败证据）。

### 已实现的 IB2d 机制

弹簧（含非线性 α 律）、阻尼弹簧、不变梁、**非不变梁**、目标点（含 `update_target`）、
**质量点/幽灵粒子 + 重力**（`mass_pts`，`dt/2` 半步 leapfrog 序列）、porous 滑移、
poroelastic、被动标量 `concentration`（IB2d 自己的分裂 upwind 对流-扩散，验证到 1e-11）、
**被动标量源项**（`source` / `k_source` / `c_inf` 三种模型：
`please_Find_Source_For_Concentration` + 未分裂 upwind 更新，pyIB2d 完全没有这一支，
在参考侧按 matIB2d 逐行补齐）、
**Boussinesq 浮力**、凝血 `coagulation`、**FV_LT 肌肉**、**3 元件肌肉**、
**电生理 `electro_phys`**、任意外力 `arb_ext_force`、示踪粒子 `tracers`、
Brinkman 阻力项、per-example 用户模块（`update_Springs` / `update_Beams` /
`update_nonInv_Beams` / `update_Target_Point_Positions` / `update_Damped_Springs`）、
以及 **restart 协议**（AFSI 侧自实现，与单次连续运行差 8e-15 m）。

### 验证方式与已知差异（如实记录）

* **帧对齐**：pyIB2d 在 `cter % print_dump == 0 and cter >= print_dump` 时 dump，
  即"第 `cter+1` 步结束时的状态、却按 `cter·dt` 打时间戳"（上游标注怪癖）；
  `main.py` 按同样步号 dump 并打真实时间，`compare.py` 据此配对帧。
* **强验证**的例子：纯扩散标量（demo_476，1e-11）、质量点轨迹（demo_703，1.75e-6 h）、
  restart（demo_620/621，8e-15 m）、凝血键（demo_625/626，23/23 与 3/3 完全一致）、
  Boussinesq + 标量（demo_702，相对 L2 2e-5）、Brinkman（demo_475，窗口 0.62%）。
* **不可比的部分**也逐条写在每个 readme 的"不能验证"里，典型来源：
  ① 低 Re 自推进流场对分辨率极敏感（水母阶梯）；
  ② 被驱动的软目标点壁在两种离散下"渗漏"不同（分叉动脉、阻抗泵）；
  ③ 上游设置本身在 CFL/稳定边缘（demo_570 的 `ω·dt≈1.96`、demo_474 的官方 dt）；
  ④ 上游算例自己标注为失败尝试（`Failed_Attempts/*`）。
* **缩减设置**：官方网格常达 `1024×1024` 甚至 `384×1280`、步数上万；为能在普通机器上跑完
  且不撑爆内存，每个 demo 都在 readme 里写明"官方 vs 本 demo"的设置，并只缩减网格与
  时间窗口（结构与加载是物理量，不随网格改变）。若需要恢复官方设置，改
  `NX/NY/TFINAL/PRINT_DUMP` 环境变量即可（`run_reference.py --nx/--tend/--print-dump`）。
* **上游 bug（已在对应 readme 与脚本注释里写明，并在参考侧打补丁）**：
  ① pyIB2d 的 tracer 移动调用少传 4 个参数（`TypeError`）；
  ② `read_In_Brinkman_Info` 返回未定义名（所有 brinkman 算例 `NameError`）；
  ③ Brinkman 分支重复 `chdir('viz_IB2d')`；
  ④ `IBM_Driver.py:711` 把 `beams_info` 误传给 `update_nonInv_Beams`；
  ⑤ 3 元件肌肉分支调用 `give_3_Element_Muscle_Activation` 却没 import（`NameError`）；
  ⑥ `Example_Boussinesq/Falling_Sphere` **缺 `boussinesq.mesh`**（生成脚本也从不写它），
     照上游跑必然 `FileNotFoundError`——本仓库用 `make_input.py` 重建全域掩码；
  ⑦ `Kelvin_Helmholtz_Qualitative_2` **完全不带数据文件**（需先跑 MATLAB 生成脚本）——
     本仓库用移植版 `make_input.py` 生成；
  ⑧ pyIB2d 的 `please_Form_Boussinesq_Forcing_Terms` 掩码索引未做 1-based→0-based；
  ⑨ `Jellyfish_Material_512x512`（`kb·dt/mu≈1e9`）两个代码第一步都 NaN；
  ⑩ `Rubberband_with_Beams` 的 `nonInv_beams` 拼写错误、Lymphangion 的 `.muscle` 里
     `FV_LT_muscle = 0`、`help_Me_Restart.m` 写死作者桌面绝对路径、pyIB2d 完全没有
     restart 实现、质量点速度不参与 restart；
  ⑪ `Example_Test_Restart_Protocol` 之外，IB2d 在 demo_620 里到 t=1.5 泄漏 26% 面积（晚期轨迹不可比）；
  ⑫ 被动标量的源项（`source = 1/2/3`）**在 pyIB2d 里完全没实现**（`IBM_Driver.py` 从不读
     `source/k_source/c_inf`，总是调用分裂 upwind 更新）——demo_680–684 在参考侧用
     `ib2d_input/please_Find_Source_For_Concentration.py` 逐行补齐 matIB2d 的
     `please_Find_Source_For_Concentration.m` 与未分裂更新；
  ⑬ `Example_Concentration_Dynamics/*` 的 `.concentration` 文件**没有表头**（pyIB2d 会读错），
     且 `Moving_Cylinders_64/input2d` 里 `source` 键**重复出现**（后一个覆盖前一个）；
  ⑭ `Example_Meshmerize_Me/*` 的 `please_Compute_External_Forcing.m` 里"首次调用"守卫恒真
     （每步都重新搜索窗口）、`give_me_source_model.m` / `test_Parabola.m` 等是死文件；
  ⑮ `Examples_First_Year_Seminar/Plates`：`plates.spring` 声明 307 根却写了 308 行，
     其中最后一根引用不存在的点 207（生成器循环 `s=1:N` 多写一根）——pyIB2d 自带的读取器
     直接崩；`update_Springs.m` 又把横向弹簧静止长度写死 `d = 1.0`，而几何间距只有 0.2
     （两个代码都 NaN）；
  ⑯ `Example_VIV_Cylinder/Single_VIV_Cylinder_Damped_Springs_High` 的 2 根阻尼弹簧
     `L0 = 0.0667` 与初始几何距离 0.1576 相差 2.4 倍（`t = 0` 就有 ~1.4e3 的预张紧力，
     算例一开始就极猛）；`update_Target_Point_Positions.m` 里 `angVec(0)` 存在
     MATLAB 1 基索引越界隐患。

### 全部 demo 一览

| Demo | 上游算例 | 名称 |
|---|---|---|
| `ib2d/demo_444` | — | IB2d 弹性圆环 |
| `ib2d/demo_445` | — | IB2d 水母游动 |
| `ib2d/demo_446` | — | IB2d "Wobbly Beam" |
| `ib2d/demo_447` | — | IB2d "Rubberband with Damped Springs" |
| `ib2d/demo_448` | — | IB2d "HeartTube Muscle" |
| `ib2d/demo_449` | — | IB2d "Wobbly Non-Invariant Beam" |
| `ib2d/demo_450` | — | Gravity Cellular Race |
| `ib2d/demo_451` | — | Tracers in Impedance Pump |
| `ib2d/demo_453` | — | Single Porous Rubberband |
| `ib2d/demo_454` | — | Poroelastic Rubberband |
| `ib2d/demo_455` | — | Rubberband with Beams |
| `ib2d/demo_456` | — | Concentration Diffusion |
| `ib2d/demo_457` | — | User-Defined Fiber Model |
| `ib2d/demo_458` | — | Thrips |
| `ib2d/demo_459` | — | IB2d `Moving_Rubberband` |
| `ib2d/demo_460` | — | IB2d `Pulsing_Heart` |
| `ib2d/demo_461` | — | IB2d `Single_Coral` |
| `ib2d/demo_462` | — | IB2d `NL_Rubberband_With_Springs` |
| `ib2d/demo_463` | — | IB2d `Compare_Porous_NonPorous_Rubberbands` |
| `ib2d/demo_464` | — | IB2d `Update_Springs_Peristaltic_Pumping` |
| `ib2d/demo_465` | — | IB2d `Moving_Circle` |
| `ib2d/demo_466` | — | IB2d `Lymphangion` |
| `ib2d/demo_467` | — | IB2d 任意外力接口 `arb_ext_force` |
| `ib2d/demo_468` | Example_1_Wall_Flow_Around_Cylinder | IB2d `Example_1_Wall_Flow_Around_Cylinder` |
| `ib2d/demo_469` | Example_Middle_Channel_Flow | IB2d `Example_Middle_Channel_Flow` |
| `ib2d/demo_470` | Example_Hole In_Channel | IB2d `Example_Hole In_Channel` |
| `ib2d/demo_471` | Example_Aneurysm | IB2d `Example_Aneurysm` |
| `ib2d/demo_472` | Example_Flow_Around_Cylinder | IB2d `Example_Flow_Around_Cylinder` |
| `ib2d/demo_473` | — | IB2d `Tracers_In_Channel_Flow` |
| `ib2d/demo_474` | — | IB2d `Tracers_In_Bifurcating_Artery` |
| `ib2d/demo_475` | Example_Brinkman_Channel | IB2d `Example_Brinkman_Channel` |
| `ib2d/demo_476` | Example_Concentration_Gradients/Pure_Diffusion | IB2d `Example_Concentration_Gradients/Pure_Diffusion` |
| `ib2d/demo_477` | Example_Concentration_Gradients/Bifurcating_Artery | IB2d `Example_Concentration_Gradients/Bifurcating_Artery` |
| `ib2d/demo_478` | Example_Concentration_Gradients/Impedance_Pumping_Tube | IB2d `Example_Concentration_Gradients/Impedance_Pumping_Tube` |
| `ib2d/demo_480` | — | IB2d `Hoover_Jellyfish` |
| `ib2d/demo_481` | — | IB2d `Two_Corals` |
| `ib2d/demo_482` | Examples_Education/Interpolation/Swimmer | IB2d `Examples_Education/Interpolation/Swimmer` |
| `ib2d/demo_483` | Example_Porous_Rubberbands/Compare_Porous_NonPorous_Rubberbands_256x256 | IB2d `Example_Porous_Rubberbands/Compare_Porous_NonPorous_Rubberbands_256x256` |
| `ib2d/demo_484` | Example_Meshmerize_Me | IB2d `Example_Meshmerize_Me` |
| `ib2d/demo_485` | Example_Meshmerize_Me/Flow_Around_Starfish | IB2d `Example_Meshmerize_Me/Flow_Around_Starfish` |
| `ib2d/demo_486` | Example_Meshmerize_Me/old | IB2d `Example_Meshmerize_Me/old` |
| `ib2d/demo_487` | Example_Meshmerize_Me/Insect_Wing_Veins | IB2d `Example_Meshmerize_Me/Insect_Wing_Veins` |
| `ib2d/demo_488` | Example_Meshmerize_Me/Starfish | IB2d `Example_Meshmerize_Me/Starfish` |
| `ib2d/demo_489` | Example_Meshmerize_Me/Dragonfly_Wing | IB2d `Example_Meshmerize_Me/Dragonfly_Wing` |
| `ib2d/demo_490` | Example_HeartTube/Insect_Heart | IB2d `Example_HeartTube/Insect_Heart` |
| `ib2d/demo_491` | Example_HeartTube/Insect_Heart/Valveless_Model | IB2d `Example_HeartTube/Insect_Heart/Valveless_Model` |
| `ib2d/demo_492` | Example_HeartTube/Sea_Spiders | IB2d `Example_HeartTube/Sea_Spiders` |
| `ib2d/demo_493` | Example_Wobbly_Beam/Beam_256 | IB2d `Example_Wobbly_Beam/Beam_256` |
| `ib2d/demo_494` | Examples_First_Year_Seminar/Leaf | IB2d `Examples_First_Year_Seminar/Leaf` |
| `ib2d/demo_495` | Examples_First_Year_Seminar/Flexible Bubbles | IB2d `Examples_First_Year_Seminar/Flexible Bubbles` |
| `ib2d/demo_496` | Examples_First_Year_Seminar/Flow_Around_Object | IB2d `Examples_First_Year_Seminar/Flow_Around_Object` |
| `ib2d/demo_497` | Examples_First_Year_Seminar/Jellyfish_Oscillator | IB2d `Examples_First_Year_Seminar/Jellyfish_Oscillator` |
| `ib2d/demo_499` | Examples_First_Year_Seminar/Beam_and_Cells | IB2d `Examples_First_Year_Seminar/Beam_and_Cells` |
| `ib2d/demo_500` | Example_VIV_Cylinder/Single_VIV_Cylinder_128 | IB2d `Example_VIV_Cylinder/Single_VIV_Cylinder_128` |
| `ib2d/demo_501` | Example_VIV_Cylinder/Single_VIV_Cylinder_256_Rectangle | IB2d `Example_VIV_Cylinder/Single_VIV_Cylinder_256_Rectangle` |
| `ib2d/demo_502` | Example_VIV_Cylinder/Single_VIV_Cylinder_256 | IB2d `Example_VIV_Cylinder/Single_VIV_Cylinder_256` |
| `ib2d/demo_503` | Example_VIV_Cylinder/Single_VIV_Cylinder_512 | IB2d `Example_VIV_Cylinder/Single_VIV_Cylinder_512` |
| `ib2d/demo_504` | Example_VIV_Cylinder/Single_VIV_Cylinder_Damped_Springs_256 | IB2d `Example_VIV_Cylinder/Single_VIV_Cylinder_Damped_Springs_256` |
| `ib2d/demo_506` | Example_VIV_Cylinder/Single_VIV_Cylinder_512_Rectangle | IB2d `Example_VIV_Cylinder/Single_VIV_Cylinder_512_Rectangle` |
| `ib2d/demo_520` | Example_Channel_Flow/Example_Poroelastic_Channel_Flow | IB2d `Example_Channel_Flow/Example_Poroelastic_Channel_Flow` |
| `ib2d/demo_521` | — | IB2d 水母收敛阶梯 `Re75/Res_192_240x72` |
| `ib2d/demo_522` | — | IB2d 水母收敛阶梯 `Re75/Res_64_80x24` |
| `ib2d/demo_523` | — | IB2d 水母收敛阶梯 `Re75/Res_32_40x12` |
| `ib2d/demo_524` | — | IB2d 水母收敛阶梯 `Re75/Res_128_160x48` |
| `ib2d/demo_525` | — | IB2d 水母收敛阶梯 `Re75/Res_1024_1280x384` |
| `ib2d/demo_526` | — | IB2d 水母收敛阶梯 `Re75/Res_96_120x36` |
| `ib2d/demo_527` | — | IB2d 水母收敛阶梯 `Re75/Res_768_960x288` |
| `ib2d/demo_528` | — | IB2d 水母收敛阶梯 `Re75/Res_512_640x192` |
| `ib2d/demo_529` | — | IB2d 水母收敛阶梯 `Re75/Res_48_60x18` |
| `ib2d/demo_530` | — | IB2d 水母收敛阶梯 `Re75/Res_384_480x144` |
| `ib2d/demo_531` | — | IB2d 水母收敛阶梯 `Re75/Res_256_320x96` |
| `ib2d/demo_532` | — | IB2d 水母收敛阶梯 `Re300/Res_192_240x72` |
| `ib2d/demo_533` | — | IB2d 水母收敛阶梯 `Re300/Res_64_80x24` |
| `ib2d/demo_534` | — | IB2d 水母收敛阶梯 `Re300/Res_32_40x12` |
| `ib2d/demo_535` | — | IB2d 水母收敛阶梯 `Re300/Res_128_160x48` |
| `ib2d/demo_536` | — | IB2d 水母收敛阶梯 `Re300/Res_1024_1280x384` |
| `ib2d/demo_537` | — | IB2d 水母收敛阶梯 `Re300/Res_96_120x36` |
| `ib2d/demo_538` | — | IB2d 水母收敛阶梯 `Re300/Res_768_960x288` |
| `ib2d/demo_539` | — | IB2d 水母收敛阶梯 `Re300/Res_512_640x192` |
| `ib2d/demo_540` | — | IB2d 水母收敛阶梯 `Re300/Res_48_60x18` |
| `ib2d/demo_541` | — | IB2d 水母收敛阶梯 `Re300/Res_384_480x144` |
| `ib2d/demo_542` | — | IB2d 水母收敛阶梯 `Re300/Res_256_320x96` |
| `ib2d/demo_543` | — | IB2d 水母收敛阶梯 `Re37pt5/Res_192_240x72` |
| `ib2d/demo_544` | — | IB2d 水母收敛阶梯 `Re37pt5/Res_64_80x24` |
| `ib2d/demo_545` | — | IB2d 水母收敛阶梯 `Re37pt5/Res_32_40x12` |
| `ib2d/demo_546` | — | IB2d 水母收敛阶梯 `Re37pt5/Res_128_160x48` |
| `ib2d/demo_547` | — | IB2d 水母收敛阶梯 `Re37pt5/Res_1024_1280x384` |
| `ib2d/demo_560` | Examples_Education/Convergence/Jellyfish/Simulation_Skeletons/Re37pt5/Res_96_120x36 | IB2d `Examples_Education/Convergence/Jellyfish/Simulation_Skeletons/Re37pt5/Res_96_120x36` |
| `ib2d/demo_561` | Examples_Education/Convergence/Jellyfish/Simulation_Skeletons/Re37pt5/Res_768_960x288 | IB2d `Examples_Education/Convergence/Jellyfish/Simulation_Skeletons/Re37pt5/Res_768_960x288` |
| `ib2d/demo_562` | Examples_Education/Convergence/Jellyfish/Simulation_Skeletons/Re37pt5/Res_512_640x192 | IB2d `Examples_Education/Convergence/Jellyfish/Simulation_Skeletons/Re37pt5/Res_512_640x192` |
| `ib2d/demo_563` | Examples_Education/Convergence/Jellyfish/Simulation_Skeletons/Re37pt5/Res_48_60x18 | IB2d `Examples_Education/Convergence/Jellyfish/Simulation_Skeletons/Re37pt5/Res_48_60x18` |
| `ib2d/demo_564` | Examples_Education/Convergence/Jellyfish/Simulation_Skeletons/Re37pt5/Res_384_480x144 | IB2d `Examples_Education/Convergence/Jellyfish/Simulation_Skeletons/Re37pt5/Res_384_480x144` |
| `ib2d/demo_565` | Examples_Education/Convergence/Jellyfish/Simulation_Skeletons/Re37pt5/Res_256_320x96 | IB2d `Examples_Education/Convergence/Jellyfish/Simulation_Skeletons/Re37pt5/Res_256_320x96` |
| `ib2d/demo_566` | Examples_Education/Convergence/Jellyfish/Simulation_Skeletons/Re150/Res_192_240x72 | IB2d `Examples_Education/Convergence/Jellyfish/Simulation_Skeletons/Re150/Res_192_240x72` |
| `ib2d/demo_567` | Examples_Education/Convergence/Jellyfish/Simulation_Skeletons/Re150/Res_64_80x24 | IB2d `Examples_Education/Convergence/Jellyfish/Simulation_Skeletons/Re150/Res_64_80x24` |
| `ib2d/demo_568` | Examples_Education/Convergence/Jellyfish/Simulation_Skeletons/Re150/Res_32_40x12 | IB2d `Examples_Education/Convergence/Jellyfish/Simulation_Skeletons/Re150/Res_32_40x12` |
| `ib2d/demo_569` | Examples_Education/Convergence/Jellyfish/Simulation_Skeletons/Re150/Res_128_160x48 | IB2d `Examples_Education/Convergence/Jellyfish/Simulation_Skeletons/Re150/Res_128_160x48` |
| `ib2d/demo_570` | Example_Mass_Points/Gravity_vs_Flow | IB2d `Example_Mass_Points/Gravity_vs_Flow` |
| `ib2d/demo_571` | Example_Mass_Points/Gravity_Cellular_Race | IB2d `Example_Mass_Points/Gravity_Cellular_Race` |
| `ib2d/demo_572` | Example_Mass_Points/Channel_Flow_Cellular_Race | IB2d `Example_Mass_Points/Channel_Flow_Cellular_Race` |
| `ib2d/demo_573` | Example_Mass_Points/Massive_Cell_In_Channel | IB2d `Example_Mass_Points/Massive_Cell_In_Channel` |
| `ib2d/demo_574` | Example_Mass_Points/Gravity_vs_Flow_256x256 | IB2d `Example_Mass_Points/Gravity_vs_Flow_256x256` |
| `ib2d/demo_575` | Example_Baby_Spider_Shower/Flow_Based | IB2d `Example_Baby_Spider_Shower/Flow_Based` |
| `ib2d/demo_576` | Example_Concentration_Dynamics/Leaf_64 | IB2d `Example_Concentration_Dynamics/Leaf_64` |
| `ib2d/demo_577` | — | IB2d `Rayleigh_Taylor_Instability` |
| `ib2d/demo_578` | Example_Boussinesq/Kelvin_Helmholtz_Instability/Kelvin_Helmholtz_Looks_Like_A_Cat | IB2d `Example_Boussinesq/Kelvin_Helmholtz_Instability/Kelvin_Helmholtz_Looks_Like_A_Cat` |
| `ib2d/demo_579` | Example_Boussinesq/Kelvin_Helmholtz_Instability/Kelvin_Helmholtz_Qualitative | IB2d `Example_Boussinesq/Kelvin_Helmholtz_Instability/Kelvin_Helmholtz_Qualitative` |
| `ib2d/demo_587` | Example_HeartTube/Electromechanical_Pumping | IB2d `Example_HeartTube/Electromechanical_Pumping` |
| `ib2d/demo_588` | Example_HeartTube/Electromechanical_Pumping_w_Ca_Dynamics | IB2d `Example_HeartTube/Electromechanical_Pumping_w_Ca_Dynamics` |
| `ib2d/demo_589` | Example_HeartTube/EM_Pump_w_Calcium_and_Muscles | IB2d `Example_HeartTube/EM_Pump_w_Calcium_and_Muscles` |
| `ib2d/demo_590` | Examples/Example_KC/Kathleen | IB2d example "Dear KC" |
| `ib2d/demo_594` | Examples/Example_Corals/Two_Corals | IB2d example "Two Corals" |
| `ib2d/demo_595` | Examples/Example_Sawtooth | IB2d example "Sawtooth" |
| `ib2d/demo_596` | — | IB2d example "Corals" |
| `ib2d/demo_597` | Examples/Example_Pulsing_Heart/512x512 | IB2d example "Pulsing Heart" |
| `ib2d/demo_598` | — | IB2d example "Insect Flight" |
| `ib2d/demo_599` | — | IB2d example "Tethered Jellyfish" |
| `ib2d/demo_600` | — | IB2d example "Tethered Jellyfish" |
| `ib2d/demo_601` | — | IB2d example "Beating Heart" |
| `ib2d/demo_602` | — | IB2d example "Nematocyst" |
| `ib2d/demo_603` | — | IB2d example "Beehive" |
| `ib2d/demo_604` | — | IB2d example "Jellyfish Material" |
| `ib2d/demo_605` | — | IB2d example "Jellyfish Material |
| `ib2d/demo_606` | — | IB2d example "Swimmer" |
| `ib2d/demo_607` | — | IB2d example "Swimmer" |
| `ib2d/demo_608` | — | IB2d example "Jellyfish Update Beam Curvature" |
| `ib2d/demo_609` | Examples/Example_KC/Kathleen_256 | IB2d example "Dear KC" |
| `ib2d/demo_610` | Examples/Example_KC/For_Brian | IB2d example "Dear KC" |
| `ib2d/demo_611` | Examples/Example_KC/For_Danh | IB2d example "Dear KC" |
| `ib2d/demo_612` | Example_HeartTube/Muscle_Model | IB2d `Example_HeartTube/Muscle_Model` |
| `ib2d/demo_613` | Example_HeartTube/3_Element_Muscle_Model | IB2d `Example_HeartTube/3_Element_Muscle_Model` |
| `ib2d/demo_614` | Example_HeartTube/128x128_Electromechanical_Pumping_w_Ca_Dynamics | IB2d `Example_HeartTube/128x128_Electromechanical_Pumping_w_Ca_Dynamics` |
| `ib2d/demo_620` | Example_Test_Restart_Protocol/Standard_Rubberband_Restart | IB2d `Example_Test_Restart_Protocol/Standard_Rubberband_Restart` |
| `ib2d/demo_621` | Example_Test_Restart_Protocol/Poroelastic_Rubberband_Restart | IB2d `Example_Test_Restart_Protocol/Poroelastic_Rubberband_Restart` |
| `ib2d/demo_625` | Example_Stirring_Coagulation | IB2d `Example_Stirring_Coagulation` |
| `ib2d/demo_626` | Example_Testing_Thru_Boundaries/Stirring_With_N_Multiple_Cells_512x64 | IB2d `Example_Testing_Thru_Boundaries/Stirring_With_N_Multiple_Cells_512x64` |
| `ib2d/demo_627` | Example_Testing_Thru_Boundaries/Works_on_512x128 | IB2d `Example_Testing_Thru_Boundaries/Works_on_512x128` |
| `ib2d/demo_628` | Example_Testing_Thru_Boundaries/Stirring_With_Multiple_Cells_512x64 | IB2d `Example_Testing_Thru_Boundaries/Stirring_With_Multiple_Cells_512x64` |
| `ib2d/demo_629` | Example_Testing_Thru_Boundaries/Works_on_64x16 | IB2d `Example_Testing_Thru_Boundaries/Works_on_64x16` |
| `ib2d/demo_630` | Example_Coagulation/Example_Testing_Thru_Boundaries/Works_on_64x16 | IB2d `Example_Coagulation/Example_Testing_Thru_Boundaries/Works_on_64x16` |
| `ib2d/demo_633` | Examples_Education/Convergence/Jellyfish/Simulation_Skeletons/Re150/Res_1024_1280x384 | IB2d `Examples_Education/Convergence/Jellyfish/Simulation_Skeletons/Re150/Res_1024_1280x384` |
| `ib2d/demo_634` | Examples_Education/Convergence/Jellyfish/Simulation_Skeletons/Re150/Res_96_120x36 | IB2d `Examples_Education/Convergence/Jellyfish/Simulation_Skeletons/Re150/Res_96_120x36` |
| `ib2d/demo_635` | Examples_Education/Convergence/Jellyfish/Simulation_Skeletons/Re150/Res_768_960x288 | IB2d `Examples_Education/Convergence/Jellyfish/Simulation_Skeletons/Re150/Res_768_960x288` |
| `ib2d/demo_636` | Examples_Education/Convergence/Jellyfish/Simulation_Skeletons/Re150/Res_512_640x192 | IB2d `Examples_Education/Convergence/Jellyfish/Simulation_Skeletons/Re150/Res_512_640x192` |
| `ib2d/demo_637` | Examples_Education/Convergence/Jellyfish/Simulation_Skeletons/Re150/Res_48_60x18 | IB2d `Examples_Education/Convergence/Jellyfish/Simulation_Skeletons/Re150/Res_48_60x18` |
| `ib2d/demo_638` | Examples_Education/Convergence/Jellyfish/Simulation_Skeletons/Re150/Res_384_480x144 | IB2d `Examples_Education/Convergence/Jellyfish/Simulation_Skeletons/Re150/Res_384_480x144` |
| `ib2d/demo_639` | Examples_Education/Convergence/Jellyfish/Simulation_Skeletons/Re150/Res_256_320x96 | IB2d `Examples_Education/Convergence/Jellyfish/Simulation_Skeletons/Re150/Res_256_320x96` |
| `ib2d/demo_640` | Example_Jellyfish_Swimming/Hoover_Jellyfish | IB2d `Example_Jellyfish_Swimming/Hoover_Jellyfish` |
| `ib2d/demo_641` | Example_Jellyfish_Swimming/Tentacle_Jelly | IB2d `Example_Jellyfish_Swimming/Tentacle_Jelly` |
| `ib2d/demo_642` | Example_Jellyfish_Swimming/Failed_Attempts/Jellyfish_Spring_RL_Interpolation | IB2d `Example_Jellyfish_Swimming/Failed_Attempts/Jellyfish_Spring_RL_Interpolation` |
| `ib2d/demo_643` | Example_Jellyfish_Swimming/Failed_Attempts/Jellyfish_Springs_Only | IB2d `Example_Jellyfish_Swimming/Failed_Attempts/Jellyfish_Springs_Only` |
| `ib2d/demo_644` | Example_Jellyfish_Swimming/Failed_Attempts/Jellyfish_With_Beams/Forward_Swimmer | IB2d `Example_Jellyfish_Swimming/Failed_Attempts/Jellyfish_With_Beams/Forward_Swimmer` |
| `ib2d/demo_645` | Example_Jellyfish_Swimming/Failed_Attempts/Jellyfish_With_Beams/Backwards_Swimmer_for_LOLz | IB2d `Example_Jellyfish_Swimming/Failed_Attempts/Jellyfish_With_Beams/Backwards_Swimmer_for_LOLz` |
| `ib2d/demo_646` | Example_Jellyfish_Swimming/Failed_Attempts/Jellyfish_IBAMR_Convert | IB2d `Example_Jellyfish_Swimming/Failed_Attempts/Jellyfish_IBAMR_Convert` |
| `ib2d/demo_670` | Examples_Education/Pendulum/PointMass-256x256 | IB2d `Examples_Education/Pendulum/PointMass-256x256` |
| `ib2d/demo_671` | Examples_Education/Pendulum/PointMass-512x512 | IB2d `Examples_Education/Pendulum/PointMass-512x512` |
| `ib2d/demo_672` | Examples_Education/Pendulum/PointMass-1024x1024 | IB2d `Examples_Education/Pendulum/PointMass-1024x1024` |
| `ib2d/demo_673` | Example_Wobbly_Beam/Beam_32 | IB2d `Example_Wobbly_Beam/Beam_32` |
| `ib2d/demo_674` | Example_Wobbly_Beam/nonInv_Beam_32 | IB2d `Example_Wobbly_Beam/nonInv_Beam_32` |
| `ib2d/demo_675` | Example_Celestial_Mechanics/Planets | IB2d `Example_Celestial_Mechanics/Planets` |
| `ib2d/demo_676` | Example_Scallop/Scallop_FV_LT_Muscle | IB2d `Example_Scallop/Scallop_FV_LT_Muscle` |
| `ib2d/demo_677` | Example_Scallop/Scallop_3Elem_Muscle | IB2d `Example_Scallop/Scallop_3Elem_Muscle` |
| `ib2d/demo_678` | Example_Scallop/Scallop_3Elem_Muscle_Almost_Swims | IB2d `Example_Scallop/Scallop_3Elem_Muscle_Almost_Swims` |
| `ib2d/demo_680` | Example_Concentration_Dynamics/Rubberband_Constant_64 | IB2d `Example_Concentration_Dynamics/Rubberband_Constant_64` |
| `ib2d/demo_681` | Example_Concentration_Dynamics/Rubberband_Limited_64 | IB2d `Example_Concentration_Dynamics/Rubberband_Limited_64` |
| `ib2d/demo_682` | Example_Concentration_Dynamics/Rubberband_Sink_64 | IB2d `Example_Concentration_Dynamics/Rubberband_Sink_64` |
| `ib2d/demo_683` | Example_Concentration_Dynamics/Corals_64 | IB2d `Example_Concentration_Dynamics/Corals_64` |
| `ib2d/demo_684` | Example_Concentration_Dynamics/Moving_Cylinders_64 | IB2d `Example_Concentration_Dynamics/Moving_Cylinders_64` |
| `ib2d/demo_685` | Example_Meshmerize_Me | IB2d `Example_Meshmerize_Me` |
| `ib2d/demo_686` | Example_Meshmerize_Me/Starfish | IB2d `Example_Meshmerize_Me/Starfish` |
| `ib2d/demo_687` | Example_Meshmerize_Me/Flow_Around_Starfish | IB2d `Example_Meshmerize_Me/Flow_Around_Starfish` |
| `ib2d/demo_688` | Example_Meshmerize_Me/Dragonfly_Wing | IB2d `Example_Meshmerize_Me/Dragonfly_Wing` |
| `ib2d/demo_689` | Example_Meshmerize_Me/Insect_Wing_Veins | IB2d `Example_Meshmerize_Me/Insect_Wing_Veins` |
| `ib2d/demo_690` | Example_Meshmerize_Me/old | IB2d `Example_Meshmerize_Me/old` |
| `ib2d/demo_691` | Example_VIV_Cylinder/Single_VIV_Cylinder_Damped_Springs_High | IB2d `Example_VIV_Cylinder/Single_VIV_Cylinder_Damped_Springs_High` ⚠️ 草稿（未通过验证，见 readme） |
| `ib2d/demo_692` | Examples_First_Year_Seminar/Corals | IB2d `Examples_First_Year_Seminar/Corals` |
| `ib2d/demo_693` | Examples_First_Year_Seminar/Plates | IB2d `Examples_First_Year_Seminar/Plates` ⚠️ 草稿（未通过验证，见 readme） |
| `ib2d/demo_701` | Example_Boussinesq/Kelvin_Helmholtz_Instability/Kelvin_Helmholtz_Qualitative | IB2d `Example_Boussinesq/Kelvin_Helmholtz_Instability/Kelvin_Helmholtz_Qualitative` |
| `ib2d/demo_702` | Example_Boussinesq/Falling_Sphere | IB2d `Example_Boussinesq/Falling_Sphere` |
| `ib2d/demo_703` | Example_Channel_Flow/Example_Turek_Hron | IB2d `Example_Channel_Flow/Example_Turek_Hron` |
### 说明

上表按 demo 编号排序；每个 demo 目录内的 `readme.md` 给出该算例的官方设置与本 demo 设置、
逐条移植要点、`compare.py` 实测结果（含"不能验证"清单）与可复制的运行命令及实测耗时。
`_template/` 是合并后的通用模板（覆盖绝大多数机制），`port_example.py` 用来从上游目录
一键生成新 demo 骨架。

## 运行环境

- `afsi-dolfinx` conda 环境（dolfinx 0.10.0，Python 3.12），`afsic` 包可编辑安装
- 各算例 `readme.md` 内附具体运行命令（`generate_mesh.py` 生成固体网格 → `main.py` 求解）
- 多数算例支持环境变量覆盖（如 `STEPS`、`CIRCLE`、`GRID`）便于短程验证
