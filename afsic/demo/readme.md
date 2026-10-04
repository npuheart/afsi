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

| Demo | 名称 | 说明 |
|---|---|---|
| `ib2d/demo_444` | IB2d 弹性圆环 | 读取 IB2d 原始输入；周期 Taylor–Hood + Peskin 两阶段（`PeskinRK2Solver`）与 fiber Chorin/IPCS 投影（grad-div γ=0/100）共 5 条 AFSI 曲线，与 IB2d (Octave) 逐帧对照 |
| `ib2d/demo_445` | IB2d 水母游动 | Hoover & Miller 水母模型（弹簧 + 非不变梁 + 目标点 + 肌肉驱动 `update_Springs`），周期 Taylor–Hood + Peskin 两阶段；减分辨率版（48×160, dt=4e-5, T=1 s），与 IB2d 参考（pyIB2d，纯 Python，无需 MATLAB）逐帧对照 |
| `ib2d/demo_446` | IB2d 摆动梁 | `Wobbly_Beam`：两端钉住的弹性拱梁（不变梁 $\kappa=7.5\times10^9$ + 目标点）在 1×1 周期盒中摆动（32×32, dt=5e-5, T=0.05）；全程标记点偏差 ≤1.8h，与 pyIB2d 对照 |
| `ib2d/demo_447` | IB2d 阻尼橡皮筋 | `Rubberband_with_Damped_Springs`：零静长强收缩闭环（64 点阻尼弹簧环，32×32, dt=1e-3, T=1.5s）；γ=0 时塌缩分岔（AFSI 全塌缩 vs 参考卡住）——由 grad-div γ=2.5 修正：t≤1.0 全程 dX≤1h、晚段 2h（详见其 readme） |
| `ib2d/demo_448` | IB2d 心管肌肉泵 | `HeartTube_Muscle`：两条平行弹性壁（弹簧 + 不变梁 + 4 角目标点）+ 153 条 Hill 肌肉带，行波激活（10 Hz）驱动的蠕动泵（128×128, dt=1e-4, T=0.25 s），与 pyIB2d 对照 |
| `ib2d/demo_449` | IB2d 非不变摆动梁 | `Wobbly_NonInv_Beam`：两端钉住的非不变梁（$\kappa=10^{10}$, C=0，62 段）+ 2 目标点在 1×1 周期盒摆动（32×32, dt=1e-5, T=0.05）；标记点平均偏差 0.74h，与 pyIB2d 对照 |
| `ib2d/demo_450` | IB2d 重力细胞赛跑 | `Gravity_Cellular_Race`：两个弹性细胞（各 81 点）+ 每点"质量弹簧"（$k=10^6$）连到幽灵粒子（$M\in\{0.05,0.2,1\}$），重力经幽灵拖拽标记（64×64, dt=5e-5, T=0.35）；含 IB2d 质量点"幽灵粒子"模型考证，细胞轨迹偏差 <8e-4 |
| `ib2d/demo_451` | IB2d 阻抗泵 + 示踪粒子 | `Tracers_In_Impedance_Pump`：上下弹性壁（弹簧+不变梁+4 目标点）+ 11 根跨径泵弹簧（每步 $RL=1-0.9|\sin(2\pi\cdot 10t)|$）+ 110 个被动示踪粒子（5×5 盒, 64×64, T=2 s）；结构 0.2–0.6h；整流流量 γ=0 差 15–20% → **grad-div γ=2.5 修正到 ~5%**；tracer 逐点为拉格朗日混沌（参考自身 dt 减半也发散），详见其 readme |
| `ib2d/demo_453` | IB2d 多孔滑移橡皮筋 | `Single_Porous_Rubberband`：64 点收缩环（弹簧 $k=10^7$、$L_0=0$）全点 porous 滑移（$\kappa=10^{-4}$，4 阶差分法向；porous 公式与 pyIB2d 逐位一致）（32×32, dt=1e-4, T=0.1）；**塌缩对 grad-div γ 极敏感**——γ=0 时 t≈0.043 崩溃（伪散度漏流），γ=2.5 时全程 dX≤1h、面积轨迹重合 |
| `ib2d/demo_454` | IB2d 孔弹性橡皮筋 | `Poroelastic_Rubberband`：同 453 的收缩环 + porous 滑移，再加 **poroelastic**（每步额外位移 $\Delta t\,F_{\rm spring}/(\mu c)$，$c=2.5\times10^5$，用 $X_h$ 处弹簧力；驱动用上一步 F 记账）（32×32, dt=1e-4, T=0.1, γ=2.5）；全程 dX≤0.75h、面积轨迹重合（交叉）|
| `ib2d/demo_455` | IB2d 不变梁橡皮筋 | `Rubberband_with_Beams`：249 点椭圆环（半轴 0.2×0.4）用**不变梁**（$\kappa=10^7$，参考曲率 = 等面积圆 $R=0.2828$）连接，在 1×1 周期盒（$\mu=0.0025$）里做"椭圆↔圆"摆动（64×64, dt=1e-2, T=25）；全程 dX≤2.0h（均值 1.24h，t≤4 内 0.83h）；**核查发现 pyIB2d 副本的 `input2d` 把开关误写成 `nonInv_beams`（其解析器不认，原样跑该算例结构力为 0）**，本移植按 matIB2d 规范版用 `beams = 1`；γ=2.5 可把面积漂移从 4.9% 压到 0.05%（详见 readme） |
| `ib2d/demo_456` | IB2d 被动标量扩散 | `Concentration_Diffusion`（`Diffusion_Chamber`）：496 点方形腔壁（全为钉死目标点）+ 中心 $3\times3$ 节点浓度团块，$k=0.025$（1×1 周期盒, $\mu=10$, 64×64, dt=1e-4, T=0.5）；腔壁被钉死 ⇒ IB 力/流速恒为 0，是**纯扩散基准**。AFSI 新增**周期 Q1 有限元被动标量输运**（复用 `PeskinRK2Solver` 的周期约化 `periodic_prolongation`，Q1 自由度 = IB2d 网格节点；集中质量显式推进）。质量两侧严格守恒（9.000000）；末态 $\max|C^{AFSI}-C^{IB2d}|=2.5\times10^{-5}$（峰值 0.18%）、相对 $L_2$ 1.3e-3；与**含像源的精确解**相比最大相对误差 AFSI 2.4e-4 / IB2d 4.3e-3（无界公式 $\sigma_0^2+2kt$ 因像源偏 +0.55%，两侧都跟着精确解走）|
| `ib2d/demo_457` | IB2d 通用用户力接口 | `User_Defined_Fiber_Model`：为 demo_444 的同一椭圆环（64 点、零静长线弹簧 $k=2.5\times10^4$、32×32、T=1.5）改走 **IB2d 通用用户力接口**（`.user_force` + 用户模块 `give_Me_General_User_Defined_Force_Densities.py`，pyIB2d 从运行目录 import）。力公式逐位复刻（校验差 1.5e-16）；结果与 demo_444 逐位一致：$\gamma=0$ 环塌缩（末态面积 2.4e-4）→ $\gamma=100$ 面积守恒（0.2423，参考自身泄漏到 0.1840），$t\le0.84$ 内 dX≤0.45h |
| `ib2d/demo_458` | IB2d 单质点重力沉降 | `Thrips`：**单个**标记点（无自身结构）经 $k=10^3$ 质量弹簧连到 $M=10^{-3}$ kg 幽灵质点，重力仅作用于幽灵（$g=9.80665$），0.5×0.5 周期盒、32×32、dt=1e-4、T=0.1 ⇒ 弹簧-质量-阻尼振子（$\omega=1000$ rad/s）缓慢下沉。结构自由度用 **DG0 "point" 单元**承载（1 自由度 = 1 粒子）；复刻 `Supp.py` 的 mass 记账。标记点轨迹最大偏差 **6.6e-7 m = 4.2e-5 h**，位移比 0.942（小信号测试） |
| `ib2d/demo_459` | IB2d 动态目标点驱动 | `Moving_Rubberband`（首个 `matIB2d` 算例）：64 点圆周橡皮筋（半轴 0.2、无弹簧/无梁，全部为 $k=10^7$ 目标点）在 1×1 周期盒（$\mu=10$、32×32、dt=1e-4、T=0.05）被 `.m` 驱动规律拖拽——每步 $y{+}{=}(s/2)dt$ 上浮、$x$ 按 0.05 s 周期左右往返（$s=20$ m/s），即一步拖 1e-3 m。AFSI 新增：**1-based→0-based 编号转换**（`load_example(index_base=1)`，pyIB2d 直接用文件编号索引会越界）、**动态目标点递推**（与 pyIB2d 同步位调用、同一递推）、以及**与参考逐帧同步的输出帧**（pyIB2d 第 $k$ 帧其实是第 $5k{+}1$ 步的状态而时间戳晚一步，属上游约定；差一步对 20 m/s 的运动就是 0.002 m 假误差）。标记点偏差典型 **2.1e-5 m（6.6e-4 h）**、最大 3.3e-4 m（$x$ 折返点）；形心最大差 8.8e-5 m；目标点递推一致到 **1.6e-16 m**；"标记点滞后锚点"曲线两侧一致到 1e-5 m（稳态 ~1.1e-3 m = $s\,dt/2$）|
| `ib2d/demo_460` | IB2d 数据文件驱动的移动目标点 | `Pulsing_Heart/32x32`：384 点心脏轮廓（无弹簧/无梁，全部 $k=10^8$ 目标点）在 1×1 周期盒（$\mu=0.5$、32×32、dt=1e-5、T=0.1=1 万步）内按 `update_Target_Point_Positions.m` 的两相多项式规律在 `All_Positions.txt` 的舒张/收缩两条形线之间插值（周期 0.05 s，锚点行程最大 0.119 m = 3.81 h）。锚点与 `.m` 递推**逐位一致**；形状质心最大差 **5.1e-6 m**；包围面积与预设多边形相差 ≤7e-4（IB2d 2.9e-3）；收缩–舒张相位完全同步。标记点逐点互差中位 1.21e-2 m（0.39 h）——经 dt/2 参考（自身仅变 ≤3.2e-3 m）与 $k/10$ 缩放（互差同步放大到 2.9e-2 m）确认这是 **$k=10^8$ 罚力柔度之差**而非时间/相位失配；本算例 `max|u|` 达 341 m/s，速度场只能在滤波意义下比较（readme 有完整证据）|

`pyIB2d/Examples` 已全部移植完毕；下列 `matIB2d/Examples` 算例正在按批次移植
（本仓库 demo 编号续排）：

| 算例 | 说明 | 新增机制 |
|---|---|---|
| `Example_Corals/Single_Coral` | 多项式拟合驱动的珊瑚触手（64²，原算例 25 万步 ⇒ 缩减 T 并记录） | 多目标点 + 大位移移动 |
| `Example_Porous_Rubberbands/Compare_Porous_NonPorous_Rubberbands` | 多孔 vs 非多孔双环 A/B | 多结构体 + 多孔介质（A/B 对照） |
| `Example_HeartTube/Update_Springs_Peristaltic_Pumping` | 变静长（`update_springs`）蠕动泵 | 时变静长驱动 |
| `Example_Skeleton` | 纯弹簧骨架（`geo_connect` + 复杂连通） | 复杂几何弹簧网络 |
| `Examples_Education/Interpolation/Moving_Circle/Linear_Interp` | 移动圆的位置由数据表插值给出 | 目标点位置插值 |
| `Example_Mass_Points/Gravity_vs_Flow` | 质量点 + 重力 vs 顶盖驱动流 | 外部体积力 `arb_ext_force` |
| `Example_Standard_Rubberband/NL_Rubberband_With_Springs` | 椭圆环 + **非线性弹簧**（`.spring` 第 5 列 $\alpha=2$） | 非线性弹簧力律 $F=\tfrac12(\alpha{+}1)k(L{-}L_0)^\alpha$ |

其余 `matIB2d` 算例需要新机制，暂缓：`Example_Boussinesq/*`、`Example_Channel_Flow/*`、
`Example_Meshmerize_Me/*`、`Example_VIV_Cylinder/*`（均需外部体积力 `arb_ext_force`）、
`Example_HeartTube/Electromechanical_*`（`electro_phys`）、`Example_Coagulation/*`（`coagulation`）、
`Example_Scallop/*`（`3_element_muscle`）、`Example_Concentration_Dynamics/*`（浓度源项 + 移动目标点，
规模 6.4e4 步以上）、`Example_Jellyfish_Swimming/Tentacle_Jelly`（poroelastic + 大网格）等。

## 运行环境

- `afsi-dolfinx` conda 环境（dolfinx 0.10.0，Python 3.12），`afsic` 包可编辑安装
- 各算例 `readme.md` 内附具体运行命令（`generate_mesh.py` 生成固体网格 → `main.py` 求解）
- 多数算例支持环境变量覆盖（如 `STEPS`、`CIRCLE`、`GRID`）便于短程验证
