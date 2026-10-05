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
| `ib2d/demo_461` | IB2d 视频拟合的弯曲移动目标点 | `Corals/Single_Coral`：90 点珊瑚（两条 45 点触手，全部 $k=10^8$ 目标点）在 0.6×0.6 周期盒（$\mu=0.01$、64×64、dt=1e-5）被 `.m` 的**五次多项式拟合**规律弯曲（只动 $x$，$X(y,t)$ 为三次弯曲模型，自变量为视频帧号 $30(\text{mod}\,2.30)$）。锚点规律 $t=0$ 与 `coral.vertex` 逐位一致（1.7e-16），周期性 0.0；原算例 25 万步 ⇒ 默认只跑 $T=0.115$ s（1.15 万步，锚点行程 4.41 h，全程 6.00 h，规律由 compare.py 解析验证）。标记点偏差典型 **4.3e-6 m（4.5e-4 h）**、最大 5.4e-6 m —— 本批移动目标点算例里最好；与 demo_460 对照可见罚力柔度误差正比于锚点速度（20/7/0.7 m/s → 1e-2/0.4/5e-4 h）|
| `ib2d/demo_462` | IB2d 非线性弹簧力律 | `NL_Rubberband_With_Springs`：与 demo_444 同几何（64 点椭圆、零静长弹簧、32×32、dt=1e-3、T=1.5），但 `.spring` 第 5 列 $\alpha=2$ ⇒ **硬化二次力律** $F=\tfrac12(\alpha{+}1)k(L-L_0)^\alpha$。逐弹簧对照误差 0.0；标记点偏差典型 4.3e-3 m（0.14 h）、最大 0.23 h；周长 1.937→1.770（IB2d 1.751）；$\gamma=0$ 时漏面积（漂移 8.0e-2，偏差 1.7 h）⇒ 默认 $\gamma=100$ |
| `ib2d/demo_463` | IB2d 多孔/不透水 A/B | `Compare_Porous_NonPorous_Rubberbands`：同一 2×2 周期盒（$\mu=0.1$、64×64、dt=5e-5、T=0.1）内**两条各 128 点**零静长环（$k=10^7$），左侧环整条标记为多孔（$\kappa=10^{-4}$，逐点差分模板）——**首个同流场双结构算例**。两侧都显示多孔环塌缩 ~55%（面积 0.2512→0.116/0.101）而不透水环仅收缩 ~20%（→0.197/0.201）；标记点偏差最大 0.57 h（多孔）/0.77 h（不透水）；默认 $\gamma=2.5$（demo_453 的 $\gamma$ 扫描结论）|
| `ib2d/demo_464` | IB2d 行波（位置相关）静长驱动 | `Update_Springs_Peristaltic_Pumping`：与 demo_451 同一根 HeartTube（156 点、$5\times5$ 盒、64×64、dt=1e-3、T=2），但 76 根跨径泵弹簧的静长由**蠕动行波** $RL=d-0.95\sin(\tfrac{2\pi}{L}(x_i-x_M-vt))$ 驱动，且 $x_i$ 取**当前**标记点坐标（本批首个位置相关驱动）；$t\ge0.3$ s 标记点偏差 ≤**0.32 h**（中位 0.06 h），管宽（射流特征）逐帧一致到 $10^{-3}$；初始 2.24 h 峰值源于上游规律在 $t=0$ 的不连续初值（一半泵弹簧直接处于最小静长 0.05），默认 $\gamma=2.5$
| `ib2d/demo_465` | IB2d 多状态插值（线性 vs 三次） | `Examples_Education/Interpolation/Moving_Circle` 的两个变体：64 点圆（$k=10^7$ 目标点）在 1×1 盒（$\mu=10$、32×32、dt=1e-4、T=0.1）内于三个预设位置 A/B/C 之间往返（周期 0.06 s，行程 0.45 m = 14.4 h），两变体只差插值参数 $g(s)$（线性 $g=s$ vs 分段三次）。一个移植 + `INTERP` 开关同时覆盖两变体，各自与自己的参考对比：linear 最大偏差 **0.027 h**、cubic **0.019 h**；圆心轨迹差 ≤3.7e-4 m |
| `ib2d/demo_466` | IB2d 行波激活 FV_LT 肌肉管 | `Example_Lymphangion`：310 点淋巴管（弹簧 $k=10^7$ + 306 不变梁 + 4 目标点 + **153 条 FV_LT 肌肉带**）在 5×5 盒（$\mu=1$、128×128、dt=5e-4、T=0.25）被 10 Hz **行波方波**激活驱动蠕动。管径（平均跨径）1.0132 → AFSI **0.5977** / IB2d 0.5957（收缩 41%），逐帧差 ≤**2.8e-3**（0.5%），激活相位同在 $t=0.245$；标记点偏差中位 0.18 h。**发现上游开关疑点**：算例自带 `.muscle` 与激活函数却把 `FV_LT_muscle` 置 0（按原样跑是无肌肉的被动管），本移植默认打开并提供 `MUSCLE=0` 对照 |
| `ib2d/demo_467` | IB2d 任意外力接口 `arb_ext_force` | `Example_Channel_Flow/Example_Parabolic_Flow_In_Empty_Channel`：1×1 周期盒（$\mu=10$、128×128、dt=1e-4、T=0.015=151 步）内两条各 231 点、全为 $k=10^7$ 目标点的槽道壁（$y=0.4/0.6$，$x\in[0.05,0.948]$，两端留口），**新增 AFSI 任意外力机制**：每步调用算例自带 `.m` 的逐行 Python 移植（AFSI 与 pyIB2d 跑同一文件），在 $x\in[0.1,0.16],y\in[0.41,0.59]$ 施加 10 Hz 调制的抛物速度目标罚力（$k=10^4$、$u_{\max}=250$；节点力密度 → Q1 节点求值 → Q2 插值叠加到 `solver.f`）。中心线速度 1.3929 / 1.3961 m/s（差 **0.03%**），窄带外 $\max|\Delta u|=6.9\%$ `max|u|`（相对 $L_2$ 5.5e-2）、槽道核心流量差 2.9e-2，壁面钉死偏差 2.4e-3 h；**发现**：$k\,dt/\rho=1$ 的边际罚力增益让参考的共置格式在 4 节点 $\delta$ 支撑带长出网格尺度尖峰（$u_x$ 达 -1.15），`dt` 减半即消失（-0.034 ≈ AFSI 的 -0.030），`compare.py` 自动做 dt/2 交叉核对（详见 readme） |
| `ib2d/demo_468` | IB2d 任意外力：单壁 + 圆柱绕流 | `Example_Channel_Flow/Example_1_Wall_Flow_Around_Cylinder`：1×1 周期盒（$\mu=0.01$、32×32、dt=1e-4、T=0.125=1250 步）内 58 点壁（$y=0.4$）+ 27 点圆柱（半径 0.0333，仅 1.07 $h$）全部 $k=10^7$ 目标点钉死；算例的 `.m` 在 $x\in[0.1,0.16],y\in[0.41,0.59]$ 施加 $k=10^2$ 的 $\tanh(t)$ 缓启抛物速度目标罚力（增益 $k\,dt/\rho=10^{-2}$，无窄带伪速度）。原样网格上 $L_2$ 误差 1.7e-1、`max|u|` 低 10% —— 经**加倍到 64×64/dt=5e-5**（圆柱 2.1 $h$）后降到 **3.8e-2**、`max|u|` 差 0.6%、窗口 `max|u|` 差 6.0e-3、流量差 5.3e-2（即差异来自不可分辨圆柱的网格尺度离散，而非实现错误）；两种分辨率下标记点偏差 ≤7.4e-4 h、罚力公式核对 ≤6e-2（`|f|` 峰值的万分之几）。本 demo 起 `main.py`/`compare.py` 模板通用化（按需装配弹簧/梁/目标点；力窗口、目标剖面、IB 窄带全部由模块与目标点导出，并自动配对 `ib2d_reference<TAG>.npz`/`plot<TAG>/` 交叉核对） |
| `ib2d/demo_469` | IB2d 任意外力：槽道中部射流 | `Example_Channel_Flow/Example_Middle_Channel_Flow`：116 点双壁（$y=0.4/0.6$，全钉死）+ 算例 `.m` 在**槽道正中** $x\in[0.47,0.53]$（32² 上仅 1 列 ×7 节点）施加 $u_{\max}=200$ 的 10 Hz 抛物射流（$k=10^4$、$\mu=10$、dt=1e-4、T=0.05=500 步）。原样网格：窗口 `max|u|` 差 8.1%、$L_2$ 1.25e-1；**加密 64²/dt=5e-5** 后窗口 3.6%、$L_2$ 5.3e-2、流量 2.8%、标记点 4.0e-3 h（差异随网格单调下降） |
| `ib2d/demo_470` | IB2d 任意外力：带孔槽道 20 Hz 射流 | `Example_Channel_Flow/Example_Hole In_Channel`：底层 58 点完整壁 + 顶层 46 点**带孔**壁（孔 $x\in(0.487,0.691)$，全为 $k=10^7$ 目标点），`.m` 施加 `20 Hz`、$u_{\max}=250$ 的抛物射流（$k=10^4$、$\mu=10$、32²、dt=1e-4、T=0.25=2500 步）；两边 `max|u|` 全程同步振荡，原样网格窗口差 17.8%、$L_2$ 3.2e-1 → **加密 64²/dt=5e-5** 后 4.8%/9.5e-2、流量 3.8e-2、标记点 7.0e-3 h |
| `ib2d/demo_471` | IB2d 任意外力 + 弹性瘤壁 | `Example_Channel_Flow/Example_Aneurysm`：116 点双壁，**101 点钉死**，上层壁中段 15 点自由（16 弹簧 $k=5\times10^5$ + 15 不变梁 $kb=5\times10^5$）；`.m` 施加 $u_{\max}=250$、**振幅 ×10**（$u^{\rm tar}$ 峰值 25 m/s）的 10 Hz 射流，$\mu=0.1$、64²、dt=1e-4、T=0.25。钉死点偏差 **4.5e-2 h**、窗口 `max|u|` 差 **0.65%**、流量 0.87%、罚力核对 0.17%；但**自由补片按原样发布不可点对点比较**：dt=1e-4 时参考自己的补片跑飞（散落全盒）而 AFSI 仍连贯；dt 减半后两边都连贯且形状一致（拱顶差 2.4 h）——已定位为时间步敏感 + 单元 $Re\approx4$ 不可分辨的暴力工况，非移植错误（详见 readme） |
| `ib2d/demo_472` | IB2d 任意外力：高 Re 射流 + 圆柱（长条域） | `Example_Channel_Flow/Example_Flow_Around_Cylinder`：1×0.25 长条盒（$\mu=10^{-3}$、512×128、dt=5e-5）内 **2273 个 $k=10^5$ 软目标点**描出上下壁与小圆柱，`.m` 用 $L_y/2$ 的抛物剖面在整个条带（$x\in[0.1,0.16]$、$y\in[0.035,0.215]$）施加 $\tanh(t)$ 缓启射流（$k=10^4$）。官方 $T=1.25$=25000 步，本 demo 默认缩到 **T=0.02=400 步**（文档说明）。窗口内剖面逐点 **~1%**、窗口 `max|u|` 2.2%、罚力核对 **0.04%**、软钉死标记点 2.0e-2 h（加密 4.8e-3 h）；但下游高 Re 剪切层（单元 $Re\approx25$）不可点对点比较（$x=0.5$ 峰值差 27%、$L_2$ 2.5e-1，加密不收敛）——readme 明确区分「能验证/不能验证」 |
| `ib2d/demo_473` | IB2d 被动示踪粒子（tracers） | `Example_Tracers/Tracers_In_Channel_Flow`：286 点钉死槽道壁 + **110 个被动示踪粒子**（不产生力，用同一 IB 核平流并按 IB2d 规则回卷），`.m` 在 $x\in[0.1,0.13]$ 施加 $\tanh(t)$ 缓启射流（$k=10^4$、$\mu=0.01$、64²、dt=5e-5、T=0.25=5000 步）。**新机制 + γ 校准**：$\gamma=0$ 时散度误差让下游射流衰减过头（$L_2$ 2.1e-1、流量差 20%），$\gamma=2.5$ 修正到 **6.9e-2 / 4.2e-2**；窗口 `max|u|` 1.1%、标记点 6.8e-2 h、110 个示踪粒子最大偏差 1.4 h（拉格朗日混沌，γ=0 时 5.7 h）。**发现上游 bug**：pyIB2d 的 tracer 更新调用少传 4 个参数（直接跑 `TypeError`），run_reference 已打补丁 |
| `ib2d/demo_474` | IB2d 分叉动脉射流 + 示踪粒子 | `Example_Tracers/Tracers_In_Bifurcating_Artery`：2×2 盒内 1144 点 $k=1.2\times10^7$ 钉死目标点描出 Y 形分叉动脉 + **378 示踪粒子**，主干内 $\tanh(t)$ 缓启射流（$k=10^4$、$\mu=0.1$、128²）。两个必修项：**dt 必须 5e-5**（官方 1e-4 时 $k\,dt^2/\rho h^2=0.49$，AFSI 在 $t\approx0.005$ 局部发散）、**$\gamma$ 必须打开**（γ=0 时射流大 1.5 倍，γ=1.0 降到 **2.7%**）。能验证：窗口 2.7%、标记点 1.0e-2 h、示踪粒子 0.49 h、罚力 0.05%；不能验证：分叉区流动结构（$L_2$ 5.5e-1、中列流量符号都不同，源于软钉死壁渗透性的离散差异，readme 如实记录）|
| `ib2d/demo_475` | IB2d Brinkman 阻力项 | `Example_Channel_Flow/Example_Brinkman_Channel`：1×0.25 条带内 1844 点 $k=5\times10^6$ 钉死双壁，**新机制：Brinkman 阻力** $F -= \mu\,Brink(x)\,u^n$（`.brinkman` 给出 $1/k=0$ 或 $10^5$ 的逐节点阻抗图，右侧 $x\in[0.25,0.90]$ 为多孔区），左侧 $\tanh(4t)$ 归一化抛物射流把流体推入多孔区（$k=10^4$、$\mu=2$、$\rho=1000$、512×128、dt=5e-5、T=0.05=1000 步）。**本族吻合最好**：窗口 `max|u|` **0.62%**、$L_2$ **1.56e-2**、流量 **0.31%**、标记点 **5.9e-4 h**、罚力核对 0.02%。**发现 pyIB2d 两个上游 bug**：`read_In_Brinkman_Info` 返回未定义名（所有 brinkman 算例 `NameError`）、driver 在 Brinkman 分支里重复 `chdir('viz_IB2d')` 直接崩——run_reference 均已打补丁 |
| `ib2d/demo_476` | IB2d 纯扩散基准（被动标量） | `Example_Channel_Flow/Example_Concentration_Gradients/Pure_Diffusion`：1×1 盒、64²、dt=1e-4、**官方完整设置 T=0.5**（5000 步）、496 点 $k=10^7$ 钉死方腔、中心 3×3 单位浓度方块以 $kDiff=0.025$ 纯扩散。**新机制：逐字复刻 IB2d 的被动标量差分推进**（`Supp.perform_Time_Split_Upwind`：x 子步 → 在更新场上求 y 子步；周期 upwind 对流 + 中心扩散；用 $u^{n+1}$）。流场恒为零（无外力、结构钉死），因此这是标量算子的**干净基准**：`max|ΔC|` **4.9e-13**、相对 $L_2$ **1.8e-11**、峰值/总量相对差 4e-12，即**逐位一致**。踩到的坑：驱动调用的是**分裂**格式（`noSplit` 版本在源码里被注释掉），按 no-split 写会差 6e-4；另外上游 `give_Necessary_Derivative` 的 `range(1,n-2)` 使次末行导数恒为 0，已逐字复现并注释 |
| `ib2d/demo_477` | IB2d 分叉动脉 + 被动标量 | `Example_Channel_Flow/Example_Concentration_Gradients/Bifurcating_Artery`：demo_474 的同一条 Y 形分叉动脉（1144 点 $k=1.25\times10^7$ 钉死软壁、3 对重合点需 1e-6 h 偏移）加**被动标量**：入口射流正下游 19×19 方块（$kDiff=5\times10^{-5}$），2×2 盒、128²、dt=5e-5、本 demo T=0.1（2000 步）。与 demo_474 相同的两项标定照旧适用：**γ=1.0 必修**（γ=0 时射流大 1.5 倍，γ=1.0 → 窗口 **2.66%**，与 demo_474 的 2.7% 完全一致，互为交叉验证）。浓度相对 $L_2$ **4.8%**、峰值 **4.9e-4**；与 demo_476 对比可知这 4.8% 完全来自速度场差异（窄带外 $L_2$ 0.55 的已知渗漏差异），标量算子本身已被 demo_476 验证到 1e-11 |
| `ib2d/demo_478` | IB2d 阻抗泵 + 被动标量 | `Example_Channel_Flow/Example_Concentration_Gradients/Impedance_Pumping_Tube`：5×5 盒内 412 点弹性管（616 弹簧 k=1e4–1e6、408 不变梁 kb=100、4 钉死目标点）+ 管内 336 节点浓度方块（kDiff=5e-5），128²、dt=1e-3、**官方 T=0.5 完整跑**。**新泵送律**：`update_Springs.m` 把 16 根跨径弹簧静长按 `RL = d - |0.9 d sin(2π·10t)|` 捏紧（每周期两次）。**能验证**：管直径 **1.14%**、被捏弹簧长度 2.7%、浓度总量 **0.29%**；**不能验证**瞬时速度/浓度场（浓度 L2 1.8e-1）：dt 减半**不改善**（0.190）、`GRAD_DIV=1` 也**不改善**（0.187）⇒ 与分叉动脉一族同源（剧烈运动的高速管壁在 FFT 投影 + 罚力 vs Taylor–Hood + 插值力下的耦合差异，CFL≈0.9 时被放大），readme 完整记录两次收敛性判别 |
| `ib2d/demo_587` | IB2d 电生理（FitzHugh–Nagumo）驱动的心管泵 | `Example_HeartTube/Electromechanical_Pumping`：5×5 盒 156 点心管（232 弹簧 k=1e7、152 不变梁、4 目标点）；**新机制 `electro_phys`**：驱动在 IBM 循环前一次性解一维 FHN（D=10、a=0.3、eps=1e-3、10 脉冲/T、非周期 Laplacian、前向 Euler），每步把 `springs_info(155:232,3)=(16v)^4` 写进**跨管弹簧刚度**（上游 `update_Springs.m` 注释说第 4 列是静长，实际写的是刚度——已考证）。pyIB2d **未实现** `electro_phys`，故参考侧用其 `update_springs` 钩子注入**同一份** FHN 模块；耦合值两侧**逐位相同**，管径收缩 1.0→0.35（相位一致）、标记点 0.35 h、速度场不可逐帧比 |
| `ib2d/demo_588` | IB2d 钙动力学驱动的电生理泵 | `Example_HeartTube/Electromechanical_Pumping_w_Ca_Dynamics`：与 demo_587 同几何同耦合，**电流改为钙信号** `I=0.5 Caf`（Williams 2008 质量作用定律，`Calcium_Dynamics.m`，物理周期 0.5 s，与 T_final 无关 → TFINAL 可安全截断）。`Calcium_Dynamics.py` 与 MATLAB 逐行转写逐位一致；耦合值两侧逐位相同（峰值 1.65e5），管径 0.32/0.34 |
| `ib2d/demo_589` | IB2d 钙电生理直接驱动 FV_LT 肌肉 | `Example_HeartTube/EM_Pump_w_Calcium_and_Muscles`：156 点心管 + 76 条 FV_LT 肌肉带；`electro_phys=1 且 FV_LT_muscle=1` 时驱动**跳过**弹簧刚度调制，改用 `give_Ca_ElectroPhys_Muscle_Activation.m` 的 `a_f = 10 v(t, leader node)` 作激活系数（移植为 pyIB2d 可调用的 `give_Muscle_Activation.py`，两侧共用） |
| `ib2d/demo_612` | IB2d `Muscle_Model`（FV_LT 肌肉心管） | `Example_HeartTube/Muscle_Model`：310 点心管（308 弹簧 k=1e7、306 不变梁、153 条 FV_LT 肌肉带，等效于 demo_448 的 matIB2d 孪生版）。**注意**：官方 `dt=5e-4` 下 AFSI 显式耦合 4 步内发散（demo_448 已记录同一现象），故双侧统一 `dt=1e-4` |
| `ib2d/demo_613` | IB2d `3_Element_Muscle_Model` | `Example_HeartTube/3_Element_Muscle_Model`：与 demo_612 同数据、`mu=1`，开关换成 `3_element_muscle=1`。**上游不一致（已考证）**：示例自带的 `give_3_Element_Muscle_Activation.m` 是 9 参 1 输出、肌肉文件只有 7 列，而 matIB2d blackbox 要求 10 参 4 输出 + 第 8/9 列（kSpr/alpha）→ **该示例按原样无法在随附 MATLAB blackbox 下运行**；pyIB2d 走的是与示例数据自洽的旧版（7 列、FV_LT 力律），但它的 3 元件分支**忘了 import 该模块**（`NameError`），本 demo 在 `run_reference.py` 里注入一行修正（已记录） |
| `ib2d/demo_614` | IB2d 高分辨率电生理泵（钙驱动） | `Example_HeartTube/128x128_Electromechanical_Pumping_w_Ca_Dynamics`：310 点心管、463 弹簧（第 309–463 行是跨管弹簧）、`5×2.5` 盒、`dt=1e-5`、`numFHN=1`、155 个电生理管点；机制同 demo_588，仅分辨率不同（官方 400000 步 → 已截断并记录） |

`pyIB2d/Examples` 已全部移植完毕；`matIB2d/Examples` 的差异部分已在本批移植
（demo_457–466，见上表）：移动/指定运动的目标点（`Moving_Rubberband`、`Pulsing_Heart`、
`Single_Coral`、`Moving_Circle` 的线性/三次插值对）、1-based 编号转换、行波静长驱动
（`Update_Springs_Peristaltic_Pumping`）、多孔 A/B（`Compare_Porous_NonPorous_Rubberbands`）、
非线性弹簧（`NL_Rubberband_With_Springs`）与行波肌肉激活（`Lymphangion`）。

以下 `matIB2d` 算例与已移植算例**机制重复**，暂不单独立项：
`Example_Standard_Rubberband/*` 与 `Example_Skeleton`（同为 pyIB2d 的
`Rubberband_with_Springs/Beams/Damped_Springs` 的 MATLAB 版）、
`Example_Porous_Rubberbands/Single_Porous_Rubberband`（= demo_453）、
`Example_HeartTube/Insect_Heart`、
`Example_Pulsing_Heart/512x512`、`Example_Corals/Two_Corals` 等放大/变体算例。

`ib2d/demo_467`–`472` 已把**任意外力 `arb_ext_force`** 这一族的主要机制落地：
AFSI 侧每步把节点力密度在 Q1 节点（= IB2d 共置流体网格）上求值、再经 Q2 插值叠加到
`solver.f`，并且**参考与 AFSI 跑同一份 `.m` 的逐行 Python 移植**，因此对比只反映求解器与
IB 机制之差。已覆盖：空槽道射流（467）、单壁 + 圆柱（468）、槽道中部射流（469）、
带孔槽道 20 Hz 射流（470）、弹簧/不变梁弹性瘤壁（471）、长条域高 Re 射流 + 圆柱（472）。
模板（`main.py`/`compare.py`/`run_reference.py`）已通用化：按需装配弹簧/梁/目标点，
力窗口、目标速度剖面、IB 窄带、罚力核对全部由被移植模块与目标点文件导出，
`compare.py` 还会自动配对 `ib2d_reference<TAG>.npz`/`plot<TAG>/` 做任意
"加密/dt 减半"交叉核对；三处上游/数值细节已在 readme 记录
（demo_467 的边际罚力增益窄带尖峰、demo_468 的不可分辨圆柱、demo_471 的时间步敏感自由补片、
demo_472 的不可分辨高 Re 剪切层）。

`Example_Channel_Flow/*` 剩余：`Example_Poroelastic_Channel_Flow`（poroelastic 多孔弹性壁）。
已完成：`Example_Brinkman_Channel`（demo_475）、`Example_Concentration_Gradients` 全部三个（demo_476/477/478）、
`Example_Turek_Hron`（1024×128 × 16 万步 + **质量点**自由刚体，
需新增 `mass_pts` 机制）。

**下一步优先级（`mass_pts` —— 剩余机制里解锁面最大的一个）**：`Example_Mass_Points/*`
（5 个，均为 64² / dt=5e-5 / T=0.35，除一个 256²）与 `Example_Channel_Flow/Example_Turek_Hron`
等都需要它。算法骨架已在 `pyIB2d/IBM_Blackbox/{IBM_Driver.py:685-745, Supp.py:330-410}`
读完：质量点是**幽灵粒子**（`mass_info` = [标记点 id, x, y, k, M]），标记点随流体走，
幽灵点满足 `M dv/dt = -F_mass + M g`；每步顺序为
(1) 幽灵点先以 `dt/2` 走半步（`please_Move_Massive_Boundary`），
(2) 在 X_h 处算力时同时返回幽灵-标记之间的弹簧力 `F_Mass_Bnd`，
(3) `please_Update_Massive_Boundary_Velocity(dt/2, …)` 半更新速度，
(4) 位置复位后以 `dt` 走整步、再整步更新速度。
尚需确认的一处：`F_Mass_Bnd` 的**产生者**（`please_Find_Lagrangian_Forces_On_Eulerian_grid`
内部，需定位到具体文件/函数）与其符号约定。

其余仍需新机制的 `matIB2d` 算例：`Example_Boussinesq/*`（`boussinesq` + 浓度耦合）、
`Example_Coagulation/*`（`coagulation`）、
`Example_Scallop/*`（`3_element_muscle`，注意其 blackbox 版本要求 9 列肌肉文件）、
`Example_Test_Restart_Protocol/*`（restart）、
`Example_Mass_Points/*`（`mass_pts`）、`Example_VIV_Cylinder/*`、`Example_Meshmerize_Me/*`、
`Example_Tracers/*`、`Example_Baby_Spider_Shower`、`Example_Insect_Flight`、
`Example_Sawtooth`、`Example_KC/*`、`Examples_First_Year_Seminar/*`、
`Example_Concentration_Dynamics/*`（浓度源项 + 移动目标点，规模 6.4e4 步以上）、
`Example_Jellyfish_Swimming/Tentacle_Jelly`（poroelastic + 大网格）等；
`Examples_Education/*` 的 51 个目录绝大多数是同一算例的收敛/网格变体，作为一组参数扫描
统一处理更合适。

## 运行环境

- `afsi-dolfinx` conda 环境（dolfinx 0.10.0，Python 3.12），`afsic` 包可编辑安装
- 各算例 `readme.md` 内附具体运行命令（`generate_mesh.py` 生成固体网格 → `main.py` 求解）
- 多数算例支持环境变量覆盖（如 `STEPS`、`CIRCLE`、`GRID`）便于短程验证
