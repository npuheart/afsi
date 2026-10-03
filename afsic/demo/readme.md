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
| `ib2d/demo_447` | IB2d 阻尼橡皮筋 | `Rubberband_with_Damped_Springs`：零静长强收缩闭环（64 点阻尼弹簧环，32×32, dt=1e-3, T=1.5s）；早段一致、长时为格式差异"应力算例"（两码 dt 均收敛，详见其 readme） |
| `ib2d/demo_448` | IB2d 心管肌肉泵 | `HeartTube_Muscle`：两条平行弹性壁（弹簧 + 不变梁 + 4 角目标点）+ 153 条 Hill 肌肉带，行波激活（10 Hz）驱动的蠕动泵（128×128, dt=1e-4, T=0.25 s），与 pyIB2d 对照 |
| `ib2d/demo_449` | IB2d 非不变摆动梁 | `Wobbly_NonInv_Beam`：两端钉住的非不变梁（$\kappa=10^{10}$, C=0，62 段）+ 2 目标点在 1×1 周期盒摆动（32×32, dt=1e-5, T=0.05）；标记点平均偏差 0.74h，与 pyIB2d 对照 |
| `ib2d/demo_450` | IB2d 重力细胞赛跑 | `Gravity_Cellular_Race`：两个弹性细胞（各 81 点）+ 每点"质量弹簧"（$k=10^6$）连到幽灵粒子（$M\in\{0.05,0.2,1\}$），重力经幽灵拖拽标记（64×64, dt=5e-5, T=0.35）；含 IB2d 质量点"幽灵粒子"模型考证，细胞轨迹偏差 <8e-4 |
| `ib2d/demo_451` | IB2d 阻抗泵 + 示踪粒子 | `Tracers_In_Impedance_Pump`：上下弹性壁（弹簧+不变梁+4 目标点）+ 11 根跨径泵弹簧（每步 $RL=1-0.9|\sin(2\pi\cdot 10t)|$）+ 110 个被动示踪粒子（5×5 盒, 64×64, T=2 s）；结构匹配 0.25–0.65h、管宽相位一致；净输运离散敏感（两码差 ~20%，128 网格又不同）、tracer 为拉格朗日混沌（参考自身 dt 减半也发散）——离散敏感应力算例，详见其 readme |
| `ib2d/demo_453` | IB2d 多孔滑移橡皮筋 | `Single_Porous_Rubberband`：64 点收缩环（弹簧 $k=10^7$、$L_0=0$）全点 porous 滑移（$\kappa=10^{-4}$，4 阶差分法向；porous 公式与 pyIB2d 逐位一致）（32×32, dt=1e-4, T=0.1）；**塌缩对 grad-div γ 极敏感**——γ=0 时 t≈0.043 崩溃（伪散度漏流），γ=2.5 时全程 dX≤1h、面积轨迹重合 |

## 运行环境

- `afsi-dolfinx` conda 环境（dolfinx 0.10.0，Python 3.12），`afsic` 包可编辑安装
- 各算例 `readme.md` 内附具体运行命令（`generate_mesh.py` 生成固体网格 → `main.py` 求解）
- 多数算例支持环境变量覆盖（如 `STEPS`、`CIRCLE`、`GRID`）便于短程验证
