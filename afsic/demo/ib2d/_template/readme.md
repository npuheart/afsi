# IB2d 移植模板（`_template/`）

本目录**不是**一个 demo，而是 `afsic/demo/ib2d/port_example.py` 生成新 demo 时复制的
模板代码。它把 `demo_444` … `demo_478` 里已经实现过的**全部机制**合并成一份与具体
例子无关的 `main.py` / `compare.py` / `run_reference.py`：每个机制都由例子自己的
`input2d` 开关（以及 `ib2d_input/` 里是否存在对应的 `.py` 用户模块）自动决定是否启用，
所以同一份模板可以服务任意一个 IB2d 例子。

## 文件

| 文件 | 作用 |
|------|------|
| `main.py` | AFSI 移植：周期 Taylor–Hood 有限元 + Peskin 两步格式，所有结构力用 numpy 按 IB2d 公式在 `X^{n+1/2}` 上计算；可用环境变量 `NX/NY/DT/TFINAL/PRINT_DUMP/OUTPUT_PATH/GRAD_DIV` 覆盖 |
| `run_reference.py` | 跑 pyIB2d 参考解（`--tend/--dt/--print-dump/--nx/--src/--out/--run`），把生成的 VTK 转成 `ib2d_reference.npz` |
| `ib2d_reference.py` | pyIB2d VTK → `npz` 转换器 |
| `ib2d_io.py` | `input2d` 与所有 `*.vertex/.spring/.d_spring/.beam/.nonInv_beam/.target/.muscle/.mass/.porous/.poroelastic/.tracer/.concentration/.brinkman/.user_force` 读取；`detect_index_base()` 自动判别 0 基（pyIB2d）还是 1 基（matIB2d）编号 |
| `compare.py` | 逐帧对齐比较：标记点、速度/压力/力场、示踪粒子、浓度标量、结构标量、外力模块自检；输出 `figures/` 与 `compare_table.csv` |

## 已覆盖的机制（全部按需自动开启）

| `input2d` 开关 | 机制 | 备注 |
|----------------|------|------|
| `springs` | 线性 / 非线性弹簧 | 支持 IB2d 的**非线性度** `alpha`：`F = (alpha+1)/2·k·(L−L0)^alpha·d̂` |
| `update_springs` | 例子的 `update_Springs.py` | 在半步位移之后、力计算之前调用，传入**整步**位置（与 `IBM_Driver.py:696` 一致） |
| `damped_springs` / `update_damp_springs` | 阻尼弹簧 + `update_Damped_Springs.py` | `sF = kappa(L−L0)d̂ − b·dV_leader` |
| `beams` / `update_beams` | 不变梁 + `update_Beams.py` | 含 `check_If_Beam_Points_Pass_Through_Boundary` 移植 |
| `nonInvariant_beams` / `update_nonInv_beams` | 非不变梁 + `update_nonInv_Beams.py` | pyIB2d 驱动 711 行把 `beams_info` 误传给该模块，模板改为传非不变梁数组，并在 `run_reference.py` 里对参考解打了同样的补丁 |
| `target_pts` / `update_target` | 目标点 + `update_Target_Point_Positions.py` | 锚点存在可变的 `target_info`（N×4）里 |
| `FV_LT_muscle` | Hill 力–速 + 长度–张力肌肉 | 激活函数来自例子自己的 `give_Muscle_Activation.py`（注意上游 Lymphangion 例子把 `FV_LT_muscle` 写成 0 的怪癖） |
| `porous_media` | 多孔介质滑移 | 四阶单侧/中心模板 + 法向滑移 `−kappa·F_lag·n/|dX/ds|`，随后所有标记点取模回周期盒 |
| `poroelastic` | 多孔弹性位移 | 额外位移 `dt·F_spring/(mu·c)`，`F_spring` 只取弹簧力 |
| `user_force_model` | 通用用户力接口 | 读 `.user_force` 原样数组，调用 `give_Me_General_User_Defined_Force_Densities` |
| `arb_ext_force` | 任意外力 | 与 pyIB2d 共用同一份 `please_Compute_External_Forcing.py`；Q1 节点力→Q2 插值加到 `solver.f` |
| `brinkman` | Brinkman 阻力 | `F −= mu·Brink(x)·u^n` |
| `tracers` | 被动示踪粒子 | 同一套插值，无反馈 |
| `concentration` | 被动标量 | `Supp.perform_Time_Split_Upwind` 分裂显式迎风/中心差分 |

另外还处理了：标记点/示踪粒子的周期取模、dolfinx 合并重合顶点导致的**重合标记点偏移**
（分叉例子需要）、与 pyIB2d 转储节奏严格对齐的输出帧。

## 尚未覆盖的机制（前四个会直接报错拒绝运行）

* `mass_pts` + `gravity`：IB2d 的“鬼粒子”质量点模型（`Turek_Hron`、`Falling_Sphere`、
  `Baby_Spider_Shower`、`Leaf_64` 等例子需要）；
* `3_element_muscle`：Hill 三元件肌肉（`Lag_Struct_Params[9]`，`.muscle3`/`read_Muscle3_Points`）；
* `electro_phys`：电生理耦合（`Example_HeartTube` 家族）；
* `boussinesq`：Boussinesq 近似（`F += rho·f_Bouss·C`，`expansion_coeff`）。

上面四个在模板开头有 `assert not flag(...)`，会**报错并指出名字**，不会静默算错。

另外两类**模板不实现但也不报错**，因为参考解 pyIB2d 自己就忽略它们
（`pyIB2d/IBM_Blackbox` 全库检索确认：`IBM_Driver.py` 里完全没有引用）：

* `coagulation = 1`（16 个 matIB2d 算例）：pyIB2d 有
  `give_Me_Coagulation_Force_Densities.py` 但**从不调用** ⇒ 参考解也是"没有凝血项"，
  两边一致，demo 因此只是该算例的**无凝血**版本（真要用需自行实现）；
* `Restart_Flag = 1`（7 个 `Example_Test_Restart_Protocol` 算例）：重启/续跑协议，
  pyIB2d 不支持 ⇒ 两边都从头开始跑。

最后一个是**有代码但无算例可验证**的：`update_damp_springs`
（`update_Damped_Springs.py`）——上游 183 个算例目录里不存在这个 `.m` 文件，
所以该分支只照驱动调用点写好，未经实测（见下"机制的验证情况"）。

## 使用注意（移植经验，来自 demo_444…482）

* **`GRAD_DIV`（grad-div 稳定化 $\gamma$）是逐算例标定的**：钉死/慢速算例用 0；
  被外力驱动的封闭射流（demo_451/473/474/477）需要 $\gamma \approx 1$–2.5，
  否则 AFSI 的射流会**偏强 1.5 倍**（散度误差，比例恒定而非相位差）。
  模板默认 0，用 `GRAD_DIV=… python main.py` 覆盖；`compare.py` 会把它算进 `plot/`。
* **`dt` 的罚力稳定性上限**：稠密钉死目标点（$k \gtrsim 10^7$）在粗网格上会碰到
  AFSI 的显式罚力稳定性边界，经验判据是 $k\,dt^2/(\rho h^2) \lesssim 0.4\text{–}0.5$
  （demo_474 官方 `dt` 下该值为 0.49 时局部发散，把 `dt` 减半即可）。
  两边都用同一个 `dt`（`--dt` / `DT=`），所以对比仍然公平。
* 例子自带的 `print_dump` 可能**缺失**（如 `Two_Corals`）：两边都用
  `max(1, 步数/20)` 兜底，建议显式传 `--print-dump` / `PRINT_DUMP=` 以对齐输出帧。
* 结构标记点密度远大于网格时（水母 1/128 间距 vs `h = 1/32`）误差由 δ 核弥散主导，
  加密网格收益很大；建议同时留一个粗网格 `--out ib2d_reference_nNN.npz`
  + `OUTPUT_PATH=plot_nNN`，`compare.py` 会自动把它们配对并报告成"cross-check"。

## 机制的验证情况

* `demo_480`（`Hoover_Jellyfish`）、`demo_481`（`Two_Corals`）、`demo_482`
  （`Examples_Education/Interpolation/Swimmer`）是本模板的完整端到端验证：三者分别覆盖
  `update_Springs.py`、`update_Target_Point_Positions.py`、`update_nonInv_Beams.py`，
  并同时覆盖 springs / targets / nonInv beams / concentration 等机制。
* 其他用同一模板生成的 demo 又补上了：`update_Beams.py`（demo_608）、
  FV_LT 肌肉（demo_612）、poroelastic（demo_621）、porous_media（demo_483）等。
* 模板自身的组合冒烟测试（把 `IB2D_EXAMPLE` 指向别的 demo 的 `ib2d_input/`，跑几步，
  见交付说明）：damped springs（demo_447 的输入）、通用用户力 `user_force_model`
  （demo_457 的输入）、porous + poroelastic 同时开启（demo_454 的输入）、
  targets + concentration（demo_476 的输入）都正常。
* **唯一没有任何上游算例验证过的路径**是 `update_damp_springs`
  （`update_Damped_Springs.py`）：上游 183 个算例目录里**不存在**这个 `.m` 文件
  （已用 `find` 全库确认），因此这段代码只是照 pyIB2d 驱动的调用点
  （`IBM_Driver.py:716`，`d_springs_info = update_Damped_Springs(dt, current_time, d_springs_info)`）
  写的，没有实测。`damped_springs` 本身（demo_447 的力学）已实测正常。

## 用法

```bash
conda activate afsi-dolfinx
python port_example.py 480 Example_Jellyfish_Swimming/Hoover_Jellyfish
cd demo_480
python run_reference.py            # pyIB2d 参考解 -> ib2d_reference.npz
python main.py                     # AFSI      -> plot/afsi_result.npz
python compare.py                  # figures/ + compare_table.csv
```

`port_example.py` 只负责复制文件与打印待办清单；例子专属 `*.m` 用户模块的 python 移植、
规模缩减、验证与中文 `readme.md` 都由调用者完成。
