# 单进程 RT immersed coupling 改动与验证

新入口和使用说明：`afsic/demo/demo_rt_serial/README.md`。

本次新增 RT 流体求解器、RT 节点求值/伴随力装配、独立离线双向 FSI 入口，以及数值
验证。保留既有 IB4、Chorin/IPCS 和 demo_402/main.py。新增接口从 afsic 顶层导出。
不含 conda 环境或平台相关已编译扩展，收到后须按 install.md 安装/编译。

## 已运行（2026-10-02）

环境：Linux，Python 3.12，DOLFINx/Basix 0.10.0，实数 PETSc，单 MPI 进程。

- 新增 `test_rt_serial.py`：8 passed。
- 既有 `test_duality.py`：9 passed, 0 failed。
- 20 步弹性方块，两种方法同背景 16×16 四边形网格、dt=0.001、含对流。
- Turek 网格使用原 `demo_402/generate_mesh.py` 与 `turek.geo` 生成，已随包附上。
- Turek：32×8 背景，dt=5e-5，10 步正常 2 秒入口加速；额外跑 20 步 0.02 秒快速
  加速的启动压力测试（这不是论文的标准时间参数）。

| 测试 | 最大散度 L2 | 最大归一化功率差 | 最终相对体积漂移 |
|---|---:|---:|---:|
| RT 弹性方块 | 4.63e-15 | 8.67e-19 | +1.37e-5 |
| 原 IB4 + Chorin 方块 | 7.03e-2 | 1.73e-18 | -2.42e-4 |
| RT Turek 正常启动 | 6.34e-18 | 5.42e-20 | -4.13e-12 |
| RT Turek 快速启动 | 8.92e-14 | 6.35e-16 | -3.25e-7 |

RT 和 IB4 流体离散、线性求解器与时间算法不同，表格不是只改变 coupling 的消融
实验；也不是同自由度/同精度的性能比较。RT 背景严格无散不等于显式 P2 固体几何
严格保体积。快速 Turek 启动的最大固体位移约 4.42e-4 cm，所有采样 Jacobian 为正。
正常 Turek 启动入口非常小，不能用它的微小体积误差宣称长时保体积优势。

当前 RT 直接求解原型明显比小规模 IB4/Chorin 对照慢。计时未作隔离性能测试且不
含全部 JIT/setup 开销，不宣称加速。没有完成长时 Turek 振幅/频率/阻力基准验证。

原始数值记录位于 `validation/rt_serial/`，包括逐步 CSV、摘要 JSON 和最终系数。
摘要中的计时只作本次运行记录，环境中部分进程并行执行，不用于性能结论。
最后增加的摘要配置字段不影响数值；旧记录的运行参数以上述说明与 CSV 为准。

## 文件改动

- 新增 `afsic/src/afsic/euler/RTFluidSolver.py`
- 新增 `afsic/src/afsic/coupling/{__init__,RTNodalCoupling}.py`
- 在 `afsic/src/afsic/__init__.py` 导出两个新类
- 新增 `afsic/demo/demo_rt_serial/{main.py,README.md}`
- 新增 `afsic/tests/test_rt_serial.py`
- 新增 Turek 生成网格和本次测试记录

直接力装配为 `b = E.T @ L`，其中 L 已是固体弱形式节点载荷。与旧 IB4 格点路径
不同，RT 路径不能再乘格点面积；无需额外全局 RT 投影或质量矩阵求逆。

## 追加：块预条件线性求解器（linear_solver="block"）

新增可选迭代求解路径 `RTFluidSolver(..., linear_solver="block")`，默认仍为直接
分解（MUMPS LU），块模式为显式切换。系统按 (RT 速度) × (DG 压力) 组装为 PETSc
MatNest，无对流时用 MINRES，含对流（lagged upwind 使矩阵非对称）时用右预条件
FGMRES（restart=150）。压力预条件策略：

- 默认 `pressure_scale=None`/`"simple"`：SIMPLE 型 fieldsplit
  （`PCFieldSplit SCHUR/UPPER` + `SELFP`，无对流时自动换成对称的 `SCHUR/DIAG`），
  PETSc 用真实非对角块构造 SIMPLE Schur 补并直接求解；速度块用 LU（或 hypre
  BoomerAMG）。
- `"auto"`：加性 fieldsplit + 压力质量阵缩放，c 由组装时按
  `median(diag(B diag(T)^-1 B^T) / diag(Mp))` 自动标定（量纲/单位制自适应）。
- 正浮点数：固定 c；`"schur-diag"`/`"lsc"`：诊断用的逐 DOF 对角 / 全 LSC 近似。

安全性：真残差范数判据（`UNPRECONDITIONED`，避免预条件范数假收敛）、
max_it 不再伪装成 `CONVERGED_ITS`，未收敛一律抛 `RuntimeError`（含原因、
迭代数、rtol）；每步重组 upwind 算子前先 `ksp.reset()`（修复 fieldsplit
缓存悬垂引用导致的段错误）。CLI 新增 `--linear-solver/--velocity-pc/
--pressure-scale/--ksp-max-it/--ksp-rtol`，摘要增加 `pressure_scale_mode/
iterations_total/iterations_max`，history 增加 `solver_iterations`。

### 验证（macOS arm64，DOLFINx 0.10.0 / petsc4py 3.25.5，单 MPI 进程）

`tests/test_rt_serial.py` 14 passed（含 block vs direct 回归，rtol=1e-12）。
与同参数直接解对比（`|du|` 为 RT 速度系数最大差）：

| 场景 | 模式 | 迭代总数/单步最大 | 最大散度 L2 | 对比直接解 |
|---|---|---:|---:|---:|
| 弹性方块 16×16, 20 步, 对流 | simple（默认） | 772/41 | 7.5e-8 | 2.1e-10 |
| 同上 | auto 加性 | 2012/154 | 9.7e-8 | 1.6e-10 |
| Turek 32×8, 10 步正常启动 | simple（默认） | 437/49 | 2.9e-13 | 1.0e-13 |
| 同上 | auto 加性 | 3936/399 | 5.7e-13 | 3.4e-11 |
| Turek 20 步快速启动 | simple（默认） | 242/14 | 1.2e-8 | 7.2e-9 |

直接解散度为 4.2e-15（方块）/1.9e-17（Turek）；块模式为迭代解法，散度下限
≈ `ksp_rtol`×右端量级（此处 rtol=1e-10），不达机器精度是预期行为。有对流的
快速启动场景在早期加性模式 2000 步不收敛（残差真值停滞 ~1e-4，非重启问题，
亦非假收敛），现改为干净报错并提示；切到默认 SIMPLE 后仅 14 步/步。
`--pressure-scale auto` 保留原标量标定路径作为对照与回退。

### 文件改动（追加）

- `afsic/src/afsic/euler/RTFluidSolver.py`：块模式、预条件策略、收敛守卫
- `afsic/demo/demo_rt_serial/main.py`：CLI 参数与摘要字段
- `afsic/tests/test_rt_serial.py`：block 回归测试（convection × pressure_scale）
