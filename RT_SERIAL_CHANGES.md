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
