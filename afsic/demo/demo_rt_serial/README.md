# 单进程 RT 无散 immersed FSI 原型

在 AFSI 现有固体节点弱力装配和显式坐标更新的基础上，新增 RT 流体和
直接节点耦合。原有 Chorin/IPCS、C++ IB4 实现及 demo_402/main.py 保留。
新入口默认离线运行，不调用 SwanLab 或邮件接口。

## 安装和运行

按项目根目录 install.md 安装 DOLFINx 0.10.0 环境，在 afsic 目录执行：

```bash
python -m pip install nanobind 'scikit-build-core[pyproject]' swanlab
python -m pip install --no-build-isolation -ve .
python -m pytest -q tests/test_rt_serial.py
python tests/test_duality.py

# 相同背景网格、固体、dt 的两条运行路径
python demo/demo_rt_serial/main.py --method rt --nx 16 --steps 20 --convection --out results/rt-box
python demo/demo_rt_serial/main.py --method ib --nx 16 --steps 20 --convection --out results/ib-box
```

只运行一个进程，不使用 `mpirun -n 2`。通常 Conda 激活即可。如果动态库查找失败，
可以设置 `export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"`。
本次受限执行环境使用 `UCX_TLS=self` 运行单进程；常规机器不要求这个变量，
也不要把它用于多进程跨节点计算。

### Turek 固体

复用 demo_402 的 CGS 几何、P2 固体、Saint Venant–Kirchhoff 本构、圆柱体积系绳。
压缩包原来只有几何和生成脚本，网格由原脚本生成：

```bash
(cd demo/demo_402 && python generate_mesh.py)
python demo/demo_rt_serial/main.py --case turek --method rt \
  --nx 32 --ny 8 --steps 10 --dt 5e-5 --convection --out results/rt-turek
```

入口为抛物线型，均值由 `--inlet-speed` 指定（默认 200 cm/s），在 `--ramp-time`
（默认 2 秒）内余弦加速。壁面和入口法向速度强施加、切向用 Nitsche；出口为零自然
伪牵引（对应 mu*grad(u)-pI 形式），不是给 DG 压力加边界节点值。
圆柱 `--kappa-hat` 默认 0.1，刻意不照搬 IB4 算例的默认 1；不同传递算子的显式
刚度限制不同。长时/高雷诺数 Turek 需要另外做网格、时间步、罚参数和出口稳定性验证。
短时启动不等于复现了 Turek 振幅、频率或阻力基准。

## 代码和接口

- `src/afsic/euler/RTFluidSolver.py`：RT/DG 混合方程，BE 时间项，SIP 黏性项，
  可选滞后速度迎风对流。四边形上 Basix RT degree=2 配 DG Q1；三角形配 DG P1。
  直接用 MUMPS 求解，闭域处理压力常数零空间，Stokes 矩阵可复用。
- `src/afsic/coupling/RTNodalCoupling.py`：`update(chi)` 重建节点求值矩阵 E；
  `interpolate(u, U)` 计算 E*u；`spread(L, b)` 计算 E.T*L。
- `demo/demo_rt_serial/main.py`：完整离线 FSI 时间步，可选择 rt 或原始 ib4 对照。
- `tests/test_rt_serial.py`：8 项数值验证（含参数组合）。

```python
coupling = RTNodalCoupling(rt_solver.V)
coupling.update(solid_coords)       # 当前物理坐标，而非参考网格坐标
b_ib = coupling.spread(solid_load)  # solid_load 已装配，不能再乘节点权重
rt_solver.solve_one_step(b_ib)
coupling.interpolate(rt_solver.u_, solid_velocity)
solid_coords.x.array[:] += dt * solid_velocity.x.array
```

`b_ib` 是 RT 速度空间的载荷向量，不是力密度 Function。
不乘质量矩阵，不乘/除辅助格点面积，不解 RT 投影。
速度求值矩阵通过互不共享单元的 DOF 着色，批量调用 DOLFINx Function.eval 建立；
Piola 映射和 DOF 方向变换由 DOLFINx 处理，避免自己假设基函数方向。
每次移动后必须更新 E；点在单元面上时选最小本地单元编号的迹，正反传递共用该迹。
点离开背景网格会明确报错。RT 切向不连续仍然存在，此规则不是光滑重构。

## 验证范围和限制

- 8 项测试包括：斜三角形/四边形上的任意系数求值、功率伴随、总力/力矩、移动点、
  域外点拒绝；有/无对流时的散度及法向跳跃；梯度体力不产生流动；开口流量平衡。
- 不修改已有 IB4 路径；已有对偶性脚本也须通过。
- 新 `--method ib` 对照使用现有 ChorinSolver，显式启用直接载荷，保留 IB4 格点面积
  抵消步骤。它和 RT 路径的流体离散及时间算法不同，差异不能全部归因于插值核。
- 固体采用 P2 三角形、节点传递和显式 Euler；背景无散不等于固体几何严格保体积。
- 同构形功率伴随不等于全离散能量守恒；本原型也不保证任意压力跳跃下静态平衡。
- `sampled_min_J` 是参考单元采样点处最小值，不是全单元正 Jacobian 的证明。
- 原型只支持实数、单进程、二维算例；不宣称 MPI 可扩展性或完整生产级边界处理。
- RT 基函数直接点力耦合缺少 IB4 平滑，必须重新评估显式稳定性与结构点密度。
- 当前 RT/MUMPS 原型比原来的小规模 Chorin/IB4 对照慢，尚未优化性能。

## 输出

`history.csv`：逐步散度、功率残差、体积漂移、采样 Jacobian、动能等。
`summary.json`：测试配置与最大误差。
`final_state.npz`：固体参考/当前节点坐标与速度、流体及压力系数。
系数必须结合相应网格和有限元空间解释，不是流体节点速度。

## 来源

RT/SIP/迎风流体形式参考 FEniCS DOLFINx 0.10.0 官方示例
https://github.com/FEniCS/dolfinx/blob/v0.10.0/python/demo/demo_navier-stokes.py
（LGPL-3.0-or-later）；固体载荷和 IB4 对照沿用本项目。
