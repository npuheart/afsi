# Three-phase FSI: gas + liquid + deformable solid

这是一个可以运行的 **串行二维实验后端**，位于 `afsic.multiphase`。
它在 AFSI 中增加了交错 MAC 网格流体与 FEniCSx TRI3 弹性固体的完整耦合。
原来的 `IPCSSolver`、`ChorinSolver` 和 C++ 插值接口保持不变。

参考 Wei, Liu & Li, *A multiphase IFED method for fluid-structure interactions*,
JCP 552 (2026), 114699, https://doi.org/10.1016/j.jcp.2026.114699 。
本文的思路适用于 AFSI，但把既有常密度 IPCS 的 `rho` 替换为一个函数不足以
得到高密度比三相求解器。因此首版采用独立 MAC 后端，直接检查质量和动量通量。

## 运行

使用已经安装 AFSI、FEniCSx 0.10、NumPy 和 SciPy 的 `afsi-dolfinx` 环境；绘图还需
Matplotlib。NumPy/SciPy 也可通过 AFSI 的 `multiphase` 可选依赖安装：

```bash
conda activate afsi-dolfinx
export AFSI_OFFLINE=1
cd afsic/demo/three_phase
python main.py --output /tmp/three-phase --max-steps 200 --max-seconds 50
python main.py --output /tmp/three-phase --max-steps 200 --max-seconds 50 --resume
python plot.py /tmp/three-phase
```

每次调用至多推进指定步数或运行指定循环秒数；网格建立、JIT、读写检查点的时间不在
循环时间上限内。`--dt` 是上限，实际取流体 CFL/显式黏性、弹性波/支撑刚度估计、
重力估计和剩余终止时间的最小值。这些估计不是浸入式耦合稳定性的证明。
不支持 MPI 多进程，尝试使用时立即报错。

快速检查（粗网格只检查执行与数值不变量，不能解析板厚）：

```bash
python main.py --nx 32 --ny 24 --time .002 --dt .0001 \
  --max-steps 30 --output /tmp/three-phase-smoke
```

从仓库根目录运行测试：

```bash
AFSI_OFFLINE=1 python -m pytest afsic/tests/test_multiphase.py -q
```

## 算例和文件

- 箱体 `[0,1.2] × [0,0.8] m`，四壁无滑移，压力固定一个代数自由度。
- 初始水柱 `x < 0.58 m, y < 0.55 m`；气体/液体/固体密度分别为
  `1.2 / 1000 / 1100 kg/m³`，动力黏度分别为 `1.8e-5 / .001 / .001 Pa s`。
- 弹性板参考范围 `[.62,.68] × [.05,.40] m`，平面应变 modified neo-Hookean，
  默认 `G=2e4 Pa, K=1e5 Pa`。板底部 `y < .09 m` 的参考单元采用体积弹簧支撑
  （默认 `1e7 N/m⁴`）和阻尼（`1e4 N s/m⁴`），不是精确固定位移边界。
- 固体网格间距不大于背景网格间距的一半；默认流体网格 `96 × 64`。
- `parameters.json` 记录材料、网格、参数和源代码哈希；续算必须匹配。
- `metrics.csv` 记录流体质量、液体体积、最大散度、密度极值、固体 Jacobian、
  能量、位移，以及输运密度相对几何密度的 L1 差异。
- `checkpoint.npz` 为最近成功保存的完整状态，临时文件写完后原子替换。
  异常不会把半步状态覆盖到有效检查点。`status.json` 明确区分完成、限额和错误。
- `plot.py` 从检查点生成 `checkpoint.png`，无需重新计算。

这是自包含的演示几何，**不是**论文 Antoci/Liao 实验的参数复现，短跑通过也不代表
冲击位移、压力峰值或大变形已经得到实验验证。

## 离散方法与边界

1. `phi < 0` 表示液体；`psi < 0` 表示固体。用正则 Heaviside 形成三种互补权重，
   固体权重覆盖气液权重，对应论文式 (20)-(22)。
2. 气液界面采用一阶迎风守恒输运；每若干步提取分片线性零等值线并重建距离。
   全局常数位移通过标量求根保持 **固体外** 液体体积。它只约束总体积，不保证局部
   界面形状精度；与论文 WENO/subcell-fix/变分约束方案不同。
3. 结构距离来自当前有限元边界线段的最短距离和奇偶射线判定。适用于封闭、无自交
   的二维边界；没有接触、自交或拓扑变化处理。
4. 密度采用单元有限体积输运；将同一组质量通量平均到交错速度控制体，输运动量。
   压力投影解 `-div((1/rho) grad p) = -div(u*)/dt`。应力使用完整的
   `mu*(grad u + grad u.T)`，四壁切向零速用反对称 ghost 值实现。
5. 输运密度自初始化后持续守恒推进，不每步用 level-set 密度覆盖。几何密度用于
   重力，几何黏度用于应力。由于两套界面表示的数值扩散不同，其差异必须用
   `density_geometry_relative_l1` 监控；这也是本基线在长时间/高密度比下的限制。
6. 固体弱载荷 `b=-∫P:grad(v)dX`，铺展为 `f=Iᵀb/(dx*dy)`，
   速度插值为 `U=Iu`。因此 `b·U = dx*dy Σ f·u`，不能再对载荷乘有限元质量矩阵。
   这是节点耦合基线，与论文的高斯积分点耦合不同。默认 `--kernel bspline`：
   u 用 `B4(x)B3(y)`，v 用 `B3(x)B4(y)`（下标是支持宽度，次数分别是 3 和 2）。
   利用 `B4'(r)=B3(r+.5)-B3(r-.5)`，内部完整支持上的插值速度保持 MAC 离散
   散度为零的性质。参考 [Gruninger & Griffith 的复合 B-spline 构造](https://arxiv.org/abs/2408.08280)。
   `--kernel peskin4` 保留各向同性四点 Peskin 核用于比较。近墙核支持截断且不
   归一化，保持功率对偶，但不保证近墙常数重现和散度保持。有限元几何与显式时间
   积分仍可产生体积误差；不能据此声称固体单元体积精确守恒。
7. 固体惯性和重力已经包含在 Eulerian 三相场，未重复添加独立固体惯性/重力。
   采用单网格显式 IFED，尚无论文双网格 TS-IFED 子步进。

当前未实现：三维、AMR、高阶对流、界面接触角、表面张力、开放边界、刚体接触、
并行流体求解，以及实验/网格时间收敛验证。不可把此版本标为论文方法的完整复现。
