# Demo 421 — 鱼游动（DFIBMFoam CircularFishSwimming 移植）

在 AFSI（FEniCSx/dolfinx）中用浸没边界法（IBM）模拟**鱼游动**。
算法继承自 `demo_339/multi_direct_forcing`（DFIBMFoam multi-direct forcing 的
FEniCSx 移植），把"固定圆柱（$U^d=0$）"扩展为"**游动鱼体**（$U^d=\mathrm{d}X/\mathrm{d}t$）"。

鱼体几何与运动学**忠实移植**原版 OpenFOAM 代码
`MsureCFD/DFIBMFoam` 的 `CircularFishSwimming` 案例
（`/tmp/DFIBMFoam/CircularFishSwimming/code/IBM.C` 的
`updateIbpCoordinate()` / `updateIbpVelocity()`）。

## 鱼体模型（DFIBMFoam 公式）

鱼身为 NACA 4 位厚度分布，叠加**行波摆动**：

$$h(x,t) = L\big(0.351\sin(x/L-1.796)+0.359\big)\cdot\sin\!\Big(2\pi\big(\tfrac{x}{\lambda}-\tfrac{t}{T_w}\big)\Big)$$

$$d(x) = \tfrac{0.125L}{0.2}\Big(0.2969\sqrt{x/L}-0.1260\,x/L-0.3516(x/L)^2+0.2843(x/L)^3-0.1015(x/L)^4\Big)$$

上表面 $y=h+d$、下表面 $y=h-d$。鱼整体绕原点以半径 $R$ 圆周游动（"Circular" 游泳），
鱼体纵轴始终切于圆周轨迹：

$$\theta(t)=\theta_0+\frac{2\pi t}{T_c},\qquad \text{rotation}=\theta-\frac{\pi}{2}$$

每个边界标记点的**期望速度**（喂给直接力的 $U^d$）由坐标变化给出（DFIBMFoam
`updateIbpVelocity`）：

$$\mathbf U^d_l = \frac{\mathbf X_l(t)-\mathbf X_l(t-\Delta t)}{\Delta t}$$

## 求解器（= multi-direct forcing + 移动体）

预测步（AB2）→ 迭代直接力（$F_l=(U^d_l-U_l)/\Delta t\cdot\Delta V_l$，Peskin 4 点
$\delta$ 核插值/扩散累加）→ 压力泊松 $\nabla^2 p=\frac{2}{3\Delta t}\nabla\cdot U$ →
速度修正（L2 投影）。与固定圆柱版唯一的本质区别：**$U^d\neq 0$ 且每步更新标记坐标并
重新 `evaluate_current_points`**（`IBInterpolation` 支持移动点）。

流体域为**闭合水槽**（四壁无滑移），压力在角落固定一个 DOF（消去零模态）。

## 文件

| 文件 | 说明 |
|------|------|
| `configuration.py` | 水槽/鱼体/IBM 参数（SI），`STEPS` 环境变量覆盖步数 |
| `fish_geometry.py` | 鱼体几何 + 摆动运动学（DFIBMFoam 公式移植，可向量化快速计算） |
| `main.py` | AB2 分步求解器 + 移动鱼体 multi-direct forcing 时间循环 |
| `output/` | `velocity.xdmf/.h5`、`pressure.xdmf/.h5`、`fish_trace.csv`（标记轨迹） |

## 运行

```bash
conda activate afsi-dolfinx
cd afsic/demo/demo_421
python main.py                  # 默认 1000 步 (T=1.0 s ≈ 2 个摆动周期)
STEPS=100 python main.py        # 短程冒烟验证
```

## 默认参数（与 DFIBMFoam 一致）

| 参数 | 值 | 说明 |
|------|-----|------|
| 鱼长 $L$ | 0.1 m | 弦长 |
| 截面数 | 200 | 上/下表面共 400 个标记（$\Delta s\approx0.5$ mm < h） |
| 波长 $\lambda$ | 0.1 m | 行波波长 |
| 摆动周期 $T_w$ | 0.5 s | |
| 游动半径 $R$ | 0.3 m | 圆周游动 |
| 绕圈周期 $T_c$ | 37.7 s | |
| 水槽 | $1.4\times1.4$ m | $N_x=N_y=140,\ h\approx0.01$ m |
| $\rho$ / $\mu$ | 1000 / 0.01 | $\text{Re}\sim10^3$（按游速 $\sim0.1$ m/s） |

## 已知局限（同 multi-direct forcing / afsic IBM）

- **均匀笛卡尔网格且域必须从原点 (0,0) 出发**（`IBMesh` 结构化索引 + **内核 bug**，
  见下）；**单进程**（MPI 多进程需重写 IBMesh 映射）
- **力积分定量性**：直接力积分 $\int f_{\mathrm{IBM}}dV$ 因 $\Delta V_l=\Delta s\cdot h$
  而 $\propto h$，随网格加密不收敛（见 demo_339 §8）。`main.py` 里打印的
  `F_thrust / F_lateral` 仅作**固定网格下的量级参考**；定量受力建议用
  控制体积动量平衡或包络面应力积分。

## ⚠️ 重要：afsic IBM 内核要求域从原点出发（已踩坑）

`afsic` 的 `IBKernel`（`kernel_helper.h`）计算格点索引用 $X_0=X/dh$（物理坐标直接除以
网格尺寸），**没有减去域原点 $x_0$**；而 `IBMesh::get_index`（build_map/extract_dofs/
assign_dofs 用）却用 $(x-x_0)/dx$。两者不一致 → **凡域不从 (0,0) 出发，IBM 施力/插值会
整体偏移 $x_0/dx$ 个格点**，力落到错误位置（本 demo 初版鱼力全落在左下角，产生
$|u|\sim0.9$ 的角点伪影、鱼区无流动）。

**规避**：流体域必须取 $[0,L_x]\times[0,L_y]$（原点出发），固体（鱼）绕圈中心相应移到
域内（本 demo 取域中心 $(0.7,0.7)$）。原 `demo_339/multi_direct_forcing` 域从原点出发
因此未触发。

**彻底修复**（改 C++，需重编译 `afsic_ext`）：`IBKernel::compute` 应改为
$X_0=(X-x_0)/dh$，并把 `IBMesh` 的 $x_0,y_0$ 传入内核。当前 demo 用"移固体"规避。
- 细长鱼身用**边界-only 标记**（忠实原版），未做内部掩码；若鱼身较胖可开
  `mask_interior`（需随体移动掩码，暂未实现）。

## 重要实现细节：C++ `solid_to_fluid` 是"替换"非"累加"

`afsic` 的 `IBInterpolation::solid_to_fluid` 内部 `IBMesh::assign_dofs` 用
`vector->setitem()`（**覆盖**），因此每次扩散调用会**覆盖**目标函数而不是累加。
multi-direct forcing 要求迭代累加体积力，故 `main.py` 在 **Python 层累加**：
每次迭代先扩散到临时场 `f_spread`，再 `f_ibm.x.array += f_spread.x.array`。
（不这么做的话只剩最后一次迭代的力，鱼身近似"透明"，流场几乎不被驱动——
这正是初版 u_L2≈0 的原因。）

> 同样的问题也存在于 `demo_339/multi_direct_forcing`（其圆柱因 `mask_interior=True`
> 掩盖了该缺陷，但 Cd 的网格依赖因此更严重）；如需严格 multi-direct forcing，建议
> 同样在 Python 层累加。

## 后续可扩展

- **直线游动 / 自推进**：把圆周运动改为直线平移，$U^d$ 由摆动+前进速度给出
  （DFIBMFoam 的 `desiredIbpVel` 机制，改 `fish_geometry.fish_surface` 即可）
- **多鱼**：`n_fish` 已支持，各鱼相位差 $2\pi i/n_{\text{fish}}$
- **定量受力**：接入控制体积动量平衡计算游动推力/阻力
