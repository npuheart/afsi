# AV2D：心动周期压力回放试算

移植 WholeLVHeartModelling/AV2D 的几何、启用的材料和支撑力到 AFSI 的单相
Q2/Q1 流体与节点 IB 耦合。单位为 CGS；周期为 0.8512 s。尚未完成整周期验证。

## 当前推荐试验

```bash
conda activate afsi-dolfinx
export AFSI_OFFLINE=1
python afsic/demo/av2d/run_cycle.py \
  --source /Users/pengfei/Documents/GitHub/WholeLVHeartModelling/AV2D \
  --output afsic/demo/av2d/results/cycle_120_refined_implicit \
  --nx 120 --dt 2.5e-6 --solid-refinement 1 \
  --convection implicit --advection-refresh 20
# 同一命令加 --resume 可恢复；源代码、输入及数值参数必须保持一致。
```

流体 120×120 个 Q2/Q1 单元；IB 使用 Q2 节点间距 h=7.68/(2×120)=0.032 cm。
固体均匀细分一次：瓣叶 2360、主动脉壁 12448 个 TRI3 单元；边界间距中位数
约 0.016 cm，因此 MFAC≈0.5，初始边界最大值分别为 0.6044、0.5633。
细分保留 Exodus side sets 和根部弹簧边界，输入文件不改动。
时间步减半后共有 340480 步；启动脚本自动设置足够的步数上限。
main.py 保留原网格、显式对流默认值，便于进行单因素对照。

混合速度–压力系统采用后向 Euler 黏性和压力牵引开口。`--convection implicit`
采用冻结的 Oseen 速度、斜对称体对流项及隐式回流耗散项；每 20 步更新一次对流速度
并重新装配/分解矩阵。该间隔是性能折中，尚需做间隔和时间步收敛检查。
节点固体载荷与位置更新仍为显式，不能据此宣称整个 FSI 无条件稳定。
原来的 `--convection explicit` 和 `--solver chorin` 保留用于诊断。

## 进度与动画

`status.json` 的 `complete=true` 才表示达到目标时间。每 200 步保存检查点和进度，
每 2000 步保存快照；日志包含 CFL、变形后边界 MFAC、J 极值、位移、二维流量及节点间距。
生成数据由 Git 忽略。程序退出后自动绘图，也可随时运行：

```bash
python afsic/demo/av2d/plot.py afsic/demo/av2d/results/cycle_120_refined_implicit --gif
```

`motion.gif` 包含每帧实际速度场的流线、速度背景与固体几何，固体内部遮蔽。
色标固定为所有保存帧的速度第 99 百分位数的最大值，极值可能超出色标。
流线通过规则 Q2 节点速度插值得到，是瞬时流线，不是粒子轨迹。
另输出 `streamlines.png`、`sequence.png`、`history.png` 和 `velocity.png`。

`--start-from checkpoint.npz` 允许创建采用不同 dt/对流格式的诊断试验；检查流体坐标、
固体参考坐标与连接关系是否匹配，并记录种子文件哈希。它不支持不同固体网格之间的转移。
这类热启动试验只用于局部诊断，不能代替从 t=0 开始的新方案整周期计算。

## 检查与已完成试验

```bash
python afsic/demo/av2d/check.py /Users/pengfei/Documents/GitHub/WholeLVHeartModelling/AV2D
python afsic/demo/av2d/check.py /Users/pengfei/Documents/GitHub/WholeLVHeartModelling/AV2D --refinement 1
python afsic/demo/av2d/check_flow.py --implicit
python afsic/demo/av2d/check_restart.py /Users/pengfei/Documents/GitHub/WholeLVHeartModelling/AV2D --convection implicit
```

均匀压力应保持零速度；19+21 步重启与连续 40 步一致到舍入误差，包括冻结对流速度。
细分网格已检查单元数、20 条根部边、初始 J=1 和近零内力，并完成显式对流的 0.01 s 测试。
原未细分、显式对流整周期尝试在 0.563120 s 失败。相同检查点、相同半时间步的对照中，
显式版本仍在 0.563195 s 失败，隐式版本完成至 0.565 s。详细数值见 [TUNING.md](TUNING.md)。

## 物理范围

保留原密度 1 g/cm³、黏度 0.035 g/(cm·s)、各向同性指数瓣叶应力、体积惩罚、
根部弹簧、tanh buttress 力及主动脉超弹性/体弹簧。平面应变使用 I1=F2:F2+1。
压力回放使用 bc_data.txt 的 LVOT/主动脉记录，以 1333.22387415 将 mmHg 转为 dyn/cm²。
没有实时 LA/LV/Windkessel 反馈。原模型关闭的纤维与反应输运项仍关闭。

流体开口位置取原参数；侧壁和开口外无滑移，开口切向速度为零。主动脉恰好位于箱体
上下边界的端节点固定参考位置，这是额外的数值约束。核函数、流体离散和时间格式与
原 IBAMR 不同；原 IBAMR FeedbackForcer 未照搬。

无瓣叶接触算法。全局入口/出口流量平衡不能证明局部不漏流；节点最小距离也不能保证
瓣叶密封或避免穿透。完整周期、局部泄漏、网格/时间收敛和压力流量精度仍需验证。
