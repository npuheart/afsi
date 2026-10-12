# AV2D：一个心动周期的压力回放试算

本目录把 WholeLVHeartModelling/AV2D 的固体网格、当前启用的材料与支撑力移植到
AFSI 单相混合 Q2/Q1 流体 + IB 耦合。目标时间 0.8512 s，dt=5e-6 s，共 170240 步。
流体先用 120×120 个 Q2/Q1 单元（原程序为 240×240 MAC 单元；两者并不等价），
固体保持原来的 590 个瓣叶和 3112 个主动脉 TRI3 单元。串行运行。

## 运行

```bash
conda activate afsi-dolfinx
export AFSI_OFFLINE=1
python afsic/demo/av2d/check.py /Users/pengfei/Documents/GitHub/WholeLVHeartModelling/AV2D
python afsic/demo/av2d/run_cycle.py \
  --source /Users/pengfei/Documents/GitHub/WholeLVHeartModelling/AV2D \
  --output afsic/demo/av2d/results/cycle_120
# 同一命令加 --resume 可从最后检查点恢复。
```

输入为两个 Exodus 网格及 bc_data.txt，保持原始文件不变；运行元数据记录输入、main.py、coupled.py 和 ChorinSolver.py 的 SHA256。每 200 步保存检查点与进度，每 2000 步保存动画帧，status.json 的 complete=true 才表示整周期
完成。错误会保留最后一次保存的检查点。plot.py 可随时绘制已保存部分，run_cycle.py
在正常结束或求解报错后自动生成 sequence.png、history.png 和 motion.gif。

## 物理和数值约定

- 使用原 CGS 单位：长度 cm、密度 1 g/cm³、黏度 0.035 g/(cm·s)、压力 dyn/cm²。
  bc_data.txt 前三列提供时间、LVOT 压力和主动脉压力，mmHg 转换系数为 1333.22387415。
  只读取第一周期，忽略文件末尾可能不完整的记录。t=0 的压力设为 0，符合原升压过程。
- 直接回放原计算记录的两端压力，不包含 LA/LV/Windkessel 的实时反馈。
- 瓣叶保留指数各向同性应力、beta*J*log(J)*F^-T、side-set 4 根部弹簧和原始
  tanh buttress 力。纤维应力与反应输运在原源码中关闭，本试算也不添加。
- 使用平面应变 I1=F2:F2+1。IBTK 的二维 jacobian 会设置 Fzz=1，见
  [IBTK 源码](https://github.com/IBAMR/IBAMR/blob/master/ibtk/include/ibtk/libmesh_utilities.h)。
- 主动脉采用原 2*C1*(F-F^-T) 和体弹簧。为避免开口端节点移出 AFSI 插值区域，
  恰好位于箱体上下边界的主动脉端节点固定在参考位置；这是相对原弹簧处理的额外约束。
- 上下压力开口的中心和半宽取原 CirculationModel/input2d 参数，侧壁及开口以外为
  无滑移，开口切向速度为零。默认联立求解速度和压力，以压力牵引施加两端波形，并加能量回流稳定项；
  不再使用最初失败的 Chorin 压力投影。原 IBAMR 的 FeedbackForcer 离散未照搬；
  开口端的离散位置由流体网格决定。
- 耦合采用 AFSI 原有插值/分布，装配固体载荷直接加入流体右端。核函数和时间格式
  与原 IBAMR 不同；不修改原始材料刚度来掩盖不稳定。
- 无瓣叶接触算法。最小节点间距只是诊断，不保证闭合密封或阻止网格穿透。

## 检查与诊断

check.py 检查单元数量、全部 side-set ID、10 个根部边、参考态 J=1 和近零初始
内力及压力转换。开发期间还验证了 16×16 网格 40 步连续与 20+20 步续算的全部
检查点字段逐位一致。60×60 网格完成 0.005 s 的短程试算。

metrics.csv 记录流速(cm/s)、两套固体的 J 极值和位移(cm)、入口/出口二维流量
(cm²/s，均以向上为正)及两瓣叶组件最小节点间距(cm)。入口/出口流量差是投影
与边界处理误差的重要诊断。不要直接把二维流量当成 L/min。

这是一周期可行性试算，不是定量复现结果；应检查完整周期内的稳定性、闭合泄漏、
流量差及局部 J，再决定是否加密至原分辨率、改进边界离散并接入循环模型。

## 2026-10-12 启动记录

120×120 短程试算完成 400 步到 0.002 s，耗时 66.55 s；瓣叶 J 范围
[0.9998484,1.0001071]，主动脉 J 范围 [0.9995729,1.0003323]。
入口/出口二维流量为 -3.4438/-3.7662 cm²/s（约 8.6% 差异，以出口绝对值为基准），
表明边界/投影误差不能忽略。按该速度估算整周期约 8 小时，实际耗时可能变化。
短程结果归档在 results/smoke_120/。

已启动 results/cycle_120/，日志 results/cycle_120.log，PID 记录
results/cycle_120.pid。本段仅记录启动，不宣称完成；请以 status.json 为准。


## 2026-10-13 数值修正

最初的 Chorin 版本在第 821 步（0.004105 s）发生主动脉单元翻转。
减半时间步仍在 0.004035 s 失败。现默认使用 `coupled.py`：联立 Q2/Q1
速度–压力方程、隐式黏性、显式斜对称对流形式和牵引开口的能量回流耗散项。
矩阵采用串行 MUMPS 直接求解；流体连续性通过同一混合系统施加。
原 Chorin 可通过 main.py 的 `--solver chorin` 保留作诊断对照。

检查命令：

```bash
python afsic/demo/av2d/check_flow.py
python afsic/demo/av2d/check_restart.py /Users/pengfei/Documents/GitHub/WholeLVHeartModelling/AV2D
```

均匀压力平衡检查要求零速度和正确常压力；新解法重启检查在舍入误差范围内一致，
不要求 MUMPS 重新分解后的结果逐位相同。status.json 现在包含 PID、UTC 更新时间、
失败的确切时间步与固体编号，检查点每 200 步保存。源代码变化后必须开新结果目录。

调参实验、数值指标和结果目录见 [TUNING.md](TUNING.md)。新整周期输出使用
`results/cycle_120_coupled/`，保留旧失败结果。
