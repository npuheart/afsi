# 弹性固体自由落水（二维实验算例）

运行入口仍是 `main.py`，使用 `--case water-entry`。固体从静止释放，没有弹簧
支撑、固体阻尼或强制位移；固体重力与惯性通过已有 Eulerian 三相方程计算。
这是可变形弹性体，不是刚体入水模型。

## 运行和查看

```bash
conda activate afsi-dolfinx
export AFSI_OFFLINE=1
cd afsic/demo/three_phase
python main.py --case water-entry --snapshots --output-every 200 \
  --max-steps 5000 --max-seconds 45 --output /tmp/water-entry
# 未到 0.25 s 时，使用同样参数续算：
python main.py --case water-entry --snapshots --output-every 200 \
  --max-steps 5000 --max-seconds 45 --output /tmp/water-entry --resume
python plot_entry.py /tmp/water-entry --gif
```

生成 `water_entry_sequence.png`、`water_entry_history.png` 和 `water_entry.gif`。
绘图依赖 Matplotlib；GIF 另需 Pillow。`--snapshots` 在输出间隔和每次调用结束时
保存时序状态；动画按保存帧播放，帧间真实时间可能不均匀，应以图中时间标记为准。
源代码/参数哈希仍严格检查；旧版本算例检查点可读取绘图，但不能直接用新代码续算。

## 默认设置

| 项目 | 默认值 |
| --- | --- |
| 水箱、流体网格 | `[0,.8] × [0,.8] m`，`64 × 64`，四壁无滑移 |
| 初始水位 | `0.35 m` |
| 弹性块范围 | `[.34,.46] × [.46,.56] m` |
| 底面距初始水位 | `0.11 m` |
| 气/水/固体密度 | `1.2 / 1000 / 1200 kg/m³` |
| 固体材料 | 平面应变 modified neo-Hookean；`G=2000 Pa, K=20000 Pa` |
| 固体网格 | TRI3，最大边方向间距不超过背景网格间距的一半 |
| 初始速度、重力 | 全部静止，`g_y=-9.81 m/s²` |
| 时间 | 默认到 `0.25 s`；`--dt` 默认上限 `1e-4 s`，自动降低 |

可设置 `--water-level`、`--release-gap`、`--rho-solid`、`--shear`、`--bulk`、
`--nx`、`--ny`、`--dt`、`--time` 和 `--kernel`。初始间隙必须大于 `3*max(dx,dy)`，
避免固体与水面的平滑过渡带在释放前就重叠。非零 `--anchor-stiffness` 对本算例
会报错；默认溃坝算例仍保留原支撑设置。

## 入水诊断的含义

- `solid_com_x/y`、`solid_com_vx/vy`：按参考单元面积（即恒定参考密度下的质量）
  加权的质心和速度，不是对不规则网格节点做简单平均。
- `nominal_waterline_gap`：当前最低固体节点相对**初始**水位的高度。
- `nominal_entry_time`：该高度首次非正的时间步末时刻；未发生时为 `-1`，并随
  检查点保存。水面实际会变形，不能把它等同于精确的首次液固接触时间。
- `boundary_phi_min`：在固体边界顶点和边中点双线性采样的最小气液 level-set；
  是几何接近指标，受界面平滑和固体内部 level-set 延拓影响，不是接触力判据。
- `solid_area_relative_error`：固体总面积相对初始面积的误差；应同时查看局部
  `solid_J_min/max`，总面积接近守恒不代表每个单元都保持不可压缩。

## 本次实际验证（2026-10-11）

环境与 [原验证记录](VALIDATION.md) 相同。默认 `64 × 64` 运行 **2511 步到 0.25 s**，
通过两次有 45 秒循环上限的调用和检查点续算完成。

| 指标 | 最终值 |
| --- | ---: |
| 首次越过初始水位 | `0.16611643 s` |
| 质心高度、竖直速度 | `0.339456 m`，`-0.461984 m/s` |
| 最低固体节点高度 | `0.290761 m` |
| 总密度质量相对误差 | `3.85e-16` |
| 几何液体面积误差 | `0`（全局修正后的值） |
| 最大流体散度 | `5.00e-12 s⁻¹` |
| 固体总面积相对误差 | `+0.478%` |
| 局部固体 J 范围 | `[0.7311, 1.9797]` |
| 输运密度/几何密度相对 L1 差异 | `6.32%` |

自动检查包括原有 14 项数值测试和 2 项新增落水检查：初始位置、无支撑/无初始力、
重力导致向下释放、横向漂移很小、非法初始间隙被拒绝。另实际比较了 32×32、
`dt=1e-4`、20 步连续运行与 7+13 步续算：全部检查点字段逐位一致；原 32×24
溃坝短跑的已有检查点字段与本次改动前逐位一致。

```bash
# 从仓库根目录执行
AFSI_OFFLINE=1 python -m pytest afsic/tests/test_multiphase.py afsic/tests/test_water_entry.py -q
```

本地结果保存在忽略目录 `results/water_entry/`，不提交生成数据。

此结果展示下落、自由面变形及下沉减速，但仍有明显的局部固体体积误差和密度表示
差异。没有网格/时间收敛或实验对比验证，不能用于可靠的冲击峰值或加速度预测。
当前也不处理三维、刚体约束、表面张力、气体压缩和固体碰壁接触。

## 加密网格

已完成 `128×128` 流体网格及同步固体网格加密，运行至 `0.25 s`。
复现命令、粗细网格对比及精度限制见 [REFINEMENT.md](REFINEMENT.md)。

## 延长已有计算

`extend_entry.py` 将已有检查点复制到新目录，保留原始结果与参数来源记录，
只修改终止时间；`main.py` 的材料、网格和求解器源代码校验仍然有效。

```bash
python afsic/demo/three_phase/extend_entry.py \
  afsic/demo/three_phase/results/water_entry_128 \
  afsic/demo/three_phase/results/water_entry_128_2s --time 2.0
```

脚本自动以 45 秒计算段续算、保存检查点，并在结束后生成时序图及动画。
每段开始前检查固体距四壁的最小距离；距离小于等于 3 个网格间距时，
以 `wall_proximity` 停止，不把未实现的固体碰壁过程继续算下去。
这只是分段检查，不是接触力模型或每步碰撞检测。
`status.json` 的 `complete: true` 才表示达到目标时间；
`max_seconds` 表示段落结束、后续仍会自动续算，`error` 表示求解失败。
进程意外中断后，可用 `main.py`、相同新终止时间和 `--resume` 手动恢复。

2026-10-11 已从 `0.25 s` 启动 `2.00 s` 计算，结果目录如上；此启动记录不代表
已经完成。日志为 `results/water_entry_128_2s.log`，PID 文件为
`results/water_entry_128_2s.pid`。按前段性能，预计额外耗时约两小时。
