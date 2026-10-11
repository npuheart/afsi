# 自由落水网格加密

在相同物理参数和求解器代码下，将流体网格从 `64×64` 加密到 `128×128`，
网格间距从 `0.0125 m` 减为 `0.00625 m`。固体同步从 640 个三角形加密为
2496 个三角形。计算终止时间仍为 `0.25 s`。

## 复现

在仓库根目录、`afsi-dolfinx` 环境下执行：

```bash
export AFSI_OFFLINE=1
python afsic/demo/three_phase/main.py --case water-entry --nx 128 --ny 128 \
  --time .25 --max-steps 10000 --max-seconds 45 --output-every 400 \
  --snapshots --output /tmp/afsi-water-entry-128
# 如 status.json 中 complete 为 false，用完全相同参数加 --resume 续算，直到完成。
python afsic/demo/three_phase/main.py --case water-entry --nx 128 --ny 128 \
  --time .25 --max-steps 10000 --max-seconds 45 --output-every 400 \
  --snapshots --output /tmp/afsi-water-entry-128 --resume
python afsic/demo/three_phase/plot_entry.py /tmp/afsi-water-entry-128 --gif
python afsic/demo/three_phase/compare_entry.py \
  afsic/demo/three_phase/results/water_entry /tmp/afsi-water-entry-128 \
  --output /tmp/afsi-water-entry-comparison
```

粗网格运行方法见 [WATER_ENTRY.md](WATER_ENTRY.md)。比较脚本检查两个算例已完成、
物理参数和求解器源代码哈希一致，并生成 `grid_comparison.png`、`comparison.json`
和 `comparison.md`。

## 比较的范围

自动稳定性限制使细网格时间步从约 `9.96e-5 s` 降到约 `5.04e-5 s`，
界面平滑宽度固定为 1.5 个网格，物理宽度也随加密缩小。因此这里是流体、固体、
时间步及界面宽度的联合分辨率比较，不能用两个网格确定空间收敛阶。
首次越过初始水位不等于首次实际接触水面；局部 Jacobian 和密度/几何差异也必须
与总质量、总面积守恒一起检查。

## 实际结果（2026-10-11）

| Metric | 64 x 64 | 128 x 128 |
| --- | ---: | ---: |
| Steps | 2511 | 4956 |
| Fluid cell width (m) | 0.0125 | 0.00625 |
| Solid triangles | 640 | 2496 |
| Initial-waterline crossing (s) | 0.16611643 | 0.16324788 |
| Final COM height (m) | 0.3394556 | 0.33250148 |
| Final COM vertical velocity (m/s) | -0.46198422 | -0.54124082 |
| Final minimum J | 0.73108665 | 0.75435507 |
| Final maximum J | 1.9797142 | 1.7895235 |
| Final solid area change (%) | 0.47771301 | 0.17401435 |
| Final density/geometry L1 (%) | 6.3177354 | 5.1070248 |
| Final relative mass error | 3.8535305e-16 | 3.8554494e-16 |
| Final maximum divergence (1/s) | 5.0036641e-12 | 2.1156188e-10 |

细网格通过检查点续算完成 4956 步，所有保存时刻的固体 Jacobian 为正。
入水前的质心轨迹接近，入水后差异增大：最终质心高度相差约 6.95 mm，
竖直速度大小相差约 17.2%（以粗网格为基准）。固体总面积误差及密度/几何差异
减小，但局部体积变化仍明显，尚不能宣称网格收敛或可靠预测冲击载荷。

本次只新增比较脚本和说明，求解器未修改。已实际执行完整细网格算例、
参数一致性检查及绘图，并检查生成图像。结果保存在本地忽略目录：

- `results/water_entry_128/`：检查点、诊断、时序图和动画。
- `results/water_entry_comparison/`：对比图及机器可读指标。

生成数据不提交 Git；上面的命令可重新生成。
