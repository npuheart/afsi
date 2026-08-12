# Benchmark 405

3D 血管壁 FSI 模拟 (Vessel Wall FSI)

血管壁在流体中响应周期性内壁压力（线性增长至收缩压并保持）。

## 网格
- 固体：楔形单元 (Wedge) 血管壁网格 (`vessel_wall_wall_3d_tet.xdmf`)
  - 141,038 节点，140,612 单元
  - 边界通过几何位置自动分类（无预设 facet tags）
- 瓣膜：`leaflets_M2_tet.xdmf`；`merge_meshes.py` 将血管壁 + 瓣膜合并为
  `combined_vessel_leaflets.xdmf`（cell tags：1 = vessel，2 = leaflets）
- 流体：结构化六面体网格 (hexahedron box mesh)，8 × 8 × 20，32 × 32 × 80

## 边界条件
- 流体域：四周无滑移边界
- 入口：正弦来流（`U_max` = 1.0，`freq` = 1 Hz）
- 固体：两端固定，内壁施加 luminal pressure，外壁自由；整体用惩罚（beta = 1e6，
  volumetric spring，demo_402 模式）固定
- 耦合：IB 方法 (IBMesh3D / IBInterpolation3D)

## 材料
- Neo-Hookean 超弹性模型

## 文件
- `fsi_paralell.py` — 主 FSI 求解
- `fsi_paralell_vessel.py` — 血管壁单独 FSI 版本
- `fsi_paralell_vessel_valves.py` — 血管壁 + 瓣膜 FSI 版本
- `merge_meshes.py` — 合并 vessel + leaflets 网格
- `configuration.py` — 参数（T = 0.2 s，dt = 1/10000，rho = 1，mu = 0.036 CGS）

## 运行
```bash
mpirun -np 4 python3 fsi_paralell.py
```
