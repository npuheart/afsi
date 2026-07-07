# Benchmark 405

3D 血管壁 FSI 模拟 (Vessel Wall FSI)

血管壁在流体中响应周期性内壁压力（线性增长至收缩压并保持）。

## 网格
- 固体：楔形单元 (Wedge) 血管壁网格 (`vessel_wall_wall_3d.xdmf`)
  - 141,038 节点，140,612 单元
  - 边界通过几何位置自动分类（无预设 facet tags）
- 流体：结构化六面体网格 (hexahedron box mesh)

## 边界条件
- 流体域：四周无滑移边界
- 固体：两端固定，内壁施加 luminal pressure，外壁自由
- 耦合：IB 方法 (IBMesh3D / IBInterpolation3D)

## 材料
- Neo-Hookean 超弹性模型

## 运行
```bash
mpirun -np 4 python3 fsi_paralell.py
```
