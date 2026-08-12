# demo_401 — 3D 精子（Sperm）IB-FSI 固体几何

3D 精子在流体中游动的浸没边界流固耦合（IB-FSI）算例。本目录提供**固体几何与网格生成**（与 demo_400 乌龟、demo_421 鱼游动同架构的 IB 前处理）。

![image-20260616190014846](https://githubimages.pengfeima.cn/images/20260616190016892.png)

## 几何（`a.py`，gmsh 生成）

三段式精子：球头 + 三段圆柱鞭毛（分段保持共形界面，便于分别标记）。

| 部件 | 形状 | 尺寸 |
|---|---|---|
| 头部 head | 球 | R = 0.05 |
| 颈部 neck（段 1） | 圆柱 | r = 0.01, L = 0.01 |
| 中段 mid（段 2） | 圆柱 | r = 0.01, L = 0.08 |
| 长尾 long（段 3） | 圆柱 | r = 0.01, L = 0.21 |

## 物理分组（facet tags）

- `15` 头部 HEAD
- `16` 尾尖 TIP（x 向平盘封盖）
- `17/18/19` 鞭毛三段侧面 NECK / MID / LONG
- 体积 tag `1`

## 文件

- `a.py` — gmsh 生成 `sperm3d.msh`（几何 + 网格 + 物理分组）
- `sperm3d.geo` — 中间 geo 文件
- `generate_mesh.py` — `sperm3d.msh` → `sperm-2.xdmf/.h5`（dolfinx 可读）
- `sperm-1.xdmf/.h5`、`sperm-2.xdmf/.h5` — 网格副本

## 运行

```bash
python a.py                # 生成 sperm3d.msh
python generate_mesh.py    # 生成 sperm-2.xdmf/.h5
```

> 注：本目录为 IB-FSI 的固体前处理；完整 FSI 求解（固定笛卡尔流体网格 + 拉格朗日固体网格经 `IBMesh`/`IBInterpolation` 耦合）可参照 `demo_400`（乌龟）、`demo_421`（鱼游动）。