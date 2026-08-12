# demo_343 — 圆盘随流体通过二维理想瓣膜（IB-FE FSI）

二维通道（8×1.61，正弦入口 $5(\sin 2\pi t+1.1)\,y(1.61-y)$）内，两片理想瓣膜 +
两个圆盘通过浸没边界法（IB-FE）与流体耦合。

## 文件结构

| 文件 | 说明 |
|------|------|
| `configuration.py` | 参数配置（网格、时间、上下瓣膜刚度等） |
| `generate_mesh.py` | 生成固体网格（上下瓣膜 + 两圆盘）→ `plot/mesh-343.xdmf` |
| `main.py` | IB-FE FSI 求解 → `plot/circle<CIRCLE>/` |
| `plot/plot_centerline.py` | 圆盘中心线速度对比画图 → `line_disk*.csv/.png` |
| `plot/` | 网格、运行结果、csv、图 |
| `readme.md` | 本文档 |

## 材料参数（上下瓣膜不同）

Neo-Hookean：$P = \mu(F - F^{-T}) + \lambda \ln J\, F^{-T}$

| 部件 | 刚度 | 说明 |
|------|------|------|
| 上瓣膜 | $\mu_s = 5.6\times10^5$ | 软 |
| **下瓣膜** | **$\mu_{s,down} = 10\,\mu_s = 5.6\times10^6$** | **更硬**（`mu_s_down_factor=10`） |
| 圆盘 | $0.01\times P$ | 软，随流 |
| 瓣膜尖端固定 | 惩罚 $\beta = 5\times10^7$ | `dss(4)`(下) / `dss(15)`(上) |

## 运行

```bash
conda activate afsi-dolfinx
cd afsic/demo/demo_343
python generate_mesh.py                  # 生成固体网格（首次）
CIRCLE=1 STEPS=500 python main.py        # 有圆盘
CIRCLE=0 STEPS=500 python main.py        # 无圆盘对照（类比 demo_339 no_cylinder）
```

环境变量：
- `STEPS`：步数覆盖（$t = \text{STEPS}\cdot dt$，$dt=1/64000$）
- `CIRCLE`：`1`=含圆盘，`0`=无圆盘（对照）

## 圆盘中心线对比画图

```bash
cd plot
python plot_centerline.py
```

沿圆盘中心线 $y=0.5$（圆盘1）、$y=1.1$（圆盘2）采样 $u_x$，把 `circle1`（有圆盘）与
`circle0`（无圆盘）结果叠加在同一图上，对比圆盘对流场的影响。

## 说明

- 输出写到 `plot/circle<CIRCLE>/`；网格与 `*.xdmf`/`*.h5` 被 `.gitignore` 忽略，不入库。
- 性能：320×64 网格约 0.2 s/步；圆盘位于 x=0.5，需较长模拟时间流场才到达并推动圆盘，
  建议先用短 `STEPS` 验证流程。
