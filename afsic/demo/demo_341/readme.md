# demo_341 — 三维方腔驱动圆球（FSI）

三维方腔 $[0,1]^3$ 内被流动驱动的圆球。入口 $u_x=1$，其余壁面无滑移；
圆球（Gmsh 生成，半径 0.2、中心 (0.6,0.5,0.5)）通过三维浸没边界法
（`IBMesh3D` / `IBInterpolation3D`，Peskin 类耦合）与流体相互作用。

## 文件结构

| 文件 | 说明 |
|------|------|
| `navier-stokes.py` | 纯 Navier-Stokes 求解（方腔驱动，无固体）→ `plot/ns_N<grid>/` |
| `fsi_paralell.py` | NS + FSI 求解（方腔驱动圆球）→ `plot/fsi_N<grid>/` |
| `generate_mesh.py` | 生成圆球固体网格 → `plot/mesh-341.xdmf` |
| `plot/plot_lines.py` | 后处理：t=1 时刻三条中心线速度插值 → csv + 三幅对比图 |
| `plot/` | 网格、各 run 运行结果、`line_*.csv`、`line_*.png` |
| `readme.md` | 本文档 |

## 运行

```bash
conda activate afsi-dolfinx
cd afsic/demo/demo_341

# 1) 生成圆球固体网格（首次）
python generate_mesh.py

# 2) 运行 NS 与 FSI（背景网格密度 GRID、步数 STEPS 可用环境变量控制）
GRID=16 STEPS=200 python navier-stokes.py   # NS，t=1s（200 步）
GRID=16 STEPS=200 python fsi_paralell.py    # FSI，t=1s
```

### 环境变量

| 变量 | 默认 | 说明 |
|------|------|------|
| `GRID` | 32 | 背景网格密度 $N_x=N_y=N_z$（常用 8/16/32） |
| `STEPS` | 完整 T/dt | 运行步数覆盖（t=1s 用 200） |
| `CASE` | ns / fsi | 输出子目录名（`plot/<case>_N<grid>/`） |

## 后处理与画图

```bash
cd plot
python plot_lines.py
```

遍历 `plot/` 下所有 `*_N<grid>/velocity.xdmf`（NS 与 FSI、各背景网格密度），
取时间最接近 t=1.0s 的帧，沿三条中心线插值速度分量：

| 线 | 采样 | 输出 |
|----|------|------|
| $(x, 0.5, 0.5)$ | $u_x$ | `line_x.csv` / `line_x.png` |
| $(0.5, y, 0.5)$ | $u_y$ | `line_y.csv` / `line_y.png` |
| $(0.5, 0.5, z)$ | $u_z$ | `line_z.csv` / `line_z.png` |

不同背景网格密度（NS 实线 / FSI 虚线）叠加画在同一幅图上，便于对比
NS 与 FSI 的差异以及网格收敛性。

## 说明

- 输出与网格均写到本地 `plot/`（`*.xdmf` / `*.h5` 被 `.gitignore` 忽略，
  不入库，可再生成）。
- swanlab 联网调用在离线运行时被屏蔽（不影响求解）。
- 性能参考（单核）：$16^3$ 约 1.4 s/步、$32^3$ 约 12 s/步，建议先用 8/16
  快速验证，再跑 32。
