# demo_340 — 二维理想瓣膜（FSI）

流体-固体耦合模拟：两片理想瓣膜在脉冲入口流动下的变形。

## 文件结构

| 文件 | 说明 |
|------|------|
| `main.py`          | 主程序：ChorinSolver 流体 + IB-FE 瓣膜结构耦合（默认 FRH 本构，45° 纤维） |
| `materials.py`     | 本构模型：`FRHMaterial`（纤维增强超弹性，默认）+ `NeoHookeanMaterial`（备用） |
| `generate_mesh.py` | 生成瓣膜固体网格 → `plot/mesh-340.xdmf`（上/下两叶片） |
| `plot/`            | 数据 + 绘图代码：参考数据 csv、`data/ani_*.csv`、绘图脚本 `plot_1/2.py` |
| `readme.md`        | 本文档 |

## 运行

```bash
conda activate afsi-dolfinx
python generate_mesh.py   # 生成固体网格 plot/mesh-340.xdmf（首次运行）
python main.py            # 运行 FSI（或 mpirun -n <N> python main.py）
```

## 说明

- 主程序默认使用 `FRHMaterial`，纤维方向 45°（`f1 = (√2/2, ±√2/2)`）；
  切换 60°/75° 纤维方向可在 `main.py` 中取消对应行注释（或重跑绘图数据时改 `f1`）。
- 运行输出（`velocity.xdmf` / `solid_force.xdmf`）写到本地 `plot/` 文件夹（被 `.gitignore` 忽略，不入库，可再生成）。
- 所有图与画图数据均可再生成：
  - `plot/plot_1.py` 对比 AFSI 与文献（Ryan M2/M3、Kamensky）位移（→ `smoothed_x/y.png`），
    AFSI 曲线读取 `plot/data/ani_*.csv`（45°）。
  - `plot/plot_2.py` 对比 45/60/75° 纤维角位移（→ `Anisotropic_x/y_displacement.png`），读取 `plot/data/ani_*.csv`。
  - `plot/data/ani_{t,x,y}.csv`：三个纤维角各跑一次 `main.py`（完整 T=3s，或 `STEPS=` 短跑），
    由位移探针导出；列名对应 `demo-340-000092/000091/000090`（45/60/75°）。
- 参考数据 CSV（`X_M2`/`X_FSI`/`x_dis_ALE`/`Y_ALE`/`Y_FSI`/`y_M2`，文献 Ryan M2/M3、Kamensky）随仓库保留。