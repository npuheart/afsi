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

## 完整 T=3s 复跑（45°）

`main.py` 增加了两个环境变量覆盖，便于在别处运行与后处理：

| 变量 | 作用 |
|------|------|
| `OUTPUT_PATH` | 覆盖输出目录（默认 `plot/`），避免必须写进本算例目录 |
| `PROBE_TRACE` | 每步把位移探针追加写为 CSV（`t,x_disp,y_disp`），供与 `data/ani_*.csv` 逐步对比 |

```bash
export OUTPUT_PATH=/path/to/out PROBE_TRACE=/path/to/probe45.csv
mpirun -n 8 python main.py       # 48000 步
```

**务必用 MPI**：同样 200 步，单进程 125 s、8 进程 7 s；单进程跑满 T=3s 约需 7 小时，
8 进程约 **12 分钟**（48000 步 / 707 s）。

实测（45°，T=3s，48000 步，无 NaN）：

- 探针位移范围：$x\in[0.00015,0.6015]$、$y\in[0,0.4476]$，峰值都出现在 $t\approx1.25$ s。
- 与仓库内归档的 AFSI 45° 序列（`data/ani_*.csv`）**逐步吻合**：最大偏差 $3.7\times10^{-4}$
  （信号幅值 0.6014，即 0.06%）。
- 周期性：$t=0.25$ s 的 $x=0.5942$ 与 $t=2.25$ s 的 $x=0.6013$ 相差 1.2%，即仍在缓慢趋近
  周期态（前几个周期振幅略增）。
- 两叶片张开而非相互靠近：叶尖自由间隙由未变形的 0.21 增至峰值 1.08、回落至谷值 0.50
  （$L_y=1.61$）。全场最大速度 ≈9.2 m/s（入口峰值 10.5 m/s）出现在 $x\approx2.7$、
  即叶片下游的收缩处。
- 叶片整体 $|u_s|_{\max}=0.7515$（叶片长 0.7）。

补充：`generate_mesh.py` 需要 `gmsh`（本环境用 `pip/mamba install python-gmsh` 补装）；
`main.py` 无条件读取 `plot/mesh-340.xdmf`，故首次运行前必须先跑一次网格生成。