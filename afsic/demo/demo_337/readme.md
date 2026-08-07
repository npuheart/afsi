# demo_337 — 理想左心室的舒张与收缩（IB-FE）

理想化左心室（LV ellipsoid）在生理性心室内压加载下的流固耦合（IB-FE）仿真：
被动 Neo-Hookean 弹性固体 + 三维 Chorin 投影流体解算器。

方法（每时间步）：
1. 在当前的体力 $f$ 下求解 NS 方程
2. 将流体速度插值到固体网格节点（fluid → solid）
3. 推进固体位置：$X \mathrel{+}= v\, dt$
4. 计算固体内力（Neo-Hookean PK1 应力 + 基底约束 + 心室内压牵引）
5. 将固体力扩散回流体网格作为 $f$（solid → fluid）

---

## 目录结构

```
demo_337/
├── readme.md                 # 本指导文档
├── configuration.py          # 全部参数（流体/固体/网格/输出路径）
├── main.py                   # 主运行程序（输出到 data/results/）
├── NeoHookean.py             # 本构：Neo-Hookean
├── PressureEndo.py           # 生理压力波形（舒张/收缩）
├── fsi_paralell*.py          # （旧版变体，供参考：纤维/主动收缩/计时）
├── data/
│   ├── mesh/                 # LV 椭圆网格（pulse-fenicsx / Docker 生成）
│   │   └── lv_ellipsoid/geometry/{mesh.xdmf, mesh.h5, markers.json}
│   ├── reference/            # 参考数据（Docker / 外部生成）
│   │   ├── ideal_middle_wall.txt       # 中壁参考线
│   │   ├── diastole-afsi.txt           # AFSI 舒张结果
│   │   ├── diastole-pulse-disp.txt     # pulse 舒张参考位移（Docker 生成）
│   │   ├── systole-pulse-disp.txt      # pulse 收缩参考位移（Docker 生成）
│   │   ├── diastole-ibamr.csv          # IBAMR 舒张参考
│   │   └── systole-ibamr.txt           # IBAMR 收缩参考
│   ├── results/              # main.py 输出（velocity/solid_force + metrics.csv）
│   ├── figures/              # 画图输出（.png）
│   ├── 32x32x32.csv          # 并行性能数据
│   ├── 64x64x64.csv
│   ├── paralell_analysis.csv
│   └── plot/                 # 网格/参考数据生成 + 画图命令
│       ├── docker-compose.yml        # fenicsx-pulse 容器
│       ├── generate_mesh.py          # 生成网格（Docker 内运行）
│       ├── bench-ilv-inflation.py    # pulse 舒张参考数据（Docker 内运行）
│       ├── bench-ilv-contraction.py  # pulse 收缩参考数据（Docker 内运行）
│       ├── middle_wall_location.py   # 中壁参考线
│       ├── plot_diastole.py          # 舒张对比图
│       ├── plot_systole.py           # 收缩对比图
│       ├── plot_systole_fit.py       # 收缩拟合图
│       ├── plot_percentage.py        # 占比图
│       └── plot_speedup.py           # 并行加速比图
```

---

## 0. 快速复现（diastole 对比图）

以下命令可直接复现 `data/figures/diastole_plot.png`
（Initial / Pulse / AFSI 三条曲线；IBAMR 数据可选，缺省时自动跳过该曲线）。

```bash
# 0) 激活环境（画图需要 pandas；本机已装）
conda activate afsi-dolfinx-v1

# 1) 生成 LV 椭圆网格（data/mesh 已存在则跳过）
#    镜像入口是 jupyter lab，必须 --entrypoint 覆盖；挂载整个 data/ 使 ../mesh 映射正确
cd demo_337/data/plot
docker run --rm --entrypoint /usr/bin/bash \
  -v "$PWD/..":/repo -w /repo/plot \
  ghcr.io/finsberg/fenicsx-pulse:v0.4.1 -c "python generate_mesh.py"

# 2) 生成 pulse 舒张参考数据 -> ../reference/diastole-pulse-disp.txt
docker run --rm --entrypoint /usr/bin/bash \
  -v "$PWD/..":/repo -w /repo/plot \
  ghcr.io/finsberg/fenicsx-pulse:v0.4.1 -c "python bench-ilv-inflation.py"

# 3) 画图 -> ../figures/diastole_plot.png
python plot_diastole.py
```

前置数据：
- `data/reference/ideal_middle_wall.txt`、`diastole-afsi.txt`（AFSI 结果）需已存在
- `data/reference/diastole-ibamr.csv` 可选（无则画图脚本自动跳过 IBAMR 曲线）

---

## 1. 生成网格与参考数据（Docker）

网格和 pulse-fenicsx 参考数据在 `fenicsx-pulse` 容器内生成
（镜像内置 `cardiac_geometries` + `pulse` + dolfinx）。

```bash
cd demo_337/data/plot

docker compose up -d
docker exec -it fenicsx-pulse-container /usr/bin/bash

# 在容器内（data/ 挂载为 /repo，工作目录 /repo/plot）：
cd /repo/plot
python generate_mesh.py            # -> ../mesh/lv_ellipsoid/geometry/（网格）
python bench-ilv-inflation.py      # -> ../reference/diastole-pulse-disp.txt
python bench-ilv-contraction.py    # -> ../reference/systole-pulse-disp.txt
python middle_wall_location.py     # -> ../reference/ideal_middle_wall.txt
exit
```

> 单次运行（不进容器交互）可改用 docker run 一键命令（见上方「快速复现」，
> 把 `bench-ilv-inflation.py` 换成目标脚本即可）：
> ```bash
> docker run --rm --entrypoint /usr/bin/bash \
>   -v "$PWD/..":/repo -w /repo/plot \
>   ghcr.io/finsberg/fenicsx-pulse:v0.4.1 -c "python bench-ilv-inflation.py"
> ```

> 网格也可用本机已装好的 `cardiac_geometries` 直接生成：
> 见 `data/plot/generate_mesh.py` 中的参数（与基准一致：r_short=7/10, r_long=17/20）。
> `markers.json` 需含 `ENDO` / `BASE` / `EPI` 三个面标记。

## 2. 运行主程序（main.py）

在 `demo_337/` 目录下运行（参数见 `configuration.py`，输出到 `data/results/`）：

```bash
# 激活环境
conda activate afsi-dolfinx-v1

python main.py                    # 单进程
mpirun -n <N> python main.py      # 并行
```

输出：
- `data/results/velocity.xdmf(.h5)` — 流体速度（P1 插值）
- `data/results/solid_force.xdmf(.h5)` — 固体力 + 固体坐标
- `data/results/metrics.csv` — 每步指标（u_L2, p_L2, 固体力范数, LV 容积, 心室内压）

> 说明：`configuration.py` 中 `mesh_dir` 指向 `data/mesh/lv_ellipsoid/geometry`。
> 若用 Docker 重新生成了网格，请确认 `mesh.xdmf`/`mesh.h5`/`markers.json` 在该目录。

## 3. 画图（程序全部运行完毕后）

参考数据齐全后，在 `demo_337/data/plot/` 下运行画图命令：

```bash
python plot_diastole.py     # -> ../figures/diastole_plot.png
python plot_systole.py      # -> ../figures/systole_plot.png
python plot_systole_fit.py  # -> ../figures/systole_plot.png
python plot_percentage.py   # -> ../figures/your_plot_2.png
python plot_speedup.py      # -> ../figures/your_plot.png
```

> 各画图脚本读取 `../reference/` 下的参考/AFSI 数据；`plot_speedup.py`、
> `plot_percentage.py` 读取 `../`（`data/` 根目录）下的性能 CSV。
>
> 可选数据：`*-ibamr.*`（外部 IBAMR 参考）、`systole-afsi-4.txt`（AFSI 收缩结果）
> 缺失时画图脚本自动跳过对应曲线，不报错；`systole-pulse-disp.txt` 由
> `bench-ilv-contraction.py`（Docker）生成。

---

## 参数（configuration.py 摘要）

| 参数 | 值 | 含义 |
|------|-----|------|
| `T` / `dt` | 0.1 s / 1e-3 s | 总时间 / 时间步 |
| `rho` / `mu` | 1.0 / 0.01 | 流体密度 / 动力黏度（SI） |
| `Lx,Ly,Lz` / `Nx,Ny,Nz` | 5,5,5 / 32,32,32 | 流体域尺寸 / 网格数 |
| `mu_s` | 0.1 | 固体剪切模量（Pa） |
| `beta` | 5e6 | 基底约束惩罚 |
| `diastole_pressure` / `systole_pressure` | 8 / 110 mmHg | 舒张/收缩压 |
| `mesh_dir` | `data/mesh/lv_ellipsoid/geometry` | 结构网格目录 |
| `output_path` | `data/results/` | 输出目录 |

## 兼容性说明

- 本 demo 面向 **dolfinx 0.10**。旧脚本 `fsi_paralell*.py` 中
  `create_vector(L_hat)` 在 0.10 下不适用，需改为 `create_vector(Vs)`
  （`main.py` 已修正）。
- `fsi_paralell_fibers*.py` 为纤维/主动收缩变体，需要网格中的纤维场
  （`cardiac_geometries` 生成的 microstructure 数据）。


