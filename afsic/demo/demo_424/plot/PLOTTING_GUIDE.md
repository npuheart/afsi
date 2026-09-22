# demo_424 画图说明

本文档说明这个目录里各种图是怎么生成的，包括：

- PyVista 场图：速度、压力、截面、流线、固体位移
- matplotlib 流量历史图
- 光滑压力过渡对比柱状图
- headless 环境下用 OSMesa 渲染的方法

---

## 1. 环境依赖

基础 Python 包：

```bash
pip install numpy matplotlib pyvista meshio h5py
```

如果是在没有 X server 的 headless 机器上跑 PyVista 渲染，还需要一个软件 OpenGL 后端，例如 OSMesa：

```bash
# Ubuntu 可以这样安装
apt-get download libosmesa6
dpkg-deb -x libosmesa6_*.deb /tmp/osmesa
```

然后设置：

```bash
export LD_LIBRARY_PATH=/tmp/osmesa/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export AFSI_USE_OSMESA=1
export MPLCONFIGDIR=/tmp/mplconfig
export HOME=/tmp/pvhome
```

简单测试：

```bash
python -c "import pyvista, meshio, h5py, matplotlib; print('ok')"
```

---

## 2. 读取 AFSI 的 XDMF

当前 2-D `XY` XDMF 用 VTK 的 `XdmfReader` 容易出问题，所以后面统一用 `meshio` 读：

```python
from meshio.xdmf import TimeSeriesReader
import numpy as np
import meshio
import pyvista as pv

with TimeSeriesReader("velocity.xdmf") as ts:
    points, cells = ts.read_points_cells()
    t, point_data, cell_data = ts.read_data(0)

mesh = pv.from_meshio(
    meshio.Mesh(points, cells, point_data=point_data)
)
```

变量名可能是：

- 速度：`f`
- 压力：`p` 或 `p_`

可以动态处理：

```python
p_key = "p" if "p" in mesh.point_data else "p_"
mesh["velocity"] = np.asarray(mesh["f"])
mesh["pressure"] = np.asarray(mesh[p_key]).ravel()
mesh["speed"] = np.linalg.norm(mesh["velocity"], axis=1)
```

---

## 3. PyVista 场图脚本

样例脚本位置：

```text
plot/closed_NY45_ipcs/example_figures/make_examples.py
```

或者各算例目录：

```text
plot/<case>/example_figures/make_examples.py
```

它做的主要事情：

1. 读取当前算例目录下的 `velocity.xdmf` 和 `pressure.xdmf`
2. 转成 `pyvista.UnstructuredGrid`
3. 生成这些图：

```text
00_overview.png              # 2x2 总览：|u|、p、速度矢量、流线
01_velocity_magnitude.png    # 速度大小云图
02_pressure.png              # 压力云图
03_velocity_vectors.png      # 速度矢量图
04_streamlines.png           # 流线图
05_cross_section_x.png       # x = 0.05 m 截面
index.html                   # 网页预览入口
```

运行方式：

```bash
cd plot/<case>/example_figures
AFSI_USE_OSMESA=1 LD_LIBRARY_PATH=/tmp/osmesa/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH \
python make_examples.py
```

如果是在有显示器的桌面环境，可以去掉 `AFSI_USE_OSMESA=1` 和 `LD_LIBRARY_PATH`。

脚本中 headless 渲染的关键代码：

```python
pv.OFF_SCREEN = True

if os.environ.get("AFSI_USE_OSMESA") == "1":
    import vtk
    from pyvista import _vtk
    _vtk.vtkRenderWindow = vtk.vtkOSOpenGLRenderWindow
```

然后即可：

```python
pl = pv.Plotter(off_screen=True, window_size=(1500, 650))
pl.add_mesh(mesh, scalars="speed", cmap="turbo")
pl.view_xy()
pl.enable_parallel_projection()
pl.screenshot("01_velocity_magnitude.png")
pl.close()
```

---

## 4. 流线和截面

流线：

```python
x_seed = mesh.bounds[0] + 0.001
ys = np.linspace(mesh.bounds[2] + 0.001, mesh.bounds[3] - 0.001, 25)
seeds = pv.PolyData(np.c_[np.full_like(ys, x_seed), ys,
                          np.zeros_like(ys)])

streamlines = mesh.streamlines_from_source(
    seeds,
    vectors="velocity",
    integration_direction="forward",
    surface_streamlines=True,
    max_length=(mesh.bounds[1] - mesh.bounds[0]) * 1.2,
)
```

画：

```python
pl.add_mesh(mesh, scalars="speed", cmap="Blues", opacity=0.85)
pl.add_mesh(streamlines, color="black", line_width=2)
pl.add_mesh(seeds, color="red", point_size=8, render_points_as_spheres=True)
```

截面：

```python
section = mesh.slice(normal="x", origin=(0.05, 0.0, 0.0))
pl.add_mesh(section, scalars="speed", cmap="turbo",
            line_width=10, render_lines_as_tubes=True)
```

---

## 5. 固体位移 PVD/VTU 图

`main.py` 输出：

```text
solid_displacement.pvd
solid_displacement_p0_000000.vtu
```

PyVista 读取：

```python
import numpy as np
import pyvista as pv

mb = pv.read("solid_displacement.pvd")
grid = mb[0]

d = np.asarray(grid["displacement"])
if d.shape[1] == 2:
    d = np.c_[d, np.zeros(len(d))]

grid["displacement"] = d
grid["d_mag"] = np.linalg.norm(d, axis=1)
```

画位移大小：

```python
pl = pv.Plotter(off_screen=True, window_size=(1400, 700))
pl.set_background("white")
pl.add_mesh(grid, scalars="d_mag", cmap="turbo",
            scalar_bar_args={"title": "|d| (m)"})
pl.view_xy()
pl.enable_parallel_projection()
pl.screenshot("solid_displacement.png")
pl.close()
```

如果要看放大变形：

```python
warped = grid.warp_by_vector("displacement", factor=200.0)
```

---

## 6. matplotlib 流量历史图

`main.py` 会在时间循环里输出：

```text
flow_history.csv
```

列：

```text
step,t,Q_gap,Q_leak_up,Q_leak_dn,max_u
```

画图脚本：

```text
plot/closed_NY45_S4_ipcs_s4_t1p0/plot_flow_history.py
```

核心代码：

```python
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows = list(csv.DictReader(open("flow_history.csv")))
t = np.array([float(r["t"]) for r in rows])
q_gap = np.array([float(r["Q_gap"]) for r in rows])
q_up = np.array([float(r["Q_leak_up"]) for r in rows])
q_dn = np.array([float(r["Q_leak_dn"]) for r in rows])
umax = np.array([float(r["max_u"]) for r in rows])

fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
axes[0].plot(t, q_gap, "o-")
axes[0].set_ylabel("Q_gap (m^2/s)")
axes[1].plot(t, q_up, "o-", label="upstream")
axes[1].plot(t, q_dn, "s-", label="downstream")
axes[1].set_ylabel("Q_leak (m^2/s)")
axes[1].legend()
axes[2].plot(t, umax, "o-")
axes[2].set_xlabel("t (s)")
axes[2].set_ylabel("max|u| (m/s)")
fig.tight_layout()
fig.savefig("flow_history_matplotlib.png", dpi=200)
```

归一化版本：

```python
i04 = int(np.argmin(np.abs(t - 0.4)))
for y, lab in ((q_gap, "Q_gap"), (q_up, "Q_leak_up"),
               (q_dn, "Q_leak_dn"), (umax, "max|u|")):
    plt.plot(t, y / y[i04], "o-", label=lab)
plt.axhline(1.0, color="k", lw=0.7)
plt.legend()
plt.savefig("flow_history_normalized_matplotlib.png", dpi=200)
```

---

## 7. 光滑压力过渡对比图

脚本：

```text
plot/plot_smooth_transition_summary.py
```

它读取：

```text
plot/closed_NY45_ipcs_t1/verify.json
plot/closed_NY45_ipcs_t5/verify.json
```

然后画柱状图：

```text
plot/smooth_transition_t1_t5_summary.png
```

包含：

- `max_u`
- `Q_gap_num`
- `Q_leak_4h upstream`
- `Q_leak_4h downstream`

运行：

```bash
cd plot
python plot_smooth_transition_summary.py
```

---

## 8. 推荐画图流程

对一个新算例 `<case_dir>`：

1. 确认有：
   ```text
   velocity.xdmf
   pressure.xdmf
   verify.json
   flow_history.csv      # 如果 main.py 开了流量诊断
   solid_displacement.pvd
   ```
2. 复制 PyVista 画图脚本：
   ```bash
   mkdir -p <case_dir>/example_figures
   cp plot/closed_NY45_ipcs/example_figures/make_examples.py \
      <case_dir>/example_figures/
   cp plot/closed_NY45_ipcs/example_figures/index.html \
      <case_dir>/example_figures/
   ```
3. 运行：
   ```bash
   cd <case_dir>/example_figures
   AFSI_USE_OSMESA=1 LD_LIBRARY_PATH=/tmp/osmesa/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH \
   python make_examples.py
   ```
4. 流量图：
   ```bash
   cd <case_dir>
   cp ../../closed_NY45_S4_ipcs_s4_t1p0/plot_flow_history.py .
   python plot_flow_history.py
   ```

---

## 9. 当前主要输出图位置

```text
plot/closed_NY45_ipcs/example_figures/index.html
plot/closed_NY45_S4_ipcs_s4/example_figures/index.html
plot/closed_NY45_S4_ipcs_s4_t1p0/example_figures/index.html
plot/closed_NY45_S4_ipcs_s4_t1p0/flow_history_matplotlib.png
plot/closed_NY45_S4_ipcs_s4_t1p0/flow_history_normalized_matplotlib.png
plot/smooth_transition_t1_t5_summary.png
```

