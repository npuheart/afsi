
# demo_400 — 2D 乌龟 FSI（IB 方法）

浸没边界流固耦合（IB-FSI）：2D 乌龟（头 + 四肢 + 尾巴）在通道流体中，受周期性压力驱动，**头尾固定**，**四肢随流摆动**。

## 流体（CGS 单位）

- 2D 通道 200 × 100，四边形网格 128 × 64
- 入口（左，tag 14）：斜坡来流 `Inlet`（Um，余弦斜坡 0.5 s 建立，见 `Inlet.update`）
- 出口（右，tag 12）：压力出口（p = 0）
- 上/下壁（tag 13/11）：无滑移
- rho = 1 g/cm³，mu = 0.01 Poise，dt = 5e-5 s，T = 30 s

## 固体（乌龟）

- 几何：pygmsh 生成的乌龟轮廓 `turtle.geo` → `turtle_mesh.xdmf/.h5`（含 cell_tags / facet_tags），整体放大 100 倍放入通道
- 材料：Neo-Hookean `P = mu_s*(F - F^-T) + lambda_s*ln(J)*F^-T`，mu_s = lambda_s = 1e4，nu_s = 0.45
- **固定头尾**：惩罚约束 `beta * (X - X0)` 施加于 facet tag 15（beta = 1e6）
- **周期性压力驱动**：`pressure_waveform`（`sin` / `fast_open`，幅值 p_amp = 100，周期 p_period = 2 s）；follower pressure 沿当前脊柱方向 `spine_dir`（tag 16 → tag 17 质心方向，每步更新）投影后施加于 tag 16/17
- **四肢摆动**：四肢速度由流体插值（IB），随流摆动

## 耦合

固定笛卡尔流体网格 + 拉格朗日固体网格（`IBMesh` / `IBInterpolation`）。每步：流体求解 → 流体速度插值到固体 → 更新固体坐标 → 计算固体力（Neo-Hookean + 头尾惩罚 + follower pressure）→ 反馈流体。

## 运行

```bash
python main.py            # 或 mpirun -n <N> python main.py
```

输出：`velocity` / `pressure` / `solid_force`（xdmf + h5）。



## 短程复跑（t ≤ 1 s，128×64，串行）与失稳位置

本算例没有离线入口：`main.py` 顶部无条件调用 `swanlab_init(..., api_key=...)`，
`configuration.py` 的 `unique_filename()` 还会往 `~/afsi-data/` 建目录。复跑时用一个
小 harness 把 `swanlab_init` / `swanlab_upload` 换成 no-op + CSV 记录
（同 `demo_339/_short_run/run_compare.py` 的做法）。另外两处必须先修：

| 问题 | 修法 |
|------|------|
| `generate_mesh.py` 从 `dolfinx.io` 导入 `gmshio`（0.10 已改名） | 改为 `from dolfinx.io import gmsh as gmshio` |
| `main.py` 把 `./turtle_mesh.xdmf` 写死 | 支持 `TURTLE_MESH` 环境变量 |
| `STEPS` 不被识别 | `configuration.py` 增加 `STEPS` 覆盖 |

运行：20000 步（$\Delta t = 5\times10^{-5}$，$t \le 1$ s，载荷周期 2 s），**2321 s** 串行。

| 量 | 值 |
|----|-----|
| 入口速度 | **全程为 0**：`Um = 0.0`，且 inlet 的 `DirichletBC` 从未传给求解器 —— 流场完全由固体变形驱动 |
| 流体速度 | t=0.35 s 时 1.2 m/s；t=1 s 时 **28 m/s** |
| 肢端位移 | ±0.60 m（体高 33.8） |
| 体积 | 319.93 → 319.78（压缩 0.05%） |

**t ≲ 0.4 s 的结果可用**：压力呈跨体偶极子，速度场在四肢周围形成四瓣结构，肢端位移约体高的 1.8%。
**再往后就不可信**：t=1 s 时流体 28 m/s，而肢端速度只有 ~1e-3 m/s，相差 1e4 倍，
属显式 IB-FE 耦合失稳，不是物理结果，且发生在第一个载荷周期之内。要跑满文档里的
30 s / 600000 步，需要更小的 $\Delta t$、子迭代耦合，或两者都要。
