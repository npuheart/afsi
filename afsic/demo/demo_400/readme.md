
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

