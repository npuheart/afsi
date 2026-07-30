# Demo 339 — Direct Forcing

经典 DFG 2D-3 基准：Re=100 圆柱绕流。

## 方法

**Direct Forcing（直接力法）**：
1. 在流体网格上标记圆柱内部的自由度
2. 每步求解 NS 方程后，强制圆柱内 $u = (0, 0)$
3. 由速度修正量反算等效体积力 $\mathbf{f} = -\tilde{u}/\Delta t$
4. 体积力积分得曳力/升力

无需固体网格、本构模型、IB 插值。

## 运行

```bash
conda activate afsi-dolfinx
cd afsic/demo/demo_339/direct_forcing
python main.py
```

## 对比

| 方法 | 目录 | 固体表示 |
|------|------|----------|
| Direct Forcing | `direct_forcing/` | 流体网格 DOF 标记 |
| IB-FE | `ibfe/` | Neo-Hookean 固体 + Peskin IB 核 |
| Body Fitted | `body_fitted/` | Gmsh 贴体网格 |
| 无圆柱 | `no_cylinder/` | 纯流体（基准对照） |
