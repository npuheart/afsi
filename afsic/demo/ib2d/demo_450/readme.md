# demo_450 — Gravity Cellular Race（质量点 + 重力，IB2d）

pure-python 参考：`third_party/ib2d/pyIB2d/Examples/Gravity_Cellular_Race`
（已拷贝为 `ib2d_input/`，0-based，无需索引换算）。

## 物理设置

1×1 周期盒，流体 $\mu=0.01,\ \rho=1$；两个弹性"细胞"（各 81 个标记点组成
闭合网络：243 根弹簧 + 162 个不变梁），每个标记点还通过一根 $k=10^6$ 的
"质量弹簧"连到一个**幽灵粒子**（质量 $M\in\{0.05,0.2,1\}$，每细胞 54 个，
分布不同）。重力 $(0,-g)$ 作用在幽灵粒子上 → 幽灵经弹簧拖拽标记 → 两个
细胞"向下赛跑"，变形由各自的质量布局决定。

### IB2d 质量点模型（易误解，此处为准）

读过 matIB2d 与 pyIB2d 源码后确认，mass 点是**幽灵粒子层**：

- 标记点照常随流体运动，力为 $F=k\,(X_\text{ghost}-X_\text{marker})$，
  没有锚到初始位置的项；
- 幽灵位置/速度是**独立状态**，ODE：
  $M\,\dot V=-F+M\,g$（重力只作用在幽灵上）；
- **幽灵位置不同步回标记点**——两者仅通过 $F$ 耦合；
- 每步顺序（driver）：标记半步 → 幽灵半步移动 → 用 $(X_h, X_{mass,h})$
  算力/散布 → 解流体 → 标记全步 $X^{n+1}=X^n+\Delta t\,U_h$ →
  幽灵 $V_h=V-\frac{\Delta t}{2}(F/M-g)$，$X_{mass}=X_{mass}^n+\Delta t\,V_h$，
  $V^{n+1}=V-\Delta t(F/M-g)$。

**实验佐证**（本目录可复现）：重力关掉 → 与标准配置标记差 $9.7\times10^{-3}$；
质量取 0（记号点脱耦）→ 差 $1.3\times10^{-2}$ —— 二者都实质耦合。
（纯重力不产生位移就"赛不动"、纯质量不产生初速度也赛不动——两个开关缺一不可。）

## 数值结果（64×64，$\Delta t=5\times10^{-5}$，$T=0.35$，7000 步）

| $t$ | $\max_k\lVert X^{AFSI}-X^{IB2d}\rVert$ | $\text{comA}_y$ AFSI/IB2d | $\text{comB}_y$ AFSI/IB2d |
|---|---|---|---|
| 0.05 | $7.8\times10^{-3}$ | 0.7840 / 0.7814 | 0.7950 / 0.7948 |
| 0.20 | $1.8\times10^{-2}$ | 0.7171 / 0.7083 | 0.6481 / 0.6478 |
| 0.35 | $2.8\times10^{-2}$（$\approx1.8h$） | 0.5913 / 0.5812 | 0.3474 / 0.3480 |

"赛跑"结果：B 细胞下沉快得多（$y:0.81\to0.35$，A 只到 $0.59$），
两码轨迹几乎重合（B 细胞中心偏差 $<8\times10^{-4}$）；AFSI 约 190 s，
pyIB2d 约 3 min。

## 文件

- `main.py`          AFSI 运行（`TFINAL=0.005 python main.py` 冒烟）；
                     输出 `plot/afsi_result_g<GRAD_DIV>.npz`
                     （`X` 标记、`Xmass` 幽灵、`u/p/`网格场、`n_cellA=81`）
- `run_reference.py` 纯 Python pyIB2d 参考（自带 lag/out_par 表，本示例
                     `tracers` 等关、`mass=1, gravity=1`）→ `ib2d_reference.npz`
- `ib2d_io.py`       0-based 输入读取（本 demo 加了 `.mass` 读取）
- `ib2d_reference.py` VTK→npz 转换
- `compare.py`       轨迹表 + `figures/`（形状、赛跑历史、涡量与速度差）

```bash
conda run -n afsi-dolfinx python run_reference.py
conda run -n afsi-dolfinx python main.py
conda run -n afsi-dolfinx python compare.py
```
