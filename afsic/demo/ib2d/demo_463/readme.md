# demo_463 — IB2d `Compare_Porous_NonPorous_Rubberbands`（多孔 / 非多孔 A/B）

IB2d 官方算例（`matIB2d/Examples/Example_Porous_Rubberbands/Compare_Porous_NonPorous_Rubberbands`）
在 AFSI 中的复现。这是本仓库**第一个在同一个算例里放两条独立浸入结构**的 demo，
也是一次同流场内的"多孔 vs 不透水"对照实验。

## 算例内容

`2 x 2` 周期盒、`mu = 0.1`、`rho = 1`、`64 x 64`、`dt = 5e-5`、`T = 0.1`（2000 步）。
两条**各 128 点**的零静长弹簧环（$k = 10^7$）：

* **环 1（标记点 1–128，左侧）是多孔的**：`.porous` 文件把它每个点的多孔系数
  $\kappa = 10^{-4}$ 与差分模板标记都列出。IB2d 在常规更新后让多孔点沿局部法向
  "滑移"

  $$\mathbf X_p \mathrel{-}= \mathrm{d}t\,U_p\,\mathbf n,\qquad
  U_p = -\frac{\kappa\,(\mathbf F\cdot\mathbf n)}{|\mathbf X_p'|}$$

  其中切向 $\mathbf X_p'$ 用 4 阶单侧/中心差分（模板标记 $c\in\{-2..2\}$ 逐点给出），
  即**流体可以穿过膜**而不必被挤出（机制实现见 demo_453）。
* **环 2（标记点 129–256，右侧）是普通闭合膜**。

于是同一流场下：多孔环塌缩得极快（面积 0.2512 → 0.1159），不透水环只收缩到 0.1968
——正是本算例想展示的对照。

## 实现

1. `matIB2d` 的 1-based 文件 → `ib2d_io.load_example(..., index_base=1)`
   （本次要给 `.spring` 与 `.porous` 两个文件重编号，`run_reference.py` 里用
   `shift_ids(..., n_id_cols=2 / 1)`）；`ib2d_io` 新增 `read_porous`。
2. 双环结构：FE 网格直接取 `.spring` 连通关系（两个互不连通的分量），
   零刚度承载 256 个标记点自由度。
3. 默认 $\gamma = 2.5$：$k=10^7$ 零静长环在 $\gamma=0$ 时会塌缩到 NaN
   （demo_453 的系统扫描结论），$\gamma = 2.5$ 是全程匹配参考的取值。
4. 输出帧与 pyIB2d 落在**同一步**（帧对齐约定同 demo_459–462）。

## 结果

参考运行 11 s、AFSI 运行 24.7 s，两边各 41 帧：

| 指标 | 多孔环 | 不透水环 |
|---|---|---|
| 标记点偏差（对 IB2d） | 最大 **1.77e-2 m = 0.57 h** | 最大 **2.40e-2 m = 0.77 h** |
| 面积轨迹（AFSI / IB2d） | 0.251226 → **0.115865** / **0.101064**，`max|ΔA|` 1.5e-2 | 0.251226 → **0.196825** / **0.201378**，`max|ΔA|` 1.3e-2 |

结论：**多孔滑移 + 双结构在同一流场内的 A/B 对照被复现**——两侧都显示多孔环面积
减少 ~55%、不透水环只减少 ~20%，面积轨迹在塌缩最剧烈前吻合到 1e-2 量级；
末段（多孔环快速塌缩、膜间距接近网格尺度）差异增大到 ~10%，与 demo_453/454
记录的零静长环塌缩敏感性一致。

## 运行

```bash
conda activate afsi-dolfinx
python run_reference.py          # pyIB2d 参考 -> ib2d_reference.npz（~11 s）
python main.py                   # AFSI        -> plot/afsi_result_g2.5.npz（~25 s）
python compare.py                # 对比 + figures/
```

环境变量：`TFINAL`、`DT`、`NX`、`PRINT_DUMP`、`GRAD_DIV`（默认 2.5）、
`POROUS_SCALE`（置 0 可关闭多孔滑移，仅用于实验）、`OUTPUT_PATH`。
