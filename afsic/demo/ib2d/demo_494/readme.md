# demo_494 — IB2d `Examples_First_Year_Seminar/Leaf`

IB2d 官方算例（`matIB2d/Examples/Examples_First_Year_Seminar/Leaf`）
在 AFSI 中的复现：**目标点（固定的叶片）+ 任意外力射流**。

## 算例内容

* 非方形盒 `Lx = 0.2`、`Ly = 0.3`（"2 倍宽 × 3 倍长"）、`mu = 5e-4`、
  `rho = 1`、`128 x 192`（**官方网格**）、`dt = 1e-5`（官方值）。
  官方 `T = 0.15`（15000 步）→ 本 demo **`T = 0.01`（1000 步）**，
  `print_dump = 100`（11 帧）。
* **结构**：316 个拉格朗日点（`leaf.vertex`，一片叶子轮廓）全部由
  **316 个目标点**（`k = 1e7`）钉在初始位置 —— 一片**固定**的叶子，
  用于研究"流体掠过叶片"（叶脉网络不复用；`leaf.vertex_OLD` 是上游遗留
  的旧几何，未被任何文件引用）。
* **驱动**（`please_Compute_External_Forcing.m` → `ib2d_input/*.py`，
  两个代码共用）：窗口 `x in [0.04, 0.16]`（`.m` 里的 `0.01/0.13 + offset`，
  `offset = 0.03`）、`y in [0.005, 0.009]` 上施加**只有 y 分量**的射流

  ```
  Fy = -kStiff (uY - uY_tar),  uY_tar = -uMax (poly_coeff tanh(100t)) ((Lx/2+w/2)-x)((Lx/2-w/2)-x),
  poly_coeff = 2.5 / |((Lx/2+w/2)-Lx/2)((Lx/2-w/2)-Lx/2)|,   kStiff = 1e3, w = 0.08, uMax = 5,
  ```

  `tanh(100t)` 几乎立刻饱和（`t = 0.05` 已到 0.99），所以叶子受到的是
  近乎恒定的 "吹风"。

## 移植要点

1. 外力模块逐行移植，索引搜索（首个 `x(j) >= xMin` / 最后一个 `x(j) <= xMax`）
   用 `np.searchsorted` 复刻，并与逐行转写逐点比对（差 `0.0`）。
   `.m` 里 `poly_coeff` 用 `Lx`（虽然射流在 y 方向），移植版保留该写法。
2. `leaf.vertex`/`leaf.target` 1-based，`ib2d_io` 自动减 1。
3. **缩减**：`T = 0.15 → 0.01`。注意这**只覆盖射流开启的头 0.01 s**；
   叶子是固定的，因此被验证的主要是"射流 + 绕流"而不是叶子的长时间响应。

## 结果

参考运行约 71 s；AFSI 模拟耗时 403.0 s（墙钟另含 1-3 min 解释器/库导入；
机器并行运行其它 demo）。
11 帧对齐（`t = 0 .. 0.01001`）。`compare.py` 原文：

| 指标（末帧） | compare.py 原文 |
|---|---|
| 标记点最大偏差 | `max|dX| 1.999e-06 m = 1.28e-03 h (316 Lagrangian points)` |
| 速度峰值 AFSI / IB2d | `5.4942e+00 / 5.5144e+00 (3.68e-03 relative)` |
| 速度场偏差 | `max|du| 1.426e+00 = 7.41e-01 max|u|`（带外）; `6.936e-02 (0.99 max|u|)`（带内）; `relative L2 7.309e-02` |
| 力场偏差（带外） | `max|df| 8.860e+02 away from the band, |f|max 8.467e+03` |
| 外力窗口 | `154 nodes, max|u| AFSI 4.8196 / IB2d 5.3139 (9.30e-02 relative); forcing check |f_x - F_x^module(u^AFSI)| <= 2.761e+00 on the core nodes` |

**能验证**：叶片完全不动（1.3e-3 h）；速度峰值一致到 **3.7e-3**；
速度场相对 $L_2$ 7.3e-2；外力模块与 AFSI 力场自洽（残差 2.8，
相对 `|f|max = 8.5e3` 约 3e-4）。
**不能验证（如实记录）**：窗口内速度峰值的相对差稍大（9.3e-2，因为
`kStiff = 1e3` 很软、窗口只有 4 行节点，罚力对角点的响应敏感）；
带外瞬时速度场逐点值（`max|du|` 0.74 `max|u|`，但注意 `du away` 是
"远离 IB 带"的节点：叶片很薄、`dy = 0.0016`，射流剪切层跨越了大部分网格）。

## 运行

```bash
conda activate afsi-dolfinx
python run_reference.py --tend 0.01 --print-dump 100            # 官方 128x192，~100 s
TFINAL=0.01 PRINT_DUMP=100 python main.py                        # ~426 s
python compare.py
```
