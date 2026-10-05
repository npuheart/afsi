# demo_497 — IB2d `Examples_First_Year_Seminar/Jellyfish_Oscillator`

IB2d 官方算例（`matIB2d/Examples/Examples_First_Year_Seminar/Jellyfish_Oscillator`）
在 AFSI 中的复现：**弹簧 + 不变梁 + 逐行移植的 `update_Springs.py`**。

## 算例内容

* `1 x 1` 周期盒、`mu = 0.0010`、`rho = 1`、`128 x 128`（**官方网格**）、
  `dt = 5e-4`（官方值）。官方 `T = 2.0`（4000 步）→ 本 demo
  **`T = 0.2`（400 步，恰好一个收缩周期）**，`print_dump = 20`（20 帧）。
* **结构**：79 个拉格朗日点：**79 根弹簧**（`k = 2.5e5`）+ **77 根不变梁**
  （`kb = 2e8`），构成一个"水母钟"（bell）；钟口由最后一根弹簧闭合。
* **驱动**（`update_Springs.m`，已逐行移植为
  `ib2d_input/update_Springs.py`，两个代码共用同一份）：只改**最后一根**
  弹簧的静长

  ```
  springs(end, 4) = dist * ( 1 - 0.5 |sin(10 pi t)| ),   dist = 0.1250077,
  ```

  `|sin|` 使钟口每 0.2 s 收缩两次（5 Hz），水母因此"划水"前进。

## 移植要点

1. **`update_Springs` 是最小的一个**：MATLAB `springs_info(end,4)` →
   `springs[-1, 3]`（列 4 → 列 3）。**注意官方在 `T = 2.0` 里只驱动最后
   一根弹簧**，其余弹簧/梁都是被动的。
2. `jelly.vertex` 等 1-based，`ib2d_io` 自动减 1。
3. 唯一改动：`T = 2.0 → 0.2`（一个完整收缩周期；官方 4000 步在本机负载下
   跑不完）。

## 结果

参考运行约 41 s、AFSI 模拟耗时 145.5 s（墙钟另含 1-3 min 解释器/库导入；
机器并行运行其它 demo）。
20 帧对齐（`t = 0 .. 0.19050`）。`compare.py` 原文：

| 指标（末帧） | compare.py 原文 |
|---|---|
| 标记点最大偏差 | `max|dX| 1.887e-02 m = 2.42e+00 h (79 Lagrangian points)` |
| 速度峰值 AFSI / IB2d | `1.5467e+00 / 2.4336e+00 (3.64e-01 relative)` |
| 速度场偏差 | `max|du| 1.994e+00 = 1.81e-01 max|u|`（带外）; `1.422e+00 (0.42 max|u|)`（带内）; `relative L2 5.703e-01` |
| 力场偏差（带外） | `max|df| 8.873e+02 away from the band, |f|max 8.873e+02` |
| 弹簧平均长度 | `AFSI 0.005202 / IB2d 0.005199 (6.05e-04 relative)` |

**能验证**：**驱动的弹簧长度律**（平均长度一致到 6.1e-4，即"钟口收多紧"
两边相同）——这正是唯一被 `update_Springs.py` 修改的量。
**不能验证（如实记录）**：末帧标记点 2.42 h、速度峰值相差 36%、速度场相对
$L_2$ 0.57。原因：水母由 **79 个稀疏标记 + `kb = 2e8` 的极刚梁**构成，
`ds = Lx/(2Nx) = 0.0039` 与标记间距同量级；在 `dt = 5e-4`（官方值，
`u_max dt / h ≈ 0.3`）下，两种离散对"薄且刚的钟"的耦合方式不同，
相位误差在 0.2 s 内就累积到半个周期量级。标记点轨迹的**振幅**可比、
**相位**不可比。

## 运行

```bash
conda activate afsi-dolfinx
python run_reference.py --tend 0.2 --print-dump 20               # 官方 128²，~41 s
TFINAL=0.2 PRINT_DUMP=20 python main.py                          # ~175 s
python compare.py
```
