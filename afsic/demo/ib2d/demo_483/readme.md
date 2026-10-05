# demo_483 — IB2d `Example_Porous_Rubberbands/Compare_Porous_NonPorous_Rubberbands_256x256`

IB2d 官方算例（`matIB2d/Examples/Example_Porous_Rubberbands/Compare_Porous_NonPorous_Rubberbands_256x256`）
在 AFSI 中的复现：**弹簧 + 多孔滑移（porous media）**。

## 算例内容

* `2 x 2` 周期盒、`mu = 0.1`、`rho = 1`、**官方 `256 x 256`、`dt = 5e-5`
  原样保留**。官方 `T = 0.5`（10000 步）→ 本 demo **`T = 0.005`（100 步）**、
  `print_dump = 25`（5 帧）。
* **结构**：1024 个拉格朗日点构成一条**闭合橡皮筋**（半径约 0.57），
  **1024 根线性弹簧**（`k = 2e7`、**静长 `L0 = 0`**，即整条环想缩成一个点），
  其中 **512 个点标记为 porous**（`kappa = 3.5e-4`，stencil 标志
  `c in {-2..2}`）。
* **porous 机制**：标记在常规 IB 更新**之后**沿局部法向额外滑移
  （流体可以"穿过"多孔带），滑移速度
  `U_p = -kappa F_lag n / |(x_s, y_s)|`，位置再减 `dt U_p n`，
  最后全部标记做周期折叠；切向 `(x_s,y_s)` 用 porous 列表相邻行的
  4 阶单侧/中心差分（`/ds`，`ds = Lx/(2Nx)`）。
* 物理：橡皮筋一边被弹性拉住收缩、一边"漏"流，多孔滑移让它能在
  不排开全部流体的情况下收缩。

## 移植要点

1. 模板的 `porous_slip_update` 覆盖该机制，**无需用户模块**
   （`input2d` 只打开 `springs = 1`、`porous_media = 1`）。
2. `rubberband.*` 为 1-based，`ib2d_io` 自动减 1。
3. **必须用官方 `NX = 256`**：先试 `NX = 128` 时 **两个代码都发散**
   （`ds` 翻倍 → 每个标记的铺力翻倍，收缩太快；参考与 AFSI 都 NaN）。
   注意这**不是**移植问题，而是"标记加密 3.5 倍于 `dx`"的极端设置。
4. 稳定化 `GRAD_DIV = 2.5`：沿用 demo_453/454 对本族（收缩的多孔环）
   的结论——粗网格 P2/P1 投影的伪散度会让环塌缩过快甚至崩溃。
   本 demo 未再单独扫描 γ。
5. **缩减**：只把 `T = 0.5 → 0.005`（10000 → 100 步）。

## 结果

参考运行约 46 s（256²，100 步）；AFSI 模拟耗时 249.8 s（墙钟另含
1-3 min 解释器/库导入；机器并行运行其它 demo）。5 帧对齐（`t = 0 .. 0.00505`）。
`compare.py` 原文：

| 指标（末帧） | compare.py 原文 |
|---|---|
| 标记点最大偏差 | `max|dX| 9.158e-04 m = 1.17e-01 h (1024 Lagrangian points)` |
| 速度峰值 AFSI / IB2d | `1.4171e+01 / 1.4746e+01 (3.90e-02 relative)` |
| 速度场偏差 | `max|du| 1.806e+00 = 1.23e-01 max|u|`（带外）; `2.328e+00 (0.16 max|u|)`（带内）; `relative L2 3.135e-02` |
| 力场偏差（带外） | `max|df| 6.840e+03 away from the band, |f|max 9.627e+04` |
| 弹簧平均长度 | `AFSI 0.003629 / IB2d 0.003630 (1.66e-04 relative)` |

**能验证**：**收缩运动学**（标记点 0.117 h）、**弹簧平均长度 1.7e-4**、
速度峰值 3.9e-2、速度场相对 $L_2$ 3.1e-2 —— 即"环缩多快、流场多快、
多孔漏流多少"两边一致。这是本族（收缩的多孔环）**唯一**被验证到
`T = 0.005` 的设置；上游的 `T = 0.5` 会把环缩到网格尺度以下
（demo_453 报告 `t ≈ 0.043` 崩溃），所以长时间不可比。
**不能验证（如实记录）**：带内速度场逐点值（`max|du|` 0.16 `max|u|`），
以及 `T > 0.01` 之后的塌缩过程（本机不可行）。

## 运行

```bash
conda activate afsi-dolfinx
python run_reference.py --tend 0.005 --print-dump 25             # 官方 256²，~180 s
GRAD_DIV=2.5 TFINAL=0.005 PRINT_DUMP=25 python main.py           # ~480 s
python compare.py
```
