# demo_476 — IB2d `Example_Concentration_Gradients/Pure_Diffusion`（纯扩散基准）

IB2d 官方算例（`matIB2d/Examples/Example_Channel_Flow/Example_Concentration_Gradients/Pure_Diffusion`）。
**新机制：被动标量（`concentration`）的显式对流-扩散推进**，与 demo_456 的有限元被动标量
互为补充——这里逐字复刻 IB2d 自己的差分格式，因此两边应当**逐位吻合**。

## 算例内容

* `1 x 1` 周期盒、`mu = 10`、`rho = 1`、`64 x 64`、`dt = 1e-4`；
  官方 `T = 0.5`（5000 步、`print_dump = 200`）→ 本 demo 用**完整设置**（约 4 分钟）。
* **结构**：496 个 `k = 1e7` 的钉死目标点 = 方形扩散腔的四壁。
* **标量**：`chamber.concentration` 首行给出 `kDiff = 0.025`，其余为 `(Ny, Nx)` 初始浓度——
  盒子中心一个 `3 x 3`、值为 1 的方块（节点值之和 9，积分 `9 h^2 = 2.197e-3`）。
* **无外力**：算例目录里**没有** `please_Compute_External_Forcing.m`（`arb_ext_force = 0`），
  且结构钉死在初始位置，因此流场**恒为零**——这正是"纯扩散"的含义，也是检验标量算子的干净基准。

## 移植要点

1. **驱动实际调用的是分裂格式**。`Supp.py` 里同时存在
   `perform_Time_noSplit_Upwind`（被注释掉）与 `perform_Time_Split_Upwind`，而
   `please_Update_Adv_Diff_Concentration` 调用的是**后者**：

   ```
   C*     = C^n + dt ( kDiff C_xx - uX Cx )
   C^{n+1}= C*  + dt ( kDiff C*_yy - uY C*_y )     ← 在 C* 上重新求导
   ```

   第一版按 no-split 写，结果与参考差 `6.3e-4`；改成 split 后立刻降到 `3.3e-12`
   （同样的算子、不同的求和次序 ⇒ 纯浮点舍入）。
2. **上游 upwind 循环边界的小毛病**：`give_Necessary_Derivative` 的 x 分支用
   `for j in range(1, lenX-2)`，**次末列**（`lenX-2`）的导数因此恒为 0；y 分支同理。
   本 demo 逐字复现（否则两边对不上），并在 `main.py` 中注释说明。
3. **速度取 `u^{n+1}`**：驱动在解完流体、更新完标记点之后才推进浓度，用的是**新**速度场；
   `main.py` 的循环顺序（标记点 → 标量 → dump）与之一致。
4. **网格**：标量活在 IB2d 的**共置节点**上，正好等于 AFSI 网格的 Q1 顶点自由度，
   因此标量更新直接在 `(Ny, Nx)` 节点数组上做，不需要任何有限元投影。
5. 模板同时支持"有/无外力模块"两种情况（本例没有模块，`main.py` 自动跳过）。

## 结果

参考运行约 65 s、AFSI 运行约 63 s，各 26 帧（`t = 0 .. 0.50010`）。

| 指标（末帧 `t = 0.50010`） | 结果 |
|---|---|
| 标记点偏差（496 点钉死） | 3.30e-16 m = **2.11e-14 h** |
| 速度场 | AFSI `max\|u\| = 1.5e-13`、IB2d `0.0`（**恒为零**，只差线性求解舍入） |
| 浓度 `max\|ΔC\|` | **4.93e-13**（峰值 0.0139 ⇒ 相对 3.5e-11） |
| 浓度相对 $L_2$ 误差 | **1.81e-11** |
| 峰值 `max C` AFSI / IB2d | 0.01392573 / 0.01392573（4.16e-12 相对） |
| 总量 `∑C h²` AFSI / IB2d | 2.1972656e-3 / 2.1972656e-3（1.45e-12 相对，守恒） |
| 壁面窄带内 `max\|ΔC\|` | 5.0e-15 |

**结论**：标量算子（upwind 对流 + 中心扩散 + 周期处理 + 分裂时间步）与 pyIB2d **逐位一致**
（误差 1e-11 量级 = 浮点舍入），是本族里最强的验证；峰值从 1.0 衰减到 0.0139、总量守恒到
1.5e-12，说明扩散量级也对。

## 运行

```bash
conda activate afsi-dolfinx
python run_reference.py                    # ~65 s
python main.py                             # ~63 s
python compare.py                          # figures/ 与 compare_table.csv
```
