# demo_495 — IB2d `Examples_First_Year_Seminar/Flexible Bubbles`

IB2d 官方算例（`matIB2d/Examples/Examples_First_Year_Seminar/Flexible Bubbles`）
在 AFSI 中的复现：**弹簧 + 不变梁 + 目标点 + 任意外力驱动**（全部由模板覆盖），
外加一个逐行移植的 `please_Compute_External_Forcing.py`。

## 算例内容

* `1 x 1` 周期盒、`mu = 0.01`、`rho = 1`、`128 x 128`（**官方网格**）、
  `dt = 5e-5`。官方 `T = 1.25`（25000 步）→ 本 demo **`T = 0.05`（1000 步）**，
  `print_dump = 50`（20 帧）。
* **结构**（666 个标记）：一条通道壁（**306 根弹簧** `k = 1e7`、
  **204 根不变梁** `kb = 1e9`、**462 个目标点** `k = 1e7` 把壁钉死），
  通道内漂浮**柔性气泡**（由同一个顶点集构成、由弹簧/梁连接）。
* **驱动**（`please_Compute_External_Forcing.m`，已逐行移植为
  `ib2d_input/please_Compute_External_Forcing.py`，两个代码共用同一份）：
  在窗口 `x in [0.1, 0.16]`、`y in [0.41, 0.59]` 上施加
  **速度靶值罚力**

  ```
  Fx = -kStiff (uX - uX_tar),  uX_tar = -uMax 5 tanh(t) ( (Lx/2+w/2) - y )( (Lx/2-w/2) - y ),
  Fy = 0,   kStiff = 1e4, w = 0.2, uMax = 250,
  ```

  即通道中央的抛物线射流，按 `tanh(t)` 缓慢开启。

## 移植要点

1. **外力模块**：该 `.m` 与 `Examples_First_Year_Seminar/Flow_Around_Object`
   的模块逐字节相同（只差注释头），所以两边用同一份 Python；
   MATLAB 的索引搜索（首个 `x(j) >= xMin` / 最后一个 `x(j) <= xMax`）用
   `np.searchsorted` 复刻，已与朴素的逐行转写逐点比对（`0.0` 差）。
2. **上游写法留存**：`.m` 里 `zeros(Nx, Ny)` 却按 `fx(j, i)`（j = y 索引）
   索引，只有在方形网格上才碰巧正确；移植版统一用 `(Ny, Nx)`。
3. 官方 `T = 1.25` 在本机负载下不可行，只跑 `T = 0.05`（见"运行"）。

## 结果

参考运行约 84 s、AFSI 运行约 296 s（含导入，机器并行运行其它 demo）。
20 帧对齐（`t = 0 .. 0.04755`）。`compare.py` 原文：

| 指标（末帧） | compare.py 原文 |
|---|---|
| 标记点最大偏差 | `max|dX| 3.948e-03 m = 5.05e-01 h (666 Lagrangian points)` |
| 速度峰值 AFSI / IB2d | `6.5210e-01 / 7.4618e-01 (1.26e-01 relative)` |
| 速度场偏差 | `max|du| 1.403e-01 = 8.12e-01 max|u|`（带外）; `4.358e-01 (0.42 max|u|)`（带内）; `relative L2 1.445e-01` |
| 力场偏差（带外） | `max|df| 7.164e+01 away from the band, |f|max 1.100e+03` |
| 外力窗口 | `184 nodes, max|u| AFSI 0.5814 / IB2d 0.5769 (7.81e-03 relative); forcing check |f_x - F_x^module(u^AFSI)| <= 4.289e-02 on the core nodes` |
| 弹簧平均长度 | `AFSI 0.014571 / IB2d 0.014573 (1.75e-04 relative)` |

**能验证**：结构运动学（标记点 0.51 h）、弹簧长度 1.8e-4、**外力窗口内
速度峰值 7.8e-3**、以及"外力模块与 AFSI 力场在窗口核心节点上一致"
（残差 4.3e-2，相对 `kStiff*uMax*5` 量级可忽略）——这验证了移植的外力模块
与 AFSI 的实现一致。
**不能验证（如实记录）**：气泡内部/壁面附近的瞬时速度场逐点值
（`max|du|` 达 0.81 `max|u|`；相对 $L_2$ 1.4e-1）。柔性气泡在两种离散下的
变形细节不同（同 demo_478 一族的原因）。

## 运行

```bash
conda activate afsi-dolfinx
python run_reference.py --tend 0.05 --print-dump 50              # 官方 128²，~84 s
TFINAL=0.05 PRINT_DUMP=50 python main.py                         # ~296 s
python compare.py
```
