# demo_459 — IB2d `Moving_Rubberband` (moving target points)

IB2d 官方算例（`matIB2d/Examples/Example_Moving_Rubberband`）在 AFSI 中的复现。
这是本仓库**第一个 `matIB2d` 算例**，也是第一个**目标点位置随时间被驱动**的算例。

## 算例内容

`1 x 1` 周期盒、`mu = 10`、`rho = 1`、`32 x 32`、`dt = 1e-4`、`T = 0.05`（500 步）。
结构是一条 **64 点圆周橡皮筋**（半轴 0.2×0.2，圆心 (0.25, 0.25)，点数 `N = 2 Nx`），
**既没有弹簧也没有梁**：每个点都是 `k = 1e7` 的**目标点**，即一条被刚性拖拽的
"载体"。目标点位置由算例的 `update_Target_Point_Positions.m` 逐步推进：

| 方向 | 规律 |
|---|---|
| $y$ | 每步 `y += (s/2) dt`（恒定上浮） |
| $x$ | `t mod period < period/2` 时右移 `+s dt`，否则左移 `-s dt` |
| 参数 | `period = 0.05 s`、`L = 0.5 m`、`s = L/(period/2) = 20 m/s` |

于是 `T = 0.05 s` 内橡皮筋**左右各扫一次单程**、同时**上浮 0.5 m**：圆心从
(0.25, 0.25) 走到 (0.75, 0.5) 再回到 (0.25, 0.75)。橡皮筋每步被拖动
`1e-3 m`（= 0.032 h，h = 1/32 网格步长；标记点间距 `ds = 1/64`），因此这个
算例同时是一个**快速运动浸入边界**的测试。

## 移植要点（AFSI 侧新增能力）

1. **1-based → 0-based 转换**。`matIB2d` 的输入文件用 MATLAB 的 1-based 编号，
   pyIB2d 却直接用文件里的编号索引 `xLag`（`IBM_Driver.py:360` 会越界）。因此
   `ib2d_io.load_example(..., index_base=1)` 统一把所有编号字段（
   `conn/p1/p2/p3/ids`）**偏移 `-index_base`** 并做上下界断言。原始文件按 1-based
   原样存放在 `ib2d_input/`（保持"算例文件不动"的原则），`run_reference.py` 在
   写入参考运行目录时另写一份 0-based 的 `.target`。
2. **动态目标点驱动**。驱动函数由算例提供（MATLAB 里是 `update_Target_Point_Positions.m`，
   pyIB2d 则从运行目录 `import update_Target_Point_Positions`）。本目录把 `.m`
   逐行翻译成 Python `ib2d_input/update_Target_Point_Positions.py` 供 pyIB2d
   参考运行使用；AFSI 侧在 `main.py:update_targets()` 里实现**同一递推**，并放在
   **同一步内相同位置**（半步标记点位置之后、力装配之前，与 `IBM_Driver.py:698`
   一致），时间取该步开始时刻。
3. **输出帧对齐**。pyIB2d 的 `cter` 从 0 开始、在步内容器自增前判断
   `cter % print_dump == 0 且 cter >= print_dump`，所以第 `k` 帧其实是**第
   `5k+1` 步结束**的状态，而它盖的时间戳 `k·print_dump·dt` 比状态**晚一步**
   （MATLAB IB2d 同样如此，属于上游的时间戳约定）。`main.py` 按**同样的步**输出
   帧（但记录真实时间），于是两边**逐帧同步**、`t_afsi[k] = t_ref[k] + dt`。
   这在静/慢结构上没有影响，但对以 20 m/s 拖动的算例若差一步就是 0.002 m 的假误差。

## 结果

参考运行 5.5 s、AFSI 运行 5.1 s，两边各 100 帧（`python compare.py` 生成
`figures/`）：

| 指标 | 结果 |
|---|---|
| 标记点偏差 | 典型 **2.1e-5 m = 6.6e-4 h**；全程最大 3.27e-4 m = 0.0105 h（出现在 `t ≈ 0.028`，即 $x$ 反向的折返点） |
| 形心位置 | 最大偏差 **8.8e-5 m**（瞬态），`t > 0.012` 后 ~1e-6 m |
| 目标点递推 | AFSI 的目标点位置与 `.m` 递推逐位一致（**1.6e-16 m**） |
| 标记点-目标点滞后 | AFSI 与 IB2d 的 `max|X−X_anchor|` 曲线一致到 ~1e-5 m（两侧都稳定在 ~1.1e-3 m ≈ `s·dt/2`，即一阶格式跟踪匀速锚点的稳态滞后）；初始瞬态峰值 6.5e-3 m 两侧同样吻合 |
| 包围面积 | AFSI 0.125462–0.125466（漂移 1.3e-6），IB2d 0.125455–0.125463（漂移 6.3e-6）；初始 64 边形面积 0.125462 与 $64$-gon 公式 $\tfrac12 N R^2\sin(2\pi/N)$ 一致 |
| 速度场 | `max\|u\|` AFSI 22.381 / IB2d 22.384；末帧 `max\|Δu\|` 2.9e-2（相对 0.13%） |

结论：**动态目标点机制移植正确**——两侧的形心轨迹、包围面积、以及"标记点滞后于
锚点"这一流固耦合量都吻合；标记点偏差（0.001–0.01 h）主要来自 20 m/s 拖动下
`k = 1e7` 刚性系绳的显式耦合滞后（该系绳频率 $\sqrt{k/m}\sim 5\times10^5$ rad/s
远高于 1/dt，两侧都靠流体阻尼维持稳定，这也是上游算例本身的性质）。

## 运行

```bash
conda activate afsi-dolfinx
python run_reference.py          # pyIB2d 参考 -> ib2d_reference.npz（~5 s）
python main.py                   # AFSI        -> plot/afsi_result.npz（~5 s）
python compare.py                # 对比 + figures/
```

环境变量：`TFINAL`、`DT`、`NX`、`PRINT_DUMP`、`GRAD_DIV`、`OUTPUT_PATH`。
`TFINAL=0.002 python main.py` 可做 smoke test（为了与参考逐帧对应，输出帧落在
`k·print_dump+1` 步及末帧上，所以在很短的算例里帧数较少）。
