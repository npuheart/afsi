# demo_462 — IB2d `NL_Rubberband_With_Springs`（非线性弹簧力律）

IB2d 官方算例（`matIB2d/Examples/Example_Standard_Rubberband/NL_Rubberband_With_Springs`）
在 AFSI 中的复现。它与 demo_444（pyIB2d `Rubberband_with_Springs`）**几何完全相同**
（64 点椭圆、半轴 0.4×0.2、零静长弹簧、32×32、dt=1e-3、T=1.5），区别只在
**`.spring` 文件第 5 列**：IB2d 的弹簧带一个非线性指数 $\alpha$，力律为

$$F = \frac{\alpha+1}{2}\,k\,(L - L_0)^{\alpha}\,\frac{\mathbf d}{L}$$

（leader 节点 $+F$、follower 节点 $-F$，段矢量做周期最小镜像处理）。本算例
$L_0 = 0$、$\alpha = 2$，于是 $F = \tfrac32 k L^2$ —— 一个**硬化二次力律**
（同 $k$ 下比线性弹簧硬 $\tfrac32 L$ 倍）。demo_444 用的是 $\alpha = 1$ 的线性律，
因此这是本仓库第一个非线性弹簧算例。

## 实现

1. `matIB2d` 的 1-based 文件 → `ib2d_io.load_example(..., index_base=1)` 重编号；
   `run_reference.py` 写出重编号后的 `.spring`（本次是第一个需要重编号 **spring**
   文件而不是 `.target` 的算例，`shift_ids(path_in, path_out, n_id_cols=2)`）。
2. AFSI 侧按 IB2d 的通用 $\alpha$ 力律实现（`spring_forces`），并用逐弹簧循环做
   逐位对照（`rel. max err = 0.0`）。
3. 结构 FE 网格直接用 `.spring` 的连通关系（闭合环），零刚度承载 64 个标记点。
4. 默认 $\gamma = 100$：零静长闭合环在 Q2/Q1 下会因无法表示跨膜压差而漏面积
   （demo_444/447 的结论）。

## 结果

参考运行 13 s、AFSI 运行 7.8 s，两边各 76 帧：

| 指标 | AFSI ($\gamma=100$) | AFSI ($\gamma=0$) | IB2d |
|---|---|---|---|
| 标记点偏差（对参考） | 典型 **4.3e-3 m = 0.14 h**，最大 7.2e-3 m = 0.23 h | 最大 5.4e-2 m = 1.7 h | — |
| 包围面积（0.250924 起） | 末值 0.247519（漂移 3.4e-3） | 末值 0.170728（漂移 **8.0e-2**） | 末值 0.241483（漂移 9.4e-3） |
| 周长（1.936912 起） | 末值 1.770396 | — | 末值 1.751186 |
| 形心 | 位移 6.9e-13 m（环对称） | — | — |
| 速度 | `max\|u\|` 0.118 m/s | — | 0.346 m/s |

结论：**非线性（$\alpha=2$）弹簧力律移植正确**——两侧的周长收缩（1.937 → 1.770/1.751）
与标记点位置一致到 0.2 h 量级；残余差异来自零静长闭合环的张力-压力平衡
（两侧的面积/周长松弛速度略有差别，与 demo_444 记录的同一机制一致），
而 $\gamma=0$ 时 AFSI 侧会明显漏面积（漂移大 8 倍、偏差升到 1.7 h），
因此默认取 $\gamma=100$。

## 运行

```bash
conda activate afsi-dolfinx
python run_reference.py          # pyIB2d 参考 -> ib2d_reference.npz（~13 s）
python main.py                   # AFSI        -> plot/afsi_result.npz（~8 s）
python compare.py                # 对比 + figures/
GRAD_DIV=0 OUTPUT_PATH=./plot_g0 python main.py   # γ=0 对照
```

环境变量：`TFINAL`、`DT`、`NX`、`PRINT_DUMP`、`GRAD_DIV`、`OUTPUT_PATH`。
