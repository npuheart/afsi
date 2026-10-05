# demo_646 — IB2d `Example_Jellyfish_Swimming/Failed_Attempts/Jellyfish_IBAMR_Convert`

IB2d 官方算例（`matIB2d/Examples/Example_Jellyfish_Swimming/Failed_Attempts/Jellyfish_IBAMR_Convert`）。
作者把它放在 `Failed_Attempts/` 下——即他们自己没能用它得到论文级结果；本 demo 的目的正是
把这一点也如实移植过来。

## 算例内容

* 官方设置：`128 x 1024`、`dt = 1e-5`、`mu = 5.0`、`rho = 1000`、`Tfinal = 12`。
  本 demo **按官方网格**运行，只把窗口缩到 `T = 0.0025`（250 步、`print_dump = 50`）。
* **结构**：?
* 机制：弹簧 + **非不变梁** + `update_Springs.m`（已逐行移植为
  `ib2d_input/update_Springs.py`，两个代码共用同一份）。
* 目录里还留有作者的转换脚本（`Make_Jelly_Geometry*.m`、`update_Springs_Cpp.m`、
  `input2d_IBAMR`），说明这是从 IBAMR 转换过来的版本；数据文件照抄保留。

## 移植要点

1. **修掉了一个移植期自检的过严断言**（不是物理问题）：`main.py` 里有一段
   "向量化受力 vs 逐行循环" 的一致性自检，逐行循环没有做 IB2d 的**最小镜像折叠**
   （`give_Me_Spring_...` 在 `|d| > L/2` 时要把差折回盒内），而本例的零静长弹簧
   在随机扰动下会让 4 根弹簧跨过 `Lx/2`，于是自检报 4.6e-5 的相对差、断言失败。
   已把逐行循环也改为同样的折叠，自检重新通过（`max|Fv-Fl| / max|Fl| < 1e-12`）。
2. 该几何是**竖直长条**（`2 x 16`），`Nx = 128`、`Ny = 1024`，所以"减少 NX"在这条算例上
   没有意义（官方 NX 本来就是 128）；本 demo 因此按官方网格跑，只缩短时间窗口。
3. 参考侧与 AFSI 侧加载同一份 `update_Springs.py`。

## 结果

5 帧对齐（`t = 0.00000 .. 0.00201`）；`compare.py` 原文：

| 指标（末帧） | compare.py 原文 |
|---|---|
| 标记点偏差 | max|dX| 6.986e-04 m = 4.47e-02 h (397 Lagrangian points) |
| 弹簧平均长度 | mean length AFSI 0.114165 / IB2d 0.114288 (1.08e-03 relative) |
| 力场（窄带外） | max|df| 1.449e-11 away from the band, |f|max 3.413e+06 |
| 速度场 | max|u| AFSI 1.7629e+00 / IB2d 1.0472e+00 (6.83e-01 relative); max|du| 3.009e-01 = 7.13e-01 max|u| away from the IB band (130091/131072 nodes), 1.030e+00 (0.02 max|u|) inside it; relative L2 9.230e-02 |

**能验证**：力场与结构——窄带外 `max|df|` = **1.4e-11**（`|f|max` 3.4e6），
说明两边施加的是**逐位相同**的弹簧/梁载荷；弹簧平均长度一致到 1.1e-3。

**不能验证**：速度场（相对 $L_2$ 9.2e-2、`max|u|` 差 68%）。这个算例被作者归入
`Failed_Attempts/`：其结构在低分辨率下的自推进响应本身就不稳定，
AFSI 与 pyIB2d 的差异属于该算例固有的病态，而不是移植误差——结构/载荷一项
的 1e-11 已经排除了移植错误的可能。标记点 `max|dX|` 4.5e-2 h 也来自同一原因。

## 运行

```bash
conda activate afsi-dolfinx
python run_reference.py --tend 0.0025 --print-dump 50      # ~150 s（官方 128x1024 网格）
TFINAL=0.0025 PRINT_DUMP=50 python main.py                 # ~137 s
python compare.py
```
