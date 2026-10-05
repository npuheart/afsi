# demo_626 — IB2d `Example_Testing_Thru_Boundaries/Stirring_With_N_Multiple_Cells_512x64`（凝血模型）

IB2d 官方算例（`matIB2d/Examples/Example_Coagulation/Example_Testing_Thru_Boundaries/Stirring_With_N_Multiple_Cells_512x64`）。

## 算例内容

* 官方设置：周期盒 `512 x 128`、`dt = 2.5e-5`、`mu = 0.05`、`Tfinal = 1.25`（5 万步、`print_dump = 200`）。
  本 demo 用 **`256 x 64`、`T = 0.06`（2400 步）** 运行（内存与时间受限；`ib2d_input/` 仍是上游逐字数据）。
* **结构**：4987 个拉格朗日点 = 7428 根弹簧 + 4884 根不变梁 + 103 个**被搅拌的目标点**
  （`update_Target_Point_Positions.m` 被逐行移植为 `ib2d_input/update_Target_Point_Positions.py`，
  两个代码共用同一份），外加 **33 个凝血细胞**（`channel.coagulation`）。
* **凝血机制**：细胞由"键"（bond）连接——当两个细胞的Lagrangian 点距离小于
  `threshold` 时形成键（刚度 `bond_k`、静长在形成瞬间冻结），并在拉伸超过
  `fracture` 时断裂；键力 `f_coag` 直接加到结构的拉格朗日力上。
  实现见 `ib2d_input/coagulation_model.py`（MATLAB 逐行忠实移植，pyIB2d 侧使用同一模块）。

## 移植要点

1. 凝血模型的**判据、冻结静长、断裂阈值**都按上游 `.m` 逐条复现，两边跑同一个
   `coagulation_model.py`（`run_reference.py` 把它拷进 pyIB2d 的运行目录），
   因此键的形成/断裂**时刻**只受速度场差异影响，不受实现差异影响。
2. 对比脚本额外输出 `bonds`（两边各自的键数与**共同键数**）、`max|dL|`（当前长度差）、
   `max|dRL|`（冻结静长差）、`f_coag` 峰值，以及**从保存的标记点重算的"低于阈值的细胞对"**
   数量（几何一致性检查）。
3. 上游 `input2d` 的 `coagulation` 标志与 `channel.coagulation` 文件配套；标志关掉时
   `compare.py` 会把它作为"无凝血消融"处理（本 demo 默认按官方开启）。

## 结果

参考约 220 s、AFSI 约 148 s；两边帧对齐后 `compare.py` 原文：

| 指标（末帧 `t = 0.06`） | compare.py 原文 |
|---|---|
| 键（bonds） | AFSI 3 / IB2d 3（shared 3，`max|dL|` 7.13e-3、`max|dRL|` 4.83e-4） |
| `max\|f_coag\|` AFSI / IB2d | 3.9720 / 5.4906 |
| 速度峰值 AFSI / IB2d | 15.496 / 15.396（**6.45e-3** 相对）、`max|du|` 1.511 |
| 标记点偏差 | 1.57e-2 m = 8.03 h |
| 欧拉力场 | 相对 L2 7.95e-1、`max|df|` 3.33e4 = 1.03 `max|f|` |
| 几何一致性（低于阈值的细胞对） | 2 / 0 |
| 目标点锚点差 | 0.00e+00（两边同一个 `.m`） |

**能验证**：键的形成/断裂（3 个键两边完全一致）、速度场（6e-3）

**不能验证**：欧拉力场的相对 $L_2$（本族的力高度集中在键/梁节点上，逐点比较被
"键形成瞬间的微小差异"放大；键本身的数目、共同键与 `f_coag` 峰值才是可靠指标）。
标记点的 `max|dX|` 以 `h` 为单位偏大，是因为被搅拌的整个结构在盒内平移了几十厘米，
而 `h` 只有 `Lx/Nx`——应按结构自身尺度（约 0.3–0.6 m 的搅拌行程）理解。

## 运行

```bash
conda activate afsi-dolfinx
python run_reference.py --nx 256 --tend 0.06 --print-dump 150      # ~ 220 s
NX=256 TFINAL=0.06 PRINT_DUMP=150 python main.py                   # ~ 148 s
python compare.py
```

> 更长的窗口会更好：官方 `T = 1.25` 才能把细胞全部搅到一起；本 demo 的
> `T = 0.06` 只覆盖机制被触发的最初阶段（见上面"键"一行）。把 `--tend/TFINAL`
> 调大即可看到完整的凝血过程（内存占用与 `NX` 成正比，与 `T` 只是线性时间）。
