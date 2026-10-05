# demo_499 — IB2d `Examples_First_Year_Seminar/Beam_and_Cells`

IB2d 官方算例（`matIB2d/Examples/Examples_First_Year_Seminar/Beam_and_Cells`）
在 AFSI 中的复现：**弹簧 + 不变梁 + 目标点**（全部由模板覆盖，无用户模块）。

## 算例内容

* `1 x 1` 周期盒、`mu = 0.01`、`rho = 1`、官方 `256 x 256` →
  本 demo **`NX = 128`**、`dt = 5e-5`。官方 `T = 1.25`（25000 步）→
  本 demo **`T = 0.05`（1000 步）**，`print_dump = 50`（20 帧）。
* **结构**：672 个拉格朗日点：**671 根弹簧**（`k = 5e5`）+
  **510 根不变梁**（`kb = 5e11`、`C = 0`）+ **2 个钉死目标点**（`k = 1.75e8`）。
  一条被两端钉住的直梁（"Beam"）加上挂在其上的若干"细胞"（"Cells"，
  由弹簧/梁构成的闭合小体）；上游目录里还有一个 `.nonInv_beam` 文件，
  但 `input2d` 的 `nonInvariant_beams = 0`，两边都**不使用**它。

## 移植要点

1. **无需移植任何 `.m` 模块**：模板的弹簧/梁/目标点已实现，且启动时与逐行
   循环对照（机器精度）。
2. `Beam_and_Cells.*` 为 1-based，`ib2d_io` 自动减 1。
3. 唯一改动是降分辨率 + 缩短 `Tfinal`（见"运行"）。

## 结果

参考运行约 56 s、AFSI 运行约 316 s（含导入，机器并行运行其它 demo）。
20 帧对齐（`t = 0 .. 0.04755`）。`compare.py` 原文：

| 指标（末帧） | compare.py 原文 |
|---|---|
| 标记点最大偏差 | `max|dX| 4.162e-03 m = 5.33e-01 h (672 Lagrangian points)` |
| 速度峰值 AFSI / IB2d | `2.8058e+00 / 2.7548e+00 (1.85e-02 relative)` |
| 速度场偏差 | `max|du| 3.848e-01 = 8.60e-01 max|u|`（带外）; `4.153e-01 (0.85 max|u|)`（带内）; `relative L2 5.254e-02` |
| 力场偏差（带外） | `max|df| 8.622e+01 away from the band, |f|max 1.022e+03` |
| 弹簧平均长度 | `AFSI 0.001251 / IB2d 0.001255 (2.84e-03 relative)` |

**能验证**：结构运动学（标记点 0.53 h）、速度峰值 1.9%、速度场相对 $L_2$
5.3e-2、弹簧平均长度 2.8e-3 —— 即"梁与细胞怎么动、流场整体多快"都一致。
**不能验证（如实记录）**：梁/细胞表面的瞬时速度场逐点值
（`max|du|` 达 0.86 `max|u|`）。`kb = 5e11` 的刚性结构 + 细胞间的
薄间隙让局部流场对离散方式敏感（同 demo_493/478 一族）。

## 运行

```bash
conda activate afsi-dolfinx
python run_reference.py --tend 0.05 --print-dump 50 --nx 128     # ~56 s
TFINAL=0.05 PRINT_DUMP=50 NX=128 python main.py                 # ~316 s
python compare.py
```
