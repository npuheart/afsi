# demo_485 — IB2d `Example_Meshmerize_Me/Flow_Around_Starfish`

IB2d 官方算例（`matIB2d/Examples/Example_Meshmerize_Me/Flow_Around_Starfish`）
在 AFSI 中的复现：**目标点（钉死的海星）+ 任意外力脉冲射流**。

## 算例内容

* 水槽盒 `Lx = 1.0`、`Ly = 0.25`、`mu = 0.1`、`rho = 1000`、
  官方 `1024 x 256`、`dt = 5e-5`。官方 `T = 1.0`（20000 步）→
  本 demo **`NX = 128`（`Ny = 32`）、`T = 0.01`（200 步）**、`print_dump = 50`。
* **结构**：3010 个拉格朗日点（`starfish_flowtank.vertex`）全部是目标点
  （`k` 见 `starfish_flowtank.target`）——钉死的海星。
* **驱动**（`please_Compute_External_Forcing.m` → `ib2d_input/*.py`）：
  窄 `x`、几乎全 `y` 的射流窗口 `x in [0.04, 0.06]`、`y in [0.02, 0.22]`，

  ```
  Fx = -kStiff (uX - uX_tar),  uX_tar = -uMax sin(freq_P 2 pi t) ((Ly/2+w/2)-y)((Ly/2-w/2)-y),
  freq_P = 5, 脉冲占空比：on 一个周期、off 一个周期（rest = 1*period）,
  w = 0.25, uMax = 2.5e3, kStiff = 1e4,
  ```

  `Ly = 0.25` 使 `Ly/2 +- w/2` = `0` 与 `0.25`（整个水槽高度），
  所以是**全高脉冲射流**，峰值靶速 `2.5e3 m/s` 量级（罚力极硬）。

## 移植要点

1. 外力模块逐行移植（含 `|d|>L/2` 无关的索引搜索），与逐行转写逐点比对
   （差 `0.0`）。
2. `starfish_flowtank.*` 为 1-based，`ib2d_io` 自动减 1。
3. **缩减**：`NX = 128`、`T = 0.01`（上游 1024x256 不可行）。

## 结果

参考运行约 203 s、AFSI 运行约 87 s（墙钟时间含导入，机器并行运行其它 demo；
`main.py` 自报的模拟耗时只有 14.6 s，网格很小）。5 帧对齐（`t = 0 .. 0.01005`）。
`compare.py` 原文：

| 指标（末帧） | compare.py 原文 |
|---|---|
| 标记点最大偏差 | `max|dX| 1.834e-05 m = 2.35e-03 h (3010 Lagrangian points)` |
| 速度峰值 AFSI / IB2d | `2.0245e-01 / 1.5082e-01 (3.42e-01 relative)` |
| 速度场偏差 | `max|du| 1.670e-01 = 1.07e-01 max|u|`（带外）; `8.375e-02 (0.44 max|u|)`（带内）; `relative L2 2.073e+00` |
| 力场偏差（带外） | `max|df| 1.654e+03 away from the band, |f|max 1.204e+05` |
| 外力窗口 | `52 nodes, max|u| AFSI 0.2024 / IB2d 0.0735 (1.75e+00 relative); forcing check |f_x - F_x^module(u^AFSI)| <= 5.678e+02 on the core nodes` |

**能验证**：海星**完全不动**（2.35e-3 h）；外力模块与 AFSI 力场自洽
（残差 5.7e2，相对 `|f|max = 1.2e5` 约 5e-3）。
**不能验证（如实记录）**：速度场完全不可比（相对 $L_2$ 2.07，窗口内靶速差
1.75 倍）。原因是上游参数本身极端：`uMax = 2.5e3`、`w = 0.25` 的目标剖面
在 `1024 x 256` 网格上才勉强实现；在 `128 x 32` 上窗口只剩 52 个节点、
`dy = 7.8e-3` 而 `w = 0.25`，罚力`kStiff = 1e4` 对离散化极其敏感，
`dt = 5e-5` 下两种实现给出完全不同的局部速度。（同时也说明：
本例**不适合**在缩减网格上验证速度场，只适合验证"射流窗口 + 结构钉死"。）

## 运行

```bash
conda activate afsi-dolfinx
python run_reference.py --tend 0.01 --print-dump 50 --nx 128     # ~47 s
TFINAL=0.01 PRINT_DUMP=50 NX=128 python main.py                 # ~15 s
python compare.py
```
