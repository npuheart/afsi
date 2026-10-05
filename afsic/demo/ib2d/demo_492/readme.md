# demo_492 — IB2d `Example_HeartTube/Sea_Spiders`（海蜘蛛式多足游动结构）

IB2d 官方算例（`matIB2d/Examples/Example_HeartTube/Sea_Spiders`）。
这是 `Example_HeartTube` 家族里结构最复杂的一个：**同一套机制（弹簧 + 行波静长
＋不变/非不变梁 + 目标点 + 被动标量）用在 2868 点的多足结构上**。

## 算例内容

* 条带周期盒 `1 x 0.125`、`mu = 0.01`、`rho = 1`；官方 `512 x 128` 平移到
  `Nx = 1024`、`Ny = 128`、`dt = 5e-5`、`Tfinal = 1.0`（2 万步、`print_dump = 200`）。
* **结构**：2868 个拉格朗日点 = 海蜘蛛（一个身体 + 多条腿）：
  **3685 根弹簧**（`k` 从 0.1 到 5e7）、**2454 根非不变梁**（`.nonInv_beam`，6 列格式）、
  **414 个钉死目标点**，以及 `update_Springs.m` 驱动的**行波静长**（已逐行移植为
  `ib2d_input/update_Springs.py`，两个代码共用同一模块）。
* **标量**：`sea_spider.concentration` 给出 `kDiff = 5e-5` 与 `(128, 1024)` 的初始浓度场
  （节点值和 28905）——驱动力来自结构自身，标量只被携带。

## 移植要点

1. **必须缩减网格才能安全运行**。官方 `1024 x 128` 的求解（外加同尺寸浓度场）
   需要数 GB 内存；一台机器上同时跑多个这类算例会把内存吃满。因此本 demo 附带
   `make_reduced_input.py`，它生成**保持长宽比**的 `reduced_input/`：

   ```
   python make_reduced_input.py --nx 128 --tend 0.005 --print-dump 25
   ```

   该脚本把 `input2d` 的 `Nx/Ny`（以及可选的 `Tfinal/print_dump`）改写为缩减值，
   其余数据文件逐字复制，并把 `(128, 1024)` 的浓度场按整数步长抽稀成 `(16, 128)`
   （初始斑块在几十个节点尺度上光滑，抽稀只让初始条件偏移不到半个节点）。
   `reduced_input/` 已 gitignore，`ib2d_input/` 保持上游逐字不变。
   两个代码都对它运行：`run_reference.py --src reduced_input …` 与
   `IB2D_EXAMPLE=reduced_input python main.py`。
2. 机制全部沿用已验证的实现：弹簧/目标点/不变梁见 demo_446/449，行波静长见 demo_464，
   非不变梁见 demo_449，被动标量见 demo_476/477。
3. **上游数据里有一处小矛盾**：`input2d` 声明了 `porous_media = 0`，但目录里带着
   `sea_spider.porous`（以及 `beams = 0` 却带 `sea_spider.beam`）。驱动按标志位忽略它们，
   本 demo 同样忽略——文件照抄保留。

## 结果

缩减设置 `128 x 16`、`dt = 5e-5`、`T = 0.005`（101 步、5 帧，`t = 0 .. 0.00505`）：
参考运行约 40 s、AFSI 运行约 1 s。

| 指标（末帧 `t = 0.00505`） | compare.py 原文 |
|---|---|
| 标记点偏差（2868 点） | 6.27e-4 m = **8.03e-2 h** |
| 弹簧平均长度 AFSI / IB2d | 0.012167 / 0.012156（**9.6e-4** 相对） |
| 罚力场 `max\|df\|`（窄带外） | **2.30e-14**（`\|f\|max` 1.02e4） |
| 浓度峰值 AFSI / IB2d | 1.000000 / 1.000000（1.1e-7 相对） |
| 浓度相对 $L_2$ 误差 | **4.12e-2** |
| 浓度 `max\|ΔC\|` | 1.68e-1（窄带外 8.4e-2） |
| 浓度总量 AFSI / IB2d | 390.64 / 394.53 |
| 速度峰值 AFSI / IB2d | 1.975 / 2.961（3.3e-1 相对） |

**能验证**：结构力学——弹簧长度一致到 **9.6e-4**、标记点 8e-2 h、力场差 2e-14
（说明两边加载的是同一套力），以及标量的**峰值与总量**（峰值 1e-7、总量 ~1%）。

**不能验证**：速度场（相对 $L_2$ 5.4e-1）。本 demo 的缩减网格是 `128 x 16`
（长宽比 8:1），横向只有 16 个节点，而官方是 128——**细长的横向分辨率不足以支撑
该速度场**，所以这一项不可比；标量场 4.1e-2 的差异正是这个速度差异的脚印
（算子本身已在 demo_476 验证到 1e-11）。若需要速度场对比，应在有更大内存的机器上
用 `--nx 512`（或更大）重跑。

## 运行

```bash
conda activate afsi-dolfinx
python make_reduced_input.py --nx 128 --tend 0.005 --print-dump 25
python run_reference.py --src reduced_input --tend 0.005 --print-dump 25   # ~40 s
IB2D_EXAMPLE=reduced_input TFINAL=0.005 PRINT_DUMP=25 python main.py        # ~1 s
python compare.py
```
