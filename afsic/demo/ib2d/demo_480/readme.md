# demo_480 — IB2d `Hoover_Jellyfish`（水母搏动：弹簧 + 目标点 + 非不变梁 + `update_Springs`）

IB2d 官方算例（`matIB2d/Examples/Example_Jellyfish_Swimming/Hoover_Jellyfish`）。
本 demo 由**合并模板** `afsic/demo/ib2d/_template/` 用
`python port_example.py 480 Example_Jellyfish_Swimming/Hoover_Jellyfish` 生成，
是模板覆盖「线性/非线性弹簧 + 目标点 + 非不变梁 + 例子自带 `update_Springs.py`」
这一组合的第一个验证算例。

## 算例内容

* `3 x 10` 周期盒、`mu = 6.667`、`rho = 1000`；官方 `192 x 640`（`dx = dy = 1/64`）、
  `dt = 1e-5`、`T = 5`（50 万步）、`print_dump = 5000`。
* **结构**（`jelly.vertex` 共 415 点）：
  * 357 根**弹簧**（`k = 1e5 … 1.28e9`，静止长度 `1/128 … 1.0`）；
  * 96 个**钉死目标点**（`k = 2.5e6`，0 基 id 319…414 / 文件里 1 基 320…415，
    全部位于 `y = 9.96875` —— 伞盖平顶）；
  * 317 根**非不变梁**（`kb = 5.24288e11`，参考曲率 `|C| ≤ 1.83e-4`）。
* **搏动律**（例子自带的 `update_Springs.m`）：`.spring` 文件 MATLAB 第 `319:357` 行
  （即 0 基 318…356，`k = 1e5` 的伞缘弹簧）的静止长度按

  ```
  RL_i = |cos(freq · pi · t)|,   freq = 2   ⇒  周期 1 s、每秒两次收缩
  ```

  周期性收缩 —— 伞盖于是反复排水推进。伞缘 39 根弹簧被捏紧，其余结构保持弹性。
* 该算例没有不变梁（`beams`）、没有肌肉、没有外力模块。

## 移植要点

* `update_Springs.m` → `ib2d_input/update_Springs.py`（逐行移植，含中文无关的常量
  `FREQ = 2`、行区间 `319:357`）。pyIB2d 从**运行目录**导入同名模块，`main.py` 从
  `ib2d_input/` 导入，两边跑的是**同一份**代码；调用位置也与驱动一致
  （半步位移之后、力计算之前，传入**整步**位置 `xLag/yLag`）。
* 文件编号是 MATLAB 的 1 基，`ib2d_io.detect_index_base()` 自动判别为 `index_base = 1`
  并在读取时统一减 1，无需为每个算例写死。
* 模板的通用机制全部自动开启：`springs`（含 IB2d 的 `alpha` 非线性度）、
  `target_pts`、`nonInvariant_beams`、`update_springs`；本例不需要任何额外代码。
* `main.py` 的两项自检（把向量化力与朴素逐行循环对比，随机扰动构型）通过：
  弹簧 `0.00e+00`、非不变梁 `1.25e-16`。

## 本 demo 的缩减设置

| 参数 | 官方 | 本 demo | 说明 |
|---|---|---|---|
| `Nx x Ny` | 192 x 640（`h = 1/64`） | **96 x 320**（`h = 1/32`） | `NX=96`，`NY` 按盒长比自动 = 320；另做了 `48 x 160` 的粗网格对照 |
| `dt` | 1e-5 | **1e-4** | 10 倍；0.05 s 内 500 步可跑完 |
| `Tfinal` | 5.0 | **0.05** | 只有 1 s 搏动周期的 5% |
| `print_dump` | 5000 | **100** | 5 帧 |

## 结果（`NX=96`、`DT=1e-4`、`T=0.05`；`compare.py` 自身输出）

参考运行 ~58 s、AFSI 运行 ~380 s，两边各 5 帧（`t = 0 … 0.0401`）。

| 帧 `t` | `max\|dX\|/h` | `max\|u\|` AFSI / IB2d | 窄带外 `max\|du\|` | 带内 `max\|du\|` | 窄带外相对 $L_2$ | 窄带外 `max\|df\|` |
|---|---|---|---|---|---|---|
| 0.01010 | 7.96e-03 | 2.118e-01 / 1.668e-01 | 4.111e-02 | 8.084e-02 | 1.613e-01 | 2.01e+00 |
| 0.02010 | 1.48e-02 | 1.905e-01 / 1.939e-01 | 3.183e-02 | 7.386e-02 | 1.292e-01 | 7.99e+00 |
| 0.03010 | 1.57e-02 | 1.159e-01 / 1.408e-01 | 2.687e-02 | 5.304e-02 | 1.310e-01 | 1.64e+01 |
| 0.04010 | 9.30e-03 | 9.214e-02 / 9.834e-02 | 3.449e-02 | 6.516e-02 | 2.364e-01 | 2.37e+01 |

末帧（`t = 0.0401`）汇总：

| 指标 | 结果 |
|---|---|
| 标记点偏差（415 点） | 2.907e-04 m = **9.30e-03 h** |
| 速度幅值 `max\|u\|` AFSI / IB2d | 0.0921 / 0.0983（**6.31%**） |
| 弹簧平均长度 AFSI / IB2d | 0.114983 / 0.114930（**4.54e-04** 相对差） |
| 带内 `max\|du\|` | 6.52e-02 = 0.66 `max\|u\|`（IB 力带内，弥散差异） |
| 力场（`\|f\|max = 5.84e+04`） | 窄带外 `max\|df\| = 23.7`（0.04%） |

粗网格对照（`NX=48`、`NY=160`、`h = 1/16`、`T=0.1`，作为 `ib2d_reference_n48.npz`
+ `plot_n48/` 被 `compare.py` 自动配对）：末帧标记点偏差 **1.03e-01 h**、
幅值差 **19.4%**、弹簧平均长度差 **1.09e-02** —— 网格加密一倍（`h` 减半）后
标记点误差改善约 11 倍、幅值误差改善约 3 倍，说明本例的结构力在粗网格上
被 `supp = 4` 的 δ 核严重抹平（伞缘弹簧间距 `1/128`，远小于 `h`）。

## **不可比较**的部分（明确声明）

1. **只有 1 s 搏动周期的前 5%**。官方 `T = 5`（5 个完整周期、50 万步），本 demo 跑
   `T = 0.05`：`t = 0.05` 时 `|cos(2π·2·0.05)| = cos(0.2π) = 0.809`，伞缘只收缩了约
   19%，**水母尚未完成一次完整搏动**，因此这里比较的是启动瞬态，不能与官方的
   周期性游动（推进速度、涡环脱落）对比。
2. **远场速度 $L_2$ 误差（0.13–0.24）不代表系统性偏差**。该瞬态里远场速度极小
   （`max|u| ≈ 0.09 m/s`），$L_2$ 由低幅区噪声主导；真正的幅值指标 `max|u|` 只差 6.3%，
   而带内（δ 核支撑区）的弥散差异 ~0.5–0.7 倍 `max|u|` 是两种投影（IB2d 的节点
   共位 FFT 投影 vs AFSI 的 Taylor–Hood $L_2$ 投影）固有的差别。
3. **`dt` 比官方粗 10 倍、`h` 粗 2 倍**：官方 `dt = 1e-5` 配 `h = 1/64`；本例 `dt = 1e-4`
   配 `h = 1/32`。粗网格对照正好量化了这一点（见上表）。
4. 结构本身的**运动学**不可独立验证：静止长度由同一份 `update_Springs.py` 决定，
   两边逐位相同（这是"共享模块"设计的必然结果），真正独立的检验是标记点偏差
   （9.3e-03 h）和弹簧平均长度（4.5e-04）。

## 运行

```bash
conda activate afsi-dolfinx
python run_reference.py --nx 96 --dt 1e-4 --tend 0.05 --print-dump 100   # ~58 s
NX=96 DT=1e-4 TFINAL=0.05 PRINT_DUMP=100 python main.py                  # ~380 s
python compare.py

# 粗网格对照（可选，compare.py 会自动把 ib2d_reference_n48.npz 与 plot_n48/ 配对）
python run_reference.py --nx 48 --dt 1e-4 --tend 0.1 --print-dump 100 \
    --out ib2d_reference_n48.npz --run ib2d_run_n48                      # ~66 s
NX=48 DT=1e-4 TFINAL=0.1 PRINT_DUMP=100 OUTPUT_PATH=plot_n48 python main.py  # ~150 s
```

（运行时间是在**单线程** `OMP_NUM_THREADS=1` 下测得的墙钟时间；测量时机器上同时
有多个 AFSI 作业，负载很高，AFSI 一侧的墙钟时间被明显放大。`python compare.py`
写出 `figures/compare_history.png`、`figures/compare_fields.png`、
`figures/compare_profiles.png` 与 `figures/compare_table.csv`。）
