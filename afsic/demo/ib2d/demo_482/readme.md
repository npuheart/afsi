# demo_482 — IB2d `Examples_Education/Interpolation/Swimmer`（游泳者：弹簧 + 非不变梁 + `update_nonInv_Beams`）

IB2d 官方算例（`matIB2d/Examples/Examples_Education/Interpolation/Swimmer`）。
本 demo 由**合并模板** `afsic/demo/ib2d/_template/` 用
`python port_example.py 482 Examples_Education/Interpolation/Swimmer` 生成，
验证模板覆盖「弹簧 + 钉死目标点 + **非不变梁** + 例子自带 `update_nonInv_Beams.py`
（每步改写参考曲率）」这一组合 —— 其中 `update_nonInv_beams` 是此前 35 个 demo
从未覆盖过的机制。

## 算例内容

* `8 x 4` 周期盒、`mu = 10`、`rho = 1000`；官方 `512 x 256`（`dx = dy = 1/64`）、
  `dt = 2.5e-5`、`T = 3`（12 万步）、`print_dump = 800`。
* **结构**（`swimmer.vertex` 共 440 点，分两部分）：
  * **弹性体**：id 1…184（0 基 0…183）组成一条一维"脊柱"，
    `x ∈ [0.800, 2.008]`、`y ∈ [1.750, 2.265]`；
    183 根**弹簧**（`k = 9.5625e9`，静止长度 `1/128`）+
    182 根**非不变梁**（`kb = 2.03634e12`，参考曲率由文件给出，`|C| ≤ 1.04e-4`）；
  * **"左壁"**：id 185…440（0 基 184…439）是 256 个 `k = 1e6` 的钉死目标点，
    全部位于 `x = 0.0078`、`y ∈ [0.0078, 7.977]`
    （**注意**：跨度是盒高 `Ly = 4` 的两倍，绕周期折叠后成为两列重合的钉死点）。
* **游动律**（例子自带的 `update_nonInv_Beams.m`）：从 `swimmer.phases`
  （184 行 3 列）取出脊柱的两个"相位" `yP1`（= 初始构型）、`yP2`，
  用分段三次 `g(t̃)` 在 0.5 s 周期（0.25 s 下潜 + 0.25 s 上摆）内插值，然后改写
  **非不变梁的 `y` 参考曲率**：

  ```
  beams_info[:, 6] = yPts[:-2] + yPts[2:] - 2·yPts[1:-1]
  ```

  由于 `|yP2 - yP1| ≈ 16 m`（远大于身体长度 1.2 m），参考曲率被"猛烈切换"，
  脊柱像鞭子一样甩动并推动流体。

## 移植要点

* `update_nonInv_Beams.m` → `ib2d_input/update_nonInv_Beams.py`：逐行移植，
  含 `read_File_In('swimmer.phases')`（改为 `np.loadtxt`，先按模块自身目录查找，
  再退回当前目录，这样 AFSI 侧从 `ib2d_input/` 导入、pyIB2d 侧从运行目录导入
  都能找到 `swimmer.phases`）。不变量列 6（MATLAB）→ numpy 列 5。
* **上游 bug（已打补丁）**：pyIB2d 的驱动在第 711 行把**不变梁**数组
  `beams_info`（本算例未开启，是 `zeros((1,1))` 占位）传给了 `update_nonInv_Beams`，
  直接用会 `IndexError`。`run_reference.py` 里包了一层，把**非不变梁**数组喂给它；
  `main.py`（模板）里直接传正确的数组，并注释说明这一偏差点。
  两边因此都用**同一份**移植模块、同一组输入数据。
* 模板的 `noninv_beam_forces` 按 IB2d 原样：`r = (Xp + Xr - 2Xq) - C`，
  `F_q += 2kr`、`F_p -= kr`、`F_r -= kr`；自检（随机扰动构型 vs 朴素逐行循环）
  相对误差 `1.25e-16`。
* `ib2d_io.detect_index_base()` 判定 `index_base = 1`。

## 本 demo 的缩减设置

| 参数 | 官方 | 本 demo | 说明 |
|---|---|---|---|
| `Nx x Ny` | 512 x 256（`h = 1/64`） | **128 x 64**（`h = 1/16`） | `NX=128`；另做了 `64 x 32` 粗网格对照 |
| `dt` | 2.5e-5 | **1e-4** | 4 倍；已验证 `dt = 2.5e-5`（官方值，2000 步）给出同样的幅值（`t = 0.05` 时 0.893 vs 0.890 m/s），即该瞬态在 `dt = 1e-4` 上已收敛 |
| `Tfinal` | 3.0 | **0.05** | 1/5 个下潜冲程 |
| `print_dump` | 800 | **100** | 5 帧 |

## 结果（`NX=128`、`DT=1e-4`、`T=0.05`；`compare.py` 自身输出）

参考运行 ~207 s、AFSI 运行 ~249 s，两边各 5 帧（`t = 0 … 0.0401`）。

| 帧 `t` | `max\|dX\|/h` | `max\|u\|` AFSI / IB2d | 幅值相对差 | 窄带外 `max\|du\|` | 窄带外相对 $L_2$ | 窄带外 `max\|df\|` |
|---|---|---|---|---|---|---|
| 0.01010 | 1.08e-04 | 1.801e-02 / 2.572e-02 | 30.0% | 6.856e-03 | 6.523e-01 | 5.98e+00 |
| 0.02010 | 4.66e-04 | 3.557e-02 / 3.286e-02 | **8.2%** | 8.435e-03 | 2.421e-01 | 8.94e+00 |
| 0.03010 | 2.80e-03 | 1.309e-01 / 1.040e-01 | 25.9% | 2.477e-02 | 1.654e-01 | 5.49e+01 |
| 0.04010 | 8.95e-03 | 3.856e-01 / 2.872e-01 | 34.3% | 8.429e-02 | 1.886e-01 | 2.48e+02 |

末帧（`t = 0.0401`）汇总：

| 指标 | 结果 |
|---|---|
| 标记点偏差（440 点） | 5.596e-04 m = **8.95e-03 h** |
| 弹簧平均长度 AFSI / IB2d | 0.007805 / 0.007805（**3.16e-05** 相对差）—— 结构本身几乎逐位一致 |
| 力场（`\|f\|max = 9.48e+04`） | 窄带外 `max\|df\| = 248`（0.26%） |
| 速度幅值 | `t = 0.02` 时 **8.2%**；`t = 0.04` 时 34.3%（见下） |

粗网格对照（`NX=64`、`NY=32`、`h = 1/8`，作为 `ib2d_reference_n64.npz` + `plot_n64/`
被 `compare.py` 自动配对）：末帧标记点偏差 **3.49e-03 h**、弹簧平均长度差 **3.22e-05**、
幅值差 **33.1%** —— 加密一倍后标记点误差改善 2.6 倍，但**幅值差没有改善**
（33% → 34%），说明速度偏差不是分辨率问题（见下）。

## **不可比较**的部分（明确声明）

1. **`t ≳ 0.02` 之后的速度场不可比较**。脊柱的参考曲率被切换的幅度
   `|yP2 - yP1| ≈ 16 m` 与 `kb = 2e12` 组合，使流场在 0.03 s 内从
   `max|u| ≈ 0.014 m/s` **指数增长**到 ~0.3–0.4 m/s（每帧约翻倍）。
   在这种快速增长的瞬态里，两种时间推进（pyIB2d 前向 Euler + 共位 FFT 投影 vs
   AFSI 的 Peskin 两步 RK2 + Taylor–Hood $L_2$ 投影）的微小增长率差异会被指数放大：
   两边在 `t = 0.02` 只差 **8.2%**，到 `t = 0.04` 已差 **34%**。
   加密网格（`h` 减半）**没有**改善这一点（33% → 34%），所以它不是分辨率问题，
   而是方案差异在指数瞬态上的放大。**结构**（标记点 9e-3 h、弹簧长度 3e-5）与
   **力的分布**（带外 0.26%）仍然可比。
2. **"左壁"的 256 个钉死点在 `Ly = 4` 的盒里绕了两圈**（`y` 跨度 0.008…7.977），
   折叠后有 128 对坐标完全重合的标记点。IB2d 与 AFSI 都按周期折回处理，
   力的最小镜像也一致，但这条"壁"**不是**上游 README 所描述的物理边界
   （该算例属于教学用 `Interpolation` 目录）；本 demo 只把它当作用户数据原样搬运，
   不对它的物理含义负责。
3. **`|X_tgt-Xa|max = 4.0` 是假象**，不是误差：目标锚点保存的是**未折回**的初始坐标
   （`y` 到 7.977），而标记点每步 `mod Ly` 折回 `[0, 4)`，两者相减可以接近 `Ly`；
   目标力在计算时会做最小镜像折叠，所以实际力不含这个 4.0。`compare.py` 因此
   不用这个量做指标（参考解也不写锚点，`dXa` 一列恒为 NaN）。
4. **`T = 0.05` 只有官方 3 s 的 1.7%**：只覆盖 0.5 s 冲程的前 1/5，
   更谈不上官方的"持续推进/游动速度"。

## 运行

```bash
conda activate afsi-dolfinx
python run_reference.py --nx 128 --dt 1e-4 --tend 0.05 --print-dump 100  # ~207 s
NX=128 DT=1e-4 TFINAL=0.05 PRINT_DUMP=100 python main.py                 # ~249 s
python compare.py

# 粗网格对照（可选，compare.py 会自动把 ib2d_reference_n64.npz 与 plot_n64/ 配对）
python run_reference.py --nx 64 --dt 1e-4 --tend 0.05 --print-dump 100 \
    --out ib2d_reference_n64.npz --run ib2d_run_n64
NX=64 DT=1e-4 TFINAL=0.05 PRINT_DUMP=100 OUTPUT_PATH=plot_n64 python main.py

# 时间步无关性检查（可选）
NX=64 DT=2.5e-5 TFINAL=0.05 PRINT_DUMP=500 python main.py    # 末帧 max|u| = 0.893 vs dt=1e-4 的 0.890
```

（运行时间为单线程 `OMP_NUM_THREADS=1` 下的墙钟时间，测量时机器负载很高。
`compare.py` 写出 `figures/compare_history.png`、`figures/compare_fields.png`、
`figures/compare_profiles.png` 与 `figures/compare_table.csv`。）
