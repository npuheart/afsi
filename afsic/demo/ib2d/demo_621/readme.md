# demo_621 — IB2d `Example_Test_Restart_Protocol/Poroelastic_Rubberband_Restart`（孔弹橡皮筋 + 重启协议）

上游算例：`third_party/ib2d/matIB2d/Examples/Example_Test_Restart_Protocol/Poroelastic_Rubberband_Restart`
（`gitignored`；本 demo 的 `ib2d_input/` 即其 `input2d`、`rubberband.vertex`、
`rubberband.spring`、`rubberband.poroelastic` 的逐字拷贝）。pyIB2d 侧参考用
`run_reference.py` 直接驱动 `third_party/ib2d/pyIB2d/IBM_Blackbox/IBM_Driver.py`。

## 物理设置

与 demo_453/demo_454 同族的橡皮筋：64 个标记构成中心在原点的椭圆（半轴
0.2），64 根**零静长**线性弹簧（`.spring` 有 5 列，$\alpha=1$，
$k=10^6$，$L_0=0$），**每个标记都开启孔弹**（`.poroelastic`，
$c=2.5\times10^5$）：

$$\Delta \mathbf X_{\rm pe} = \Delta t\,\frac{\mathbf F_{\rm spring}}{\mu\,c}$$

其中 $\mathbf F_{\rm spring}$ **只含弹簧力**、在 $\mathbf X_h$ 处求值；
半步步移用上一帧的 $F_{\rm Poro}$（driver 的记账方式），全步用本步值。
与 demo_454 不同，本算例 `porous_media = 0`（**没有 `.porous` 文件**）：
标记随局部弹性载荷在流体中"蠕变"，但不沿曲线法向滑移。几何/流体：
$1\times1$ 周期盒，$\mu=0.01$、$\rho=1$、$32\times32$、$\mathrm dt=10^{-4}$、
$T=0.2$（2000 步）、`print_dump = 20`。孔弹项使环在 0.2 s 内从面积
0.2509 塌缩到 0.02–0.04 量级。

上游这个算例唯一的区别是 `input2d` 打开了 `Restart_Flag = 1`。

## 上游重启协议（IB2d 是怎么做的）

与 demo_620 完全相同的机制（详见 demo_620 readme）：`Restart_Flag = 1` →
`IBM_Driver.m` 调用算例自带的、**必须手工写死** `ctsave / print_dump / 绝对路径`
的 `help_Me_Restart.m` → `pass_Back_Data_For_Restart.m` 从
`viz_IB2d/lagsPts.NNNN.vtk`（当前 + 前一帧位置）与 `u.NNNN.vtk`（欧拉速度）
重建状态，时间计数取 `current_time = ctsave*print_dump*dt` 再 +1 步。

本算例的额外一点（孔弹特有）：**上一帧的孔弹力记忆 $F_{\rm Poro}$ 从
`viz_IB2d/Data_For_Restart/poroelastic_Restart_<ctsave-1>.txt` 读回**
（`print_vtk_files` 每帧用 `print_Matrix_to_TXT_for_Restart` 写出，仅当
`poroelastic_Yes == 1`）。浓度、质量点等同 demo_620（本算例都不涉及）。

自带数据：`viz_IB2d/` 预置 `lagsPts.0000..0003.vtk`、`u.0001..0003.vtk`
（**缺 `u.0000.vtk`**，不影响重启，因为只读 `u.0003`）与
`Data_For_Restart/poroelastic_Restart_0..3.txt`，对应 `help_Me_Restart.m`
写死的 `ctsave = 3`：即从第 61 步（$t=6\times10^{-3}$，整段的 3%）续跑。

### 上游的坑 / bug（详见 demo_620，同源）

1. `help_Me_Restart.m` 逐算例手工写死，且路径是作者机器的绝对路径
   （`/Users/battistn/Desktop/IB2d/...`）——不改就跑不起来；
2. 重启粒度 = 最后一个 dump 帧，最多丢 `print_dump*dt` 的进度；
3. **pyIB2d 没有任何重启功能**（`grep -ri restart pyIB2d/IBM_Blackbox` 零命中），
   `Restart_Flag` 对 Python 端静默忽略；
4. **质量点速度不参与重启**（`mVelocity = zeros(...)`，
   “RESTART MASS * STILL NEEDS IMPLEMENTATION”）——本算例无质量点；
5. 帧时间戳 off-by-one（写的是第 `cter+1` 步的状态，标 `cter*dt`）；
6. `clear restart_Flag` 使同一 session 无法二次重启；
7. matIB2d 的标记移动需要 `xLag` 与 `xLag_P` 两帧，故重启必须存两帧位置；
   AFSI 的两阶段格式无此历史依赖。

## AFSI 侧的重启实现

同 demo_620，但**多存一项孔弹力记忆**：

```bash
python main.py
TFINAL=0.1001 RESTART_AT=1001 RESTART_DUMP=state.npz python main.py
RESTART_FROM=state.npz python main.py
```

dump 内容：`X_dof`（标记自由度）、`u_dof`（$u_n=u_$）、`p_dof`、**`F_Poro_prev`**
（上一帧孔弹力，半步步移要用），外加 `step/t/dt/X_markers`。
`RESTART_AT` 以已完成步数计，重启点可为任意步（不限于 print_dump 帧）。

## 结果

### 1. 重启一致性（run 0..t1 → dump → 重启 → T，对比单次跑到 T）

`python compare.py --noise-floor`（$K=1001$，$t_1=0.1001$，$T=0.2$，
51 个公共帧）：

| 指标 | 重启 vs 连续 | 噪声地板（同配置重跑，0..T） | 噪声地板（分段 vs 连续，$t\le t_1$） |
|---|---|---|---|
| $\max_k\lvert\Delta X_k\rvert$ | **5.00e-15 m**（1.6e-13 $h$） | 1.60e-15 m | 1.47e-15 m |
| $\max\lvert\Delta u\rvert$ | 1.03e-12 | 1.16e-12 | 1.10e-12 |
| $\max\lvert\Delta p\rvert$ | 5.07e-11 | 5.39e-11 | 4.05e-11 |

**重启偏差在 1e-15 m 量级（$10^{-13}h$，$h=1/32$），与"同配置两次独立运行"
的噪声地板同量级；速度/压力偏差已低于噪声地板。** 标记位置约比噪声地板大
3 倍，是因为 $t_1$ 之后孔弹环进入快速塌缩（强不稳定）阶段，round-off 被
动力学放大——即便如此仍比 AFSI-vs-IB2d 的差异（0.3–1.1 $h$）小 12 个数量级。
（不是逐位相同：线性求解的 BLAS 归约在不同进程间不可逐位复现。）

### 2. AFSI vs IB2d（pyIB2d 单次连续运行，101 个公共帧）

| $t$ | $\max_k\lvert X^{\rm AFSI}_k-X^{\rm IB2d}_k\rvert$ | 相对 $h$ | 面积 AFSI / IB2d |
|---|---|---|---|
| 0.0000 | 5.6e-12 | 1.8e-10 | 0.250924 / 0.250924 |
| 0.0161 | 1.08e-02 | 0.34 | 0.216120 / 0.220622 |
| 0.0481 | 1.17e-02 | 0.37 | 0.161496 / 0.170932 |
| 0.0801 | 2.05e-02 | 0.66 | 0.121071 / 0.130535 |
| 0.1121 | 8.96e-03 | 0.29 | 0.090197 / 0.092345 |
| 0.1441 | 1.65e-02 | 0.53 | 0.066886 / 0.059740 |
| 0.1761 | 3.00e-02 | 0.96 | 0.048904 / 0.036738 |
| 0.2000 | 3.54e-02 | 1.13 | 0.038460 / 0.024851 |

标记偏差中位数 1.55e-2 m（0.5 $h$），全程 $\le1.27h$；面积在
$t\lesssim0.13$ 一致到 ~5%，之后 AFSI 塌缩偏慢（末期 0.0385 vs 0.0249）。
与 demo_454 的孔弹算例精度同一档次（demo_454：dXmax 0.17–0.76 $h$）。

## 什么**不可**比

- **末期塌缩不可比**：$t\gtrsim0.13$ 后环进入自由落体式塌缩，周长→0，
  AFSI（$\gamma=2.5$ 的 grad-div 稳定 + Q2/Q1 无法表示跨膜压差）与
  pyIB2d（同位中心差分 + FFT）对同一强不稳定阶段的相位差被指数放大；
  末期面积相差 ~55%。
- **参考侧没有重启**：pyIB2d 无重启实现，故"重启结果"无法与 IB2d 的正统
  重启对比；本 demo 的重启一致性验证只针对 **AFSI 自身的连续运行**
  （这也是最强的一种验证，且证明了 AFSI 重启实现自洽无损）。
- 上游自带 dump 可以让 MATLAB 端"重启"，但仓库里**没有** MATLAB 重启后的轨迹，
  因此无法在 AFSI 侧复现"IB2d 重启 vs IB2d 连续"这条基线。

## 命令与实测耗时

```bash
conda activate afsi-dolfinx
python run_reference.py                      # pyIB2d 参考（单次连续，无重启）
python main.py                               # AFSI 连续运行 T=0.2（2000 步）
python compare.py --noise-floor              # 对比 + 重启验证
```

本机（macOS arm64，10 核；实测时机器被其它算例占满，load average ≈100，故为“重负载上界”；空闲时 main.py 只需 ~31 s）实测：

| 命令 | 规模 | 墙钟 |
|---|---|---|
| `python main.py`（T=0.2，2000 步） | 32×32 | ~37 s（含 ~5 s 导入/装配） |
| `python run_reference.py` | 32×32，2000 步，101 帧 VTK | ~2.5 min |
| `python compare.py --noise-floor` | 1 个 1001 步段 + 3 个 2000 步运行 | ~3.5 min |

## 文件

- `main.py`          AFSI 运行 + 重启协议（含 `F_Poro_prev`）
- `run_reference.py` pyIB2d 参考（`.spring`/`.poroelastic` 重编号后运行；**无重启**）
- `ib2d_io.py`       1-based 输入读取（`index_base=1`，含 `porous`/`poroelastic`）
- `ib2d_reference.py` VTK → npz 转换
- `compare.py`       AFSI vs IB2d 对比 + 重启一致性验证 + 图
- `ib2d_input/`      上游输入文件逐字拷贝
