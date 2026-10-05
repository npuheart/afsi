# demo_620 — IB2d `Example_Test_Restart_Protocol/Standard_Rubberband_Restart`（标准橡皮筋 + 重启协议）

上游算例：`third_party/ib2d/matIB2d/Examples/Example_Test_Restart_Protocol/Standard_Rubberband_Restart`
（`gitignored`，本 demo 的 `ib2d_input/` 即其 `input2d` / `practice.vertex` /
`practice.spring` 的逐字拷贝）。pyIB2d 侧参考用 `run_reference.py` 直接驱动
`third_party/ib2d/pyIB2d/IBM_Blackbox/IBM_Driver.py`。

## 物理设置

与 demo_444 / demo_462 完全相同的橡皮筋：64 个拉格朗日点构成中心在原点的
椭圆（$x\in[0.3,0.7]$，半轴 0.2），64 根**零静长**线性弹簧（`.spring` 只有
4 列 $\Rightarrow$ IB2d 默认 $\alpha=1$，$F=k(L-L_0)\,\mathbf d/L$，
$k=2.5\times10^4$）在 $1\times1$ 周期盒内（$\mu=0.01$，$\rho=1$，
$32\times32$，$\mathrm ds=1/(2N_x)$，$\mathrm dt=10^{-3}$，$T=1.5$
即 1500 步）。上游这个算例**唯一的区别**是 `input2d` 打开了
`Restart_Flag = 1`（`print_dump = 40`，`Tfinal = 1.5`）。

## 上游重启协议（IB2d 是怎么做的）

1. `input2d` 的 `Temporal_Information` 里加一行 `Restart_Flag = 1`。matIB2d
   把它读成 `Time_Params(3)`（`please_Initialize_Simulation.m`，缺省 0）。
2. `IBM_Driver.m` 在 `restart_Flag == 1` 时**跳过** `U=V=0` 的冷启动，
   转而调用**算例目录下用户自己写的** `help_Me_Restart(dt)`
   （本算例自带 `help_Me_Restart.m`）：里面必须**手工写死**上一帧的
   `ctsave`（dump 序号）、`print_dump` 和 `viz_IB2d/` 的**绝对路径**。
3. `pass_Back_Data_For_Restart.m` 据此重建状态：
   - `cter = ctsave*print_dump`，`current_time = cter*dt`，随后 `ctsave/cter/
     current_time` 各 +1（“假装最后一帧刚刚写完”）；
   - 从 `viz_IB2d/lagsPts.NNNN.vtk` 读**当前帧和前一帧**两组标记位置
     （`xLag, yLag, xLag_P, yLag_P`）；
   - 从 `viz_IB2d/u.NNNN.vtk` 读欧拉速度 `(U,V)`；
   - 若开浓度，从 `viz_IB2d/concentration.NNNN.vtk` 读 `C`；
   - 若开 poroelastic，从 `viz_IB2d/Data_For_Restart/poroelastic_Restart_*.txt`
     读上一帧的孔弹力记忆 `F_Poro`。
4. 弹簧/梁/靶点等**参数**不存盘，重启时重新读输入文件、逐步重算力；
   即“从最后一帧的热启动”。

**关键结论：IB2d 的重启粒度是"最后一个 print_dump 帧"**，dump 之后、崩溃之前
算出的所有步都丢失（最多损失 `print_dump*dt` 的时间）。

### 上游的坑 / bug

1. **`help_Me_Restart.m` 必须逐算例手工改，且路径是作者机器上的绝对路径**
   （`/Users/battistn/Desktop/IB2d/...`，两个算例都是）——不先把路径改成自己
   机器上的 `viz_IB2d/`，`Restart_Flag = 1` 就直接跑不起来（`cd` 失败/读不到
   .vtk）。仓库里也因此**没有**任何自动化的一致性检查。
2. **自带数据本身是真实的，可以让重启"开箱"跑起来，但只能从固定的一帧**：两个算例的
   `viz_IB2d/` 里预置了 4 帧 dump（`lagsPts.0000..0003.vtk`、`u.0000..0003.vtk`；
   poroelastic 另有 `Data_For_Restart/poroelastic_Restart_0..3.txt`），正好对应
   `help_Me_Restart.m` 里写死的 `ctsave = 3`：
   - Standard：`print_dump = 40`，`dt = 1e-3` → 从 dump #3（第 121 步，
     $t=0.12$）续跑到 $T=1.5$（1380 步）；
   - Poroelastic：`print_dump = 20`，`dt = 1e-4` → 从第 61 步
     （$t=6\times10^{-3}$，仅 3% 处）续跑到 $T=0.2$。

   两个小坑：poroelastic 的 `viz_IB2d/` **缺 `u.0000.vtk`**（不影响，因为
   `help_Me_Restart` 只读 `u.0003`）；`help_Me_Restart` 里路径写死在作者机器上
   （见下一条）。
   顺带验证了这批 dump 的真实性：本 demo 的 pyIB2d 参考（`ib2d_reference.npz`）
   第 0..3 帧首点 $x$ 为 `0.700 / 0.733188 / 0.850863 / 0.874047`，与仓库里
   MATLAB 生成的 dump `0.700 / 0.732854 / 0.850585 / 0.873697` 相差
   $\lesssim3\times10^{-4}$（matIB2d 与 pyIB2d 的常见量级差异）。
3. **pyIB2d 完全没有重启功能**：`grep -ri restart pyIB2d/IBM_Blackbox` 无任何
   命中；`IBM_Driver.main` 的 `Time_Params` 只有 `[Tfinal, dt]`，所以
   `input2d` 里的 `Restart_Flag` 对 Python 端是**静默忽略**的。
4. **质量点（mass）的速度不参与重启**：driver 里
   `mVelocity = zeros(...)` 并注明 “RESTART MASS * STILL NEEDS IMPLEMENTATION”。
   本算例不含质量点，故无影响（demo_621 亦无）。
5. **帧时间戳 off-by-one**：driver 在 0 基计数器 `cter` 为 `print_dump` 正整数倍
   时 dump，写出的是第 `cter+1` 步结束后的状态，却把时间戳标成 `cter*dt`
   （比状态晚一步）。`main.py` 按同样的步 dump，但把时间标成真实时间
   （见 `compare.py` 的说明）。
6. `IBM_Driver.m` 里 `clear restart_Flag`：重启标志用完即清，无法在同一
   session 内二次重启。
7. matIB2d 的 `please_Move_Lagrangian_Point_Positions` 除 `xLag` 外还需要
   `xLag_P`，所以重启必须存**两帧**标记位置；而 AFSI 的两阶段 Peskin 格式是
   “无记忆”的（见下），只需一帧。

## AFSI 侧的重启实现

`main.py` 用环境变量驱动，与上游 `Restart_Flag` 无关：

```bash
# 连续跑 T=1.5（默认）
python main.py
# 跑到 t1=0.721 并在第 721 步结束后 dump 完整状态
TFINAL=0.721 RESTART_AT=721 RESTART_DUMP=state.npz python main.py
# 从该状态热启动继续到 T=1.5
RESTART_FROM=state.npz python main.py
```

- `RESTART_AT` 以**已完成步数**计，dump 写在第 `RESTART_AT-1` 步之后；
- dump 存的是**积分器全部状态**：标记自由度 `X_dof`、流体速度自由度
  `u_n(=u_)`、压力自由度 `p_dof`（外加 `step/t/dt` 与便于人读的
  `X_markers`）。Peskin (2002) 两阶段格式下一步只依赖 $(X^n, u^n)$，
  所以这些就是全部内存；孔弹算例再多一项 `F_Poro_prev`（见 demo_621）；
- 与上游不同，AFSI 的重启点是**任意步**，不限于 print_dump 帧。

## 结果

### 1. 重启一致性（run 0..t1 → dump → 重启 → T，对比单次跑到 T）

`python compare.py --noise-floor`（K=721，$t_1=0.721$，$T=1.5$，21 个公共帧）：

| 指标 | 重启 vs 连续 | 噪声地板（同配置重跑） | 噪声地板（分段 vs 连续，$t\le t_1$） |
|---|---|---|---|
| $\max_k\lvert\Delta X_k\rvert$ | **8.17e-15 m**（2.6e-13 h） | 7.90e-15 m | 3.38e-15 m |
| $\max\lvert\Delta u\rvert$ | 2.32e-13 | 2.49e-13 | 3.15e-13 |
| $\max\lvert\Delta p\rvert$ | 1.07e-11 | 2.66e-11 | 2.61e-11 |

**重启与连续运行的偏差 ≈ 同配置两次独立运行的噪声地板（≈1e-14 m，
$10^{-13}h$ 量级），即重启是"无损"的**。注意不是逐位相同：线性求解
（PETSc/BLAS 归约）在不同进程间不可逐位复现，因此我们量化的是
“重启偏差 vs 进程间噪声地板”，而不是声称 bit-exact。

### 2. AFSI vs IB2d（pyIB2d 单次连续运行，38 个公共帧）

| $t$ | $\max_k\lvert X^{\rm AFSI}_k-X^{\rm IB2d}_k\rvert$ | 相对 $h$ | 面积 AFSI / IB2d |
|---|---|---|---|
| 0.0000 | 5.6e-12 | 1.8e-10 | 0.250924 / 0.250924 |
| 0.1210 | 4.2e-03 | 0.14 | 0.249456 / 0.251096 |
| 0.3610 | 2.8e-03 | 0.09 | 0.246825 / 0.248054 |
| 0.7210 | 7.4e-03 | 0.24 | 0.244555 / 0.233669 |
| 1.0810 | 3.1e-02 | 0.98 | 0.243246 / 0.212283 |
| 1.4810 | 4.8e-02 | 1.53 | 0.242301 / 0.185346 |

末期面积：AFSI 0.250924 → 0.242301（漏 3.5%），IB2d 0.250924 → 0.185346
（漏 **26.1%**）；周长 AFSI 1.9369 → 1.7457，IB2d → 1.5292。

## 什么**不可**比 / 上游的离散伪影

- **后期轨迹不可比**：IB2d（同位于格心的中心差分 + FFT）在零静长闭合环上
  **持续漏面积/漏体积**，$t=1.5$ 已丢 26%。这不是本移植的误差：demo_444
  用 **matIB2d（Octave）** 跑同一算例得到 $t=1.5$ 面积 **0.1840**（漏 26.7%），
  与本次 pyIB2d 参考的 **0.1853**（漏 26.1%）一致，说明是 IB2d 自身的离散
  误差。AFSI 侧 $\gamma=100$ 的 grad-div 稳定项把面积守恒到 3.5%，于是
  $t\gtrsim0.7$ 之后两者的标记偏差被 IB2d 的泄漏主导，长到 $1.5h$。
- 因此本表只应读作“**同一物理问题、两套离散各自的守恒性**”，不是“AFSI 复现
  IB2d 轨迹”。早期（$t\lesssim0.4$，面积差别还小）偏差在 $0.1h$ 以内。
- 上游的**重启协议本身无法与 pyIB2d 对比**（pyIB2d 无重启），所以重启一致性
  只能对 AFSI 自身的连续运行验证；这是本 demo 主要、也是最强的验证。

## 命令与实测耗时

```bash
conda activate afsi-dolfinx
python run_reference.py                      # pyIB2d 参考（单次连续，无重启）
python main.py                               # AFSI 连续运行 T=1.5
python compare.py --noise-floor              # 对比 + 重启验证（会自行再跑 3 次主程序）
```

本机（macOS arm64，10 核；实测时机器被其它算例占满，load average ≈100，故为“重负载上界”；空闲时 main.py 只需 ~31 s）实测：

| 命令 | 规模 | 墙钟 |
|---|---|---|
| `python main.py`（T=1.5，1500 步） | 32×32 | ~45 s（含 ~14 s 导入/装配） |
| `python run_reference.py` | 32×32，1500 步 | ~2 min |
| `python compare.py --noise-floor` | 1 个 721 步段 + 3 个 1500 步运行 | ~9 min |

## 文件

- `main.py`          AFSI 运行 + 重启协议（`RESTART_AT/RESTART_DUMP/RESTART_FROM`）
- `run_reference.py` pyIB2d 参考（1-based 文件重编号后运行；**无重启**）
- `ib2d_io.py`       1-based 输入读取（`index_base=1`）
- `ib2d_reference.py` VTK → npz 转换
- `compare.py`       AFSI vs IB2d 对比 + 重启一致性验证 + 图
- `ib2d_input/`      上游输入文件逐字拷贝
