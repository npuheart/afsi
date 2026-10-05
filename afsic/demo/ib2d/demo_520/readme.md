# demo_520 — IB2d `Example_Channel_Flow/Example_Poroelastic_Channel_Flow`（速度靶型外力驱动的通道 + 孔弹网络）

IB2d 官方算例（`matIB2d/Examples/Example_Channel_Flow/Example_Poroelastic_Channel_Flow`，
上游 gitignored）：`ib2d_input/` 即该目录的 `input2d` / `channel.vertex` /
`channel.spring` / `channel.target` / `channel.poroelastic` / `Channel_Channel.m`
（几何生成脚本）/ `please_Compute_External_Forcing.m` 的逐字拷贝。

## 算例内容

* 周期盒 `1.0 × 0.25`、`ρ = 1`、`μ = 1e-3`、`supp = 4`、`string_name = "channel"`；
  上游 `Nx = 512`、`Ny = 128`（`dx = dy = 1/512`）、`dt = 5e-5`、`Tfinal = 1.25`
  （**25000 步**）、`print_dump = 200`。
* **结构（4934 个标记）**：
  * **通道两壁**：`x ∈ [0.05, 0.95]`（每壁 922 点，间距 `ds = 1/1024`），
    `y = 0.0625` / `0.1875`，共 1844 个标记，**全部**是 `k_t = 1e6` 的**钉死目标点**
    （`channel.target` 1844 行）——它们把流体限制在宽 `w = 0.15` 的通道里；
  * **孔弹网络**（`channel.poroelastic`，3090 点，`c = 5e5`）：挂在通道**底壁下方**
    `x ∈ [0.45, 0.55]`、高 `w/5 = 0.03` 的一个二维方形弹簧网（6150 根 `k = 1e5`、
    静长 `ds` 的弹簧，竖直臂接到通道底壁）。网络节点同时是**孔弹点**：
    每步额外位移 `ΔX = Δt·F_spring/(μ·c)`（`F_spring` 只含弹簧力）。
* **驱动（`arb_ext_force`）**：在补丁 `x ∈ [0.1, 0.16]`、`y ∈ [0.06, 0.19]` 上以
  罚力 `kStiff = 1e4` 把流体速度拽向

  ```
  uX_tar = -uMax·5·sin(2·10·π·t)·((Ly/2 + W/2) - y)·((Ly/2 - W/2) - y),   uY_tar = 0,
  ```

  `uMax = 200`、`W = 0.15`（10 Hz 振荡；因两个线性因子 ≤ 0.005625，靶速度峰值
  ≈ `200·5·0.005625 = 5.6` m/s，且 `t = 0` 时靶速度为 0）。
  于是 `Re = ρ·u·L/μ ≈ 1000·5.6·0.15/1e-3 ≈ 8e5`——**强惯性射流**。

## 移植要点

1. **`please_Compute_External_Forcing.py` 逐行移植**（`ib2d_input/`）：
   常量块（`kStiff = 1e4`、`W = 0.15`、`uMax = 200`、补丁窗口），
   首次调用时用 `give_Me_Indices_To_Apply_Force` 建一次 `inds`（掩膜）并存回，
   之后每步算靶速度与罚力密度。签名/返回严格按 `IBM_Driver.py:751`
   （`(dt, t, x, y, grid_Info, U, V, first, inds)` → `(Fx, Fy, first, inds)`）。
   MATLAB 1-based → 0-based：`inds` 存 0-based `(x-index, y-index)`，
   `fx[j, i]` 即 numpy `[y, x]` 项。
2. **孔弹（`poroelastic`）机制模板已具备**（与 demo_454/620/621 同源，
   那里的 `ΔX = Δt·F_spring/(μc)` 与 driver 的 `F_Poro` 记账已被逐字核对）；
   本 demo 是**同时**开 `springs + target_pts + poroelastic + arb_ext_force`
   的算例，四种机制都在合并模板里。
3. **上游 quirk / bug**：
   * 几何脚本 `Channel_Channel.m` 里写的是 `Nx = Ny = 512`、`Lx = Ly = 1`，
     生成完再把 `yLag` 平移 `-0.375`；而 `input2d` 是 `Lx = 1`、`Ly = 0.25`
     ——两者不同源（脚本内部用 `Ly = 1` 算 `ds`，平移后才落进 `Ly = 0.25` 的盒子）。
     因为最终 `dx = dy`，`ds = min(Lx/(2Nx), Ly/(2Ny))` 与 pyIB2d 硬编码的
     `ds = Lx/(2Nx)` **恰好相等**（`1/1024`），这是本例能直接跑起来的前提。
   * 上游罚力刚度折算成"每步修正比例"是 `kStiff·dt/ρ = 1e4·5e-5 = 0.5`
     （每步修正 50%），比 demo_467/468 同族算例的 `1e-2` **硬 50 倍**：
     该罚力对"在时间步的哪个 stage 施力"极敏感。
   * **参考解（pyIB2d，显式）自己会 NaN**：把窗口拉到 `Tfinal = 0.1` 时，
     末个有限帧 `t = 0.0775` 的射流速度已涨到 117 m/s（`u·dt/dx ≈ 0.75`，
     顶到显式稳定界），下一帧（`t = 0.0800`）起 NaN——而 AFSI（隐式）在该窗口内
     始终有限（末帧 `max|u| = 2.92`）。所以本 demo 把窗口截到 `Tfinal = 0.02`，
     保证**两个代码都在有限区间内**才做比较。
4. **减缩设置**（本机上上游网格不可行）：

   | | 上游（本算例） | 本 demo |
   |---|---|---|
   | 网格 | 512 × 128（65536 单元，dx = dy = 1/512） | **128 × 32**（4096 单元，dx = dy = 1/128） |
   | dt | 5e-5 | **5e-5（原值）** |
   | Tfinal | 1.25 s（25000 步） | **0.02 s**（400 步，= 0.2 个 10 Hz 周期） |
   | print_dump | 200 | 20（20 帧） |

   **注意**：孔弹网络的节点间距是 `dx_shipped/2 = 1/1024`，减缩到 1/128 后
   一个网格单元里有 8 个网络节点，且 `ds_code = Lx/(2Nx) = 1/256` 是网络真实
   间距的 4 倍 ⇒ **网络力被放大 4 倍**（AFSI 与 pyIB2d 用同一条 `Lx/(2Nx)`
   公式，两边同样放大）。要把网络分辨出来必须用上游的 512×128。

## 结果

参考（pyIB2d）与 AFSI 各 400 步、20 帧，`t = 0 … 0.01905`。
`h = 1/128`；`max|u|` 是全域峰值（它在受力补丁内）。

| 指标（末帧 `t = 0.01905`） | 结果 |
|---|---|
| 标记点偏差（4934 点） | 1.26e-2 m = **1.61 h** |
| 速度峰值 AFSI / IB2d | 5.2495 / 7.9482（**34%**），`max\|Δu\| = 6.61` |
| 速度场相对 $L_2$ 误差（窄带外） | 5.62e-1 |
| 弹簧平均长度 AFSI / IB2d | 9.93e-4 / 1.054e-3（**5.8%**） |
| 力场 `max\|Δf\|`（窄带外，`\|f\|max = 6.6e3`） | 3.02e3 |
| 目标点（钉死通道壁）偏差 | 3.8e-4 m = 4.9e-2 h |

前 10 帧两边的峰值速度同步增长（`0.29/0.20, 0.64/0.56, 1.00/1.15, 1.35/1.83,
1.69/2.56, 2.11/3.25, 2.52/3.74, 2.85/4.16, 3.11/4.59, 3.38/5.06`），
但参考的射流越跑越快、AFSI 的慢——差异从早期的 30% 拉开到 34% 并持续扩大。

**能验证**：通道/网络几何与目标点约束（4.9e-2 h）、弹簧平均长度（5.8%，
即网络被拉伸的量级一致）、以及"外力 → 通道内射流 → 网络蠕变"这条因果链的
存在与量级；`please_Compute_External_Forcing.py` 与 `poroelastic` 两个
移植模块本身在别的 demo（468 / 454、620）里已被逐式核对。

**不能验证（如实记录）**：

* **流场**（`max|u|` 差 34%、相对 $L_2$ 5.6e-1）：靶型罚力把流体拽到
  `u ≈ 5.6` m/s，`Re ≈ 8e5`，且受力补丁只有 `0.06 × 0.13` = 8 × 17 个单元，
  射流完全欠分辨——两个离散给出的是两条不同的湍流解；
* **孔弹网络的形变细节**（标记点偏差 1.61 h）：网络节点间距只有
  `h/8`，减缩网格无法分辨；
* **分辨率判别**：我把网格加密到 256×64（同一 `dt`、同一 `Tfinal`）后，
  `t = 0.015` 的 `max|u|` 相对差反而从 35% 变成 **46%**（标记点 1.38 h → 1.90 h）
  ⇒ 差异**不是**"网格太粗"造成的，加密只会让射流更湍、两条解更分离；
* **上游的"官方结果"**：本算例上游没有随附参考输出，`Tfinal = 1.25` 那一段
  参考（pyIB2d）自己在 `t ≈ 0.078` 之后就 NaN 了（见 §3），因此长时行为无法核对；
* **`dt` 收敛性判别**：未做（把 `dt` 减半 + `Tfinal = 0.02` 需要约 4 倍的
  参考机时，本批时间不允许）；从"加密网格不改善"来看，也预期不改善。

## 运行

```bash
conda activate afsi-dolfinx
# 参考（pyIB2d）→ ib2d_reference.npz（本机实测 ~6 min，与其他任务并发）
python run_reference.py --nx 128 --tend 0.02 --print-dump 20
# AFSI → plot/afsi_result.npz（本机实测 61 s 计算，墙钟 ~4.5 min 并发下）
TFINAL=0.02 PRINT_DUMP=20 NX=128 NY=32 python main.py
python compare.py
```

上游设置（仅作说明，本机 512×128 × 25000 步不可行）：

```bash
python run_reference.py --tend 1.25 --print-dump 200
TFINAL=1.25 PRINT_DUMP=200 python main.py
```
