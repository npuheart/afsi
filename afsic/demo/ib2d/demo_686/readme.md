# demo_686 — IB2d `Example_Meshmerize_Me/Starfish`（海星网格 + 流槽振荡射流）

IB2d 官方算例（`matIB2d/Examples/Example_Meshmerize_Me/Starfish`）。
Meshmerize 家族第 2 个：同一个"流槽 + 小窗口罚力"框架（**共享移植要点见
demo_685 的 readme**），但结构换成**海星轮廓网格**（3010 标记点，
**同时**用目标点 `k = 1e5` 和弹簧 `k = 1e6` 约束），驱动换成**振荡的 x 方向射流**。

## 上游路径

* 数据：`third_party/ib2d/matIB2d/Examples/Example_Meshmerize_Me/Starfish/`
  （`starfish_flowtank.vertex/spring/target`、`input2d`；
  `please_Compute_External_Forcing.m` → 已逐行移植为同名 `.py`；
  `main2d.m`、`modify_read_Vertex_Points_and_Plot_Them_and_Print_Them.m`
  （SVG→网格生成脚本）、`starfish_*.svg`、`starfish_initial.pkl`（0 字节）未随 demo 提供）。
* 算法源码：`third_party/ib2d/pyIB2d/IBM_Blackbox/`。

## 算例内容

* `1 x 0.25` 周期盒、`mu = 0.1`、`rho = 1000`（`nu = 1e-4`）、官方 `1024 x 256`
  （方格 `h = 9.77e-4`）、`dt = 5e-5`、`supp = 4`；官方 `Tfinal = 3.0`（60000 步）、
  `print_dump = 100`。
* **结构**：`starfish_flowtank.vertex` 3010 点；`starfish_flowtank.spring`
  （`k = 1e6`）**和** `starfish_flowtank.target`（`k = 1e5`，`update_target = 0`）
  同时作用 → 海星轮廓既被弹簧维持、又被钉在初始位置。
* **驱动**：`x ∈ [0.015, 0.065]`、`y ∈ [0.011, 0.239]`（占满左侧）窗口内
  ```
  uX_tar = -uMax sin(4πt) ((Ly/2+w/2)-y)((Ly/2-w/2)-y),   uY_tar = 0
  ```
  `uMax = 500`、`w = 0.23`、`freq = 2`（即 `sin(2·2πt)` → 周期 0.5 s）。
  **注意**：该模块没有 `scale` 归一化（与 685/690/688/689 不同），
  型线幅值因此随 `Ly` 变化（上游原样保留）。

## 移植要点

* 家族共享的移植要点（外力模块的 3 个函数、半分辨率规则、`T` 缩减、
  `print_dump` 调整）见 **demo_685 的 readme**；本例特有的是：
  * `springs = 1` **且** `target_pts = 1`：模板同时施加弹簧力与目标点力
    （与 pyIB2d 的求和次序一致），弹簧平均长度被单独对比；
  * 外力模块**无 `scale`**、时间因子为 `sin(4πt)`、型线是 y 的抛物线（中心 `Ly/2`）；
  * 窗口覆盖整个左边界（26 × 117 个节点 @512×128），所以射流比 685/690 宽得多，
    速度场对比也好得多（见结果）。
* 官方网格 `1024 x 256` → 本 demo **两个方向同时减半** = `512 x 128`
  （正方形格子，窗口仍非空）；`T = 0.02`（400 步，官方 60000 步的 1/150，
  覆盖 `sin(4πt)` 的 1/6 个周期——驱动刚启动的阶段）。

## 结果

参考运行 ~25 s、AFSI 运行 ~75 s（各 400 步、41 帧，`t = 0 .. 0.01955`）。

| 指标（末帧 `t = 0.01955`） | 结果 |
|---|---|
| 标记点偏差（3010 点，弹簧 + 目标点） | 5.64e-6 m = **2.9e-3 h** |
| 弹簧平均长度 AFSI / IB2d | 0.019789 / 0.019789（**2.4e-6** 相对） |
| 窗口内 `max\|u\|` AFSI / IB2d | 4.432e-2 / 4.733e-2（**6.4e-2** 相对） |
| 速度场相对 $L_2$（窄带外） | 1.39e-1 |
| 窄带外 `max\|du\|` | 1.75e-2 = 6.3e-1 `max\|u\|` |
| 外力场（窄带外） | `max\|df\| = 8.4e1`（`\|f\|max = 4.0e4`，差异来自 `u` 的不同） |

**能验证**：
* **弹簧 + 目标点同时作用的网格结构**：3010 个标记点偏差 **2.9e-3 h**、
  弹簧平均长度差 **2.4e-6**（相对）——本例的结构是家族里最"软"的
  （有弹簧），仍然逐位级吻合；
* **外力模块**（两侧共用同一份移植代码，窗口/型线一致），
  且速度场比 685/690 好得多（`relL2 = 0.14`，幅值差 6.4%），
  说明宽窗口时两套离散是一致的——反过来印证 685/690 的大偏差来自
  "单格点宽 + 临界阻尼"的极端驱动（见 demo_685 的"不能验证"）。

**不能验证 / 已知限制**：

1. **`T` 只跑 1/150**（400/60000 步）。官方 3 s 在 512×128 上需要
   AFSI ~3 小时（实测 ~0.19 s/步）。本 demo 只覆盖 `sin(4πt)` 周期的 1/6
   （驱动幅值到 `sin(0.04π) ≈ 0.13`）；`t = 0` 之后的早期响应是主要检验内容。
2. **网格减半**（1024×256 → 512×128）：格子仍为正方形，但结构解析度降低
   （3010 标记点在 512×128 上约每格 0.05 个标记——上游是每格 0.01 个，
   依旧是"远稀于网格"的分布，两侧一致）。
3. **速度场仍只做半定量比较**（窄带外 `relL2 = 0.14`，`max|du| = 0.63 max|u|`）：
   本例 `kStiff·dt = 1e4 × 5e-5 = 0.5`（未到临界），比 685/690 好，
   但两套投影/时间推进在"薄射流"上的差异依旧可见。
4. 本机无 MATLAB/Octave，无法跑上游 matIB2d 做第三方核对。
5. `starfish_initial.pkl` 是上游的 **0 字节**占位文件（可能用于重启），
   未被任何驱动的非重启路径读取，故未提供。

## 运行

```bash
conda activate afsi-dolfinx
python run_reference.py --tend 0.02 --nx 512 --print-dump 10   # pyIB2d，~25 s
NX=512 TFINAL=0.02 PRINT_DUMP=10 python main.py                # AFSI，~75 s
python compare.py                                              # figures/ + compare_table.csv
```
