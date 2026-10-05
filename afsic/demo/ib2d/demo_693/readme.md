# demo_693 — IB2d `Examples_First_Year_Seminar/Plates`（泵动平板/心管）— **草稿，未通过验证**

IB2d 官方算例（`matIB2d/Examples/Examples_First_Year_Seminar/Plates`）。
两块竖直平板（206 点）由 103 根**横向弹簧**相连，其静止长度被 10 Hz 泵动
（`update_Springs.m`），模拟"心管"搏动把流体挤出去。

> **状态：草稿（draft）。** 上游数据与用户模块有**两处 bug**（见下），
> 修好之后 pyIB2d 能跑（但很剧烈，`max|u| ≈ 20–25`），AFSI 在官方 `dt` 下
> 仍立即发散；把 `dt` 缩小 4 倍后 AFSI 能跑，但结构已经完全跑飞
> （标记点偏差 ~7.3 h）。故本 demo **未通过验证**，仅作记录。

## 上游路径

* 数据：`third_party/ib2d/matIB2d/Examples/Examples_First_Year_Seminar/Plates/`
  （`plates.vertex`（206 点 + 表头）、`plates.spring`、`input2d`；
  `update_Springs.m` → 已逐行移植为同名 `.py`；
  `main2d.m`、`Make_Your_Plates_and_Input_Files.m`（几何生成器）、
  `Make_Your_Plates_and_Input_Files.m~`（编辑器备份）未随 demo 提供）。
* 算法源码：`third_party/ib2d/pyIB2d/IBM_Blackbox/`。

## 算例内容

* `1 x 1` 周期盒、`mu = 0.01`、`rho = 1`、官方 `256 x 256`、`dt = 1e-4`、
  `supp = 4`；官方 `Tfinal = 1.5`（15000 步）、`print_dump = 200`。
  本 demo 用 `128 x 128`（两个方向同时减半）、`dt = 2.5e-5`（官方 `1e-4` 的 1/4）、
  `T = 0.025`（1000 步）、`print_dump = 100`；对齐输出帧覆盖 `t = 0 .. 0.0225`
  （两侧各 10 帧）。
* **结构**：206 点（每块平板 103 点，竖直排在 `x = 0.4 / 0.6`，间距 `ds = dx/2`）；
  `plates.spring` 共 307 行有效弹簧 = 204 根链式弹簧（`k = 2.5e4`，
  `L0 = 1.95e-3`）+ 103 根横向弹簧（`k = 2.5e4`，`L0 = 0.2`）。
* **驱动**：`update_Springs.py` 把**最后 103 行**（= 横向弹簧）的静止长度设为
  `RL = d − |0.9 d sin(2π·10 t)|`，即 10 Hz 的"泵"，其余弹簧不变。

## 移植要点（含两处上游 bug）

1. **上游数据 bug（`plates.spring`）**：文件里有一行弹簧是 `206 → 207`，
   而 `plates.vertex` 只有 206 个点（上游生成器 `Make_Your_Plates_and_Input_Files.m`
   的链式弹簧循环写成 `for s = 1:N` 且条件 `s > N/2`，于是最后多写出一根
   `(N, N+1)`），同时文件声明的弹簧数（307）比实际行数（308）少 1。
   后果：**pyIB2d 自带的读取器（忽略表头计数）直接崩**
   （`IndexError: index 206 is out of bounds for axis 0 with size 206`）。
   处理（两侧一致，不改上游文件）：按**表头声明的条数**读，并丢掉引用不存在点的行
   （`ib2d_io.load_example` 里加了这段过滤；`run_reference.py` 写运行目录时同样过滤）。
2. **上游参数 bug（`update_Springs.m`）**：模块硬编码 `d = 1.0`（注释"heart tube
   diameter"），但本算例生成的几何两块平板只相距 **0.2**，于是横向弹簧的静止长度被
   泵到最大 1.0（= 实际间距的 5 倍）⇒ 弹簧力 `2.5e4 × 0.8 = 2e4`，
   **两个代码都在几步内 NaN**（实测）。本 demo 把 `d` 换成实际间距 **0.2**，
   并在模块 docstring 里写明这是有意的修补（否则无法运行）。
3. **仍需缩小 `dt`**：`d = 0.2` 之后，官方 `dt = 1e-4` 下 AFSI 仍 NaN
   （参考解有限但 `max|u| ≈ 27`，即"泵"非常剧烈）；`dt = 2.5e-5` 时两者都能跑。
4. **网格减半**：`256 x 256` → `128 x 128`；`T` 取 0.025（覆盖 1/4 个泵周期，
   泵周期 `1/f = 0.1 s`）。

## 结果（实测，未通过）

| 指标 | 结果 |
|---|---|
| 上游原始参数（`d = 1.0`）| **两个代码都 NaN**（几帧内发散） |
| `d = 0.2`、官方 `dt = 1e-4` | 参考解有限（`max\|u\| ≈ 27`），**AFSI NaN** |
| `d = 0.2`、`dt = 2.5e-5`、`T = 0.025`（本 demo 默认）| 参考 `max\|u\| = 24.53`、AFSI `20.17`（幅值差 **17.8%**、`relL2 = 4.16e-1`），但**标记点偏差 7.29 h**（结构跑飞）；弹簧平均长度 AFSI 0.018226 / IB2d 0.018760（2.85e-2 相对）；`max\|df\| = 3.14e4` |
| 运行时间 | 参考 **7.4 s**、AFSI **44.9 s**（各 1000 步） |

## 不能验证 / 需要排查

1. **结构不可比**（标记点偏差 7.3 h）：泵动幅度（0.1→0.2 的静止长度摆动）
   配上 `rho = 1` 的流体，使平板的响应极其剧烈（`max|u| ~ 20`，
   CFL 近似 `u·dt/dx ≈ 20 × 2.5e-5 / 7.8e-3 ≈ 0.06`，即标记点每步移动
   ~5e-4 m ≈ 0.06 `dx`，弹簧每步被拉伸/压缩很多）⇒ 两套显式耦合的轨迹完全分道扬镳。
2. **官方 `dt` 下 AFSI 发散**（上游 "d = 1.0" 更是两个代码都发散）：
   本算例的泵动参数与官方离散化不兼容，属于上游算例本身的问题。
3. 未做进一步"修补"（例如把泵动幅度改小、或把 `update_Springs` 的 `d`
   再调整），以避免脱离上游；如需把这个算例做成"已验证"的 demo，
   建议先与上游作者确认 `update_Springs.m` 里 `d` 的取值与所配几何。
4. 本机无 MATLAB/Octave，无法跑上游 matIB2d 交叉验证。

## 运行

```bash
conda activate afsi-dolfinx
python run_reference.py --tend 0.025 --dt 2.5e-5 --nx 128 --print-dump 100   # ~8 s
NX=128 DT=2.5e-5 TFINAL=0.025 PRINT_DUMP=100 python main.py                  # ~45 s
python compare.py
```
