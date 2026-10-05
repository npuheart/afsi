# demo_613 — IB2d `Example_HeartTube/3_Element_Muscle_Model`（三元件 Hill 肌肉心管）

IB2d 官方算例（`matIB2d/Examples/Example_HeartTube/3_Element_Muscle_Model`）在 AFSI 中的复现。
与 demo_612（`Muscle_Model`）**共用同一份几何与肌肉数据、同一套力律**，
区别只在 `input2d` 的开关（`3_element_muscle = 1`）与流体黏度（`mu = 1` 而非 0.1）。

## 1. 算例内容

| 项目 | 值（官方 `input2d`） |
|---|---|
| 周期盒 / 网格 | `5 x 5`，`128 x 128`，`mu = 1`，`rho = 1` |
| 时间 | `dt = 5e-4`，`T = 0.25`（500 步） |
| 结构 | 310 点（上下各 155）；308 根弹簧（`k = 1e7`）、306 根不变梁、4 个目标点、**153 条肌肉带**（`i -> i+155`，`Fmax = 1e5`、`LFO = 1`、`SK = 0.3`、`a = 0.25`、`b = 4`） |
| 开关 | `3_element_muscle = 1`，`FV_LT_muscle = 0` |

`HeartTube.muscle`、`HeartTube.spring`、`HeartTube.beam`、`HeartTube.target`、`HeartTube.vertex`
与本目录名下的 `Muscle_Model` 版本**逐个数值相同**（只有 1-based 编号），已核对。

## 2. 上游不一致（本算例最重要的发现，已考证）

* 示例自带的 `give_3_Element_Muscle_Activation.m` 是**单输出**函数，签名
  `(v, LF, LFO, SK, a, b, Fmax, current_time, xPt, xLag)`（9 个参数），与其
  `give_Muscle_Activation.m` **逐字节相同**（只改了函数名）；它算的就是 FV_LT 力律
  `Fm = a_f Fmax exp(-((Q-1)/SK)^2) (1/P0)(b P0 - a vc)/(vc+b)`。
* 随附的 `matIB2d` blackbox（`please_Find_Lagrangian_Forces_On_Eulerian_grid.m:701-780`）
  却按**另一代**接口调用它：

  ```matlab
  [Fm, on_Parallel, PE_Coeff, af_Val] = give_3_Element_Muscle_Activation(v,LF,LFO,sk,a,b,Fmax,current_time,xPt,xLag,i);
  ```

  即 **10 个参数、4 个输出**，并额外读取肌肉文件的**第 8、9 列**（`kSpr`、`alpha`，用来构造
  串联阻尼 `bDamp` 与并联非线性弹性 `PE`）。而本示例的 `HeartTube.muscle` **只有 7 列**。
  ⇒ **该示例按原样无法在随附 MATLAB blackbox 下运行**（输入/输出个数不匹配 + 列越界；
  即便 MATLAB 自动扩列，新增列也全是 0）。
* `pyIB2d` 实现的是**与示例数据自洽的旧版**：`read_Hill_3Muscle_Points` 只读 7 列，
  `give_3_Element_Muscle_Force_Densities` 与 FV_LT 版逐行相同，只是调用
  `give_3_Element_Muscle_Activation`。这正是示例的 `.m` 与数据所对应的版本，
  所以本移植以 **pyIB2d 为参考**，力律 = FV_LT（三元件模型里的收缩元 CE 部分；
  串联/并联元在该版本里不存在）。
* **pyIB2d 自身的 bug（本 demo 必须绕开）**：`please_Find_Lagrangian_Forces_On_Eulerian_grid.py`
  的 FV_LT 分支在函数内部 `from give_Muscle_Activation import ...`，而 3 元件分支
  **直接调用 `give_3_Element_Muscle_Activation` 却从未 import** → 参考运行必然抛
  `NameError: name 'give_3_Element_Muscle_Activation' is not defined`。
  `run_reference.py` 里把示例移植后的模块注入到该模块的命名空间即可修复（代码里有注释）。

## 3. 结果（`compare.py`）

参考 12 帧、AFSI 13 帧，对齐 12 帧。AFSI 运行参数：`NX=64`（`h = 0.0781`）、
官方 `dt = 5e-4`（本例 `mu = 1` 稳定，无需像 demo_612 那样降 `dt`）、
`TFINAL = 0.15 s`（300 步，实测 **12.9 s**）；
参考 `--tend 0.15 --nx 64 --print-dump 25`。

| 指标（末帧 `t = 0.1380 s`） | 结果 |
|---|---|
| 管径（153 条肌肉带平均长度，静息 1.0） | AFSI **0.5652** / IB2d **0.5581**（行波把管子收缩约 44%）；逐帧 `max\|Δw\| = 9.5e-3` |
| 收缩（激活）相位 | 两侧管径极小值同在 `t = 0.1380`；管径历史逐帧一致（`t = 0.05` 时 0.880/0.883，`t = 0.10` 时 0.661/0.652） |
| 长度-张力因子 `F1` 均值 | AFSI 0.4542 / IB2d 0.4446；`max\|ΔF1\| = 9.3e-2` |
| 标记点偏差（310 点） | 中位 1.7e-2 m = **0.22 h**，最大 2.3e-2 m = 0.29 h（`h = 0.0781`） |
| 末帧速度场 | `max\|u\|` AFSI 13.4 / IB2d 14.1，`max\|Δu\| = 2.40` |

**能验证**：肌肉驱动本身——管径收缩幅度（0.565/0.558）、收缩相位（同一帧极小）、
管径历史逐帧一致（`max|Δw| = 9.5e-3`，约为收缩行程的 2%）、标记点中位 0.22 h/最大 0.29 h，
以及末帧速度场 `max|Δu| = 2.4`（约为峰值的 17%，是本批 6 个 demo 里最好的一例，
因为本例 `mu = 1` 较大、流速较低）。

**不能验证（如实记录）**：
* 官方 **`128×128`** 网格与 **`T = 0.25 s`**：本 demo 用 `NX = 64`、`T = 0.15 s`
  （`h` 与 IB2d 的 `ds` 随之改变），因此不是官方设置的全参数复现。
* 三元件模型里**串联元（含 `bDamp` 阻尼）与并联非线性元**在本示例数据里**不存在**
  （见 §2），所以本 demo **没有**实现它们——这也是"上游示例与 blackbox 版本不匹配"
  的直接后果，而非移植缺漏。

## 4. 运行

```bash
conda activate afsi-dolfinx
python run_reference.py --tend 0.15 --nx 64 --print-dump 25   # pyIB2d 参考
TFINAL=0.15 NX=64 PRINT_DUMP=25 python main.py                # AFSI（~13 s）
python compare.py
```

环境变量：`TFINAL`、`DT`、`NX`/`NY`、`PRINT_DUMP`、`OUTPUT_PATH`、`GRAD_DIV`（默认 0）。
