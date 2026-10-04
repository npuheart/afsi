# demo_469 — IB2d `Example_Middle_Channel_Flow`（槽道中部射流）

IB2d 官方算例（`matIB2d/Examples/Example_Channel_Flow/Example_Middle_Channel_Flow`）
的复现，沿用 demo_467/468 的 `arb_ext_force` 模板。

## 算例内容

* `1 x 1` 周期盒、`mu = 10`、`rho = 1`、`32 x 32`、`dt = 1e-4`、`T = 0.05`（500 步）。
* **结构**：两条各 58 点的完整槽道壁（`y = 0.4 / 0.6`，`x in [0.05, 0.94]`），
  116 个标记点全部为 `k = 1e7` 的钉死目标点。
* **外力**：窗口在**槽道正中** `x in [0.47, 0.53]`、`y in [0.4025, 0.5975]`
  （32² 网格上只有 1 列 × 7 行 = 7 个节点），

  $$F_x = -k(u_x - u_x^{\rm tar}),\qquad
  u_x^{\rm tar} = -u_{\max}\,5|\sin(10\pi t)|\,(\tfrac{L_x}{2}+\tfrac{W}{2}-y)(\tfrac{L_x}{2}-\tfrac{W}{2}-y),$$

  `k = 1e4`、`u_max = 200`；`F_y = 0`。射流朝两端开口排出。

## 结果

参考运行 ~5 s、AFSI 运行 ~4 s。原样网格与 `64 x 64`/`dt = 5e-5` 交叉核对：

| 指标（末帧） | 原样 `32 x 32`, `dt = 1e-4`（`t = 0.049`） | 加密 `64 x 64`, `dt = 5e-5`（`t = 0.045`） |
|---|---|---|
| 标记点偏差（壁钉死） | 3.83e-4 m = 1.2e-2 h | 6.2e-5 m = **4.0e-3 h** |
| 施力窗口 `max\|u\|` AFSI / IB2d | 1.2869 / 1.3998（8.1%） | 1.7738 / 1.8396（**3.6%**） |
| 速度场相对 $L_2$ 误差（窄带外） | 1.25e-1 | **5.3e-2** |
| 窄道流量（中列、窗口行）相对差 | 8.1%（仅 1 行可测） | **2.8%**（7 行） |
| 罚力公式核对 | ≤8.2（`\|f\|_{\max}` 8.6e4） | ≤2.0e1（`\|f\|_{\max}` 8.1e4） |

说明：原样 32² 网格上"中部射流"只落在**单列节点**上（窗口宽 0.06 只有 ~2 格），
因此驱动量与流量口径都很粗（流量口径退化为 1 行），6% 量级的相对差主要来自
这个"一个节点宽的喷口"；加密到 64²（射流 3 列、窗口 39 节点）后各项误差一致下降
（窗口 3.6%、$L_2$ 5.3e-2、流量 2.8%、标记点 4e-3 h）。

## 运行

```bash
conda activate afsi-dolfinx
python run_reference.py                  # pyIB2d 参考 -> ib2d_reference.npz
python main.py                           # AFSI -> plot/afsi_result.npz
python compare.py                        # 对比 + figures/（自动带 64^2 交叉核对）

python run_reference.py --nx 64 --dt 5e-5 --tend 0.05 --print-dump 100 \
    --out ib2d_reference_dt5e-5.npz --run ib2d_run_dt5e-5
NX=64 DT=5e-5 TFINAL=0.05 PRINT_DUMP=100 OUTPUT_PATH=./plot_dt5e-5 python main.py
```

环境变量（`main.py`）：`TFINAL`、`DT`、`NX`/`NY`、`PRINT_DUMP`、`GRAD_DIV`、
`OUTPUT_PATH`、`IB2D_EXAMPLE`。
