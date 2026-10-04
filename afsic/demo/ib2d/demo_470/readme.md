# demo_470 — IB2d `Example_Hole In_Channel`（带孔槽道，20 Hz 射流）

IB2d 官方算例（`matIB2d/Examples/Example_Channel_Flow/Example_Hole In_Channel`）
的复现，沿用 demo_467/468 的 `arb_ext_force` 模板。

## 算例内容

* `1 x 1` 周期盒、`mu = 10`、`rho = 1`、`32 x 32`、`dt = 1e-4`、`T = 0.25`
  （2500 步，`print_dump = 20`）。
* **结构**：104 个标记点，全部为 `k = 1e7` 的钉死目标点——底层 58 点完整壁
  （`y = 0.4`）+ 顶层 46 点壁**带孔**（`y = 0.6`，孔在 `x in (0.487, 0.691)`）。
* **外力**：`x in [0.1, 0.16]`、`y in [0.41, 0.59]`，

  $$u_x^{\rm tar} = -u_{\max}\,5|\sin(20\pi t)|\,(\tfrac{L_x}{2}+\tfrac{W}{2}-y)(\tfrac{L_x}{2}-\tfrac{W}{2}-y),$$

  `k = 1e4`、`u_max = 250`：**20 Hz** 的抛物射流，孔让流体从顶上泄出——这正是本算例的主题。

## 结果

参考运行 ~25 s、AFSI 运行 ~15 s，各 126 帧。原样网格与 `64 x 64`/`dt = 5e-5`
（`T = 0.05`）交叉核对：

| 指标（末帧） | 原样 `32 x 32`, `dt = 1e-4`（`t = 0.250`） | 加密 `64 x 64`, `dt = 5e-5`（`t = 0.045`） |
|---|---|---|
| 标记点偏差 | 3.16e-4 m = 1.0e-2 h | 1.10e-4 m = **7.0e-3 h** |
| 施力窗口 `max\|u\|` AFSI / IB2d | 0.1645 / 0.2001（17.8%） | 0.8948 / 0.9396（**4.8%**） |
| 速度场相对 $L_2$ 误差（窄带外） | 3.22e-1 | **9.5e-2** |
| 流量（中列、窗口行）相对差 | 3.2e-1（1 行） | **3.8e-2**（7 行） |
| 罚力公式核对 | ≤9.5e2（`\|f\|_{\max}` 1.6e4） | ≤2.8e2（`\|f\|_{\max}` 4.0e4） |

趋势：整段过程中两边的 `max|u|` 同步振荡（20 Hz 驱动），原样网格上峰值差
5%–17%、加密后 5%；孔区的剪切层在粗网格上尤其敏感，$L_2$ 误差从 3.2e-1 降到 9.5e-2
（3.4 倍），标记点稳定在 1e-2 h 以内。罚力公式核对始终是 `|f|` 峰值的 1% 量级。

## 运行

```bash
conda activate afsi-dolfinx
python run_reference.py                  # pyIB2d 参考（~25 s）
python main.py                           # AFSI（~15 s）
python compare.py

python run_reference.py --nx 64 --dt 5e-5 --tend 0.05 --print-dump 100 \
    --out ib2d_reference_dt5e-5.npz --run ib2d_run_dt5e-5
NX=64 DT=5e-5 TFINAL=0.05 PRINT_DUMP=100 OUTPUT_PATH=./plot_dt5e-5 python main.py
```
