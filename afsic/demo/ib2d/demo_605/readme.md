# demo_605 — IB2d example "Jellyfish Material (512x512)" (matIB2d
``Examples/Examples_First_Year_Seminar/Jellyfish_Material_512x512``) in AFSI.

## 中文小结

* **算例**：`Jellyfish_Material_512x512` —— 与 demo_604 同一族的 512² 版本
  （255 个点、`kb` 最大 `1.04e15`）。上游 `Tfinal = 1 s`、`128 x 512`。
* **缩减**：`NX = 64`（`64 x 256`）、`T = 0.005 s`（500 步）。
* **结果（重要）**：**两个代码都在第一个时间步就 NaN**（`kb dt / mu ≈ 1e9`），
  除了 `t = 0` 之外没有任何可比帧 —— 上游这一版的参数本身不可跑。
  demo_604 才是这一族的有效成员。


> 下面 "Physics / porting notes" 部分是英文叙述（与 `main.py` 的 docstring 同源）：
> 上游算例的标识符、方程都来自英文源码，保持原样以免转述出错。

The high-resolution member of the pair (demo_604 is the 64 x 256 run): 255
markers in a 2 x 8 periodic box, mu = 10, rho = 1000, dt = 1e-5, shipped grid
128 x 512, 254 springs (k = 1.3e7) and 253 non-invariant beams (kb = 1.0e15).
The update law is the *same* analytic Alben-Peng-Miller bell model as demo_604
(its ``update_nonInv_Beams.m`` differs only in the hardcoded grid constants
``N1 = 512`` and ``xRef = L1/8``), ported line-by-line to
``ib2d_input/update_nonInv_Beams.py``.

The same pyIB2d driver bug affects this example (it passes ``beams_info`` --
the scalar 0, since ``beams = 0`` -- to ``update_nonInv_Beams``), so
``run_reference.py`` patches that call; see demo_604 for the details.

Cost: the shipped ``Tfinal = 1 s`` (100000 steps) at 128 x 512 is out of reach,
so the demo runs a **reduced** ``NX = 64`` (64 x 256) and ``T = 0.005 s``
(500 steps) with ``print_dump = 50`` (10 frames) instead of 2500.

## 结果（`python compare.py` 的完整输出）

```
        t   dXmax/h            max|u| a/r    du away    du band      relL2    df away
  0.00000  2.12e-09  0.000e+00/0.000e+00   0.000e+00  0.000e+00  0.000e+00  0.000e+00
  0.00051       nan        nan/nan               nan        nan        nan        nan
  0.00101       nan        nan/nan               nan        nan        nan        nan
  0.00151       nan        nan/nan               nan        nan        nan        nan
  0.00201       nan        nan/nan               nan        nan        nan        nan
  0.00251       nan        nan/nan               nan        nan        nan        nan
  0.00301       nan        nan/nan               nan        nan        nan        nan
  0.00351       nan        nan/nan               nan        nan        nan        nan
  0.00401       nan        nan/nan               nan        nan        nan        nan
  0.00451       nan        nan/nan               nan        nan        nan        nan
  0.00501       nan        nan/nan               nan        nan        nan        nan
markers          : max|dX| nan m = nan h (255 Lagrangian points)
velocity         : max|u| AFSI nan / IB2d nan (nan relative); max|du| nan = nan max|u| away from the IB band (15947/16384 nodes), nan (nan max|u|) inside it; relative L2 nan
force field      : max|df| nan away from the band, |f|max nan
/Users/pengfei.ma/afsi/afsic/demo/ib2d/demo_605/compare.py:312: UserWarning: Data has no positive values, and therefore cannot be log-scaled.
  fig.tight_layout()
wrote /Users/pengfei.ma/afsi/afsic/demo/ib2d/demo_605/figures/compare_table.csv and the figures
```

这张表就是 `compare.py` 的原始输出：每帧的 `dXmax/h` 是标记点偏差（以网格步长 h
为单位），`max|u| a/r` 是 AFSI/IB2d 的速度峰值，`du away` / `du band` 是 IB 带外/
带内的速度差，`relL2` 是速度场相对 $L_2$，`df away` 是带外外力场差（有被动标量时
还有 `max|dC|` 与 `relL2(C)`）。它与 `figures/compare_table.csv` 一致。

## 不可比 / 需要打折看的地方

* **这个上游算例在一步之内就爆炸（两个代码都一样）**：`.nonInv_beam` 文件里梁刚度
  `kb` 最大到 `1.04e15`，而 `dt = 1e-5`、`mu = 10`，罚力/刚度的显式稳定量
  `kb dt / mu ≈ 1e9`。实测：AFSI 与 pyIB2d **在第一个输出帧（`t = 5e-4`）就已经是
  NaN**（`compare.py` 的表里全是 `nan`；`plot/afsi_result.npz` 与
  `ib2d_reference.npz` 的 `X/u` 除 `t=0` 外全部非有限）。这不是移植 bug，而是
  **上游参数本身不可跑**：`.m` 给出的参考曲率与 512² 版几何不匹配，一步就产生巨大
  的非不变梁力。
* **能说的结论只有**：两个代码在 `t = 0` 完全一致（`max|dX|/h = 0`）—— 因为
  `t = 0` 的初始离散是同一份数据；此后两边同时发散，**本 demo 无法给出任何流固
  耦合的定量对比**。
* **规模缩减**：上游 `Tfinal = 1 s`（100000 步）在 `128 x 512`，本 demo 用
  `NX = 64`（`64 x 256`）和 `T = 0.005 s`（500 步）—— 但如上所述，实际只有第 0 帧
  是有限的。
* **pyIB2d 驱动 bug**：与 demo_604 相同（`IBM_Driver.py:711` 把 `beams_info`——
  本算例 `beams = 0` 时是标量 `0`——传给 `update_nonInv_Beams`）；本 demo 的
  `run_reference.py` 已包含最小补丁，否则参考解连第一步都进不去。
* **建议**：把 demo_604（`Jellyfish_Material`，`64 x 256`，`kb = 1e13`）当作这一族
  的有效成员；605 保留下来作为"上游 512² 版本不可用"的记录。

## 运行方法（实测耗时，本机）

```bash
conda activate afsi-dolfinx
cd afsic/demo/ib2d/demo_605
```

| 命令 | 实测 `real` |
|---|---|
| `python run_reference.py --nx 64 --tend 0.005 --print-dump 50` | 4m11.548s |
| `NX=64 python main.py` | 2m8.977s |
| `python compare.py` | - |

（`real` 取自 `ref.log` / `afsi.log`；本机同一时刻还有大量其他算例在跑，
绝对耗时波动很大。参考解写入 `ib2d_reference.npz`，AFSI 写入
`plot/afsi_result.npz`，`compare.py` 生成 `figures/` 与 `compare_table.csv`。）
