# demo_449 — IB2d "Wobbly Non-Invariant Beam"（非不变梁摆动）的 AFSI 复现

## 1. 问题描述

IB2d 的 `Examples/Wobbly_NonInv_Beam`（pyIB2d 版，输入文件逐字复制自
`third_party/ib2d/pyIB2d/Examples/Wobbly_NonInv_Beam`）：与 demo_446 相同的拱形梁
几何（(0.25, 0.5) → (0.75, 0.5)，顶点 y ≈ 0.625），但扭转弹簧是**非不变梁**——
每条梁带存储生成时刻的参考二阶差分
$\mathbf C=(\mathbf X_{p}+\mathbf X_{r}-2\mathbf X_{q})$（本例文件中 $\mathbf C=0$，
即释放时是受力状态）；$\kappa_{\rm beam}=10^{10}$。1×1 周期盒、μ = 0.1、ρ = 1、
32×32 网格、$\mathrm dt=10^{-5}$（示例原值）。

结构（0 基索引，直接读 `ib2d_input/`）：

* **非不变梁**：62 组三点、$\kappa=10^{10}$：
  $\mathbf r=(\mathbf X_p+\mathbf X_r-2\mathbf X_q)-\mathbf C$，
  $\mathbf F_q \mathrel{+}= 2\kappa\,\mathbf r$、$\mathbf F_p,\mathbf F_r \mathrel{-}= \kappa\,\mathbf r$
  （IB2d `give_Me_nonInv_Beam_Lagrangian_Force_Densities`，力校核 $1.4\times10^{-16}$）；
* **目标点**：两端（id 0、63），$k=2\times10^8$。

流体/耦合：与其它 ib2d demo 相同的管线（周期 Q2/Q1 + Peskin 两阶段 + 4 点核 +
零刚度 FE 链承载自由度）。

## 2. 结果

![shapes](figures/compare_shapes.png)

![history](figures/compare_history.png)

* 梁以 ~0.02 s 周期在高频振荡（中点 y 在 0.42–0.57 间往返）——$k=10^{10}$ 的近刚性
  振子；AFSI 与 pyIB2d 追踪同一振荡；
* 全程 100 帧标记点最大偏差 **均值 0.023（0.74h）、峰值 0.048（1.55h）**
  （$h=1/32$），与 demo_446 同量级；周期内相位有小漂移；
* $\gamma=0$（默认）。

## 3. 运行

```bash
conda activate afsi-dolfinx
cd afsic/demo/ib2d/demo_449

python main.py                    # AFSI（本机约 40 s）-> plot/afsi_result_g0.npz
TFINAL=0.001 python main.py       # 冒烟
python run_reference.py           # pyIB2d 参考（约 40 s）-> ib2d_reference.npz
python compare.py                 # -> figures/*.png + compare_table.csv
```

## 4. 文件

| 文件 | 说明 |
|---|---|
| `main.py` | AFSI 计算（非不变梁 + 目标点） |
| `run_reference.py` | 参考解（纯 Python，pyIB2d） |
| `ib2d_io.py` / `ib2d_reference.py` | 0 基输入读取 / VTK→npz 转换 |
| `ib2d_input/` | `input2d`、`BeamCurve.vertex/.nonInv_beam/.target`（逐字复制） |
| `compare.py` | 形状/历史/场对照与 csv |

## 5. 备注

- 与 demo_446（不变梁）对照可以看出两种梁模型的差别：不变梁的零力构型由
  生成器写出（$C=0$ 恰好为拱形），非不变梁则在释放时记忆 $\mathbf C$；
- 同样的近刚性高频振荡特性 ⇒ 存在 ~1 格的相位漂移，属正常离散差异。
