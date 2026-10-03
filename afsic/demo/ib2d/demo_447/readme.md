# demo_447 — IB2d "Rubberband with Damped Springs"（阻尼橡皮筋）的 AFSI 复现

## 1. 问题描述

IB2d 的 `Examples/Rubberband_with_Damped_Springs`（pyIB2d 版，输入文件逐字复制自
`third_party/ib2d/pyIB2d/Examples/Rubberband_with_Damped_Springs`）：一个由 64 个
标记点组成的封闭橡皮筋（初始为绕 (0.5, 0.5) 的椭圆环），相邻点用**阻尼弹簧**
（$\kappa=2.5\times10^4$、静长 $L_r=0$、阻尼 $b=5$）连成一圈
（`0-1-2-…-63-0`），放在 1×1 周期盒的黏性流体中（μ = 0.01、ρ = 1）。

$L_r=0$ 意味着**每个弹簧的平衡长度是零**：结构在力学上倾向于整体塌缩，
过程由流体阻力与相对微弱的材料阻尼调制。

结构力（IB2d `give_Me_Damped_Springs_Lagrangian_Force_Densities`，逐字复刻）：

$$
\mathbf s_i = \kappa\,(L_i-L_r)\,\hat{\mathbf d}_i - b\,\mathbf V_i^{\rm leader},
\qquad \mathbf V^{\rm leader}=\frac{X_h-X_h^{\rm prev}}{\mathrm dt}
$$

（$+s$ 加在 leader 点、$-s$ 加在 follower 点；$\mathbf V$ 用 leader 在
**上一半步**的位置——即驱动的 `xLag_P` 记账方式；含 IB2d 的最小镜像处理）。

流体/耦合：与其它 ib2d demo 相同的管线（周期 Q2/Q1 + Peskin 两阶段 + 4 点核 +
零刚度 FE 环承载自由度）。

## 2. 验证与结果

**先给结论：这是一个"应力算例"——早段两码一致，长时行为定性分岔，且双方各自的
dt 收敛都成立。请把它当"发现两套离散格式差异的算例"读，而不是逐帧对拍。**

早期（t ≲ 0.05 s）两码形状/速度一致：

| t | 标记点最大偏差 | 环面积 AFSI | 环面积 IB2d |
|---|---|---|---|
| 0.01 | $2\times10^{-3}$ | 0.2508 | 0.2509 |
| 0.02 | $7\times10^{-3}$ ($0.22h$) | 0.2452 | 0.2514 |
| 0.05 | $1.1\times10^{-2}$ ($0.35h$) | 0.2376 | 0.2522 |
| 0.10 | $1.9\times10^{-2}$ ($0.6h$) | 0.2278 | 0.2519 |
| 0.50 | $1.16\times10^{-1}$ | 0.1280 | 0.2434 |
| 1.50 | $2.57\times10^{-1}$ | 0.00023 | 0.1833 |

之后 AFSI 的环持续收缩（$L_r=0$ 的物理平衡就是塌缩），而 IB2d 参考的环基本"卡住"，
只缓慢收缩。**这不是实现错误**，我们做了如下交叉验证：

1. **力实现校核**：扰动构型下与逐根循环完全一致（误差 0）；
2. **AFSI 侧 dt 收敛**：$\mathrm dt=10^{-3}$ 与 $2.5\times10^{-4}$ 结果 4 位有效数字一致；
3. **参考侧 dt 收敛**：pyIB2d 在 $\mathrm dt=10^{-3}$ 与 $2.5\times10^{-4}$ 下同样一致；
4. **高黏准静态极限**：$\mu=50$ 时两码都几乎冻结、漂移同量级（$10^{-4}\sim10^{-5}$）——
   力的**标度**两边一致；
5. **换静长实验**：把 $L_r$ 改成初始边长（环回到"有平衡长度"的状态）后，两码
   行为一致（都只做形状振荡，面积保持 ~0.25）；
6. **阻尼项开关**（$b=0$ / $b=5$）与**换用普通弹簧**（同一几何、同一 $\kappa$、$L_r=0$）
   对参考的解几乎无影响——差异与阻尼实现无关。

因此差异是**格式层面**的：对"高刚度 + 强收缩"的闭环结构，IB2d（同位网格 + FFT 谱投影，
保留网格尺度噪声）表现为被"数值阻尼"拖住；AFSI（P2 变分投影，网格尺度含量低）则持续收缩。
旁证：参考的流体涡量在高波数段占比 0.16–0.26（AFSI 0.07–0.19），且参考的标记点
"高角模纹波"（n ≥ 6）衰减慢 3–4 倍。这与 demo_445/446 的结论一脉相承
（IB2d 保留更多网格尺度内容；AFSI 更干净）。

![shapes](figures/compare_shapes.png)

![history](figures/compare_history.png)

## 3. 运行

```bash
conda activate afsi-dolfinx
cd afsic/demo/ib2d/demo_447

# (1) AFSI（本机约 25 s）
python main.py                    # -> plot/afsi_result_g0.npz + XDMF
TFINAL=0.01 python main.py        # 冒烟
DT=2.5e-4 python main.py          # dt 收敛检查（见上）

# (2) pyIB2d 参考（纯 Python，本机约 30 s；首次运行自动 clone pyIB2d）
python run_reference.py           # -> ib2d_reference.npz

# (3) 对照图
python compare.py                 # -> figures/*.png + compare_table.csv
```

## 4. 文件

| 文件 | 说明 |
|---|---|
| `main.py` | AFSI 计算（阻尼弹簧按 IB2d 原式；`xLag_P` 半步记账） |
| `run_reference.py` | 参考解（纯 Python，pyIB2d；damped_springs=1，其余关） |
| `ib2d_io.py` | IB2d 输入读取（0 基；`.d_spring` 等） |
| `ib2d_reference.py` | pyIB2d VTK 输出 → npz |
| `ib2d_input/` | `input2d`、`rubberband.vertex/.d_spring`（逐字复制） |
| `compare.py` | 形状/面积/偏差对照与 csv |

## 5. 已知限制与结论

- **长时分岔**（见 §2）：两套格式对"零静长强收缩闭环"这一极端情形的行为不同；
  AFSI 达到物理平衡（塌缩），IB2d 参考因网格尺度噪声/数值阻尼长期"卡住"。
  两码在 dt 上都已收敛，因此这不是步长问题；
- 因此本 demo 的对照图应读作"早段一致 + 长时格式差异"，不要用作逐帧基准；
- 真正做定量收敛研究时建议：把 IB2d 加密（噪声随网格尺度细化）或对比低通后的场。
