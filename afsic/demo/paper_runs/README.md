# paper_runs — IJNMBE 论文所需 2D 计算

```bash
conda activate afsi-dolfinx
pip install -e ../../   # 若 afsic 未以可编辑方式安装，先装上（求解器有改动）
cd afsic/demo/paper_runs
NP=8 bash run_paper.sh            # 全部（annulus 约 10 min；disk 5 个工况；valve 4 × ~12 min）
NP=8 bash run_paper.sh annulus    # 单组：annulus | disk | valve
```

## 工况与对应论文内容

| 组 | 工况 | 论文用途 |
|---|---|---|
| annulus | IPCS / Chorin 网格收敛（N=16–128），对偶铺展，gamma=0 | Sec. 4.1 收敛表与收敛图（重跑） |
| annulus | N=64，dt = 1e-4 / 2.5e-5 / 6.25e-6（T=0.01） | dt 图 |
| annulus | N=64，IPCS，direct=1/0 × gamma=0/0.1/1/10 | 铺展方式与 grad-div 对稳态压力和面积的影响 |
| disk | IPCS/Chorin × gamma=0/GAMMA_DISK（对偶），IPCS dof-to-dof | Sec. 4.2 面积守恒、质心轨迹 |
| valve | IPCS/Chorin × direct=1/0，45° 纤维，T=3 s | 对偶 vs dof-to-dof 长时间稳定性（能量、叶尖位移） |

## 环境变量

- `NP`：MPI 进程数（默认 8）
- `GAMMA_DISK`：圆盘算例的 grad-div 系数（默认 1.0）
- `GAMMA_VALVE`：瓣膜算例的 grad-div 系数（默认 0）

## 代码改动（相对 test-ipcs-turek）

- `IPCSSolver` / `ChorinSolver`：新增 `grad_div` 参数（移植自 integrate-IB2d）；
  IPCS 对流项补 `rho`（rho=1 时结果不变）；IPCS 第一步取 `u^{-1}=u^0`。
- demo_423：`IB_DIRECT_LOAD`（默认 1，对偶铺展）、`GRAD_DIV`，输出 `area_ratio`。
- demo_336：`GRAD_DIV`、`T_END`、`NX`、`DT`、`MESH_PATH`、`OUTPUT_PATH`，
  输出 `disk_history_*.csv`（t, area_ratio, cx, cy, ...）；新增 `generate_mesh.py`
  （圆心 (0.6,0.5)，半径 0.2）。
- demo_340：`SOLVER`、`IB_DIRECT_LOAD`、`GRAD_DIV`、`FIBER_ANGLE`、`HISTORY`
  （t, KE, W, KE+W, x_tip, y_tip, u_max）。

## 先做短程冒烟测试

```bash
cd ../demo_423 && N=16 python generate_mesh.py && N=16 STEPS=5 SOLVER=ipcs python main.py
cd ../demo_336 && OUT=/tmp/d.xdmf python generate_mesh.py && MESH_PATH=/tmp/d.xdmf T_END=0.02 SOLVER=ipcs IB_DIRECT_LOAD=1 GRAD_DIV=1 OUTPUT_PATH=/tmp/d336 python fsi_paralell.py
cd ../demo_340 && STEPS=20 SOLVER=ipcs IB_DIRECT_LOAD=1 HISTORY=/tmp/v.csv python main.py
```
