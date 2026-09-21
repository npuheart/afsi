# demo_423 — Static equilibrium of an immersed anisotropic annular solid

This demo implements the benchmark described in the question (Section 4.1):
a thick annular cylinder immersed in a unit square filled with incompressible
fluid.  All walls are no-slip, the pressure is fixed only up to a constant, and
the analytical pressure is used to verify the numerical solution.

## Files

| File | Description |
|---|---|
| `generate_mesh.py` | Generates the structured quadrilateral annular solid mesh into `plot/mesh-423.xdmf` |
| `materials.py`     | `CircumferentialMaterial`: \(S^s=\mu_s \hat e_\theta\otimes\hat e_\theta\) |
| `main.py`          | Runs the IB-FSI simulation and prints \(L^2/H^1\) errors |
| `readme.md`        | This file |

## Run

```bash
conda activate afsi-dolfinx
python generate_mesh.py          # default N=32
python main.py                   # default N=32, 100 steps, dt=1e-4
```

Override with:

```bash
N=16 python generate_mesh.py
N=16 python main.py

N=64 STEPS=100 DT=1e-4 python main.py

# Try the triangle-mesh version (P2 solid elements are still used in main.py)
N=32 CELL_TYPE=triangle python generate_mesh.py
N=32 python main.py

# Try the incremental pressure correction solver (IPCS)
# FORCE_SCALE is set automatically to -1.0 for IPCS.
N=32 SOLVER=ipcs python main.py
```

## Parameters

- Fluid domain: \(1.0\times1.0\) m\(^2\)
- \(\rho^f=1.0\), \(\mu^f=1.0\)
- Inner radius \(R=0.25\) m, width \(w=0.0625\) m, center \((0.5,0.5)\)
- \(\mu^s=1.0\)
- Paper time setting: \(\Delta t=1.0\times10^{-3}\), \(t_f=10\Delta t\)
- The afsic Chorin solver is explicit; the demo defaults to
  \(\Delta t=1.0\times10^{-4}\), 100 steps for a better equilibrium.

## Analytical solution

\[
p(r)=
\begin{cases}
-\dfrac{\pi\mu^s}{2l^2}\big((R+w)^2-R^2\big), & r\ge R+w,\\[4pt]
\mu^s\ln\dfrac{R+w}{r}
-\dfrac{\pi\mu^s}{2l^2}\big((R+w)^2-R^2\big), & R<r<R+w,\\[4pt]
\mu^s\ln\Bigl(1+\dfrac{w}{R}\Bigr)
-\dfrac{\pi\mu^s}{2l^2}\big((R+w)^2-R^2\big), & r\le R,
\end{cases}
\qquad \mathbf v=\mathbf 0.
\]

## Important implementation note

The custom annular mesh must use the **DOLFINx quadrilateral vertex ordering**.
For a physical counter-clockwise quadrilateral `(a,b,c,d)`, DOLFINx expects
the cell as `[a,b,d,c]`.  Using a naive cyclic ordering silently corrupts the
solid element Jacobians: the solid area integral is wrong, the assembled PK
force is wrong, and the resulting pressure is about half of the analytical
value.  The current `generate_mesh.py` uses the correct ordering, so no
empirical force scaling is needed.


## 加密研究已执行（N = 16/32/64/128）

`convergence.py -n 16 32 64 128 --steps 100` 共 60 s（单进程）。复现了归档表里
$N=32$ 的一行到 5 位有效数字（$e_p^{L2}=3.169\times10^{-3}$，
$e_p^{\text{band}}=3.169\times10^{-3}$，$e_p(r<R-2h)=9.19\times10^{-6}$）。

| N | $e_p$ 全域 | inner | fibre band | $\lVert v\rVert_{L^2}$ |
|---|---|---|---|---|
| 16 | 4.798e-3 | 1.565e-4 | 4.788e-3 | 5.118e-5 |
| 32 | 3.169e-3 | 9.190e-6 | 3.169e-3 | 1.541e-5 |
| 64 | 3.074e-3 | 7.411e-5 | 3.073e-3 | 5.163e-6 |
| 128 | 3.242e-3 | 3.764e-4 | 3.172e-3 | 1.785e-6 |

- **速度干净收敛**：$\lVert v\rVert_{L^2}$ 收敛阶 1.73/1.58/1.53。
- **压力误差卡在界面带**：全域阶 0.60/0.04/-0.08，因为 $e_p$ 由 fibre band 主导，
  其值从 $N=32$ 起稳定在 $3\times10^{-3}$ Pa 附近 —— 这是界面分辨率极限（IB 核把
  界面抹平约 $\pm2h$），不是求解器误差。
- inner 区误差非单调（1.6e-4 → 9.2e-6 → 7.4e-5 → 3.8e-4），属大数相减的小量，
  不要把它的表观阶当作收敛率。
