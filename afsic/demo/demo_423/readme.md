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
