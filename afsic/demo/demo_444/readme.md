# demo_444 — IB2d rubberband: fiber and thick-band comparison

First IB2d comparison case for the AFSI solver: the standard IB2d example
`Example_Standard_Rubberband/Rubberband_with_Springs` (an ellipsoidal
"rubberband" of 64 Lagrangian points closed by 64 zero-rest-length springs,
k = 2.5e4, in a unit box with rho = 1, mu = 0.01, dt = 1e-3, t <= 1.5 s),
run with AFSI in **two structural models x three solvers** (RT, Chorin,
IPCS - six runs with identical parameters) and against the registered IB2d
run:

- `SHAPE=fiber` (default): the IB2d structure itself — 64 markers with the
  IB2d spring law f_i = k (X_{i+1} + X_{i-1} - 2 X_i), loads L_i = f_i * DS.
- `SHAPE=thick`: the same problem as a continuum band — a neo-Hookean
  annulus (stress-free inner/outer radii 0.2328 / 0.2828, thickness 0.05)
  prestrained by the affine map that puts its outer edge exactly on the
  IB2d ellipse (semi-axes 0.2 x 0.4, area preserving).  Shear modulus
  calibrated (mu_s = 750, x18.75 from the original 40) so the band rings
  with IB2d: first a_x swing 0.106 s / 0.382 vs IB2d 0.10 / 0.382, period
  0.23 s vs 0.20 s.  The stiff band needs `DT=1e-4`: at 1e-3 the explicit
  fluid-structure coupling is unstable.

- `FLUID=rt` (default): divergence-conforming RT/DG fluid + nodal coupling
  (E / E^T).  The marker-sampled velocity is (to the coupling's accuracy)
  divergence-free, so the enclosed area behaves.
- `FLUID=chorin`, `FLUID=ipcs`: the P2/P1 projection backgrounds with the
  four-point IB kernel (demos 441/442/443 pipeline).  Kept as the solver
  matrix: both projection flavours leak the enclosed area - the fiber
  collapses by t = 0.14 s and the thick annulus loses 3.7 %; see the site
  page.  Non-RT runs get an automatic tag suffix (`fiber_N16_chorin`, ...)
  so they never overwrite the RT outputs.

Shared matching: fluid mesh N = 16 whose P2-style velocity nodes (33 x 33,
spacing 1/32) sit on IB2d's 32^2 grid; dt = 1e-3 (the thick runs use
dt = 1e-4 for stiff-solid stability).

## Running

```bash
conda activate afsi-dolfinx
python main.py                    # fiber, RT (~80 s)
SHAPE=thick DT=1e-4 python main.py        # thick annulus, RT (~25 min)
FLUID=chorin python main.py       # fiber, Chorin (~5 s)
FLUID=ipcs python main.py         # fiber, IPCS (~5 s)
SHAPE=thick DT=1e-4 FLUID=chorin python main.py   # thick, Chorin
SHAPE=thick DT=1e-4 FLUID=ipcs python main.py     # thick, IPCS

# seven-curve comparison figures (needs the IB2d example directory):
IB2D_DIR=/path/to/IB2d/matIB2d/Examples/Example_Standard_Rubberband/Rubberband_with_Springs \
    python plot_compare.py        # writes figures/demo444-{area,shapes,velocity}.png
```

Environment: `SHAPE`, `FLUID`, `N`, `T_END`, `DT`, `OUT_EVERY`, `M_MEM`,
`K_SPRING`, `DS`, `M`, `K_RAD`, `T_WALL`, `MU_S`, `KAPPA_STAB`, `RT_SOLVER`,
`TAG`, `OUTPUT_PATH` (see the docstring of `main.py`).

## The `DS` weight (important)

IB2d's spring output is a force **density**; before spreading it is
multiplied by the constant `ds = min(Lx/(2Nx), Ly/(2Ny))` ("Peskin constant
ds", IBM_Driver.m:203; the multiplication sits in
`please_Find_Lagrangian_Forces_On_Eulerian_grid.m` right before the 4-point
kernel spread), `1/64` here. The default `DS = 1/(4 N) = 1/64` uses the
same convention and reproduces the IB2d dynamics: marker oscillation
period 0.201 s vs IB2d's 0.190 s; first/second a_x peaks 0.106 / 0.302 s
vs IB2d's 0.10 / 0.32 s.

An earlier version of this demo used `DS = 1/(16 N)`, based on a probe that
seemed to show IB2d's realized Eulerian force was 4x below its documented
spread (max 302.7 vs 1155). That was a comparison artifact: 302.7 is the
max of the spread field's **x-component** in the instrumented run, 1155 is
the 2-D **magnitude** of the same field - on this tall ellipse they differ
by ~4x. A direct replica of the documented pipeline (`F*ds`, 4-point
kernel) reproduces both numbers exactly at the initial state, so IB2d has
no hidden normalization; `1/256` under-forces the fiber 4x and doubles the
oscillation period.

## Results

| run | a_x(0.2 s) | area at 1.5 s | final shape |
|---|---:|---:|---|
| IB2d fiber (registered) | 0.2159 | 0.1840 (−26.8 %) | still creeping |
| AFSI fiber (RT) | 0.2235 | 0.2335 (−7.1 %) | circle R = 0.279 = equal-area circle |
| AFSI fiber (Chorin) | 0.018 | 0.000004 (−100 %) | collapsed by t = 0.14 s |
| AFSI fiber (IPCS) | 0.020 | 0.000000 (−100 %) | collapsed by t = 0.14 s |
| AFSI thick (RT) | 0.2207 | 0.2505 (−0.35 %) | ringing toward the stress-free circle R = 0.283 |
| AFSI thick (Chorin) | 0.3286 | 0.2421 (−3.7 %) | rings 0.63 s (2.7x slow) |
| AFSI thick (IPCS) | 0.3289 | 0.2421 (−3.7 %) | rings 0.63 s (2.7x slow) |

A zero-rest-length band at fixed enclosed area minimises its spring energy
(perimeter²) at the **equal-area circle**, R = sqrt(A0/pi) = 0.2826; the
AFSI fiber settles there through a large damped aspect-ratio oscillation
(a_x rings between ~0.20 and ~0.36 with a 0.19-s period; IB2d rings at
~0.20 s).  The calibrated thick band rings with the same ~0.2-s family and
relaxes toward its stress-free circle 0.2828 (still ±0.01 at 1.5 s).
The projection-based variants instead leak the enclosed area at the rate
of their velocity's residual (sub-cell) divergence - the fiber collapses
by t = 0.14 s while the thick annulus loses 3.7 % over the window and
rings 2.7x slower (0.63 s).  At
t = 0.02 s the leak is −20.8 % / −9.4 % / −7.7 % for N = 16/32/64 (IB2d's
spectral projection at 0.02 s: +0.03 %; the RT solver at N = 16: −0.56 %),
it is dt-converged (−20.8 % / −20.9 % / −20.9 % for dt = 1e-3 / 2e-4 /
1e-4) and the incremental pressure-correction flavour (FLUID=ipcs)
behaves identically (−20.5 % / −8.9 % at N = 16/32).  On quadrilaterals the
Q2/Q1 pair is not inf-sup stable, so no pressure can remove every
divergence the velocity space produces.  This is why the RT solver is the
default here.

See `figures/` for `demo444-area.png`, `demo444-shapes.png`,
`demo444-velocity.png` and the page `demo-444` on the AFSI notes site
(tag **IB2d**) for the full discussion.
