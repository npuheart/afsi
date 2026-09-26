# demo_426 — flow through a slanted channel (2-D), IB benchmark

Steady plane-Poiseuille flow through a channel inclined at `theta = pi/6` about
the origin, in a rectangular box, with the two channel walls represented as
immersed boundaries so that they are deliberately **not** grid-aligned.  This
follows the two-dimensional slanted-channel benchmark of Grüninger et al.,
whose purpose is to compare how different IB kernels reproduce the exact
solution inside a confined, stationary geometry (their Fig. 22 velocity field
and Fig. 23 profile at `x = 0.5`).

## Exact solution

With the coordinate across the channel,

```
xi(x, y) = y*cos(theta) - x*sin(theta)          walls at xi = +-D/2
```

the exact steady state is unidirectional plane Poiseuille along the axis:

```
u(x, y) = (dP/L)/(2*mu) * ( (D/2)^2 - xi^2 ) * (cos(theta), sin(theta))
v(x, y) = (dP/L)/(2*mu) * ( (D/2)^2 - xi^2 ) * (sin(theta), ...)
```

driven by the constant body force `f = -(dP/L)*(cos(theta), sin(theta))`
(the axial pressure gradient points opposite to the flow).  This field is
divergence-free and satisfies the momentum balance exactly, and it vanishes on
both plates by construction, so no-slip on the walls is exactly compatible with
it.  `configuration.py` exposes all three identities and they were checked
numerically before writing the solver.

## Parameters

The benchmark text lists `h = 1`, `D = h/cos(theta)`, `mu = 0.5`, `rho = 1`,
`dP/L = 1.0`, `U_max = 0.25`, `N = 32`, `dt = 0.15*dx`.

**These are not all mutually consistent.**  Plane Poiseuille gives

```
u_max = (dP/L) * (D/2)^2 / (2*mu) = 1.0 * 0.57735^2 / 1.0 = 1/3 = 0.33333
```

so `U_max = 0.25` disagrees with `dP/L = 1`, `mu = 0.5` by 25 %.  The
coefficient as printed in the benchmark, `(dP/L)/(2*mu*L)`, is also not
dimensionally consistent (it must carry a squared length in the denominator).

**Resolution used here:** keep the benchmark's *primary* parameters
(`mu`, `dP/L`, `h`, `theta`, box, `N`, `dt`) and treat `U_max` as a derived
quantity.  The discrete problem is then driven by exactly the same body force
as the reference solution, which is what makes the comparison a genuine
discretisation-error study.  Set `DP_DL=0.75` to reproduce `U_max = 0.25`
instead.

| quantity | value |
|---|---|
| `theta` | 30° (`cos = 0.866025`, `sin = 0.5`) |
| channel width `D` | 1.154701 |
| radius `D/2` | 0.577350 |
| `rho`, `mu` | 1.0, 0.5 |
| `dP/L` | 1.0 (body force) |
| **`u_max` (derived)** | **0.333333** (benchmark text: 0.25) |
| `N`, `dx` | 32, 0.03125 |
| `dt` | 0.0046875 (`0.15*dx`) |
| channel width in cells | 36.95 |
| grid | `112 x 154` on box `3.5 x 4.8125` |
| `Re = rho*u_max*D/mu` | 0.770 |
| Courant | 0.050 |
| `nu*dt/dx^2` | 2.400 |

`nu*dt/dx^2 = 2.4` exceeds the explicit-diffusion limit of 0.25, but the
viscous term is treated **implicitly** by both solvers, so there is no
diffusion restriction here; only advection matters, and the Courant number is
small.

## The box: the quoted one does not contain the channel

The benchmark quotes `Omega = [0,1] x [-0.25,2]`.  That rectangle does **not**
contain the slanted channel.  The walls are `y = (x*sin +- D/2)/cos`, so over
`x in [0,1]` the channel spans `y in [-0.66667, 1.24402]` (width 1.9107 =
1.6547*D): the quoted box cuts the lower plate off below `y = -0.25`, truncating
the inlet cross-section by 37 %.

The benchmark's own setup figure settles the intent: **both** plates appear as
dotted Lagrangian-marker lines running the full length of the box, and both
ends are open channel cross-sections.  The box must therefore be extended
**downward** (and, to let the lower plate reach the left face, leftward) — and
the figure is also clearly *wider* than `[0,1]`, showing the channel end to end.

Domain used here, matched to that figure:

```
Omega = [-0.9, 2.6] x [-1.4, 3.4]        (box 3.5 x 4.8125)
```

| | `y` at `x = -0.9` | `y` at `x = 2.6` |
|---|---|---|
| lower plate `xi = -D/2` | −1.18628 | +0.83444 |
| upper plate `xi = +D/2` | −0.03167 | +2.16778 |

Both plates run the full width, stay strictly inside the box, and terminate
**exactly** on the inlet and outlet faces, so both ends are complete 1.3333-tall
cross-sections and no plate needs clipping.  `X_MIN`, `X_MAX`, `Y_MIN`, `Y_MAX`
are all environment-overridable; `X_MIN=0 X_MAX=1 Y_MIN=-0.25` reproduces the
quoted truncation.

Note `dx` is fixed by the benchmark's `L/N` convention with `L = 1`, and the
cell counts are *derived* from the box (`NX`, `NY`), not the reverse — otherwise
widening the box would silently coarsen the grid.  At `N = 32`,
`dx = 0.03125` and the grid is `112 x 154`.

## Boundary conditions
## Boundary conditions

| boundary | condition |
|---|---|
| inlet `x=0`, channel interval `y in [-2/3, 2/3]` | velocity Dirichlet, **analytic** |
| outlet `x=1`, channel interval `y in [-0.0893, 1.24402]` | velocity Dirichlet, **analytic** |
| every other boundary segment | no-slip `u = 0` |
| the two plates inside the box | immersed-boundary penalty |

The flow is driven by the **body force**, not by a pressure condition, so no
pressure BC is imposed anywhere: the pressure has no Dirichlet data, its null
space is fixed by the projection step, and only incompressibility determines
the field.  The inlet/outlet Dirichlet dofs are selected by the geometric
channel interval on those faces, so the no-slip wall segments share the faces
with the analytic openings.

## The plate penalty: explicit vs implicit

The benchmark (their Eq. 14) uses penalty stiffness plus a penalty body force
plus damping, with the parameters "empirically determined as the largest values
that maintain numerical stability".  That statement is the crux, and it matters
which term the penalty enters:

* **The body force `ns_solver.f` is frozen during the momentum solve**, i.e. it
  is applied **explicitly**.  The spread kernel maps a Lagrangian force to a
  body force per unit volume with total weight `1/(dx*dy)`, so an effective
  damping rate is `DAMP/(dx*dy)` and explicit stability requires

  ```
  DAMP * dt / (rho * dx * dy) <~ 1     =>     DAMP <~ rho*dx*dy/dt = 0.208
  ```

  at `dx = 1/32`, `dt = 0.15*dx`.  That is far too weak to pin no-slip.
  Measured behaviour: with `DAMP = 1e3` the solution blew up to `max|u| ~ 1e143`
  within 300 steps.

* **The `drag` term is assembled into the momentum LHS**, so it is **implicit**
  and carries no time-step restriction.  This demo therefore imposes the plates
  as a thin band of large implicit drag,
  `PLATE_DRAG = 1e4` over `PLATE_DRAG_BAND = 1.5*dx` around each plate
  (`USE_IMPLICIT_DRAG=1`, the default).

The explicit Peskin route is still implemented (`USE_IMPLICIT_DRAG=0`) and is
correct in form — rigid stationary plates are *not* integrated in time, so
`X == X_ref` identically and the penalty reduces to `-DAMP*u_ib` — but its
usable `DAMP` is capped as above.

**Consequence for the benchmark's stated purpose.** Fig. 23 compares IB
*kernels* (BS vs CBS21 vs CBS32 ...), and narrower support gives a thinner
numerical boundary layer.  The kernels live in the C++ coupling layer and the
shared solvers expose only a constant scalar `drag`, so this demo cannot vary
the kernel support width: it reproduces the *geometry, parameters and reference
solution*, and the channel-accuracy numbers below are for one fixed penalty
band.  Reproducing Fig. 23 needs the kernel choice to be selectable in
`IBMesh`/`IBInterpolation` (or the explicit route to be made implicit).

## Run

```bash
cd afsic/demo/demo_426
python main.py                                  # full run: T_END=20 (~50 min)
SMOKE=1 SMOKE_STEPS=300 python main.py          # quick check
USE_IMPLICIT_DRAG=0 DAMP=0.1 SMOKE=1 python main.py   # explicit-Peskin route
```

Environment overrides: `N`, `DP_DL`, `THETA_DEG`, `Y_MIN`, `DT_FACTOR`,
`T_END`, `SMOKE`, `SMOKE_STEPS`, `PLATE_DRAG`, `PLATE_DRAG_BAND`,
`USE_IMPLICIT_DRAG`, `BETA`, `DAMP`, `SOLVER`.

## Files

| file | description |
|---|---|
| `configuration.py` | parameters, exact solution, channel clipping to the box |
| `verify.py` | analytic references, sampling, error metrics |
| `main.py` | box, boundaries, IB plates, time loop, report |

## Results (N = 32, `dx = 0.03125`, `dt = 0.15*dx`, T = 20)

Steady state is reached early: `max|u|` settles to 0.34690 by t ~ 2 and is
unchanged to 5 digits through t = 20, so the metrics below are converged values.

| metric | value |
|---|---|
| relative L2 error in the channel | **6.76 %** |
| Linf error in the channel | 3.79e-2 m/s |
| profile relative L2 (at `x = 0.5`) | 7.10 % |
| profile Linf | 3.82e-2 m/s |
| numerical `u_max` | 0.34307 m/s (analytic 0.33333, **+2.92 %**) |
| `u_max` location in `xi` | 0.000 (exactly on the centreline) |
| fluid `|u|` on the plates | max 1.27e-2, mean 2.61e-3 m/s |
| plate displacement from reference | identically 0 |
| elapsed | 2922 s (4267 steps, 112 x 154 grid) |

The velocity is a few percent **high** with the peak on the centreline, and the
residual velocity on the plates is the numerical boundary layer produced by the
penalty band: the analytic profile vanishes on the plates, so any non-zero
`|u|` there is smearing error.  Widening the box from `[-0.2,1]x[-0.8,2]` to
`[-0.9,2.6]x[-1.4,3.4]` improved the peak error from +5.4 % to **+2.9 %**,
presumably because the inlet/outlet now sit further from the measurement
station and the plates are longer relative to the channel width.

### Why the whole-box error is not reported as an accuracy measure

The whole-box relative L2 error is ~1.0 with Linf ~11.1 m/s.  That is **not** a
solver error.  The exact parabola `A*((D/2)^2 - xi^2)` grows without bound
*outside* the channel -- in this box it reaches 11.1 -- while the physical flow
outside the plates is stagnant.  The metric is therefore dominated by a
spurious reference value, which is why the channel-restricted numbers are the
meaningful ones and the profile at `x = 0.5` is the benchmark's own choice.

## Figures

`plot/N32_ipcs/example_figures/` (see demo_424/plot/PLOTTING_GUIDE.md for the
workflow; the XDMF is read with meshio and rendered off-screen with OSMesa):

| figure | contents |
|---|---|
| `paper_setup.png` | **reproduction of the benchmark's setup figure**: the box, the `|u|` colour map, the two plates as rows of Lagrangian-marker dots, and the white profile-measurement line at `x = 0.5` |
| `paper_setup_with_exact.png` | the same, AFSI vs the exact solution side by side |
| `00_overview.png` | 2x2: `|u|`, `p`, vectors, streamlines |
| `01_velocity_magnitude.png`, `02_pressure.png` | field plots with the plates drawn |
| `03_velocity_vectors.png`, `04_streamlines.png` | vectors and streamlines |
| `05_profile_x.png` | `|u|` across the channel at `x = 0.5` vs the exact parabola, plus the error (the benchmark's Fig. 23 comparison) |
| `06_history_matplotlib.png` | `max|u|` history against the analytic `u_max` |
| `index.html` | preview page |

Pressure is plotted as `p - mean(p)`: with no pressure Dirichlet datum anywhere
(the flow is body-force driven) `p_` is fixed only up to an additive constant
and comes out around `8e11` while its standard deviation is `0.84` and its range
`14.5`, all of which are physical.

## Status / open items

* **Fig. 23 cannot be reproduced as shipped** — the kernel-support comparison
  needs a selectable kernel in the IB layer, or an implicit treatment of the
  Peskin penalty force.  See above.
* `u_max` is +4.5 % high; a thinner or stronger penalty band, or the implicit
  Peskin route once available, should reduce it.  A small sweep over
  `PLATE_DRAG_BAND` (1.0 to 3.0 `dx`) would show whether the boundary layer
  thickness behaves as the benchmark describes.
* Only `N = 32` is implemented; the benchmark's fine grid is `N = 32` with
  `dx = L/N`, so a convergence sweep would vary `N` (and keep `dt = 0.15*dx`).
* The `U_max` inconsistency between the benchmark text and its own parameters
  is documented rather than silently patched; if the intended target really is
  `U_max = 0.25`, run with `DP_DL=0.75`.

## Immersed-boundary spreading normalisation (RETRACTED — the operators are fine)

**The "`1/h^2` over-injection" reported below was a measurement artifact and has
been retracted; `SPREAD_NORM` now defaults to `none` (no scaling).**  A proper IB
discretisation needs the interpolation and spreading operators to be adjoints,
and they are: the fluid-side inner product must carry the cell area,

    <u, S F>_Omega = sum_ij u_ij (S F)_ij * dx * dy ,   dx = 1/(2N)

With that weighting the measured ratio is **exactly 1.000000** at
N = 16/32/64/128, i.e. no correction is needed
(`afsic/tests/test_duality.py` has always used this weighting and passes).  The
ratios quoted in the table below are precisely `dx*dy` — the factor that was
missing from the *unweighted* node sum used at the time:

| N | dx | `<S*u,F>` / `<u,SF>` (node sum, unweighted) | `dx*dy` | ratio with the volume weight |
|---|---|---|---|---|
| 16 | 0.03125 | 9.765625e-04 | 9.765625e-04 | 1.000000 |
| 32 | 0.015625 | 2.441406e-04 | 2.441406e-04 | 1.000000 |
| 64 | 0.0078125 | 6.103516e-05 | 6.103516e-05 | 1.000000 |
| 128 | 0.00390625 | 1.525879e-05 | 1.525879e-05 | 1.000000 |

Consequently `SPREAD_NORM=h2` did not restore adjointness — it multiplied the
Lagrangian force by `DX^2` = 9.77e-4, weakening the immersed coupling by 1024x.
With it the plates exerted essentially no force: in the committed
`plot/N32_ipcs_smoke` run the plates were passive tracers carried downstream at
~0.37 `u_max`, the fluid *outside* the channel reached 0.37 `u_max` (i.e. there
was no confined channel at all), and the in-channel error was 34.7 %.

`SPREAD_NORM` remains as a calibration knob (`none` = as shipped, the default;
`h2` = the old 1024x attenuation; `<number>` = explicit factor).

### What actually limits the explicit (tether) route

The explicit, frozen-force penalty carries a per-step gain
`~ beta*dt^2/rho`.  At `dt = 0.2*DX` a 0.5-cell perturbation of the plates is
*amplified* for `beta >= ~1e5` (at `beta = 1e6` the run diverges within a few
steps), while the same tether in demo_424 (`beta = 1e7`, `dt = 2e-4`) relaxes
the perturbation — giving demo_424 the same 31x larger time step would blow it
up too.  So the tether is not usable at this demo's time step, and the
**implicit** drag band (`USE_IMPLICIT_DRAG=1`, the default) is what holds the
plates: measured in-channel error 5.3 %, plate slip 0.005 `u_max`, zero drift.

The measurements behind this section (hold tests, the 2x2x2 driving/plate sweep,
the duality re-measurement) are recorded in
`docs/demo-426-ib-coupling-findings.md`; how to run the demos on this machine is
in `docs/run-demo-424-426.md`.

Diagnostic: with `h2`, `kappa` = 60/200/600 give delta = 0.12455/0.12459/0.12470
—— i.e. **delta is independent of kappa**, so the tether is not what is holding
the plates.  The measured scale is consistent with the plate being dragged until
the tether stress balances the fluid load on it
(`kappa*delta ~ mu*u_plate/(h/2)`), but neither raising `kappa` (checked to
2e5) nor reducing `dt` (checked 16x) moves delta toward `h/2`.

Fixed-point coupling within the step (`IB_ITERATIONS > 1`) was also tried: at
`T = 0.625`, 8 iterations change delta by only 0.8 %, because `solve_one_step()`
does not reset `u_n`/`u_n1`, so the loop does not actually iterate on a frozen
fluid state.  Closing the coupling properly needs a solver-level change.
