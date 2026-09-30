# demo_426 — flow through a slanted channel (2-D), IB benchmark

Steady plane-Poiseuille flow through a channel inclined at `theta = pi/6`, in a
rectangular box, with the two channel walls represented as immersed boundaries
so that they are deliberately **not** grid-aligned.  This follows the
two-dimensional slanted-channel benchmark as used by Li et al. (2025, §4.3.1;
originating from Grüninger et al. 2024, §4.1), whose purpose is to compare how
different IB kernels reproduce the exact solution inside a confined, stationary
geometry (Fig. 22 velocity field, Fig. 23 profile at `x = 0.5`).

> **Geometry corrected 2026-09-30.**  The domain, channel position and
> parameters now match the benchmark text exactly: `Omega = [0,1] x [-0.25,2]`
> (translated to `[0,1] x [0,2.25]` here), walls at `xi~ = 0` and `xi~ = h = 1`,
> `mu = 0.5`, `U_max = 0.25`, `dt = 0.15*dx`.  An earlier version of this demo
> used a larger box and placed the channel *centreline* through the origin
> (perpendicular width `D = h/cos(theta)`), which made the quoted `U_max` look
> inconsistent; that reading was wrong.  Results below were re-measured on the
> corrected geometry (runs of 2026-09-30; older outputs are kept in
> `velocity_oldbox/` and `pressure_oldbox/`).

## Exact solution

With the wall coordinate

```
xi~(x, y) = y*cos(theta) - x*sin(theta)         walls at xi~ = 0 and xi~ = h
```

the exact steady state (benchmark Eq. (48)) is unidirectional plane Poiseuille
along the axis:

```
u(x, y) = (dP/L)/(2*mu) * xi~ * (h - xi~) * (cos(theta), sin(theta))
```

equivalently `(dP/L)/(2*mu) * ((h/2)^2 - xi^2) * (cos, sin)` with `xi` measured
from the centreline.  It is divergence-free, satisfies the momentum balance
exactly, and vanishes on both plates by construction, so no-slip on the walls
is exactly compatible with it.  `configuration.py` exposes the identities and
they were checked numerically before writing the solver.

## Parameters

The benchmark lists `h = 1` (perpendicular channel width), `D = h/cos(theta)`
(vertical width between the plates), `mu = 0.5`, `rho = 1`, `dP/L = 1.0`,
`U_max = 0.25`, `N = 32`, `dt = 0.15*dx`.  These are mutually consistent:
Eq. (48) peaks at

```
U_max = (dP/L) * h^2 / (8*mu) = 1.0 * 1 / (8 * 0.5) = 0.25
```

| quantity | value |
|---|---|
| `theta` | 30° (`cos = 0.866025`, `sin = 0.5`) |
| perpendicular width `h` | 1.0 |
| vertical width `D = h/cos` | 1.154701 |
| `rho`, `mu` | 1.0, 0.5 |
| `dP/L` | 1.0 |
| `U_max` | 0.25 |
| `N`, `dx` | 32, 0.03125 |
| `dt` | 0.0046875 (`0.15*dx`) |
| channel width in cells | 36.95 |
| grid | `32 x 72` on box `1 x 2.25` |
| `Re = rho*u_max*D/mu` | 0.5774 |
| Courant | 0.0375 |
| `nu*dt/dx^2` | 2.400 |

`nu*dt/dx^2 = 2.4` exceeds the explicit-diffusion limit of 0.25, but the
viscous term is treated **implicitly** by both solvers, so there is no
diffusion restriction here; only advection matters, and the Courant number is
small.

## The box and channel position

The benchmark box is `Omega = [0,1] x [-0.25,2]`, with the channel entering the
left face and leaving the right face.  Eq. (48) vanishes on `xi~ = 0` and
`xi~ = h`, i.e. the **lower wall passes through the origin**; on the left face
the channel therefore spans `y in [0, 1/cos(theta)] = [0, 1.1547]`, and at
`x = 1` it spans `[0.5774, 1.7321]` — the whole channel is inside the quoted
box (0.25 of clearance below the channel at the inlet, 0.268 above it at the
outlet).

Like every AFSI case, the demo translates the box so that its lower-left
corner sits on the origin:

```
X = x ,  Y = y + 0.25      =>   X in [0, 1],  Y in [0, 2.25]
```

so the channel's lower-left corner sits at `(0, 0.25)`.  At `N = 32` the box is
exactly `32 x 72` cells of `dx = 1/32` (no padding).  Both plates run the full
width and terminate **exactly** on the inlet and outlet faces, so both ends are
complete channel cross-sections.

`X_MIN`, `X_MAX`, `Y_MIN`, `Y_MAX` and `SHIFT_Y` are environment-overridable.

## Boundary conditions

| boundary | condition |
|---|---|
| inlet `X=0`, channel interval `Y in [0.25, 1.4047]` | velocity Dirichlet, **analytic** |
| outlet `X=1`, channel interval `Y in [0.8274, 1.9821]` | velocity Dirichlet, **analytic** |
| every other boundary segment | no-slip `u = 0` |
| the two plates inside the box | immersed-boundary penalty |

The openings carry the analytic **velocity** Dirichlet (the `f_body` route is
off by default) and every other boundary segment is no-slip; the inlet/outlet
dofs are selected by the geometric channel interval on those faces, so the
wall segments share the faces with the openings.

Pressure has no *physical* datum in the velocity-driven runs — only `grad(p)`
enters the momentum equations — but the solver pins ONE reference dof instead
of leaving the pure-Neumann system singular.  The IPCS stores the ACCUMULATED
correction (`p_ += phi`); with no datum the singular system lets that constant
drift every step, and the saved `pressure.xdmf` degenerated into a ~`7.8e11`
constant (measured 2026-09-28).  Since 2026-09-29 the dof 3 cells diagonally
inside the inlet opening is fixed to `p = 0` (on the corrected geometry: node
`(0.09375, 0.34375)`, dof 106), which makes the system non-singular and the
stored field physical: in-channel it equals `1.017 * p_analytic + 0.209`
(corr `0.99947`, measured on the 2026-09-30 run).  Pressure-driven runs
prescribe the analytic opening pressure instead (see below).

### Driving modes (2026-09-28; re-measured 2026-09-30)

As run, the flow is driven by the analytic **velocity** Dirichlet on the two
openings (the `f_body` route is off by default).  A **pressure**-driven variant
was added:

* `DRIVING=velocity` (default) — analytic velocity on the openings; one
  reference pressure dof is pinned to `p = 0` (see *Boundary conditions*).
* `DRIVING=pressure` — pressure Dirichlet on the openings (velocity free
  there); `P_FACE=exact` prescribes the linear analytic pressure
  `p = -DP_DL*((X-SHIFT_X) cos + (Y-SHIFT_Y) sin)` on each opening,
  `P_FACE=const` prescribes one constant per face.  IPCS uses the
  `ds_p`/`p_traction` fix (demo_424 scheme: the volume form drops the opening
  pressure-traction term, so the predictor would otherwise be inconsistent
  there).

Measured on the corrected benchmark geometry (T = 2 s, 427 steps, tether
`BETA=8e3`, ipcs; outputs in `velocity/`, `pressure/`): in-channel relative L2
= **0.46 % (velocity)** vs **11.9 % (pressure, exact)**.  Both are stable; the
pressure mode is less accurate **by construction**: the openings are oblique
30-degree cuts, and the predictor's remaining natural condition there
(`mu du/dn = 0`) is not what the exact solution satisfies on such a cut, so
the boundary mismatch propagates inward (end `max|u|` runs +6.8 % high).  On
the benchmark's (smaller) box the openings sit closer to the measurement
station, which is why the error is larger here than on the earlier extended
box (7.9 %).  For benchmark-grade accuracy with pressure driving, extend the
channel and/or impose the full exact traction (pressure + viscous part) on the
openings.

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
python generate_mesh.py               # (re)build the plate mesh; needed after
                                      # N / PLATE_THICKNESS / geometry changes
python main.py                        # 427 steps, T = 2 s (~15 s per run)
USE_IMPLICIT_DRAG=0 BETA=8e3 python main.py                     # tether route,
                                      # velocity-driven (default DRIVING)
USE_IMPLICIT_DRAG=0 BETA=8e3 DRIVING=pressure P_FACE=exact python main.py
```

Outputs go to `<demo>/<DRIVING>/` directly under the demo folder
(`velocity/`, `pressure/`): `velocity.xdmf/.h5`, `pressure.xdmf/.h5`,
`solid_coords.xdmf/.h5`, `solid_displacement.xdmf/.h5`, `history.csv`,
`verify.json`, `run.log`.

Environment overrides: `N`, `DP_DL`, `MU`, `THETA_DEG`, `X_MIN`, `X_MAX`,
`Y_MIN`, `Y_MAX`, `SHIFT_Y`, `DT_FACTOR`, `T_END`, `SMOKE`, `SMOKE_STEPS`,
`PLATE_DRAG`, `PLATE_DRAG_BAND`, `USE_IMPLICIT_DRAG`, `BETA`, `DAMP`,
`SOLVER`, `IB_DIRECT_LOAD`, `DRIVING`, `P_FACE`, `X_PROFILE`.

## Files

| file | description |
|---|---|
| `configuration.py` | parameters, exact solution, channel geometry |
| `verify.py` | analytic references, sampling, error metrics |
| `main.py` | box, boundaries, IB plates, time loop, report |
| `generate_mesh.py` | builds the 2-D triangular plate strips read by `main.py` |
| `aaa.md` | benchmark text excerpt (Li et al. §4.3.1 and Figs. 22/23) |
| `velocity/`, `pressure/` | outputs of the two driving-mode runs (2026-09-30) |
| `velocity_oldbox/`, `pressure_oldbox/` | the same runs on the pre-2026-09-30 geometry |

## Results (N = 32, `dx = 0.03125`, `dt = 0.15*dx`, T = 2)

Both driving modes, tether plates (`USE_IMPLICIT_DRAG=0 BETA=8e3`), ipcs, 427
steps; outputs in `velocity/` and `pressure/` (runs of 2026-09-30 on the
corrected benchmark geometry):

| metric | velocity-driven | pressure-driven (`exact`) |
|---|---|---|
| relative L2 error in the channel | **0.464 %** | 11.87 % |
| Linf error in the channel | 3.26e-3 m/s | 4.50e-2 m/s |
| profile relative L2 (at `x = 0.5`) | 0.538 % | 7.48 % |
| numerical `u_max` | 0.25072 m/s (**+0.29 %**) | 0.26710 m/s (+6.84 %) |
| `u_max` location in `xi` | 0.000 (centreline) | 0.000 |
| fluid `\|u\|` on the plates (max) | 3.39e-3 m/s | 5.83e-3 m/s |
| plate displacement from reference (max) | 1.34e-2 m | 2.02e-2 m |
| stored in-channel pressure vs analytic | `1.017*p_an + 0.209` (corr 0.99947) | `0.984*p_an - 0.136` (corr 0.99379) |
| elapsed | 15.5 s | 14.1 s |

Notes:

* The velocity-driven run reproduces the analytic peak on the centreline to
  +0.29 %; the residual `|u|` on the plates is the numerical boundary layer
  produced by the IB kernel (the exact profile vanishes there).
* The tether route leaves a slow plate **creep** (see *the tether route*
  below); over these 2-s runs the maximum displacement is ~0.4-0.7 cells.
  For a strictly stationary geometry use the implicit drag band
  (`USE_IMPLICIT_DRAG=1`).
* The profile sampling line at `x = 0.5` was fixed on 2026-09-30 (it used to
  ignore the coordinate shift and did not lie on the centreline); the profile
  metrics above are now meaningful.

### Why the whole-box error is not reported as an accuracy measure

The whole-box relative L2 error is ~0.93 with Linf ~1.27 m/s.  That is **not**
a solver error.  The exact parabola `A*((h/2)^2 - xi^2)` keeps growing
*outside* the channel -- in this box it reaches 1.27 -- while the physical flow
outside the plates is stagnant.  The metric is therefore dominated by a
spurious reference value, which is why the channel-restricted numbers are the
meaningful ones and the profile at `x = 0.5` is the benchmark's own choice.

## Figures

The archived example figure set (`plot/N32_ipcs/example_figures/`, removed in
the 2026-09-30 cleanup) can be regenerated from the XDMF outputs in
`velocity/` with the demo_424/plot workflow (meshio + off-screen rendering):

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

Pressure provides only `grad(p)` to the momentum equations, so plots show
`p - mean(p)`; with the reference dof pinned (see *Boundary conditions*) the
stored `p_` itself is physical (in-channel it matches the analytic pressure up
to an affine fit, see *Results*) rather than an arbitrary drifting constant.

## Status / open items

* **Fig. 23 cannot be reproduced as shipped** — the kernel-support comparison
  needs a selectable kernel in the IB layer, or an implicit treatment of the
  Peskin penalty force.  See above.
* Pressure-driven accuracy (~12 % in-channel) is limited by the oblique
  openings (see *Driving modes*); imposing the full exact traction there would
  help and is the natural next step.
* Only `N = 32` is implemented; the benchmark's fine grid is `N = 32` with
  `dx = L/N`, so a convergence sweep would vary `N` (and keep `dt = 0.15*dx`).
* The apparent `U_max` inconsistency reported in earlier revisions of this file
  was an artifact of a wrong channel placement (centreline instead of lower
  wall through the origin) and is resolved by the 2026-09-30 geometry
  correction.

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
With it the plates exerted essentially no force: in the archived (since
removed) `plot/N32_ipcs_smoke` run the plates were passive tracers carried
downstream at ~0.37 `u_max`, the fluid *outside* the channel reached 0.37
`u_max` (i.e. there
was no confined channel at all), and the in-channel error was 34.7 %.

`SPREAD_NORM` remains as a calibration knob (`none` = as shipped, the default;
`h2` = the old 1024x attenuation; `<number>` = explicit factor).

### What limits the explicit (tether) route — revised 2026-09-28

Two independent effects were conflated in the earlier analysis:

1. **A sign error (fixed 2026-09-28).**  The tether force used to be stored
   with the opposite sign on both solver branches (net effect `-J^T F`), so the
   spring *pushed the plates away* when the fluid displaced them — an
   unconditional amplifier at any stiffness.  Controlled check: `BETA = -8e3`
   (mathematically identical to the old sign) diverges to plate displacement
   209x `h/2` within 320 steps, while `BETA = +8e3` is stable and accurate.
   With the unified `-∫f·v` convention (same as demo_424) the tether is stored
   as the physical force and the plates are held: the 320-step smoke gives
   in-channel error **0.63 %**, plate slip <= 2 % of `u_max` — better than the
   drag band.  (Measured on the earlier extended box; on the corrected
   benchmark box the same recipe gives 0.46 %, see *Results*.)
2. **An explicit-coupling time-step budget (real).**  The frozen-force penalty
   carries a per-step gain `~ BETA*dt^2/rho`.  Measured stable at
   `BETA = 8e3` and `1.2e4` (`BETA*dt^2/rho = 0.31` and `0.47` at
   `dt = 0.2*DX`); larger stiffness approaches the explicit stability limit.

Within the budget the tether route is now usable, but the plates keep
**creeping** at the residual slip velocity (~0.1 cell/s; 1.3 cells after
`t = 10 s` at `BETA = 8e3`) — they are not pinned.  For a stationary geometry
use the **implicit** drag band (`USE_IMPLICIT_DRAG=1`, the default): measured
in-channel error **5.33 %** at `PLATE_DRAG = 1e4`, zero drift (reproduced on
the current code).

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
