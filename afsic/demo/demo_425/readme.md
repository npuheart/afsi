# demo_425 — tethered aorta in a box, 3-D round tube

demo_424 extended from a 2-D planar idealisation to a genuine 3-D round pipe.

demo_424 cut the cylindrical aorta along a plane through its axis, which turned
the lumen into a parallel-plate channel and the wall into two flat strips.
demo_425 keeps the circular cross-section:

| | demo_424 (2-D) | demo_425 (3-D) |
|---|---|---|
| lumen | parallel-plate channel | circular pipe, radius `A_LUMEN` |
| wall | two flat strips | cylindrical shell, thickness `T_WALL` |
| occluder | flat membrane strip | coaxial disc, thickness `DISC_T` |
| fluid domain | box, width `1.5·2a` | square duct, side `BOX_SIDE` |
| solid outline | reads as the letter **H** | the same H, in an axial cut |
| IB machinery | `IBMesh` / `IBInterpolation` | `IBMesh3D` / `IBInterpolation3D` |

Everything else is carried over from demo_424 unchanged: the solid still has
**no constitutive law** (each Lagrangian point is held only by a tether
`f = β(X_ref − X)`), and the duct is still driven by a **pressure Dirichlet on
both open ends**.

## Scope of this first stage

This stage delivers the **geometry, the mesh generator and a small smoke test**.
It does *not* yet deliver a validated physics run — see *Status / open items*.

| Delivered | Not yet done |
|---|---|
| `configuration.py` — all geometry/physics, `p_inlet(t)` | full 0.4 s transient run |
| `generate_mesh.py` — structured solid mesh, both cases, with a volume guard | `verify.py` — analytic error metrics |
| `main.py` — 3-D IB coupling, 6 velocity + 2 pressure duct BCs, smoke test | refinement study (NY 45 → 90) |
| smoke test passing in both cases | figures (`example_figures/`) |
| | resolving the coupled disc/wall tether balance |

## Geometry

| Quantity | Value |
|---|---|
| aorta length `L_AORTA` | 0.1 m |
| lumen radius `A_LUMEN` | 0.015 m |
| wall thickness `T_WALL` | 0.002 m |
| duct side `BOX_SIDE` | 0.045 m |
| outer fluid layer `GAP` | 0.0055 m, uniform all around the tube |
| occluding disc | radius `A_LUMEN`, thickness `DISC_T` = 0.002 m, at mid-length |

**Why a square duct of side 0.045 m.** `BOX_SIDE = 2(a+t) + 2·GAP` is chosen so
the outer fluid layer around the round tube matches demo_424's box gap
(`0.0055 m`). A square duct with the pipe axis at its centre gives the same gap
in every direction and keeps the fluid cells cubic, and it puts all four duct
side walls at a constant distance from the tube — the cleanest 3-D counterpart
of the 2-D box.

**Grid.** Cube cells, `h = BOX_SIDE/NY`. As in demo_424 the duct is half a cell
longer than the aorta at each end, so the aorta's open ends sit at cell centres
in x and no pressure Dirichlet node coincides with a Lagrangian end node:

    NX = round(L_AORTA/h) + 1 ,  BOX_L = NX·h = L_AORTA + h ,  X_OFF = h/2

At the default `NY = 45`, `h = 1.0 mm`, `NX = 101`, duct `0.101 × 0.045 × 0.045 m`,
fluid mesh `101 × 45 × 45 = 204 525` cells.

## Solid mesh — how it is built

`generate_mesh.py` lays out a **structured node grid** `(i_a, i_r, i_t)`:

* `i_a` — axial index, with `r = a` and the two disc faces as grid planes
* `i_r` — radial layer index, `r = 0 … a` (core) then `a … a+t` (wall)
* `i_t` — angular index, `N_THETA_DIV` divisions around the full circle

Every solid cell is a hexahedron spanning one step in each index, emitted from
the block list in `configuration.solid_node_blocks()`. The blocks are
non-overlapping and together tile exactly the tube plus (closed case) the disc.

Two details that are easy to get wrong, and are therefore documented in the code:

1. **The axis is degenerate.** The four cells around `r = 0` share all four of
   their inner corners at one point. They are dropped, and the cells next to
   them become wedges (zero-volume tetrahedra are filtered out automatically).
2. **Hexahedra are split into 6 tetrahedra along the main diagonal 0–6.**
   DOLFINx has no built-in hex→tet conversion, and `create_mesh` does not accept
   hexahedra, so the split is done here. The rule is *globally conforming on a
   structured grid*: both hexahedra sharing a face pick the same face diagonal,
   so no hanging nodes appear. `_check_cube_rule()` verifies at import that the
   six tetrahedra tile a unit cube exactly with positive volume — an earlier
   hand-written table was silently degenerate and this check now makes that
   class of mistake impossible.

### Sizing

One target solid cell size drives both resolutions
(`SOLID_TARGET_FRACTION`, default 0.5 → `SOLID_TARGET = h/2 = 0.5 mm`):

* wall: `N_R_WALL = round(T_WALL/SOLID_TARGET)` = 4 layers
* inner region (disc + tube core): `N_R_CORE = round(A_LUMEN/SOLID_TARGET)` = 30
* angular: `N_THETA_DIV ≈ π(a+t)/(2·HS_R)` rounded to a multiple of 8 = 216

The inner region is sized from `h`, **not** from the finer wall spacing. This
matters: every inner layer is replicated over the whole tube length *and* the
full circumference, so sizing it from `HS_R` blew the closed-case mesh up to
7.8 M tetrahedra; sizing it from `SOLID_TARGET` gives 4.1 M.

### Volume guard

`generate_mesh.py` asserts that the assembled solid volume matches
`configuration.exact_solid_volume()` to `1e-2` relative. That reference is
computed analytically from the same block list, treating each cell as the
pseudotriangle ring of area

    0.5 · (r_hi² − r_lo²) · sin(2π/N_theta)

times `N_theta` (one sector → the full ring) times the axial step. Because the
mesh inscribes exactly these polygons in the true circles, agreement is at
machine precision, which is what makes the guard a genuine cross-check rather
than a tautology:

| case | cells | tets | volume | analytic | rel |
|---|---|---|---|---|---|
| open | 86 400 | 518 400 | 2.0103357595e-05 | 2.0103357595e-05 | −1.8e-12 |
| closed | 700 272 | 4 138 128 | 8.4825175028e-05 | 8.4825175028e-05 | −3.3e-12 |

This guard earned its keep: during development it caught a wrong tetrahedral
split, and a spurious core block that silently filled the lumen with solid in
the `open` case.

## Boundary conditions

* **four duct side walls** (`y = 0`, `y = BOX_SIDE`, `z = 0`, `z = BOX_SIDE`) —
  no-slip, `u = 0`
* **the two open ends** (`x = 0`, `x = BOX_L`) — pressure Dirichlet
  `p = p_in(t)` and `p = 0`, with **no velocity condition**; the natural
  condition there is zero traction
* `p_in(t)` ramps linearly `0 → Δp` over `RAMP_T = 0.05 s` then holds

As in demo_424 this is a prescribed-normal-traction (pressure-driven) open
boundary realised in the projection method: the pressure datum is an essential
condition on the pressure Poisson problem (the `bcp` argument). Because both
ends carry Dirichlet data, the pressure null space is fixed and no gauge point
is needed.

GAP_DRAG (the optional porous damping used in demo_424's closed case) is
available here too and defaults to **0** for the open case.

## Run

```bash
source /home/deepseek-harness/afsi/.tools/afsi-run.sh   # or activate afsi-dolfinx
cd afsic/demo/demo_425

CASE=open   NY=45 python generate_mesh.py && CASE=open   NY=45 python main.py
CASE=closed NY=45 python generate_mesh.py && CASE=closed NY=45 python main.py

# smoke test: a few steps only
CASE=closed NY=45 SMOKE=1 SMOKE_STEPS=3 python main.py
```

Environment overrides: `CASE`, `NY`, `NZ`, `T_END`, `DT`, `RAMP_T`, `SOLVER`,
`DP_MMHG`, `BETA`, `SOLID_TARGET_FRACTION`, `N_THETA_DIV`, `SMOKE`,
`SMOKE_STEPS`, `GAP_DRAG`.

## Files

| File | Description |
|---|---|
| `configuration.py` | every parameter, the layer/block definitions, `p_inlet(t)` |
| `generate_mesh.py` | structured solid mesh (tube ± disc) + volume guard |
| `main.py` | fluid duct, 8 BCs, tether-only solid, 3-D IB coupling, smoke test |

## Smoke test results

Both cases, `NY=45`, `SOLVER=ipcs`, `SMOKE=1 SMOKE_STEPS=3`, `dt = 2e-4`:

| Check | `open` | `closed` |
|---|---|---|
| solid tets | 518 400 | 4 138 128 |
| fluid cells / velocity dofs | 204 525 / 1 681 043 | same |
| fields finite | yes | yes |
| inlet Dirichlet satisfied | `4.4e-10 Pa` | `3.1e-09 Pa` |
| outlet Dirichlet satisfied | exactly 0 | exactly 0 |
| solid responds to pressure | wall moves | disc `max‖u‖ = 8.0e-4 m/s` |
| setup: fluid + solver | 41.5 s | 41.2 s |
| setup: solid mesh load | 22.3 s | 22.3 s |
| setup: IB mesh + coords | 8.0 s | 8.8 s |
| **setup total** | **50 s** | **73 s** |
| **total (3 steps)** | **822 s** | **887 s** |

The physics moves the right way: with `p_in > 0` upstream and `p = 0`
downstream the sealed disc is pushed downstream, which is demo_424's closed-case
behaviour. At step 2 the closed case gives `max|u| = 2.4966e-02 m/s` and
`p ∈ [0, 1.968e-1] Pa` — identical to the values obtained from a
four-times-coarser solid mesh, i.e. refining the solid does not perturb the
early-time solution.

### Cost model (important for planning)

Setup is cheap and scales gently with the solid: 73 s for the 4.1 M-tet closed
mesh against 50 s for the 0.5 M-tet open one (solid load 22 s either way).

**The time loop is the bottleneck.** Both cases take ~820 s for 3 steps, i.e.
about 270 s per step, essentially independent of the solid mesh size. That is
the fluid solve: 1.68 M velocity dofs with PETSc's default `BCGS`+`Jacobi`
momentum solve and hypre BoomerAMG on the pressure Poisson problem, assembled
from scratch every step. A full `T = 0.4 s` run is 2000 steps, which at this
rate is **~150 hours** — so the full transient / refinement study needs a
faster fluid solve (better preconditioners, larger `dt`, or coarser `NY`)
before it is practical.

## Status / open items

* **Full transient run not yet done.** `T = 0.4 s` at `dt = 2e-4` is 2000 steps;
  the smoke test covers 2–3 steps only.
* **No `verify.py` yet.** The analytic references demo_424 used must be
  re-derived for the round geometry. In the *open* case the lumen is a circular
  pipe, so the fully developed profile is the **Hagen–Poiseuille parabola**

      u(r) = G/(4μ)·(a² − r²) ,   Q = π G a⁴/(8μ)

  instead of demo_424's `G/(2μ)(a² − y²)`, `2Ga³/(3μ)`. The outer passage is
  now an *annulus* between the duct wall and the tube, not a plane channel, and
  its exact solution is the annular-Poiseuille profile — worth deciding whether
  to compare against it or to keep demo_424's regularisation of damping the
  outer region.
* **The disc's tether equilibrium is not the 1-D estimate.** demo_424 used
  `δ = Δp/(β·T_WALL)`. In 3-D the sealed disc is held by *two* springs in
  series: the disc itself and the annulus of wall it is attached to, which is
  pulled inward by the disc and resists through its own tether. `DISC_DELTA_SCALE`
  in `configuration.py` is therefore only a scale, **not** the expected
  equilibrium; the coupled balance must be derived before any δ comparison.
* **Solid/fluid resolution mismatch.** At `NY=45` the solid's angular cell is
  ~0.35 mm against a fluid `h = 1.0 mm`, and the closed mesh is 4.1 M tets for
  205 k fluid cells. The solid could be coarsened substantially
  (`SOLID_TARGET_FRACTION`) before the two are well matched.
* **Duct side walls are close.** With `BOX_SIDE = 0.045` the duct wall is only
  5.5 mm from the tube (5.5 fluid cells). A wider duct, or exploiting symmetry,
  would reduce the 3-D wall effect on the outer passage.
* Only `NY = 45` has been generated; `NY = 90` (wall 8 cells, `h = 0.5 mm`)
  would be the refinement step.
