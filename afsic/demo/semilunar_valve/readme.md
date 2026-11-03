# semilunar_valve — aortic valve FSI with a rigid, tethered vessel wall (3D)

Three-dimensional semilunar (aortic) valve in an idealised aortic root, after the last example of
Black et al., *An open-source computational framework for immersed fluid–structure interaction
modeling using FEBio and MFEM* (arXiv:2601.08266, Sec. 4.5). The authors' model file
(`ValveFSI.fsm`, FEBio Model Repository project 135) is currently not downloadable, so the geometry
is rebuilt from the model the paper says it modified: the LS-DYNA ICFD example *Advanced: Heart
Valve* (https://lsdyna.ansys.com/heartvalve/), downloaded by `generate_mesh.py`.

Units: CGS (cm, g, s; pressure dyn/cm²). Flow along +z.

## Model

| | |
|---|---|
| fluid | Cartesian box \|x\|,\|y\| ≤ 2.0, z ∈ [−4.0, 14.2] (inlet 4 cm upstream of the root, length 18.2 cm), Q2/Q1, Chorin projection; computed shifted to the origin (see below) |
| blood | ρ = 1.1 g/cm³, μ = 0.036 dyn·s/cm² (paper) |
| vessel wall | immersed solid, 0.2 cm thick, **rigid: tether only**, f = β (X_ref − x), β = 2×10⁷ dyn/cm⁴ |
| leaflets | immersed solid, 0.015 cm thick, 4 elements across, compressible neo-Hookean ψ = μ/2 (I₁−3) − μ ln J + λ/2 (ln J)², E = 2×10⁶ dyn/cm², ν = 0.4 (paper) |
| leaflet attachment | attachment edges extended 0.10 cm into the wall; that extension (part 5) is tethered too (`VALVE_TETHER=1`) |
| inlet z = −4 | plug inflow v = Q(t)/A e_z inside the lumen (r < 1.15), zero outside the lumen |
| outlet z = 14.2 | p = 0, zero traction |
| box sides | no-slip |
| time | Δt = 1×10⁻⁴ s, one beat T = 1 s (paper) |

**Inflow waveform.** The paper uses the flow rate of Oks et al. (2022), given only as a figure.
`configuration.q_inlet()` is a placeholder (one systolic half-sine, Q_max = 250 cm³/s over 0.3 s,
zero in diastole). Supply the digitised curve as a two-column CSV `t,Q` via `INFLOW_CSV=...`.

**Coupling.** `IBInterpolation3D.solid_to_fluid` overwrites the fluid force, so each solid spreads
into its own fluid function and `main.py` sums the two before the next fluid step.

**Box at the origin.** The IB kernels in `IBMesh3D` (`interpolation`, `distribution`) are built from
the absolute particle coordinates, `LKernel({particle.x, particle.y, particle.z}, {dx, dy, dz})`,
without subtracting the box origin (x0, y0, z0). With a box that does not start at the origin,
points with negative coordinates map to out-of-range nodes and are skipped (zero velocity, no force)
and the others are shifted. In a first run with the box [−2, 2]² × [−4, 14.2], 76 % of the wall
points and 62 % of the leaflet points received no velocity. `main.py` therefore computes in
shifted coordinates (box [0, 4] × [0, 4] × [0, 18.2], solids moved by (2, 2, 4) on reading, axis at
(2, 2)) and checks at start-up that a uniform fluid velocity reaches at least 95 % of each solid's
points (99.2 % of the wall: the rest are the end-plane rings, whose kernels leave the grid; 100 % of
the leaflets). Output is in the shifted frame. The other demos use boxes at the origin and are not
affected; subtracting (x0, y0, z0) in `IBMesh3D` would remove the restriction.

## Differences from the paper

- Vessel: the paper's vessel is body-fitted and rigid (no-slip); here it is an immersed, tethered
  wall in a Cartesian box, so fluid also fills the gap between the wall and the box.
- Geometry is the LS-DYNA original at its own size: inflow diameter 2.3 cm (paper: 2.8 cm), leaflet
  height ≈ 1.53 cm (paper: 1.6 cm). How far the paper extended the leaflet edges is not stated.
- Inflow waveform: placeholder (see above).

## Run

```bash
conda activate afsi-dolfinx
cd afsic/demo/semilunar_valve

python generate_mesh.py                    # H=0.2: smoke-test grid and matching solids
STEPS=300 OUTPUT_EVERY=50 mpiexec -n 4 python main.py

H=0.1 python generate_mesh.py              # finer: solids follow H (marker spacing H/2)
H=0.1 mpiexec -n 16 python main.py
```

Environment overrides (see `configuration.py`): `H`, `H_SOLID`, `WALL_T`, `WALL_LAYERS`,
`VALVE_LAYERS`, `VALVE_REFINE`, `FORCE_ORDER`, `BETA`, `VALVE_TETHER`, `DT`, `T_END`, `STEPS`,
`OUTPUT_EVERY`, `Q_MAX`, `T_SYS`, `INFLOW_CSV`, `MESH_DIR`, `OUTPUT_PATH`, `LSDYNA_DIR`.

Cost: Q2 velocity on a hexahedral box is heavy. At H = 0.2 (20×20×91 cells, 0.92 M velocity
DOFs) one step takes about 40 s on one core and 10 GB of memory; use MPI. H = 0.1 has 8× the
cells.

## Output (`data/run_h<H>_dt<DT>/` or `OUTPUT_PATH`)

`velocity.xdmf`, `pressure.xdmf`, `wall.xdmf` and `valve.xdmf` (displacements), and
`history.csv`: prescribed Q, max |u|, max |p|, max wall and valve displacement, solid force norms.

## Files

| File | |
|---|---|
| `configuration.py` | parameters, inflow waveform, paths |
| `generate_mesh.py` | downloads the LS-DYNA example, writes the wall and valve meshes |
| `mesh_tools.py` | vessel surface (with straight extensions, gmsh re-meshing), wall extrusion, leaflet solid |
| `split.py` | conforming quad → triangle and prism → tetrahedron splits |
| `main.py` | fluid, tethered wall, neo-Hookean leaflets, IB coupling, time loop, output |
