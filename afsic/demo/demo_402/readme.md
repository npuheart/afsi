# Turek FSI2 benchmark

Reference: https://kratosmultiphysics.github.io/Examples/fluid_structure_interaction/validation/fsi_turek_FSI2/

2D channel flow around a fixed cylinder with a flexible flag attached to its rear. This
demo runs the benchmark's physical parameters (SI units) through this repo's own solver,
which is an **immersed boundary method** (fixed Cartesian fluid mesh + Lagrangian solid
mesh advected by interpolated fluid velocity, coupled via `IBMesh`/`IBInterpolation`) —
the same architecture as `demo_400` (turtle) and `demo_401` (sperm), not Kratos's
body-fitted ALE mesh.

## Parameters (`configuration.py`)

- Channel: 2.5 m x 0.41 m; cylinder D=0.1 m at (0.2, 0.2); flag 0.35 m x 0.02 m
- Fluid: rho_f=1000 kg/m^3, mu_f=1.0 Pa·s (Re=100), Ubar=1.0 m/s parabolic inflow, 2 s ramp
- Flag: Saint-Venant-Kirchhoff, E=1.4e6 Pa, nu=0.4 (mu_s=5e5 Pa, lambda_s=2e6 Pa)
- "Rigid" cylinder region: same constitutive law, stiffness scaled by `cyl_stiffness_factor`
- Gravity: g=2 m/s^2, applied only on the flag

## Known limitations vs. the published benchmark

- This is a **qualitative** replication, not a precision match to published displacement/
  frequency numbers.
- The IB scheme has no separate solid inertia term (classic massless Peskin IB) — the real
  FSI2 case relies partly on the flag's excess inertia (rho_s = 10x rho_f). `rho_s` here is
  only used for the gravity body force.
- The cylinder is approximated as a much stiffer elastic region, not an exact rigid
  constraint.
- `dt`, mesh resolution (`Nx`/`Ny`), and `cyl_stiffness_factor` are the most likely knobs to
  need empirical tuning for stability — start a short run before committing to the full `T`.

## Run

```bash
python generate_mesh.py   # turek.geo -> turek_mesh.xdmf/.h5
python main.py            # or mpirun -n <N> python main.py
```

## Note 

the density of and is the same
the solid is added with the same viscosity with fluid