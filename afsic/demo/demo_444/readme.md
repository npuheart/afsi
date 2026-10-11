# demo_444 — Bonet-type torsion (initial serial implementation)

Number reserved: **444**. This is the three-dimensional immersed beam torsion
case of Section 4.4 of `arXiv:2111.09958v2`, based on Bonet and Vadala-Roth.
It is an implementation under verification, not yet a validated 450-degree result.

## Model

CGS units. Fluid cube `[0,9]^3`, no-slip walls, pressure fixed at the origin.
Beam `[4,5] × [1.5,7.5] × [4,5]`; its long axis is y.
The lower y face is held by a displacement penalty; the upper y face is rotated
about `(x,z)=(4.5,4.5)` by the right-handed y-axis rotation matrix.
The target angle increases linearly to `2.5*pi` at 2 s, then stays constant to 5 s.
Both faces use the same surface penalty; other solid faces have natural traction.

Modified Mooney–Rivlin energy per reference volume:

`W = c1*(J^(-2/3)*I1-3) + c2*(J^(-4/3)*I2-3) + kappa/2*(ln J)^2`.

`c1=c2=9000`, `kappa=168000`, `rho=1`, `mu=.04`.
PK1 stress is obtained by UFL differentiation, with **no half factor** on c1/c2.
See equations (47)–(48) of https://arxiv.org/pdf/1811.06620 .
Reference-volume solid damping `-eta_B*U`, `eta_B=3600`, follows the later
2111.09958 benchmark; the earlier 1811.06620 torsion case has no body damping.

Fluid Q2/Q1, hexahedral solid Q1 (Q2 selectable), Chorin projection,
explicit solid advection. The solid weak force vector is spread and inserted
as a direct fluid load with the Eulerian **nodal** volume `(h/2)^3`.
No extra fluid weak body-force term is applied.

## Bounded execution

Use the `afsi-dolfinx` Python environment; initial implementation is **serial**.
From this directory, for example:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 AFSI_OFFLINE=1 \
OUTPUT_PATH=/tmp/bonet_444 N=6 M=6 THETA=.1 TL=.02 T=.03 \
MAX_STEPS=100 MAX_SECONDS=50 python main.py
```

Repeat the same command to resume automatically. Never reuse the output directory
with changed parameters or changed source. Full solver and solid fields are saved
atomically; source hash and parameters guard against incompatible restarts.
`MAX_STEPS` is a per-invocation cap, not the final simulation length.
`MAX_SECONDS` bounds the time loop; setup/JIT and checkpoint writing are additional.

Defaults: `N=18`, `M=12` longitudinal cells, `max(1,round(M/6))` cells across each
side, `ORDER=1`, `T=5`, `TL=2`, `THETA=2.5*pi`, `MAX_STEPS=200`, `OUT_EVERY=20`.
The provisional conservative time step is `min(.001*h,.2/max(eta_B,1))`;
this is **not a proven stability bound**. Set `DT` explicitly for convergence work.
`BETA` defaults to `1.25*h/DT`; keep `BETA` fixed when isolating time-step errors.
Other controls: `ETA_B`, `MAX_SECONDS`, `OUTPUT_PATH`.

Outputs: `parameters.json`, `metrics.csv`, `checkpoint.npz`, `solid_*.npz`,
`status.json`. Monitors include volume, sampled J extrema, elastic energy,
fluid speed, endpoint penalty error, probe displacement and PETSc convergence.
J extrema are sampled at 27 reference points per cell, not certified cell extrema.
The diagnostic probe is the mesh node nearest `[5,4.5,5]`; its identity with the
published encircled measurement location is **not established**.

## Verification and remaining work

Run `python verify_material.py` for identity, rigid rotation, objectivity,
simple shear and a finite-difference energy/stress check.
Short coupled tests and checkpoint comparisons are recorded in the paper workspace
`run_results/2026-10-10/formal/BONET_444_IMPLEMENTATION.md`.

Before using results in the manuscript: run the full loading/relaxation period;
check endpoint errors, positive J, force balance, mesh/time-step convergence;
identify the published probe precisely and compare with reference values.
MPI and formal Q2 convergence studies remain unverified.
