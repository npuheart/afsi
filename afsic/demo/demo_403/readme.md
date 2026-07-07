# demo_403 — Beam in Cross Flow (Section 4.5)

Reference: Tuković et al. (2018), *OpenFOAM Finite Volume Solver for Fluid-Solid Interaction*, Section 4.5.

## Case Description

3D channel flow over an elastic thick plate attached to the bottom surface.
Half-domain simulation with symmetry plane at the channel centre-line.

| Parameter | Value (SI) | Value (CGS) |
|---|---|---|
| Channel | 1.5 × 0.8 × 0.8 m³ | 150 × 40 × 40 cm³ (half) |
| Plate | 0.1 × 0.2 × 0.4 m³ | 10 × 20 × 20 cm³ (half) |
| Peak inlet velocity | 0.2 m/s | 20 cm/s |
| Fluid density | 1000 kg/m³ | 1 g/cm³ |
| Fluid viscosity | 1.0 Pa·s | 10 dyne·s/cm² |
| Solid Young's modulus | 1.4 MPa | 1.4×10⁷ dyne/cm² |
| Solid Poisson's ratio | 0.4 | 0.4 |
| Reynolds number | 40 | (based on plate height) |

## Run

```bash
# 1. Generate solid plate mesh
python generate_mesh.py

# 2. Run simulation (serial or parallel)
python main.py
# or: mpirun -n 4 python main.py
```

## Notes

- Uses **massless Peskin IB** (no solid inertia) — this is a steady-state case,
  so the lack of solid inertia is less critical than for oscillatory benchmarks.
- The plate bottom (y=0) is fixed via a penalty constraint.
- The original Richter benchmark uses Saint Venant-Kirchhoff; here we approximate
  with Neo-Hookean (acceptable for the small strains expected at Re=40, E=1.4 MPa).
- The modified form (U=0.3 m/s, E=10 kPa) produces larger displacements and may
  be more numerically tractable for the IB method.