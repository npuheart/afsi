# Flow past a rigid cylinder — IB-FE method

Standard Peskin Immersed Boundary feedback force coupling (same architecture
as demo_402 Turek FSI), with the cylinder modelled as a Neo-Hookean solid
disk of extremely high stiffness (steel-like, E ≈ 200 GPa).

## Method

```
Solve NS → interpolate u to solid → X += v*dt →
compute Neo-Hookean stress + penalty force → spread force back as f
```

The cylinder is made effectively rigid by:
1. Very high shear modulus (μ_s ≈ 7.7×10¹⁰ Pa, ~1000× stiffer than demo_402)
2. Strong penalty (β = 10¹² Pa) fixing the disk to its initial position

## Files

| File | Purpose |
|------|---------|
| `cylinder_solid.geo` | Gmsh geometry for the solid disk |
| `generate_mesh.py` | .geo → .xdmf/.h5 |
| `configuration.py` | Parameters |
| `main.py` | IB-FE time loop |

## Run

```bash
python generate_mesh.py       # creates cylinder_solid.xdmf
python main.py                # or mpirun -n <N> python main.py
```

## Expected results

At Re=100, the cylinder wake should show periodic vortex shedding
(Kármán vortex street). The solid should remain nearly undeformed
due to the high stiffness + penalty fixation.

## Tuning

- `mu_s`, `lambda_s`: stiffness (higher → more rigid, but may need smaller dt)
- `beta`: penalty strength for fixation
- `dt`: may need to be reduced for very stiff solids
