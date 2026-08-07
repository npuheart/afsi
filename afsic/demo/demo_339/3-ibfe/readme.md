# Flow past a cylinder — IB-FE immersed boundary method

Neo-Hookean solid disk + penalty fixation to approximate a rigid cylinder.
Standard Peskin IB feedback force coupling (same as demo_402).

## Run

```bash
python generate_mesh.py       # creates cylinder_solid.xdmf
python main.py                # or mpirun -n <N> python main.py
```

## Notes

- The solid disk is ~1000× stiffer than a typical elastic solid
  (E ≈ 200 GPa) with strong penalty fixation (β = 10¹² Pa)
- The fluid mesh is a full rectangle — the cylinder is IMMERSED,
  not a hole
