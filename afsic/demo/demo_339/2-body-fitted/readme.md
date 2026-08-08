# Flow past a cylinder — body-fitted mesh

The cylinder is a hole in the fluid mesh.  No-slip is a Dirichlet BC
on the cylinder boundary.  No immersed boundary method.

## Gmsh curve tags

| Tag | Boundary |
|-----|----------|
| 11  | Inlet (parabolic inflow) |
| 12  | Outlet (p=0) |
| 13  | Bottom wall (no-slip) |
| 14  | Top wall (no-slip) |
| 15  | Cylinder surface (no-slip) |

## Run

```bash
python generate_mesh.py       # creates channel_hole.msh
python main.py                # or mpirun -n <N> python main.py
```
