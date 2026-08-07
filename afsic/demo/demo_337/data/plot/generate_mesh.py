"""Generate the idealized LV ellipsoid structure mesh for demo_337.

Runs inside the fenicsx-pulse Docker image (which ships
cardiac_geometries + pulse + dolfinx). Writes the mesh + markers
into ../mesh/lv_ellipsoid/geometry/ so that main.py can read it.

Usage (from data/plot/, inside the Docker container):

    docker compose up -d                     # start container
    docker exec -it fenicsx-pulse-container bash
    cd /repo && python generate_mesh.py

The generated files (mesh.xdmf, mesh.h5, markers.json, ...) land in
    ../mesh/lv_ellipsoid/geometry/
"""

import math
from pathlib import Path
from mpi4py import MPI

import cardiac_geometries

outdir = Path("../mesh/lv_ellipsoid")

# Same parameter set as the pulse-fenicsx benchmark (bench-ilv-*.py):
# after the /10 scaling + (3, 2.5, 2.5) shift in main.py, the LV fits
# inside the 5 x 5 x 5 fluid box.
cardiac_geometries.mesh.lv_ellipsoid(
    outdir=outdir,
    r_short_endo=7.0,
    r_short_epi=10.0,
    r_long_endo=17.0,
    r_long_epi=20.0,
    mu_apex_endo=-math.pi,
    mu_base_endo=-math.acos(5 / 17),
    mu_apex_epi=-math.pi,
    mu_base_epi=-math.acos(5 / 20),
    comm=MPI.COMM_WORLD,
)
print("Done creating geometry:", outdir.resolve())
