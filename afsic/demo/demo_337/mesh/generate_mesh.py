
# sudo docker run --rm -v $PWD:/home/shared -w /home/shared -it ghcr.io/finsberg/fenicsx-ldrb bash
# geox lv-ellipsoid lv --create-fibers


from pathlib import Path
from mpi4py import MPI
import dolfinx
from dolfinx import log
import cardiac_geometries
import cardiac_geometries.geometry

outdir = Path("lv_ellipsoid")
outdir.mkdir(parents=True, exist_ok=True)
geodir = outdir / "geometry"
if not geodir.exists():
    cardiac_geometries.mesh.lv_ellipsoid(outdir=geodir, create_fibers=True, fiber_space="P_2")

geo = cardiac_geometries.geometry.Geometry.from_folder(
    comm=MPI.COMM_WORLD,
    folder=geodir,
)

with open("f0.txt", "w") as f:
    f.write("\n".join(map(str, geo.f0.x.array[:])))

with open("s0.txt", "w") as f:
    f.write("\n".join(map(str, geo.s0.x.array[:])))