
import logging
from pathlib import Path
from typing import Tuple
from mpi4py import MPI

import ipyparallel as ipp

def f(x):
    return (x[0] + x[1]) * (x[0] < 0.5), x[1], x[2] - x[1]


def write_function(
    mesh_filename: Path, function_filename: Path, element: Tuple[str, int, Tuple[int,]]
):

    import dolfinx

    import adios4dolfinx

    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, mesh_filename, "r") as xdmf:
        mesh = xdmf.read_mesh()
    V = dolfinx.fem.functionspace(mesh, element)
    u = dolfinx.fem.Function(V)
    u.interpolate(f)

    adios4dolfinx.write_function_on_input_mesh(
        function_filename.with_suffix(".bp"),
        u,
        mode=adios4dolfinx.adios2_helpers.adios2.Mode.Write,
        time=0.0,
        name="Output",
    )
    print(
        f"{mesh.comm.rank + 1}/{mesh.comm.size} Function written to ",
        f"{function_filename.with_suffix('.bp')}",
    )




# Read in mesh and write function to file
mesh_file = Path("MyMesh.xdmf")
element = ("DG", 4, (3,))
function_file = Path("MyFunction.bp")
write_function(mesh_file, function_file, element)


# def verify_checkpoint(
#     mesh_filename: Path, function_filename: Path, element: Tuple[str, int, Tuple[int,]]
# ):
#     from mpi4py import MPI

#     import dolfinx
#     import numpy as np

#     import adios4dolfinx

#     with dolfinx.io.XDMFFile(MPI.COMM_WORLD, mesh_filename, "r") as xdmf:
#         in_mesh = xdmf.read_mesh()
#     V = dolfinx.fem.functionspace(in_mesh, element)
#     u_in = dolfinx.fem.Function(V)
#     adios4dolfinx.read_function(function_filename.with_suffix(".bp"), u_in, time=0.0, name="Output")

#     # Compute exact interpolation
#     u_ex = dolfinx.fem.Function(V)
#     u_ex.interpolate(f)

#     np.testing.assert_allclose(u_in.x.array, u_ex.x.array)
#     print(
#         "Successfully read checkpoint onto mesh on rank ",
#         f"{in_mesh.comm.rank + 1}/{in_mesh.comm.size}",
#     )


# # Verify checkpoint by comparing to exact solution

# with ipp.Cluster(engines="mpi", n=5, log_level=logging.ERROR) as cluster:
#     cluster[:].push({"f": f})
#     query = cluster[:].apply_async(verify_checkpoint, mesh_file, function_file, element)
#     query.wait()
#     assert query.successful(), query.error
#     print("".join(query.stdout))
