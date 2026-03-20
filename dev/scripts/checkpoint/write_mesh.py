# pip install -i https://pypi.tuna.tsinghua.edu.cn/simple adios4dolfinx[test]


# # Checkpoint on input mesh
# As we have discussed earlier, one can choose to store function data in a way that
# is N-to-M compatible by using `adios4dolfinx.write_checkpoint`.
# This stores the distributed mesh in it's current (partitioned) ordering, and does
# use the original input data ordering for the cells and connectivity.
# This means that you cannot use your original mesh (from `.xdmf` files) or mesh tags
# together with the checkpoint. The checkpoint has to store the mesh and associated
# mesh-tags.

# An optional way of store an N-to-M checkpoint is to store the function data in the same
# ordering as the mesh. The write operation will be more expensive, as it requires data
# communication to ensure contiguous data being written to the checkpoint.
# The method is exposed as `adios4dolfinx.write_function_on_input_mesh`.
# Below we will demonstrate this method.

import logging
from pathlib import Path
from typing import Tuple

import ipyparallel as ipp


def locate_facets(x, tol=1.0e-12):
    return abs(x[0]) < tol


def create_xdmf_mesh(filename: Path):
    from mpi4py import MPI

    import dolfinx

    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 10, 10)
    facets = dolfinx.mesh.locate_entities_boundary(mesh, mesh.topology.dim - 1, locate_facets)
    facet_tag = dolfinx.mesh.meshtags(mesh, mesh.topology.dim - 1, facets, 1)
    facet_tag.name = "FacetTag"
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, filename.with_suffix(".xdmf"), "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_meshtags(facet_tag, mesh.geometry)
    print(f"{mesh.comm.rank + 1}/{mesh.comm.size} Mesh written to {filename.with_suffix('.xdmf')}")


mesh_file = Path("MyMesh.xdmf")
with ipp.Cluster(engines="mpi", n=4, log_level=logging.ERROR) as cluster:
    # Create a mesh and write to XDMFFile
    cluster[:].push({"locate_facets": locate_facets})
    query = cluster[:].apply_async(create_xdmf_mesh, mesh_file)
    query.wait()
    assert query.successful(), query.error
    print("".join(query.stdout))


# Next, we will create a function on the mesh and write it to a checkpoint.


