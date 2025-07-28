from mpi4py import MPI

import gmsh  # type: ignore

from dolfinx.io import XDMFFile, gmshio
def create_mesh(comm: MPI.Comm, model: gmsh.model, name: str, filename: str, mode: str):
    """Create a DOLFINx from a Gmsh model and output to file.

    Args:
        comm: MPI communicator top create the mesh on.
        model: Gmsh model.
        name: Name (identifier) of the mesh to add.
        filename: XDMF filename.
        mode: XDMF file mode. "w" (write) or "a" (append).
    """
    mesh_data = gmshio.model_to_mesh(model, comm, rank=0)
    mesh_data.mesh.name = name
    if mesh_data.cell_tags is not None:
        mesh_data.cell_tags.name = f"{name}_cells"
    if mesh_data.facet_tags is not None:
        mesh_data.facet_tags.name = f"{name}_facets"
    if mesh_data.ridge_tags is not None:
        mesh_data.ridge_tags.name = f"{name}_ridges"
    if mesh_data.peak_tags is not None:
        mesh_data.peak_tags.name = f"{name}_peaks"
    with XDMFFile(mesh_data.mesh.comm, filename, mode) as file:
        mesh_data.mesh.topology.create_connectivity(2, 3)
        mesh_data.mesh.topology.create_connectivity(1, 3)
        mesh_data.mesh.topology.create_connectivity(0, 3)
        file.write_mesh(mesh_data.mesh)
        if mesh_data.cell_tags is not None:
            file.write_meshtags(
                mesh_data.cell_tags,
                mesh_data.mesh.geometry,
                geometry_xpath=f"/Xdmf/Domain/Grid[@Name='{name}']/Geometry",
            )
        if mesh_data.facet_tags is not None:
            file.write_meshtags(
                mesh_data.facet_tags,
                mesh_data.mesh.geometry,
                geometry_xpath=f"/Xdmf/Domain/Grid[@Name='{name}']/Geometry",
            )
        if mesh_data.ridge_tags is not None:
            file.write_meshtags(
                mesh_data.ridge_tags,
                mesh_data.mesh.geometry,
                geometry_xpath=f"/Xdmf/Domain/Grid[@Name='{name}']/Geometry",
            )
        if mesh_data.peak_tags is not None:
            file.write_meshtags(
                mesh_data.peak_tags,
                mesh_data.mesh.geometry,
                geometry_xpath=f"/Xdmf/Domain/Grid[@Name='{name}']/Geometry",
            )



def gmsh_sphere(model: gmsh.model, name: str, center=[0.6,0.5,0.5], radius=0.2,  mesh_size=0.03) -> gmsh.model:
    """Create a Gmsh model of a sphere and tag sub entitites
    from all co-dimensions (peaks, ridges, facets and cells).

    Args:
        model: Gmsh model to add the mesh to.
        name: Name (identifier) of the mesh to add.

    Returns:
        Gmsh model with a sphere mesh added.

    """
    model.add(name)
    model.setCurrent(name)

    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)

    sphere = model.occ.addSphere(center[0], center[1], center[2], radius, tag=1)

    # Synchronize OpenCascade representation with gmsh model
    model.occ.synchronize()

    # Add physical tag for sphere
    model.add_physical_group(dim=3, tags=[sphere], tag=1)

    # Embed all sub-entities from the GMSH model into the sphere and tag
    # them
    for dim in [0, 1, 2]:
        entities = model.getEntities(dim)
        entity_ids = [entity[1] for entity in entities]
        model.mesh.embed(dim, entity_ids, 3, sphere)
        model.add_physical_group(dim=dim, tags=entity_ids, tag=dim)

    # Generate the mesh
    model.mesh.generate(dim=3)
    return model


gmsh.initialize()
model = gmsh.model()
model = gmsh_sphere(model, "Sphere")
model.setCurrent("Sphere")

model_rank = 0
gdim = 3
mesh_data = gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, model_rank, gdim=gdim)
mesh = mesh_data[0]
ct = mesh_data[1]
ft = mesh_data[2]
ft.name = "Facet markers"

import dolfinx
with dolfinx.io.XDMFFile(MPI.COMM_WORLD, 'mesh-341.xdmf', "w", encoding=dolfinx.io.XDMFFile.Encoding.HDF5) as file:
    file.write_mesh(mesh)
    file.write_meshtags(ft, mesh.geometry)