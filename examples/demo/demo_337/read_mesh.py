from mpi4py import MPI
from petsc4py import PETSc
import dolfinx
import numpy as np
import json
import ufl

with open('/home/dolfinx/afsi/afsic/demo/demo_337/mesh/lv_ellipsoid/geometry/markers.json', 'r', encoding='utf-8') as file:
    mesh_markers = json.load(file)

with dolfinx.io.XDMFFile(MPI.COMM_WORLD, '/home/dolfinx/afsi/afsic/demo/demo_337/mesh/lv_ellipsoid/geometry/mesh.xdmf', "r", encoding=dolfinx.io.XDMFFile.Encoding.HDF5) as file:
    structure = file.read_mesh(name="Mesh")
    structure.topology.create_connectivity(structure.topology.dim-1, structure.topology.dim)
    ft = file.read_meshtags(structure, "Facet tags")
    endo_facets = ft.find(mesh_markers['ENDO'][0])
    base_facets = ft.find(mesh_markers['BASE'][0])
    epi_facets = ft.find(mesh_markers['EPI'][0])
    marked_facets = np.hstack([endo_facets, base_facets, epi_facets])
    marked_values = np.hstack([np.full_like(endo_facets, mesh_markers['ENDO'][0]), np.full_like(base_facets, mesh_markers['BASE'][0]), np.full_like(epi_facets, mesh_markers['EPI'][0])])
    sorted_facets = np.argsort(marked_facets)
    facet_tag = dolfinx.mesh.meshtags(structure, ft.dim, marked_facets[sorted_facets], marked_values[sorted_facets])
    facet_tag.name = ft.name

metadata = {"quadrature_degree": 5}
ds = ufl.Measure("ds", domain=structure, subdomain_data=facet_tag, metadata=metadata)
structure._geometry._cpp_object.x[:,0] = structure._geometry._cpp_object.x[:,0]/10.0 + 3.0
structure._geometry._cpp_object.x[:,1] = structure._geometry._cpp_object.x[:,1]/10.0 + 2.5
structure._geometry._cpp_object.x[:,2] = structure._geometry._cpp_object.x[:,2]/10.0 + 2.5

with dolfinx.io.XDMFFile(MPI.COMM_WORLD, 'mesh-scale.xdmf', "w", encoding=dolfinx.io.XDMFFile.Encoding.HDF5) as file:
    file.write_mesh(structure)
