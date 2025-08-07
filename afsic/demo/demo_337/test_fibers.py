from afsic import unique_filename
from mpi4py import MPI
from petsc4py import PETSc

import time
import requests
import numpy as np

import dolfinx
from dolfinx import fem, default_scalar_type
from dolfinx import log
from dolfinx.fem import (Function, functionspace,
                         dirichletbc, locate_dofs_topological)
from dolfinx.mesh import CellType, GhostMode, locate_entities, meshtags
from basix.ufl import element

from ufl import (FacetNormal, Identity, Measure, TestFunction, TrialFunction, inv, ln, det,
                 as_vector, div, dot, ds, dx, inner, lhs, grad, nabla_grad, rhs, sym, system)
from dolfinx.fem import form, assemble_scalar

from afsic import IPCSSolver, ChorinSolver, TimeManager
from afsic import swanlab_init, swanlab_upload
from dolfinx.fem.petsc import create_vector, assemble_vector

# 定义本构关系
from NeoHookean import NeoHookeanMaterial
Material = NeoHookeanMaterial()

from PressureEndo import  calculate_pressure_linear, mmHg

from ReadFibers import fun_fiber_v1, CoordinateDataMap

from dolfinx.fem.petsc import (apply_lifting, assemble_matrix, assemble_vector,
                               create_vector, create_matrix, set_bc)


# Define the configuration for the simulation
config = {"nssolver": "chorinsolver",
          "project_name": "demo-337",
          "tag": "parallel",
          "velocity_order": 2,
          "force_order": 2,
          "pressure_order": 1,
          "num_processors": MPI.COMM_WORLD.size,
          "T": 0.1,
          "dt": 1/1000,
          "rho": 1.0,
          "Lx": 5.0,
          "Ly": 5.0,
          "Lz": 5.0,
          "Nx": 32,
          "Ny": 32,
          "Nz": 32,
          "Nl": 20,
          "mu": 0.01,
          "mu_s": 0.1,  # Solid elasticity
          "diastole_pressure": 8.0*mmHg,
          "systole_pressure": 110.0*mmHg,
          "beta": 5e6,
          }





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


v_cg2 = element("Lagrange", structure.topology.cell_name(),
                 config["force_order"], shape=(structure.geometry.dim, ))
v_cg1 = element("Lagrange", structure.topology.cell_name(),
                 1, shape=(structure.geometry.dim, ))
Vs = functionspace(structure, v_cg2)
Vs_io = functionspace(structure, v_cg1)

solid_coords = Function(Vs, name="solid_coords")
solid_coords_io = Function(Vs_io, name="solid_coords_io")
solid_force = Function(Vs, name="solid_force")
solid_force_io = Function(Vs_io, name="solid_force_io")
solid_velocity = Function(Vs, name="solid_velocity")
solid_fiber = Function(Vs, name="fiber")
solid_sheet = Function(Vs, name="sheet")
solid_fiber_io = Function(Vs_io, name="fiber_io")
solid_sheet_io = Function(Vs_io, name="sheet_io")
solid_coords.interpolate(lambda x: np.array([x[0], x[1], x[2]]))

# 这里假设所有进程都能访问到同样的纤维数据，每个进程都要读取完整的纤维数据，然后各个进程根据DoFs分配来提取对应的纤维数据
dict_fibers = fun_fiber_v1(
    '/home/dolfinx/afsi/afsic/demo/demo_337/mesh/f0.txt', 
    '/home/dolfinx/afsi/afsic/demo/demo_337/mesh/s0.txt', 
    '/home/dolfinx/afsi/afsic/demo/demo_337/mesh/cdm.txt')

cdm = CoordinateDataMap()
solid_coords_np = solid_coords.x.array.reshape(-1, 3)
for i, coord in enumerate(solid_coords_np):
    coord_key = cdm.hash_floats(coord)
    if np.int64(coord_key) not in dict_fibers:
        raise KeyError(f"coord_key {coord_key} for coordinate {coord} not found in dict_fibers.")
    solid_fiber.x.array[3*i:3*i+3] = dict_fibers[coord_key][:3]
    solid_sheet.x.array[3*i:3*i+3] = dict_fibers[coord_key][3:6]

solid_fiber_io.interpolate(solid_fiber)
solid_sheet_io.interpolate(solid_sheet)
file_velocity = dolfinx.io.XDMFFile(structure.comm, "fiber-1.xdmf", "w")
file_velocity.write_mesh(structure)
file_velocity.write_function(solid_fiber_io, 0.0)
file_velocity.write_function(solid_sheet_io, 1.0)

structure._geometry._cpp_object.x[:,0] = structure._geometry._cpp_object.x[:,0]/10.0 + 3.0
structure._geometry._cpp_object.x[:,1] = structure._geometry._cpp_object.x[:,1]/10.0 + 2.5
structure._geometry._cpp_object.x[:,2] = structure._geometry._cpp_object.x[:,2]/10.0 + 2.5

import ufl

solid_sheet_normal = ufl.cross(solid_sheet, solid_fiber)
u, v = ufl.TrialFunction(Vs), ufl.TestFunction(Vs)
a = form(inner(u, v) * dx)
L = form(inner(solid_sheet_normal, v) * dx)
# L = inner(solid_sheet_normal, v) * dx
        # a3 = form(rho * dot(u, v) * dx)
        # L3 = form(rho * dot(u_s, v) * dx - k * dot(nabla_grad(phi), v) * dx)
        # A3 = assemble_matrix(a3)
        # A3.assemble()
        # b3 = create_vector(L3)
# b = assemble_vector(L)
# b.scatter_reverse(la.InsertMode.add)


A = assemble_matrix(a)
# A.scatter_reverse()
A.assemble()
b = create_vector(L)

solver2 = PETSc.KSP().create(structure.comm)
solver2.setOperators(A)
solver2.setType(PETSc.KSP.Type.MINRES)
pc2 = solver2.getPC()
pc2.setType(PETSc.PC.Type.HYPRE)
pc2.setHYPREType("boomeramg")


n0 = Function(Vs, name="n0")

A.zeroEntries()
assemble_matrix(A, a)
A.assemble()
with b.localForm() as loc:
    loc.set(0)
assemble_vector(b, L)
b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
solver2.solve(b, n0.x.petsc_vec)


solid_fiber_io.interpolate(n0)
file_velocity.write_function(solid_fiber_io, 2.0)
