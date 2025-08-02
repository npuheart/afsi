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
solid_fiber_io = Function(Vs_io, name="fiber_io")

solid_coords.interpolate(lambda x: np.array([x[0], x[1], x[2]]))


import numpy as np

def fun_fiber_v1():
    # 读取文本文件到 float64 类型的 NumPy 数组
    data_f0 = np.loadtxt('/home/dolfinx/afsi/afsic/demo/demo_337/mesh/f0.txt', dtype=np.float64).reshape(-1, 3)
    data_s0 = np.loadtxt('/home/dolfinx/afsi/afsic/demo/demo_337/mesh/s0.txt', dtype=np.float64).reshape(-1, 3)
    data_cdm = np.loadtxt('/home/dolfinx/afsi/afsic/demo/demo_337/mesh/cdm.txt', dtype=np.int64)
    dict_fibers = {}
    for data in zip(data_f0, data_s0, data_cdm):
        # print(data)
        f0, s0, cdm = data
        dict_fibers[cdm] = np.hstack((f0, s0))

    return dict_fibers

import hashlib
class CoordinateDataMap:
    def __init__(self):
        self.data_map = []
    
    def add_data(self, coord):
        coord_key = self.hash_floats(coord)
        self.data_map.append(coord_key)
        return coord_key

    def get_data(self, coord):
        coord_key = self.hash_floats(coord)
        if coord_key in self.data_map:
            return self.data_map[coord_key]
        else:
            log.error(f"Coordinate {coord} not found in data map.")
        return None

    def __contains__(self, coord):
        return self.hash_floats(coord) in self.data_map

    def hash_floats(self, coords, precision=6):
        # str_repr = f"{round(coords[0], precision):.{precision}f},{round(coords[1], precision):.{precision}f},{round(coords[2], precision):.{precision}f}"
        str_repr = f"{round(100*coords[0]+10*coords[1]+coords[2], precision):.{precision}f}"
        return int(hashlib.sha256(str_repr.encode()).hexdigest()[:8], 16)


dict_fibers = fun_fiber_v1()

cdm = CoordinateDataMap()
# x_attr_f.attr("index_map").attr("size_local")) * nb::cast<int>(x_attr_f.attr("bs")) / 3

# print(solid_coords.x.index_map.size_local*solid_coords.function_space.dofmap.bs/3)
# ijnhu = solid_coords.x.index_map.size_local*solid_coords.function_space.dofmap.bs//3
# print(solid_coords.x.bs)

solid_coords = solid_coords.x.array.reshape(-1, 3)



print("Number of fibers:", len(dict_fibers))
for i, coord in enumerate(solid_coords):
    # if ijnhu <= i:
        # break
    coord_key = cdm.hash_floats(coord)
    print(coord_key, coord)
    if np.int64(coord_key) not in dict_fibers:
        raise KeyError(f"coord_key {coord_key} for coordinate {coord} not found in dict_fibers.")
    # If found, you can assign or process as needed
    # dict_fibers[coord_key]
    solid_fiber.x.array[3*i:3*i+3] = dict_fibers[coord_key][:3]

# print(dict_fibers)
# xieug = {}
# 
#     xieug[] = 
#     # print(coord)
#     # print(cdm.hash_floats(coord))
#     # print(cdm.__contains__(coord))

solid_fiber_io.interpolate(solid_fiber)
file_velocity = dolfinx.io.XDMFFile(structure.comm, "fiber.xdmf", "w")
file_velocity.write_mesh(structure)
file_velocity.write_function(solid_fiber_io, 0.0)




structure._geometry._cpp_object.x[:,0] = structure._geometry._cpp_object.x[:,0]/10.0 + 3.0
structure._geometry._cpp_object.x[:,1] = structure._geometry._cpp_object.x[:,1]/10.0 + 2.5
structure._geometry._cpp_object.x[:,2] = structure._geometry._cpp_object.x[:,2]/10.0 + 2.5

