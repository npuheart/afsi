####################################################################################
######################### 二 维 ####################################################
####################################################################################
from petsc4py import PETSc
from afsic import IBMesh, IBInterpolation

Nx = 32
Ny = 30
ibmesh = IBMesh(0.0,1.0, 0.0,1.0, Nx,Ny,2)

from mpi4py import MPI
import ufl
import dolfinx
from dolfinx.mesh import CellType, GhostMode
from basix.ufl import element
import numpy as np
from dolfinx.fem import (Function, functionspace,
                         dirichletbc, locate_dofs_topological)
mesh = dolfinx.mesh.create_rectangle(
    comm=MPI.COMM_WORLD,
    points=((0.0, 0.0), (1.0, 1.0)),
    n=(Nx, Ny),
    cell_type=CellType.triangle,
    ghost_mode=GhostMode.shared_facet,
)

v_cg2 = element("Lagrange", mesh.topology.cell_name(),
                2, shape=(mesh.geometry.dim, ))
v_cg1 = element("Lagrange", mesh.topology.cell_name(),
                1, shape=(mesh.geometry.dim, ))
V = functionspace(mesh, v_cg2)
coords = Function(V)
fluid_empty = Function(V)
coords.interpolate(lambda x: np.array([x[0], x[1]]))

ibmesh.build_map(coords._cpp_object)
ibmesh.evaluate(0.5,0.5, coords._cpp_object)


# 固体
structure = dolfinx.mesh.create_rectangle(
    comm=MPI.COMM_WORLD,
    points=((0.3, 0.3), (0.7, 0.7)),
    n=(Nx, Ny),
    cell_type=CellType.triangle,
    ghost_mode=GhostMode.shared_facet,
)

v_cg2 = element("Lagrange", structure.topology.cell_name(),
                2, shape=(structure.geometry.dim, ))
v_cg1 = element("Lagrange", structure.topology.cell_name(),
                1, shape=(structure.geometry.dim, ))
V = functionspace(structure, v_cg2)
V_io = functionspace(structure, v_cg1)
solid_coords = Function(V)
solid_coords.interpolate(lambda x: np.array([x[0], x[1]]))


ib_interpolation = IBInterpolation(ibmesh)
ib_interpolation.evaluate_current_points(solid_coords._cpp_object)

solid_fun = Function(V)
solid_fun_io = Function(V_io)
ib_interpolation.fluid_to_solid(coords._cpp_object, solid_fun._cpp_object )
solid_fun.x.scatter_forward()

xdmf_file = dolfinx.io.XDMFFile(structure.comm, "x.xdmf", "w")
xdmf_file.write_mesh(structure)
solid_fun_io.interpolate(solid_fun)
xdmf_file.write_function(solid_fun_io, 0.1)


ib_interpolation.solid_to_fluid(fluid_empty._cpp_object, solid_fun._cpp_object)
fluid_empty.x.scatter_forward()


V1 = functionspace(mesh, v_cg1)
fluid_empty_out = Function(V1)
fluid_empty_out.interpolate(fluid_empty)


xdmf_file = dolfinx.io.XDMFFile(mesh.comm, "y.xdmf", "w")
xdmf_file.write_mesh(mesh)
xdmf_file.write_function(fluid_empty_out, 0.1)

####################################################################################