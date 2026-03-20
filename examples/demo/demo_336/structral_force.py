from afsic import unique_filename
from mpi4py import MPI
from petsc4py import PETSc

import time
import requests
import numpy as np

from dolfinx import la

import dolfinx
from dolfinx.fem import (Function, functionspace,
                         dirichletbc, locate_dofs_topological, assemble_vector)
from dolfinx.mesh import CellType, GhostMode, locate_entities, meshtags
from basix.ufl import element

from ufl import (FacetNormal, Identity, Measure, TestFunction, TrialFunction, inv, ln, det,
                 as_vector, div, dot, ds, dx, inner, lhs, grad, nabla_grad, rhs, sym, system)
from dolfinx.fem import form

from afsic import IPCSSolver, TimeManager
from afsic import swanlab_init, swanlab_upload
from dolfinx.fem.petsc import create_vector, assemble_vector


config = {"nssolver": "ipcssolver",
          "project_name": "demo-2000",
          "tag": "parallel",
          "velocity_order": 2,
          "force_order": 1,
          "pressure_order": 1,
          "T": 10.0,
          "dt": 1/200,
          "rho": 1.0,
          "Lx": 1.0,
          "Ly": 1.0,
          "Nx": 32,
          "Ny": 32,
          "Nl": 20,
          "mu": 0.001,
          }


with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "/home/dolfinx/afsi/data/336-lid-driven-disk/mesh/circle_20.xdmf", "r", encoding=dolfinx.io.XDMFFile.Encoding.HDF5) as file:
    structure = file.read_mesh()

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

# 定义弱形式
dVs = TestFunction(Vs)
mu_s = 0.1
lambda_s = 100

FF = grad(solid_coords)

solid_coords.interpolate(lambda x: np.array([x[0]*x[0]*x[1], x[1]*x[0]*x[1]]))
L_hat = form(- inner((FF-inv(FF).T), grad(dVs))*dx)
b1 = create_vector(L_hat)
solid_force.x.scatter_forward()


assemble_vector(b1, L_hat)
b1.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
with b1.getBuffer() as arr:
    solid_force.x.array[:len(arr)] = arr[:]

solid_force_norm = dolfinx.la.norm(solid_force.x, dolfinx.la.Norm.l2)
print(f"Initial solid force norm: {solid_force_norm}")

file_solid = dolfinx.io.XDMFFile(structure.comm, "x.xdmf", "w")
file_solid.write_mesh(structure)
file_solid.write_function(solid_force, 0.0)
