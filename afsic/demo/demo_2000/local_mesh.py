from fenics import VectorFunctionSpace, FunctionSpace, Point
from fenics import DirichletBC, Expression, Constant, interpolate
from fenics import BoxMesh, Mesh, MeshEditor, CellType, XDMFFile
from ibfenics1.cpp import IBMesh, IBInterpolation
import numpy as np
import os

# Parameters
nv = 0.01
T = 10.0
dt = 2e-4
num_steps = int(T/dt)
rho = 1.0
Nl = 20
Ne = 32
dt_minimum = 1e-5

# Mesh
order_velocity = 2
order_pressure = 1
order_displacement = 2


ib_mesh = IBMesh([Point(0, 0), Point(1, 1)], [Ne, Ne], order_velocity)
fluid_mesh = ib_mesh.mesh()
home_dir = os.path.expanduser("~")
solid_mesh = Mesh()
with XDMFFile(home_dir + f"/mesh-benchmark/B011-circle/mesh/circle_{Nl}.xdmf") as infile:
    infile.read(solid_mesh)

# Define function spaces
Qf = FunctionSpace(fluid_mesh, "P", order_pressure)
Vf = VectorFunctionSpace(fluid_mesh, "P", order_velocity)
Vs = VectorFunctionSpace(solid_mesh, "P", order_displacement)

# 初始化
uf = interpolate(Expression(("x[0]", "x[1]"), degree=2), Vf)
ib_mesh.build_map(uf._cpp_object)
inter = IBInterpolation(ib_mesh, solid_mesh)
us = interpolate(Expression(("x[0]", "x[1]"), degree=2), Vs)
inter.evaluate_current_points(us._cpp_object)





def calculate_fluid_boundary_conditions(Vf, Qf):
    flow_velocity = Expression(("1.0", "0.0"), degree=1)
    bcu_inflow = DirichletBC(Vf, flow_velocity, "near(x[1],1.0)")
    bcu_wall = DirichletBC(Vf, Expression(("0.0", "0.0"), degree=1), 
                           "near(x[0],0.0) || near(x[0],1) || near(x[1],0.0)")
    bcp_outlet = DirichletBC(Qf, Expression("0.0", degree=1), "near(x[0],1.0) && near(x[1],1.0)", "pointwise")
    bcu = [bcu_inflow, bcu_wall]
    bcp = [bcp_outlet]
    return bcu, bcp

