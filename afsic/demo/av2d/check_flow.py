"""Uniform traction pressure must produce zero velocity, without a pressure pin."""
import argparse
import numpy as np
import ufl
from mpi4py import MPI
from dolfinx import fem,mesh
from basix.ufl import element,mixed_element
from petsc4py import PETSc
from coupled import CoupledFlow
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--implicit',action='store_true')
args=parser.parse_args()
bg=mesh.create_rectangle(MPI.COMM_SELF,[[0.,0.],[1.,1.]],[8,8],cell_type=mesh.CellType.quadrilateral)
W=fem.functionspace(bg,mixed_element([element('Lagrange','quadrilateral',2,shape=(2,)),element('Lagrange','quadrilateral',1)]))
V,vm=W.sub(0).collapse();Q,pm=W.sub(1).collapse()
bg.topology.create_connectivity(1,2)
walls=mesh.locate_entities_boundary(bg,1,lambda x:np.isclose(x[0],0)|np.isclose(x[0],1))
ends=mesh.locate_entities_boundary(bg,1,lambda x:np.isclose(x[1],0)|np.isclose(x[1],1))
bcs=[fem.dirichletbc(fem.Function(V),fem.locate_dofs_topological((W.sub(0),V),1,walls),W.sub(0)),fem.dirichletbc(PETSc.ScalarType(0),fem.locate_dofs_topological(W.sub(0).sub(0),1,ends),W.sub(0).sub(0))]
tags=mesh.meshtags(bg,1,np.sort(ends),np.ones(len(ends),dtype=np.int32))
traction=fem.Function(Q);traction.x.array[:]=100
solver=CoupledFlow(W,V,Q,(vm,pm),bcs,5e-6,traction,ufl.Measure('ds',domain=bg,subdomain_data=tags)(1),implicit_convection=args.implicit)
solver.ib_load.set(0);solver.solve_one_step()
assert np.abs(solver.u_n.x.array).max()<1e-9
np.testing.assert_allclose(solver.p_.x.array,100,rtol=1e-9,atol=1e-8)
print('Uniform pressure equilibrium passed; max velocity:',np.abs(solver.u_n.x.array).max())
