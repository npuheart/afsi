"""Serial mixed Q2/Q1 backward-Euler flow with pressure traction boundaries."""
import numpy as np
import ufl
from dolfinx import fem
from dolfinx.fem.petsc import assemble_matrix,assemble_vector,create_vector,set_bc
from petsc4py import PETSc

class CoupledFlow:
    def __init__(self,W,V,Q,maps,bcs,dt,traction,ds):
        self.V,self.Q=V,Q
        self.vm,self.pm=maps
        self.u_n=fem.Function(V);self.u_=fem.Function(V);self.p_=fem.Function(Q)
        self.f=fem.Function(V);self.solution=fem.Function(W)
        u,p=ufl.TrialFunctions(W);v,q=ufl.TestFunctions(W)
        # Same density and viscosity as AV2D, CGS units.
        a=ufl.inner(u,v)/dt*ufl.dx + .035*ufl.inner(ufl.grad(u),ufl.grad(v))*ufl.dx
        a+=(-p*ufl.div(v)+q*ufl.div(u))*ufl.dx
        # Explicit skew convection limits energy production by discrete divergence.
        conv=ufl.dot(ufl.grad(self.u_n),self.u_n)+.5*ufl.div(self.u_n)*self.u_n
        rhs=ufl.inner(self.u_n/dt-conv,v)*ufl.dx
        normal=ufl.FacetNormal(V.mesh)
        un=ufl.dot(self.u_n,normal)
        # Energy backflow term on traction openings; activates for inward flow.
        rhs-=.5*ufl.max_value(-un,0)*ufl.inner(self.u_n,v)*ds
        rhs-=traction*ufl.inner(normal,v)*ds
        self.L=fem.form(rhs);self.b=create_vector(W)
        self.A=assemble_matrix(fem.form(a),bcs=bcs);self.A.assemble()
        self.bcs=bcs
        self.solver1=PETSc.KSP().create(V.mesh.comm)
        self.solver1.setOperators(self.A);self.solver1.setType('preonly')
        self.solver1.getPC().setType('lu')
        self.solver1.getPC().setFactorSolverType('mumps')
        self.solver2=self.solver3=self.solver1
        self.ib_load=create_vector(V)
    def solve_one_step(self):
        with self.b.localForm() as b:b.set(0)
        assemble_vector(self.b,self.L)
        self.b.array[self.vm]+=self.ib_load.array
        set_bc(self.b,self.bcs)
        self.solver1.solve(self.b,self.solution.x.petsc_vec)
        self.u_.x.array[:]=self.solution.x.array[self.vm]
        self.u_n.x.array[:]=self.u_.x.array
        self.p_.x.array[:]=self.solution.x.array[self.pm]
