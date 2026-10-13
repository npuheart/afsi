"""Serial mixed Q2/Q1 backward-Euler flow with pressure traction boundaries."""
import numpy as np
import ufl
from dolfinx import fem
from dolfinx.fem.petsc import assemble_matrix,assemble_vector,create_vector,set_bc
from petsc4py import PETSc

class CoupledFlow:
    def __init__(self,W,V,Q,maps,bcs,dt,traction,ds,implicit_convection=False,advection_refresh=20):
        self.V,self.Q=V,Q
        self.implicit_convection=implicit_convection
        self.advection_refresh=advection_refresh
        self.advection_age=0
        self.advector=fem.Function(V)
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
        if implicit_convection:
            # Frozen Oseen advector: skew volume term plus implicit backflow
            # gives nonnegative boundary contribution to the kinetic-energy balance.
            w=self.advector
            a+=ufl.inner(ufl.dot(ufl.grad(u),w)+.5*ufl.div(w)*u,v)*ufl.dx
            a+=.5*ufl.max_value(-ufl.dot(w,normal),0)*ufl.inner(u,v)*ds
            rhs=ufl.inner(self.u_n/dt,v)*ufl.dx-traction*ufl.inner(normal,v)*ds
        self.a=fem.form(a)
        self.L=fem.form(rhs);self.b=create_vector(W)
        self.A=assemble_matrix(self.a,bcs=bcs);self.A.assemble()
        self.bcs=bcs
        self.solver1=PETSc.KSP().create(V.mesh.comm)
        self.solver1.setOperators(self.A);self.solver1.setType('preonly')
        self.solver1.getPC().setType('lu')
        self.solver1.getPC().setFactorSolverType('mumps')
        self.solver2=self.solver3=self.solver1
        self.ib_load=create_vector(V)
    def solve_one_step(self):
        if self.implicit_convection and self.advection_age % self.advection_refresh == 0:
            self.advector.x.array[:]=self.u_n.x.array
            self.A.zeroEntries()
            assemble_matrix(self.A,self.a,bcs=self.bcs);self.A.assemble()
            self.solver1.setOperators(self.A)
        self.advection_age+=1
        with self.b.localForm() as b:b.set(0)
        assemble_vector(self.b,self.L)
        self.b.array[self.vm]+=self.ib_load.array
        set_bc(self.b,self.bcs)
        self.solver1.solve(self.b,self.solution.x.petsc_vec)
        self.u_.x.array[:]=self.solution.x.array[self.vm]
        self.u_n.x.array[:]=self.u_.x.array
        self.p_.x.array[:]=self.solution.x.array[self.pm]
