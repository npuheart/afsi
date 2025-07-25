import gmsh
import os
import numpy as np
import matplotlib.pyplot as plt
import tqdm.autonotebook

from mpi4py import MPI
from petsc4py import PETSc

from basix.ufl import element

from dolfinx.cpp.mesh import to_type, cell_entity_type
from dolfinx.fem import (Constant, Function, functionspace,
                         assemble_scalar, dirichletbc, form, locate_dofs_topological, set_bc)
from dolfinx.fem.petsc import (apply_lifting, assemble_matrix, assemble_vector,
                               create_vector, create_matrix, set_bc)
from dolfinx.graph import adjacencylist
from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells
from dolfinx.io import (VTXWriter, distribute_entity_data, gmshio)
from dolfinx.mesh import create_mesh, meshtags_from_entities
from ufl import (FacetNormal, Identity, Measure, TestFunction, TrialFunction,
                 as_vector, div, dot, ds, dx, inner, lhs, grad, nabla_grad, rhs, sym, system)


class ChorinSolver:
    def __init__(self, V, Q, bcu, bcp, dt_raw, rho_raw, mu_raw):
        self.bcu = bcu
        self.bcp = bcp
        
        self.V = V
        self.Q = Q
        
        mesh = V.mesh
        self.mesh = mesh
        k = Constant(mesh, PETSc.ScalarType(dt_raw))
        mu = Constant(mesh, PETSc.ScalarType(mu_raw)) 
        rho = Constant(mesh, PETSc.ScalarType(rho_raw))

        # Define trial and test functions
        u = TrialFunction(V)
        v = TestFunction(V)
        u_ = Function(V,name = "u")
        u_s = Function(V)
        u_n = Function(V)
        u_n1 = Function(V)
        p = TrialFunction(Q)
        q = TestFunction(Q)
        p_ = Function(Q,name = "p")
        phi = Function(Q)


        f = Function(V)
        # Tentative velocity step
        F1 = rho / k * inner(u - u_n, v) * dx
        F1 += rho * inner(grad(u_n) * u_n, v) * dx
        F1 += mu*inner(grad(u), grad(v))*dx
        F1 -= inner(f, v)*dx
        a1 = form(lhs(F1))
        L1 = form(rhs(F1))
        A1 = create_matrix(a1)
        b1 = create_vector(L1)
        # Pressure update
        a2 = form(inner(grad(p), grad(q)) * dx)
        L2 = form(-(1 / k) * div(u_s) * q * dx)
        A2 = assemble_matrix(a2, bcs=self.bcp)
        A2.assemble()
        b2 = create_vector(L2)
        # Velocity update
        a3 = form(inner(u, v) * dx)
        L3 = form(inner(u_s, v) * dx - k * inner(grad(p_), v) * dx)
        A3 = assemble_matrix(a3)
        A3.assemble()
        b3 = create_vector(L3)

        # Solver for step 1
        solver1 = PETSc.KSP().create(mesh.comm)
        solver1.setOperators(A1)
        solver1.setType(PETSc.KSP.Type.BCGS)
        pc1 = solver1.getPC()
        pc1.setType(PETSc.PC.Type.JACOBI)

        # Solver for step 2
        solver2 = PETSc.KSP().create(mesh.comm)
        solver2.setOperators(A2)
        solver2.setType(PETSc.KSP.Type.MINRES)
        pc2 = solver2.getPC()
        pc2.setType(PETSc.PC.Type.HYPRE)
        pc2.setHYPREType("boomeramg")

        # Solver for step 3
        solver3 = PETSc.KSP().create(mesh.comm)
        solver3.setOperators(A3)
        solver3.setType(PETSc.KSP.Type.CG)
        pc3 = solver3.getPC()
        pc3.setType(PETSc.PC.Type.SOR)
        
        self.solver1 = solver1
        self.solver2 = solver2
        self.solver3 = solver3
        self.u_ = u_
        self.u_s = u_s
        self.u_n = u_n
        self.u_n1 = u_n1
        self.p_ = p_
        self.phi = phi
        self.a1 = a1
        self.L1 = L1
        self.A1 = A1
        self.b1 = b1
        self.a2 = a2
        self.L2 = L2
        self.A2 = A2
        self.b2 = b2
        self.a3 = a3
        self.L3 = L3
        self.A3 = A3
        self.b3 = b3
        self.f = f

    def solve_one_step(self):
        # Step 1: Tentative velocity step
        self.A1.zeroEntries()
        assemble_matrix(self.A1, self.a1, bcs=self.bcu)
        self.A1.assemble()
        with self.b1.localForm() as loc:
            loc.set(0)
        assemble_vector(self.b1, self.L1)
        apply_lifting(self.b1, [self.a1], [self.bcu])
        self.b1.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        set_bc(self.b1, self.bcu)
        self.solver1.solve(self.b1, self.u_s.x.petsc_vec)
        self.u_s.x.scatter_forward()

        # Step 2: Pressure corrrection step
        with self.b2.localForm() as loc:
            loc.set(0)
        assemble_vector(self.b2, self.L2)
        apply_lifting(self.b2, [self.a2], [self.bcp])
        self.b2.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        set_bc(self.b2, self.bcp)
        self.solver2.solve(self.b2, self.p_.x.petsc_vec)
        self.p_.x.scatter_forward()

        # Step 3: Velocity correction step
        with self.b3.localForm() as loc:
            loc.set(0)
        assemble_vector(self.b3, self.L3)
        self.b3.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        self.solver3.solve(self.b3, self.u_.x.petsc_vec)
        self.u_.x.scatter_forward()

        # Update variable with solution form this time step
        with self.u_.x.petsc_vec.localForm() as loc_, self.u_n.x.petsc_vec.localForm() as loc_n, self.u_n1.x.petsc_vec.localForm() as loc_n1:
            loc_n.copy(loc_n1)
            loc_.copy(loc_n)