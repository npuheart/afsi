"""Scan mu_s x dt for BOTH schemes (implicit also reduces dt).  Report each
scheme's best (smallest) wall time per simulated second over stable dt.
"""
import os, sys, time, numpy as np
os.environ.setdefault('NX', '32'); os.environ.setdefault('NY', '32')
os.environ.setdefault('SOLID_H', '0.02')
from config import make_config
from immersed import ImmersedFEM

def run422(mu_s, dt, nsteps=15):
    cfg=make_config(); cfg['num_steps']=nsteps; cfg['frozen']=2; cfg['dt']=dt; cfg['mu_s']=mu_s
    s=ImmersedFEM(cfg); s.X[:s.n_u]=s._initial_velocity()
    t0=time.perf_counter()
    for k in range(nsteps):
        s.solve_monolithic()
        W=s.X[s.n_u+s.n_p:]
        if not np.isfinite(W).all() or abs(W).max()>1.5:
            return False, (time.perf_counter()-t0)/(k+1)
    return True, (time.perf_counter()-t0)/nsteps

def run336(mu_s, dt, nsteps=15):
    sys.path.insert(0, '/home/Pengfei.Ma@glasgow.ac.uk/afsi/afsic/src')
    from petsc4py import PETSc
    from mpi4py import MPI
    from afsic import ChorinSolver, IBMesh, IBInterpolation
    import dolfinx
    from dolfinx.fem import Function, functionspace, dirichletbc, locate_dofs_topological, form
    from dolfinx.mesh import CellType, GhostMode, locate_entities, meshtags
    from basix.ufl import element
    from ufl import TestFunction, grad, inv, dx, inner
    from dolfinx.fem.petsc import create_vector, assemble_vector
    mesh = dolfinx.mesh.create_rectangle(MPI.COMM_WORLD, ((0,0),(1,1)), (32,32),
        cell_type=CellType.quadrilateral, ghost_mode=GhostMode.shared_facet)
    mesh.topology.create_connectivity(1,2); fdim=mesh.topology.dim-1
    boundaries=[(1,lambda x:np.isclose(x[0],0)),(2,lambda x:np.isclose(x[0],1)),
                (3,lambda x:np.isclose(x[1],0)),(4,lambda x:np.isclose(x[1],1))]
    fi,fm=[],[]
    for (m,loc) in boundaries:
        fs=locate_entities(mesh,fdim,loc); fi.append(fs); fm.append(np.full_like(fs,m))
    fi=np.hstack(fi).astype(np.int32); fm=np.hstack(fm).astype(np.int32)
    tag=meshtags(mesh,fdim,fi[np.argsort(fi)],fm[np.argsort(fi)])
    V=functionspace(mesh,element('Lagrange',mesh.topology.cell_name(),2,shape=(2,)))
    Q=functionspace(mesh,element('Lagrange',mesh.topology.cell_name(),1))
    class Up:
        def __call__(self,x):
            v=np.zeros((2,x.shape[1]),dtype=PETSc.ScalarType); v[0]=1.0; return v
    u_up=Function(V); u_up.interpolate(Up())
    ns=ChorinSolver(V,Q,[dirichletbc(u_up,locate_dofs_topological(V,fdim,tag.find(4))),
        dirichletbc(np.zeros(2,dtype=PETSc.ScalarType),locate_dofs_topological(V,fdim,tag.find(1)),V),
        dirichletbc(np.zeros(2,dtype=PETSc.ScalarType),locate_dofs_topological(V,fdim,tag.find(2)),V),
        dirichletbc(np.zeros(2,dtype=PETSc.ScalarType),locate_dofs_topological(V,fdim,tag.find(3)),V)],
        [dirichletbc(0.0,locate_dofs_topological(Q,0,dolfinx.mesh.locate_entities_boundary(mesh,0,
            lambda x:np.isclose(x[0],0)&np.isclose(x[1],0))),Q)], dt,1.0,0.01)
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, os.path.expanduser('~/afsi-data/336-lid-driven-disk/mesh/circle_15.xdmf'),
            'r', encoding=dolfinx.io.XDMFFile.Encoding.HDF5) as f:
        structure=f.read_mesh()
    Vs=functionspace(structure,element('Lagrange',structure.topology.cell_name(),2,shape=(2,)))
    sc=Function(Vs); sv=Function(Vs); sf=Function(Vs)
    sc.interpolate(lambda x: np.array([x[0],x[1]]))
    dVs=TestFunction(Vs); FF=grad(sc)
    Lhat=form(-inner(mu_s*(FF-inv(FF).T), grad(dVs))*dx)
    b1=create_vector(Vs)
    ibmesh=IBMesh(0.0,1.0,0.0,1.0,32,32,2); ib=IBInterpolation(ibmesh)
    cbg=Function(V); cbg.interpolate(lambda x: np.array([x[0],x[1]]))
    ibmesh.build_map(cbg._cpp_object); ib.evaluate_current_points(sc._cpp_object)
    t0=time.perf_counter()
    for k in range(nsteps):
        ns.solve_one_step()
        ib.fluid_to_solid(ns.u_._cpp_object, sv._cpp_object)
        sc.x.array[:] += sv.x.array[:]*dt; sc.x.scatter_forward()
        ib.evaluate_current_points(sc._cpp_object)
        b1.set(0); assemble_vector(b1,Lhat)
        b1.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        with b1.getBuffer() as arr: sf.x.array[:len(arr)] = arr[:]
        ib.solid_to_fluid(ns.f._cpp_object, sf._cpp_object); ns.f.x.scatter_forward()
        if not np.isfinite(sc.x.array).all() or np.abs(sc.x.array-0.5).max()>1.0:
            return False, (time.perf_counter()-t0)/(k+1)
    return True, (time.perf_counter()-t0)/nsteps

dts = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002]
print(f"{'mu_s':>6} | {'336 dt*':>8} {'336 s/sim':>9} | {'422 dt*':>8} {'422 s/sim':>9} | winner")
for mu in [0.1, 1.0, 10.0, 100.0]:
    best336=(None,1e30); best422=(None,1e30)
    for dt in dts:
        ok, per = run336(mu, dt)
        if ok and per/dt < best336[1]: best336=(dt, per/dt)
    for dt in dts:
        ok, per = run422(mu, dt)
        if ok and per/dt < best422[1]: best422=(dt, per/dt)
    winner = '422' if best422[1] < best336[1] else '336'
    print(f"{mu:>6} | {str(best336[0]):>8} {best336[1]:>9.1f} | {str(best422[0]):>8} {best422[1]:>9.1f} | {winner}")
