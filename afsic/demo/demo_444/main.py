"""demo_444: Bonet-type 3D torsion, serial baseline, bounded/restartable."""
import os, json, csv, time, itertools, hashlib
from pathlib import Path
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import dolfinx
from dolfinx import fem, mesh
from dolfinx.fem.petsc import create_vector, assemble_vector
from basix.ufl import element
import ufl
from afsic import ChorinSolver, IBMesh3D, IBInterpolation3D

comm = MPI.COMM_WORLD
if comm.size != 1:
    raise RuntimeError('Initial demo_444 baseline supports serial execution only')
def env(name, default, cast=float): return cast(os.environ.get(name, default))
N, M, order = env('N',18,int), env('M',12,int), env('ORDER',1,int)
if min(N,M)<1 or order not in (1,2): raise ValueError('N,M positive; ORDER=1 or 2')
h = 9/N
eta = env('ETA_B',3600)
dt = env('DT',min(.001*h,.2/max(eta,1)))
T, TL, theta = env('T',5), env('TL',2), env('THETA',2.5*np.pi)
beta = env('BETA',1.25*h/dt)
if dt<=0 or T<=0 or TL<=0 or beta<0 or eta<0: raise ValueError('Invalid time/material parameters')
steps = int(np.ceil(T/dt))
limit, every = env('MAX_STEPS',200,int), env('OUT_EVERY',20,int)
seconds = env('MAX_SECONDS',50)
if min(limit,every)<1 or seconds<=0: raise ValueError('Invalid chunk limits')
out = Path(os.environ.get('OUTPUT_PATH','output')); out.mkdir(parents=True,exist_ok=True)
params = dict(N=N,M=M,order=order,dt=dt,T=T,TL=TL,theta=theta,beta=beta,eta=eta,c1=9000.,c2=9000.,kappa=168000.,rho=1.,mu=.04)
source_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
metadata=out/'parameters.json'
if metadata.exists():
    old=json.loads(metadata.read_text())
    if old['parameters']!=params or old['source_sha256']!=source_hash:
        raise ValueError('Output directory parameter/source mismatch; use a new directory')
else: metadata.write_text(json.dumps(dict(parameters=params,source_sha256=source_hash),indent=2))
fluid=mesh.create_box(comm,[[0.,0.,0.],[9.,9.,9.]],[N]*3,cell_type=mesh.CellType.hexahedron)
V=fem.functionspace(fluid,element('Lagrange','hexahedron',2,shape=(3,)))
Q=fem.functionspace(fluid,element('Lagrange','hexahedron',1))
wall=mesh.locate_entities_boundary(fluid,2,lambda x: np.any(np.isclose(x,0)|np.isclose(x,9),axis=0))
zero=fem.Function(V)
bcu=[fem.dirichletbc(zero,fem.locate_dofs_topological(V,2,wall))]
bcp=[fem.dirichletbc(PETSc.ScalarType(0),fem.locate_dofs_geometrical(Q,lambda x: np.all(np.isclose(x,0),axis=0)),Q)]
ns=ChorinSolver(V,Q,bcu,bcp,dt,1.,.04,ib_body_force=False)
load=ns.b1.duplicate(); ns.ib_load=load
for ksp in (ns.solver1,ns.solver2,ns.solver3):
    ksp.setErrorIfNotConverged(True)
    ksp.setTolerances(rtol=1e-10,atol=1e-12)
nc=max(1,int(round(M/6)))
solid=mesh.create_box(comm,[[4.,1.5,4.],[5.,7.5,5.]],[nc,M,nc],cell_type=mesh.CellType.hexahedron)
Vs=fem.functionspace(solid,element('Lagrange','hexahedron',order,shape=(3,)))
chi=fem.Function(Vs,name='coordinates'); chi.interpolate(lambda x:x)
U=fem.Function(Vs,name='solid_velocity'); force=fem.Function(Vs,name='solid_force')
target=fem.Function(Vs)
ref=Vs.tabulate_dof_coordinates().copy()
def set_target(t):
    a=theta*min(t/TL,1.)
    def rotate(x):
        r=x.copy(); r[0]=4.5+np.cos(a)*(x[0]-4.5)+np.sin(a)*(x[2]-4.5)
        r[2]=4.5-np.sin(a)*(x[0]-4.5)+np.cos(a)*(x[2]-4.5)
        return r
    target.interpolate(rotate)
solid.topology.create_connectivity(2,3)
indices=[]; values=[]
for tag,y in ((1,1.5),(2,7.5)):
    f=mesh.locate_entities_boundary(solid,2,lambda x,y=y: np.isclose(x[1],y))
    indices.extend(f); values.extend([tag]*len(f))
idx=np.argsort(indices)
tags=mesh.meshtags(solid,2,np.asarray(indices,dtype=np.int32)[idx],np.asarray(values,dtype=np.int32)[idx])
ds=ufl.Measure('ds',domain=solid,subdomain_data=tags)
dx=ufl.Measure('dx',domain=solid,metadata={'quadrature_degree':6})
F=ufl.variable(ufl.grad(chi)); J=ufl.det(F); C=F.T*F
I1=ufl.tr(C); I2=(I1**2-ufl.tr(C*C))/2
energy=9000*(J**(-2/3)*I1-3)+9000*(J**(-4/3)*I2-3)+84000*ufl.ln(J)**2
P=ufl.diff(energy,F); v=ufl.TestFunction(Vs); X=ufl.SpatialCoordinate(solid)
# Damping per reference volume; endpoint penalties per reference surface area.
L=fem.form(-ufl.inner(P,ufl.grad(v))*dx-eta*ufl.inner(U,v)*dx
           -beta*ufl.inner(chi-X,v)*ds(1)-beta*ufl.inner(chi-target,v)*ds(2))
b=create_vector(Vs)
ibmesh=IBMesh3D(0.,9.,0.,9.,0.,9.,N,N,N,2)
ib=IBInterpolation3D(ibmesh)
coords=fem.Function(V); coords.interpolate(lambda x:x); ibmesh.build_map(coords._cpp_object)
fields={k:x for k,x in vars(ns).items() if isinstance(x,fem.Function)}
fields.update(chi=chi,U=U,force=force)
checkpoint=out/'checkpoint.npz'; start=0
if checkpoint.exists():
    with np.load(checkpoint) as data:
        if not np.array_equal(ref,data['ref']): raise ValueError('Checkpoint mesh mismatch')
        start=int(data['step'])
        for k,x in fields.items(): x.x.array[:]=data[k]; x.x.scatter_forward()
ib.evaluate_current_points(chi._cpp_object)
points=np.array(list(itertools.product([0.,.5,1.],repeat=3)))
jexpr=fem.Expression(J,points)
cells=np.arange(solid.topology.index_map(3).size_local,dtype=np.int32)
volume=fem.form(J*dx); elastic=fem.form(energy*dx)
left=np.isclose(ref[:,1],1.5); right=np.isclose(ref[:,1],7.5)
# Explicitly defined midspan corner, not yet identified with the published probe.
probe=np.argmin(np.linalg.norm(ref-[5,4.5,5],axis=1))
logpath=out/'metrics.csv'
header=['step','time','angle','volume','volume_change_pct','J_min','J_max','elastic_energy','u_max','fixed_slip','rotating_error','probe_dx','probe_dy','probe_dz','ksp1','ksp2','ksp3']
if not logpath.exists():
    with logpath.open('w') as f: csv.writer(f).writerow(header)
def metrics(step):
    set_target(step*dt)
    j=jexpr.eval(solid,cells).reshape(-1)
    pos=chi.x.array.reshape(-1,3); tar=target.x.array.reshape(-1,3)
    vol=fem.assemble_scalar(volume)
    row=[step,step*dt,theta*min(step*dt/TL,1),vol,100*(vol/6-1),j.min(),j.max(),fem.assemble_scalar(elastic),np.linalg.norm(ns.u_.x.array.reshape(-1,3),axis=1).max(),np.linalg.norm(pos[left]-ref[left],axis=1).max(),np.linalg.norm(pos[right]-tar[right],axis=1).max(),*(pos[probe]-ref[probe]),*[k.getConvergedReason() for k in (ns.solver1,ns.solver2,ns.solver3)]]
    if not np.all(np.isfinite(row)) or j.min()<=0: raise RuntimeError('Invalid deformation/diagnostic')
    with logpath.open('a') as f: csv.writer(f).writerow(row)
    print(f'step={step} t={step*dt:.8g} J=[{j.min():.8g},{j.max():.8g}] dV={row[4]:.5g}% rotating_error={row[10]:.5g}',flush=True)
    return row
if start==0: metrics(0)
started=time.monotonic(); step=start
for step in range(start+1,min(steps,start+limit)+1):
    set_target((step-1)*dt)
    with b.localForm() as loc: loc.set(0)
    assemble_vector(b,L); b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,mode=PETSc.ScatterMode.REVERSE)
    force.x.array[:]=b.array; force.x.scatter_forward()
    ib.solid_to_fluid(ns.f._cpp_object,force._cpp_object); ns.f.x.scatter_forward()
    ns.f.x.petsc_vec.copy(result=load); load.scale((h/2)**3)
    ns.solve_one_step()
    ib.fluid_to_solid(ns.u_._cpp_object,U._cpp_object)
    chi.x.array[:]+=dt*U.x.array; chi.x.scatter_forward()
    for k,x in fields.items():
        if not np.isfinite(x.x.array).all(): raise RuntimeError(f'Nonfinite {k} at step {step}')
    ib.evaluate_current_points(chi._cpp_object)
    if step%every==0: metrics(step)
    if time.monotonic()-started>=seconds: break
if step%every or step==start: metrics(step)
# Atomic checkpoint: no partial file replaces a previous valid state.
with (out/'checkpoint.tmp').open('wb') as f:
    np.savez_compressed(f,step=step,ref=ref,**{k:x.x.array for k,x in fields.items()})
os.replace(out/'checkpoint.tmp',checkpoint)
np.savez_compressed(out/f'solid_{step:08d}.npz',reference=ref,coordinates=chi.x.array.reshape(-1,3),velocity=U.x.array.reshape(-1,3))
(out/'status.json').write_text(json.dumps(dict(step=step,time=step*dt,complete=step>=steps,chunk_seconds=time.monotonic()-started),indent=2))
print(f'Checkpoint saved: {checkpoint}; complete={step>=steps}',flush=True)
