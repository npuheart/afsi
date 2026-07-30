"""Quick comparison: no-cylinder vs direct forcing."""
import numpy as np
from petsc4py import PETSc
from mpi4py import MPI
comm=MPI.COMM_WORLD;rank=comm.rank
import dolfinx
from dolfinx.fem import Function,functionspace,dirichletbc,locate_dofs_topological
from dolfinx.mesh import CellType,GhostMode
from basix.ufl import element
from afsic import ChorinSolver
from afsic.common import tag_boundaries,rectangle_boundaries,TurekInlet,MARKER_LEFT,MARKER_RIGHT,MARKER_BOTTOM,MARKER_TOP

Lx,Ly=2.2,0.41;Nx,Ny=110,21;Um,rho,mu=100.,1.,0.1;dt=0.0001;nsteps=5000;D=0.1
mesh=dolfinx.mesh.create_rectangle(comm=comm,points=((0,0),(Lx,Ly)),n=(Nx,Ny),cell_type=CellType.quadrilateral,ghost_mode=GhostMode.shared_facet)
ftag=tag_boundaries(mesh,rectangle_boundaries(Lx,Ly))
v2=element('Lagrange',mesh.topology.cell_name(),2,shape=(2,));s1=element('Lagrange',mesh.topology.cell_name(),1)
V=functionspace(mesh,v2);Q=functionspace(mesh,s1);fd=mesh.topology.dim-1
inl=TurekInlet(Um=Um,Ly=Ly);ui=Function(V);ui.interpolate(inl)
bci=dirichletbc(ui,locate_dofs_topological(V,fd,ftag.find(MARKER_LEFT)))
u0=Function(V);u0.x.array[:]=0.
bcb=dirichletbc(u0,locate_dofs_topological(V,fd,ftag.find(MARKER_BOTTOM)))
bct=dirichletbc(u0,locate_dofs_topological(V,fd,ftag.find(MARKER_TOP)))
bcpo=dirichletbc(PETSc.ScalarType(0.),locate_dofs_topological(Q,fd,ftag.find(MARKER_RIGHT)),Q)

print(f'No cylinder: {nsteps} steps...')
sol1=ChorinSolver(V,Q,[bci,bcb,bct],[bcpo],dt,rho,mu)
for st in range(nsteps):t=st*dt;inl.update(t);ui.interpolate(inl);sol1.solve_one_step()
u1=sol1.u_.x.array.reshape(-1,2);vm1=np.sqrt(u1[:,0]**2+u1[:,1]**2)
print(f'  max|u|={vm1.max():.1f}')

print(f'With cylinder: {nsteps} steps...')
sol2=ChorinSolver(V,Q,[bci,bcb,bct],[bcpo],dt,rho,mu)
Vc=V.tabulate_dof_coordinates();bs=V.dofmap.index_map_bs
sd=np.where((Vc[:,0]-0.2)**2+(Vc[:,1]-0.2)**2<0.05**2)[0]
sd=np.array(sorted(sd),np.int32);dV=(Lx/Nx)*(Ly/Ny)
for st in range(nsteps):
    t=st*dt;inl.update(t);ui.interpolate(inl)
    fs=sol2.solve_one_step_df(sd,bs)
    if st%1000==0:
        dg=comm.allreduce(fs[0]*dV/dt,op=MPI.SUM)
        if rank==0:print(f'  st={st} t={t:.3f} Cd={2*abs(dg)/(rho*Um**2*D):.3f}')

u2=sol2.u_.x.array.reshape(-1,2);vm2=np.sqrt(u2[:,0]**2+u2[:,1]**2)
if rank==0:
    dg=comm.allreduce(fs[0]*dV/dt,op=MPI.SUM);lf=comm.allreduce(fs[1]*dV/dt,op=MPI.SUM)
    diff=np.abs(vm1-vm2)
    x=Vc[:,0]
    print(f'\n=== t={t:.3f}s ===')
    print(f'No cyl  |u|max={vm1.max():.1f}')
    print(f'With cyl|u|max={vm2.max():.1f}')
    print(f'cyl|u|max={vm2[sd].max():.4f}')
    print(f'Cd={2*abs(dg)/(rho*Um**2*D):.3f}  Cl={2*abs(lf)/(rho*Um**2*D):.3f}')
    print(f'Δu>1: {np.sum(diff>1)}/{len(diff)} ({100*np.sum(diff>1)/len(diff):.0f}%)')
