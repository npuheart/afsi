"""AV2D prescribed-pressure pilot in AFSI. All numerical units are CGS."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scipy.io import netcdf_file
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem, mesh
from dolfinx.fem.petsc import create_vector, assemble_vector
from basix.ufl import element
import ufl
from afsic import ChorinSolver, IBMesh, IBInterpolation

PERIOD = .8512
L = 7.68


def read_exodus(path):
    with netcdf_file(path, 'r', mmap=False) as f:
        x = np.column_stack([f.variables['coordx'][:], f.variables['coordy'][:]]).copy()
        cells = f.variables['connect1'][:].copy().astype(np.int64)-1
        sides = {}
        for i, marker in enumerate(f.variables['ss_prop1'][:], 1):
            e = f.variables[f'elem_ss{i}'][:].astype(int)-1
            s = f.variables[f'side_ss{i}'][:].astype(int)-1
            pairs = np.array([[0,1],[1,2],[2,0]])[s]
            sides[int(marker)] = np.array([cells[j, pair] for j, pair in zip(e,pairs)])
    if cells.shape[1] != 3:
        raise ValueError('Expected TRI3 Exodus mesh')
    return np.clip(x+L/2,0,L), cells, sides


def pressures(path):
    rows = [(0.,0.,0.)]
    with path.open() as stream:
        next(stream)
        for line in stream:
            values = line.split(',')
            t = float(values[0])
            rows.append((t,float(values[1])*1333.22387415,float(values[2])*1333.22387415))
            if t >= PERIOD:
                break
    a = np.asarray(rows)
    if a[-1,0] < PERIOD or np.any(np.diff(a[:,0]) <= 0):
        raise ValueError('Pressure data must cover a full cycle with increasing time')
    return a


class Solid:
    def __init__(self, path, leaflet):
        points, cells, sides = read_exodus(path)
        domain = mesh.create_mesh(MPI.COMM_SELF, cells,
            ufl.Mesh(element('Lagrange','triangle',1,shape=(2,))), points)
        self.domain = domain
        self.V = fem.functionspace(domain, element('Lagrange','triangle',1,shape=(2,)))
        self.chi = fem.Function(self.V)
        self.chi.interpolate(lambda x:x[:2])
        self.ref = self.chi.x.array.copy()
        self.end_nodes = np.isclose(self.ref.reshape(-1,2)[:,1],0) | np.isclose(self.ref.reshape(-1,2)[:,1],L)
        self.velocity = fem.Function(self.V)
        self.force = fem.Function(self.V)
        self.cells = np.array([self.V.dofmap.cell_dofs(i) for i in range(domain.topology.index_map(2).size_local)])
        v = ufl.TestFunction(self.V)
        X = ufl.SpatialCoordinate(domain)
        F = ufl.variable(ufl.grad(self.chi))
        J = ufl.det(F)
        invT = ufl.inv(F).T
        dx = ufl.Measure('dx',domain=domain,metadata={'quadrature_degree':5})
        self.root_count = 0
        edges = np.concatenate([self.cells[:,[0,1]],self.cells[:,[1,2]],self.cells[:,[2,0]]])
        graph=coo_matrix((np.ones(len(edges)),(edges[:,0],edges[:,1])),shape=(len(self.ref)//2,len(self.ref)//2))
        _, self.components=connected_components(graph,directed=False)
        if leaflet:
            # IBTK jacobian embeds the 2D gradient in 3D with Fzz=1.
            I1 = ufl.inner(F,F)+1
            C10, C01 = 1.5*5*3018*1e2, 3.2495
            beta = .5*32.5*.5*7187.5*1e3
            P = C10*ufl.exp(C01*(J**(-2/3)*I1-3))*C01*(2*J**(-2/3)*(F-I1/3*invT))
            P += beta*J*ufl.ln(J)*invT
            # Exodus side set 4 carries the original root surface spring.
            domain.topology.create_connectivity(1,2)
            facets = mesh.exterior_facet_indices(domain.topology)
            gn = mesh.entities_to_geometry(domain,1,facets,False)
            mid = domain.geometry.x[gn,:2].mean(axis=1)
            distances, which = cKDTree(mid).query(points[sides[4]].mean(axis=1))
            if distances.max() > 1e-9:
                raise ValueError('Cannot map root side set')
            roots = np.sort(facets[which])
            self.root_count = len(roots)
            tags = mesh.meshtags(domain,1,roots,np.full(len(roots),4,dtype=np.int32))
            ds = ufl.Measure('ds',domain=domain,subdomain_data=tags)
            ks = .5*1.843750*.5*2828.125*1e6
            center = ufl.as_vector([ufl.conditional(ufl.gt(X[0]-L/2,-.74),.1,-1.7)+L/2,
                                    ufl.conditional(ufl.gt(X[0]-L/2,-.74),-.6,-.5)+L/2])
            rest = ufl.sqrt(ufl.inner(X-center,X-center))
            cur = ufl.sqrt(ufl.inner(self.chi-center,self.chi-center))
            extension = cur-rest
            # Exact kappa_fcn from the source.
            activation = (ufl.tanh((extension-.05)/.015)+np.tanh(.05/.015))/(1+np.tanh(.05/.015))
            buttress = ufl.conditional(ufl.gt(extension,0),1e7*activation*(1-rest/cur),0)*(center-self.chi)
            load = -ufl.inner(P,ufl.grad(v))*dx-ks*ufl.inner(self.chi-X,v)*ds(4)+ufl.inner(buttress,v)*dx
        else:
            c1 = .5*1.843750*.5*4375*1e3
            kb = .5*1.843750*.5*4375*1e7
            P = 2*c1*(F-invT)
            load = -ufl.inner(P,ufl.grad(v))*dx-kb*ufl.inner(self.chi-X,v)*dx
        self.load_form = fem.form(load)
        self.b = create_vector(self.V)
        self.je = fem.Expression(J,np.array([[1/3,1/3]]))
        self.ids = np.arange(len(self.cells),dtype=np.int32)

    def jacobians(self):
        return self.je.eval(self.domain,self.ids).ravel()

    def assemble(self):
        j = self.jacobians()
        if not np.isfinite(j).all() or j.min() <= 0:
            raise RuntimeError('Solid inversion/nonfinite Jacobian')
        with self.b.localForm() as local:
            local.set(0)
        assemble_vector(self.b,self.load_form)
        self.force.x.array[:] = self.b.array
        if not np.isfinite(self.force.x.array).all():
            raise RuntimeError('Nonfinite solid force')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--nx',type=int,default=120)
    p.add_argument('--dt',type=float,default=5e-6)
    p.add_argument('--time',type=float,default=PERIOD)
    p.add_argument('--max-steps',type=int,default=200000)
    p.add_argument('--output-every',type=int,default=1000)
    p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    if MPI.COMM_WORLD.size != 1 or a.nx<8 or not 0<a.time<=PERIOD or a.dt<=0:
        p.error('Serial pilot, nx>=8, positive dt, time within one cycle required')
    if abs(a.time/a.dt-round(a.time/a.dt)) > 1e-6:
        p.error('End time must be an integer number of steps')
    a.output.mkdir(parents=True,exist_ok=True)
    files=[a.source/n for n in ['2d_leaflets_sideset.e','2d_aorta_sideset.e','bc_data.txt']]
    meta=dict(nx=a.nx,dt=a.dt,time=a.time,source=str(a.source.resolve()),
              sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in files+[Path(__file__)]})
    metadata=a.output/'parameters.json'
    if a.resume:
        if json.loads(metadata.read_text())!=meta: raise ValueError('Resume parameter/source mismatch')
    elif metadata.exists(): raise ValueError('Output exists; use --resume')
    else: metadata.write_text(json.dumps(meta,indent=2)+'\n')
    data=pressures(files[2])
    bg=mesh.create_rectangle(MPI.COMM_SELF,[[0.,0.],[L,L]],[a.nx,a.nx],cell_type=mesh.CellType.quadrilateral)
    V=fem.functionspace(bg,element('Lagrange','quadrilateral',2,shape=(2,)))
    Q=fem.functionspace(bg,element('Lagrange','quadrilateral',1))
    bg.topology.create_connectivity(1,2)
    walls=mesh.locate_entities_boundary(bg,1,lambda x:np.isclose(x[0],0)|np.isclose(x[0],L))
    ends=mesh.locate_entities_boundary(bg,1,lambda x:np.isclose(x[1],0)|np.isclose(x[1],L))
    bcu=[fem.dirichletbc(np.zeros(2,dtype=PETSc.ScalarType),fem.locate_dofs_topological(V,1,walls),V),
         fem.dirichletbc(PETSc.ScalarType(0),fem.locate_dofs_topological(V.sub(0),1,ends),V.sub(0))]
    low=fem.locate_dofs_topological(Q,1,mesh.locate_entities_boundary(bg,1,lambda x:np.isclose(x[1],0) & (np.abs(x[0]-(L/2-1.541837105))<1.9)))
    high=fem.locate_dofs_topological(Q,1,mesh.locate_entities_boundary(bg,1,lambda x:np.isclose(x[1],L) & (np.abs(x[0]-(L/2+.00039225))<1.4)))
    closed=mesh.locate_entities_boundary(bg,1,lambda x:(np.isclose(x[1],0) & (np.abs(x[0]-(L/2-1.541837105))>=1.9)) | (np.isclose(x[1],L) & (np.abs(x[0]-(L/2+.00039225))>=1.4)))
    bcu.append(fem.dirichletbc(PETSc.ScalarType(0),fem.locate_dofs_topological(V.sub(1),1,closed),V.sub(1)))
    def pressure_bc(t):
        return [fem.dirichletbc(PETSc.ScalarType(np.interp(t,data[:,0],data[:,1])),low,Q),
                fem.dirichletbc(PETSc.ScalarType(np.interp(t,data[:,0],data[:,2])),high,Q)]
    boundary_facets=np.concatenate([mesh.locate_entities_boundary(bg,1,lambda x:np.isclose(x[1],0)),mesh.locate_entities_boundary(bg,1,lambda x:np.isclose(x[1],L))])
    boundary_values=np.concatenate([np.full(len(mesh.locate_entities_boundary(bg,1,lambda x:np.isclose(x[1],0))),1),np.full(len(mesh.locate_entities_boundary(bg,1,lambda x:np.isclose(x[1],L))),2)]).astype(np.int32)
    order=np.argsort(boundary_facets)
    tags=mesh.meshtags(bg,1,boundary_facets[order],boundary_values[order])
    ds_flow=ufl.Measure('ds',domain=bg,subdomain_data=tags)
    solver=ChorinSolver(V,Q,bcu,pressure_bc(0),a.dt,1.,.035,ib_body_force=False)
    for ksp in [solver.solver1,solver.solver2,solver.solver3]: ksp.setTolerances(rtol=1e-9,atol=1e-11)
    solids=[Solid(files[0],True),Solid(files[1],False)]
    ib=IBMesh(0.,L,0.,L,a.nx,a.nx,2)
    coord=fem.Function(V); coord.interpolate(lambda x:x[:2]); ib.build_map(coord._cpp_object)
    interps=[IBInterpolation(ib),IBInterpolation(ib)]
    solver.ib_load=create_vector(V)
    flow_forms=[fem.form(solver.u_n[1]*ds_flow(i)) for i in [1,2]]
    step=0
    if a.resume:
        with np.load(a.output/'checkpoint.npz') as state:
            step=int(state['step']); solver.u_n.x.array[:]=state['u']; solver.p_.x.array[:]=state['p']
            for i,s in enumerate(solids): s.chi.x.array[:]=state[f'x{i}']
    def save():
        state=dict(step=step,time=step*a.dt,u=solver.u_n.x.array,p=solver.p_.x.array)
        for i,s in enumerate(solids): state.update({f'x{i}':s.chi.x.array,f'ref{i}':s.ref,f'cells{i}':s.cells})
        with (a.output/'checkpoint.tmp').open('wb') as stream: np.savez_compressed(stream,**state)
        (a.output/'checkpoint.tmp').replace(a.output/'checkpoint.npz')
        (a.output/'status.json').write_text(json.dumps(dict(complete=False,reason='running',step=step,time=step*a.dt),indent=2))
        with (a.output/f'snapshot_{step:08d}.npz').open('wb') as stream: np.savez_compressed(stream,**state)
    def log():
        row=dict(step=step,time=step*a.dt,max_speed=float(np.linalg.norm(solver.u_n.x.array.reshape(-1,2),axis=1).max()))
        row.update(Q_in_cm2_s=float(fem.assemble_scalar(flow_forms[0])),Q_out_cm2_s=float(fem.assemble_scalar(flow_forms[1])))
        leaf=solids[0]; nodes=leaf.chi.x.array.reshape(-1,2)
        if len(np.unique(leaf.components))==2:
            row['leaflet_gap_cm']=float(cKDTree(nodes[leaf.components==0]).query(nodes[leaf.components==1])[0].min())
        for i,s in enumerate(solids):
            j=s.jacobians(); row.update({f'Jmin{i}':float(j.min()),f'Jmax{i}':float(j.max()),f'displacement{i}':float(np.linalg.norm((s.chi.x.array-s.ref).reshape(-1,2),axis=1).max())})
        path=a.output/'metrics.csv'; exists=path.exists()
        with path.open('a') as stream:
            w=csv.DictWriter(stream,fieldnames=row.keys())
            if not exists:w.writeheader()
            w.writerow(row)
        print(json.dumps(row),flush=True)
    start=time.monotonic(); reason='max_steps'
    try:
        log(); save()
        for _ in range(a.max_steps):
            if step>=round(a.time/a.dt):reason='complete';break
            solver.ib_load.set(0)
            for s,ip in zip(solids,interps):
                s.assemble(); ip.evaluate_current_points(s.chi._cpp_object)
                ip.solid_to_fluid(solver.f._cpp_object,s.force._cpp_object)
                solver.ib_load.axpy((L/(2*a.nx))**2,solver.f.x.petsc_vec)
            solver.bcp=pressure_bc((step+1)*a.dt)
            solver.solve_one_step()
            if any(k.getConvergedReason()<0 for k in [solver.solver1,solver.solver2,solver.solver3]):raise RuntimeError('Fluid linear solve failed')
            if not np.isfinite(solver.u_n.x.array).all():raise RuntimeError('Nonfinite fluid velocity')
            for s,ip in zip(solids,interps):
                ip.fluid_to_solid(solver.u_._cpp_object,s.velocity._cpp_object)
                s.chi.x.array[:]+=a.dt*s.velocity.x.array
                xy=s.chi.x.array.reshape(-1,2)
                xy[s.end_nodes]=s.ref.reshape(-1,2)[s.end_nodes]
                if xy.min() < -1e-9 or xy.max() > L+1e-9:raise RuntimeError(f'Solid left fluid domain: {xy.min()}, {xy.max()}')
                j=s.jacobians()
                if not np.isfinite(j).all() or j.min()<=0:raise RuntimeError('Solid inversion/nonfinite Jacobian')
            step+=1
            if step%a.output_every==0:log();save()
        if step>=round(a.time/a.dt):reason='complete'
        log();save()
    except Exception as exc:
        reason='error'; print(repr(exc),flush=True)
        (a.output/'status.json').write_text(json.dumps(dict(complete=False,reason=reason,error=str(exc),last_good_checkpoint='checkpoint.npz'),indent=2))
        raise
    (a.output/'status.json').write_text(json.dumps(dict(complete=reason=='complete',reason=reason,step=step,time=step*a.dt,wall_seconds=time.monotonic()-start),indent=2))

if __name__=='__main__':main()
