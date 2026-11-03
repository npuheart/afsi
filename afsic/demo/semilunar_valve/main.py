"""Semilunar (aortic) valve FSI with a rigid, tethered vessel wall.

Fluid: Cartesian box [-W, W]^2 x [Z_IN, Z_OUT], Q2/Q1, Chorin projection.
  inlet  z = Z_IN : plug inflow v = Q(t)/A e_z inside the lumen (r < r_inlet), zero outside
  outlet z = Z_OUT: p = 0, zero traction (natural)
  sides           : no-slip
Immersed solids (IBInterpolation3D, one per solid; their spread forces are summed):
  wall  : tether only, f = BETA (X_ref - x)  -> rigid vessel
  valve : compressible neo-Hookean leaflets (E, nu from the paper); the attachment extension
          (part 5), embedded in the wall, is tethered as well (VALVE_TETHER=1)

    python generate_mesh.py && python main.py                 # smoke test, H=0.2
    STEPS=200 python main.py                                  # a few hundred steps
    mpirun -n 4 python main.py
"""
import json
import os
import sys
import time

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

import dolfinx
from dolfinx import log
from dolfinx.fem import (Function, functionspace, dirichletbc, locate_dofs_topological, form)
from dolfinx.fem.petsc import create_vector, assemble_vector
from dolfinx.io import XDMFFile
from dolfinx.mesh import CellType, GhostMode, locate_entities_boundary
from basix.ufl import element
from ufl import TestFunction, SpatialCoordinate, grad, inner, dx, variable, det, tr, ln, diff

from afsic import ChorinSolver, IBMesh3D, IBInterpolation3D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import configuration as cfg          # noqa: E402

comm = MPI.COMM_WORLD
rank = comm.rank
with open(os.path.join(cfg.MESH_DIR, "meta.json")) as f:
    meta = json.load(f)
R_IN = meta["r_inlet"]
A_IN = np.pi * R_IN ** 2
OUT = cfg.output_path()
W = cfg.HALF_WIDTH

# =================================================================================================
# fluid
# =================================================================================================
mesh = dolfinx.mesh.create_box(comm, ((-W, -W, cfg.Z_IN), (W, W, cfg.Z_OUT)), (cfg.NX, cfg.NY, cfg.NZ),
                               cell_type=CellType.hexahedron, ghost_mode=GhostMode.shared_facet)
gdim, fdim = 3, 2
V = functionspace(mesh, element("Lagrange", mesh.topology.cell_name(), cfg.VELOCITY_ORDER, shape=(gdim,)))
Q = functionspace(mesh, element("Lagrange", mesh.topology.cell_name(), cfg.PRESSURE_ORDER))
V_io = functionspace(mesh, element("Lagrange", mesh.topology.cell_name(), 1, shape=(gdim,)))

f_in = locate_entities_boundary(mesh, fdim, lambda x: np.isclose(x[2], cfg.Z_IN))
f_out = locate_entities_boundary(mesh, fdim, lambda x: np.isclose(x[2], cfg.Z_OUT))
f_side = locate_entities_boundary(mesh, fdim, lambda x: np.isclose(np.abs(x[0]), W) | np.isclose(np.abs(x[1]), W))

# plug profile with unit mean velocity in the lumen; scaled by Q(t)/A every step
u_in = Function(V)
u_in.interpolate(lambda x: np.vstack([0 * x[0], 0 * x[0], (np.hypot(x[0], x[1]) < R_IN).astype(float)]))
plug = u_in.x.array.copy()
u_zero = np.zeros(gdim, dtype=PETSc.ScalarType)
bcu = [dirichletbc(u_in, locate_dofs_topological(V, fdim, f_in)),
       dirichletbc(u_zero, locate_dofs_topological(V, fdim, f_side), V)]
bcp = [dirichletbc(PETSc.ScalarType(0.0), locate_dofs_topological(Q, fdim, f_out), Q)]
ns = ChorinSolver(V, Q, bcu, bcp, cfg.DT, cfg.RHO, cfg.MU)

# =================================================================================================
# immersed solids
# =================================================================================================
def read_solid(name):
    with XDMFFile(comm, os.path.join(cfg.MESH_DIR, name), "r") as fx:
        s = fx.read_mesh(name="Grid")
    Vs = functionspace(s, element("Lagrange", s.topology.cell_name(), cfg.FORCE_ORDER, shape=(gdim,)))
    return s, Vs


class Solid:
    def __init__(self, name, ibmesh):
        self.mesh, self.V = read_solid(name)
        self.x = Function(self.V, name="position")          # current position
        self.x.interpolate(lambda X: np.vstack([X[0], X[1], X[2]]))
        self.x_ref = self.x.x.array.copy()
        self.v = Function(self.V)
        self.force = Function(self.V, name="force")
        self.b = create_vector(self.V)
        self.ib = IBInterpolation3D(ibmesh)
        self.ib.evaluate_current_points(self.x._cpp_object)
        self.f_fluid = Function(V)

    def set_form(self, L):
        self.L = form(L)

    def advance(self):
        self.ib.fluid_to_solid(ns.u_._cpp_object, self.v._cpp_object)
        self.x.x.array[:] += self.v.x.array * cfg.DT
        self.x.x.scatter_forward()
        self.ib.evaluate_current_points(self.x._cpp_object)
        with self.b.localForm() as loc:
            loc.set(0)
        assemble_vector(self.b, self.L)
        self.b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        with self.b.getBuffer() as arr:
            self.force.x.array[:len(arr)] = arr[:]
        self.ib.solid_to_fluid(self.f_fluid._cpp_object, self.force._cpp_object)

    def max_disp(self):
        d = np.abs(self.x.x.array - self.x_ref).reshape(-1, 3)
        local = np.linalg.norm(d, axis=1).max() if len(d) else 0.0
        return comm.allreduce(local, op=MPI.MAX)


ibmesh = IBMesh3D(-W, W, -W, W, cfg.Z_IN, cfg.Z_OUT, cfg.NX, cfg.NY, cfg.NZ, cfg.VELOCITY_ORDER)
coords_bg = Function(V)
coords_bg.interpolate(lambda x: np.vstack([x[0], x[1], x[2]]))
ibmesh.build_map(coords_bg._cpp_object)

# wall: tether only
wall = Solid("wall.xdmf", ibmesh)
Xw = SpatialCoordinate(wall.mesh)
wall.set_form(-cfg.BETA * inner(wall.x - Xw, TestFunction(wall.V)) * dx(domain=wall.mesh))

# valve: compressible neo-Hookean (psi = mu/2 (I1-3) - mu ln J + lam/2 (ln J)^2) + tether on part 5
valve = Solid("valve.xdmf", ibmesh)
part_file = np.load(os.path.join(cfg.MESH_DIR, "valve_part.npy"))
tdim = valve.mesh.topology.dim
n_local = valve.mesh.topology.index_map(tdim).size_local + valve.mesh.topology.index_map(tdim).num_ghosts
part_local = part_file[np.asarray(valve.mesh.topology.original_cell_index)[:n_local]]
DG0 = functionspace(valve.mesh, ("DG", 0))
chi_ext = Function(DG0)
chi_ext.x.array[:] = (part_local == 5).astype(float) * (1.0 if cfg.VALVE_TETHER else 0.0)
mu_s = cfg.E_LEAFLET / (2 * (1 + cfg.NU_LEAFLET))
lam_s = cfg.E_LEAFLET * cfg.NU_LEAFLET / ((1 + cfg.NU_LEAFLET) * (1 - 2 * cfg.NU_LEAFLET))
F = variable(grad(valve.x))
J = det(F)
psi = mu_s / 2 * (tr(F.T * F) - 3) - mu_s * ln(J) + lam_s / 2 * ln(J) ** 2
P = diff(psi, F)
Xv = SpatialCoordinate(valve.mesh)
dv = TestFunction(valve.V)
dxv = dx(domain=valve.mesh)
valve.set_form(-inner(P, grad(dv)) * dxv - cfg.BETA * chi_ext * inner(valve.x - Xv, dv) * dxv)

# =================================================================================================
# diagnostics
# =================================================================================================
u_io = Function(V_io, name="velocity")
fu = XDMFFile(comm, OUT + "velocity.xdmf", "w")
fu.write_mesh(mesh)
fp = XDMFFile(comm, OUT + "pressure.xdmf", "w")
fp.write_mesh(mesh)
fw = XDMFFile(comm, OUT + "wall.xdmf", "w")
fw.write_mesh(wall.mesh)
fv = XDMFFile(comm, OUT + "valve.xdmf", "w")
fv.write_mesh(valve.mesh)
Vw_io = functionspace(wall.mesh, element("Lagrange", wall.mesh.topology.cell_name(), 1, shape=(gdim,)))
Vv_io = functionspace(valve.mesh, element("Lagrange", valve.mesh.topology.cell_name(), 1, shape=(gdim,)))
dw_io, dv_io = Function(Vw_io, name="displacement"), Function(Vv_io, name="displacement")
disp_w, disp_v = Function(wall.V), Function(valve.V)

hist = open(OUT + "history.csv", "w") if rank == 0 else None
if rank == 0:
    hist.write("step,t,Q_in,max_u,max_p,wall_max_disp,valve_max_disp,wall_force_norm,valve_force_norm\n")
    print(f"semilunar valve: grid {cfg.NX}x{cfg.NY}x{cfg.NZ} (h={cfg.H}), dt={cfg.DT}, steps={cfg.NSTEPS}, "
          f"wall {meta['wall_tets']} tets, valve {meta['valve_tets']} tets, r_in={R_IN:.4f}, out={OUT}", flush=True)

log.set_log_level(log.LogLevel.WARNING)
t0 = time.time()
for step in range(cfg.NSTEPS):
    t = (step + 1) * cfg.DT
    u_in.x.array[:] = plug * (cfg.q_inlet(t) / A_IN)
    ns.solve_one_step()
    wall.advance()
    valve.advance()
    ns.f.x.array[:] = wall.f_fluid.x.array + valve.f_fluid.x.array
    ns.f.x.scatter_forward()

    umax = comm.allreduce(np.abs(ns.u_.x.array).max() if ns.u_.x.array.size else 0.0, op=MPI.MAX)
    finite = comm.allreduce(bool(np.isfinite(ns.u_.x.array).all() and np.isfinite(valve.x.x.array).all()),
                            op=MPI.LAND)
    if step % max(cfg.NSTEPS // 200, 1) == 0 or step == cfg.NSTEPS - 1 or not finite:
        wd, vd = wall.max_disp(), valve.max_disp()
        wf = comm.allreduce(float(np.square(wall.force.x.array).sum()), op=MPI.SUM) ** 0.5
        vf = comm.allreduce(float(np.square(valve.force.x.array).sum()), op=MPI.SUM) ** 0.5
        pin = comm.allreduce(float(np.abs(ns.p_.x.array).max()) if ns.p_.x.array.size else 0.0, op=MPI.MAX)
        if rank == 0:
            hist.write(f"{step + 1},{t:.6f},{cfg.q_inlet(t):.4f},{umax:.6e},{pin:.6e},{wd:.6e},{vd:.6e},{wf:.6e},{vf:.6e}\n")
            hist.flush()
            print(f"step {step + 1:6d} t={t:.4f} Q={cfg.q_inlet(t):7.2f} max|u|={umax:9.3f} max|p|={pin:10.2f} "
                  f"wall disp={wd:.2e} valve disp={vd:.2e}  ({time.time() - t0:.0f}s)", flush=True)
    if not finite:
        if rank == 0:
            print(f"non-finite values at step {step + 1}, stopping", flush=True)
        break
    if (step + 1) % cfg.OUTPUT_EVERY == 0 or step == cfg.NSTEPS - 1:
        u_io.interpolate(ns.u_)
        fu.write_function(u_io, t)
        fp.write_function(ns.p_, t)
        disp_w.x.array[:] = wall.x.x.array - wall.x_ref
        disp_v.x.array[:] = valve.x.x.array - valve.x_ref
        dw_io.interpolate(disp_w)
        dv_io.interpolate(disp_v)
        fw.write_function(dw_io, t)
        fv.write_function(dv_io, t)

for fx in (fu, fp, fw, fv):
    fx.close()
if rank == 0:
    hist.close()
    print(f"done in {time.time() - t0:.0f}s -> {OUT}", flush=True)
