"""Control case for demo_424: the same box and the same two-end pressure
Dirichlet conditions, but with NO immersed solid.

For a plane channel of half-width A = BOX_W/2 with no-slip walls the exact
solution of the incompressible Navier-Stokes system is the parallel flow

    u(y) = G/(2 mu) (A^2 - (y-A)^2),   G = DP / BOX_L,

because the convective term vanishes identically.  This isolates the open
(pressure-driven) boundary treatment from the immersed-boundary coupling: any
spurious boundary layer here is a property of the solver, not of the solid.

Run:
    NY=45 T_END=0.2 python test_channel.py
"""
import time

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

import dolfinx
from dolfinx import log
from dolfinx.fem import (functionspace, dirichletbc, locate_dofs_topological)
from dolfinx.mesh import (CellType, GhostMode, locate_entities, meshtags)
from basix.ufl import element

import configuration as cfg
import verify as vf
from afsic import ChorinSolver, IPCSSolver

t0 = time.time()
A = 0.5 * cfg.BOX_W

mesh = dolfinx.mesh.create_rectangle(
    comm=MPI.COMM_WORLD,
    points=((0.0, 0.0), (cfg.BOX_L, cfg.BOX_W)),
    n=(cfg.NX, cfg.NY),
    cell_type=CellType.quadrilateral,
    ghost_mode=GhostMode.shared_facet,
)
fdim = mesh.topology.dim - 1
mesh.topology.create_connectivity(fdim, mesh.topology.dim)

MARKERS = [(1, lambda x: np.isclose(x[0], 0.0)),
           (2, lambda x: np.isclose(x[0], cfg.BOX_L)),
           (3, lambda x: np.isclose(x[1], 0.0)),
           (4, lambda x: np.isclose(x[1], cfg.BOX_W))]
idx, mkr = [], []
for m, loc in MARKERS:
    f = locate_entities(mesh, fdim, loc)
    idx.append(f)
    mkr.append(np.full_like(f, m))
idx = np.hstack(idx).astype(np.int32)
mkr = np.hstack(mkr).astype(np.int32)
o = np.argsort(idx)
facet_tag = meshtags(mesh, fdim, idx[o], mkr[o])

V = functionspace(mesh, element("Lagrange", mesh.topology.cell_name(),
                                cfg.VELOCITY_ORDER, shape=(mesh.geometry.dim,)))
Q = functionspace(mesh, element("Lagrange", mesh.topology.cell_name(),
                                cfg.PRESSURE_ORDER))

u_zero = np.zeros(mesh.geometry.dim, dtype=PETSc.ScalarType)
bcu = [dirichletbc(u_zero, locate_dofs_topological(V, fdim, facet_tag.find(m)), V)
       for m in (3, 4)]
dofs_in = locate_dofs_topological(Q, fdim, facet_tag.find(1))
dofs_out = locate_dofs_topological(Q, fdim, facet_tag.find(2))
bc_out = dirichletbc(PETSc.ScalarType(0.0), dofs_out, Q)


def make_bcp(v):
    return [dirichletbc(PETSc.ScalarType(v), dofs_in, Q), bc_out]


bcp = make_bcp(0.0)
if cfg.SOLVER == "chorin":
    ns = ChorinSolver(V, Q, bcu, bcp, cfg.DT, cfg.RHO, cfg.MU)
else:
    ns = IPCSSolver(V, Q, bcu, bcp, cfg.DT, cfg.RHO, cfg.MU)

log.set_log_level(log.LogLevel.WARNING)
p_prev = 0.0
for step in range(cfg.NSTEPS):
    t = step * cfg.DT
    pt = cfg.p_inlet(t)
    ns.bcp = make_bcp(pt if cfg.SOLVER == "chorin" else pt - p_prev)
    p_prev = pt
    ns.solve_one_step()

G = cfg.DP / cfg.BOX_L
y = np.arange(cfg.NY + 1) * cfg.H
u_an = G / (2.0 * cfg.MU) * (A**2 - (y - A) ** 2)
p_an = cfg.DP * (1.0 - np.arange(cfg.NX + 1) * cfg.H / cfg.BOX_L)

xs = np.arange(cfg.NX + 1) * cfg.H
pn = vf.sample_p(ns.p_, mesh, xs, A)
un = vf.sample_u(ns.u_, mesh, 0.5 * cfg.BOX_L, y)[:, 0]

print(f"pure-fluid plane channel control (no solid), NY={cfg.NY}, "
      f"T_end={cfg.T_END:g}, {cfg.SOLVER}")
print(f"exact: G={G:g} Pa/m, u_max={G*A**2/(2*cfg.MU):.6g} m/s")
print("\npressure along y=A:")
for k in range(0, min(12, len(xs))):
    print(f"   x={xs[k]:8.5f}  p_num={pn[k]: .6e}  p_exact={p_an[k]: .6e}")
print("   ...")
print(f"   x={xs[-1]:8.5f}  p_num={pn[-1]: .6e}  p_exact={p_an[-1]: .6e}")

ok = np.isfinite(pn)
slope = -np.polyfit(xs[ok], pn[ok], 1)[0]
print(f"\nfitted G over the whole box = {slope:.6g} Pa/m (exact {G:.6g})")

ok = np.isfinite(un)
e = un[ok] - u_an[ok]
print(f"u_max num={np.nanmax(un):.6e}  exact={u_an.max():.6e}  "
      f"rel={np.nanmax(un)/u_an.max()-1:.3e}")
print(f"profile |e|_inf={np.max(np.abs(e)):.6e} m/s  "
      f"rel-L2={np.sqrt(np.trapezoid(e**2, y[ok]))/np.sqrt(np.trapezoid(u_an[ok]**2, y[ok])):.4e}")
print(f"max|u| on nodes = {np.max(np.linalg.norm(un.reshape(-1,1),axis=1)):.6e}")
print(f"elapsed {time.time()-t0:.1f} s")
