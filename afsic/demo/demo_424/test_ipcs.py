"""IPCS vs Chorin on the pressure-driven plane channel, with instrumentation.

Same box, same two-end pressure Dirichlet conditions as test_channel.py, but
tiny grids so the diagnosis is cheap.  Prints the accumulated pressure p_ and
the pressure increment phi along the centreline as the ramp proceeds, which
shows exactly where the two schemes diverge.

Run:
    NY=9 T_END=0.2 python test_ipcs.py
    NY=9 T_END=0.2 SOLVER=chorin python test_ipcs.py
"""
import os
import time

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

import dolfinx
from dolfinx import log
from dolfinx.fem import functionspace, dirichletbc, locate_dofs_topological
from dolfinx.mesh import CellType, GhostMode, locate_entities, meshtags
from basix.ufl import element

import configuration as cfg
import verify as vf
from afsic import ChorinSolver, IPCSSolver

t0 = time.time()
A = 0.5 * cfg.BOX_W
LOG = os.environ.get("IPCS_LOG", "1") == "1"

mesh = dolfinx.mesh.create_rectangle(
    comm=MPI.COMM_WORLD, points=((0.0, 0.0), (cfg.BOX_L, cfg.BOX_W)),
    n=(cfg.NX, cfg.NY), cell_type=CellType.quadrilateral,
    ghost_mode=GhostMode.shared_facet)
fdim = mesh.topology.dim - 1
mesh.topology.create_connectivity(fdim, mesh.topology.dim)

MARKERS = [(1, lambda x: np.isclose(x[0], 0.0)),
           (2, lambda x: np.isclose(x[0], cfg.BOX_L)),
           (3, lambda x: np.isclose(x[1], 0.0)),
           (4, lambda x: np.isclose(x[1], cfg.BOX_W))]
idx, mkr = [], []
for m, loc in MARKERS:
    f = locate_entities(mesh, fdim, loc)
    idx.append(f); mkr.append(np.full_like(f, m))
idx = np.hstack(idx).astype(np.int32); mkr = np.hstack(mkr).astype(np.int32)
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


# DRIVE=pressure : p = p_in(t) at the inlet and 0 at the outlet (both ends)
# DRIVE=bodyforce: uniform body force G*e_x plus a pressure Dirichlet 0 at the
#                  OUTLET only -- the only pressure Dirichlet is HOMOGENEOUS.
DRIVE = os.environ.get("DRIVE", "pressure").lower()
_IPCS = IPCSSolver if cfg.SOLVER == "ipcs" else None
if cfg.SOLVER == "ipcs_fix":
    from ipcs_traction import IPCSSolverTraction
    from ufl import Measure
    ds_inlet = Measure("ds", domain=mesh, subdomain_data=facet_tag)(1)
    p_traction = dolfinx.fem.Constant(mesh, PETSc.ScalarType(0.0))
    ns = IPCSSolverTraction(V, Q, bcu, make_bcp(0.0), cfg.DT, cfg.RHO, cfg.MU,
                            ds_inlet, p_traction)
elif DRIVE == "bodyforce":
    ns = (ChorinSolver if cfg.SOLVER == "chorin" else IPCSSolver)(
        V, Q, bcu, [bc_out], cfg.DT, cfg.RHO, cfg.MU)
else:
    ns = (ChorinSolver if cfg.SOLVER == "chorin" else IPCSSolver)(
        V, Q, bcu, make_bcp(0.0), cfg.DT, cfg.RHO, cfg.MU)

# where to watch: node columns along the centreline
probe_j = [0, 1, 2, cfg.NX // 4, cfg.NX // 2, 3 * cfg.NX // 4, cfg.NX]
probe_x = np.array([j * cfg.H for j in probe_j])

log.set_log_level(log.LogLevel.WARNING)
G = cfg.DP / cfg.BOX_L
hdr = "  step      t     " + "".join(f"x={x:<9.4f}" for x in probe_x)
if DRIVE == "bodyforce":
    # f enters the momentum equation with opposite signs in the two solvers
    # (Chorin: -f.v ; IPCS: +f.v), so the sign is chosen per solver.
    sgn = 1.0 if cfg.SOLVER == "chorin" else -1.0
    gx = sgn * (cfg.DP / cfg.BOX_L)
    ns.f.interpolate(lambda x: np.array([gx + 0.0 * x[0], 0.0 * x[0]]))
    ns.f.x.scatter_forward()

print(f"# solver={cfg.SOLVER}  drive={DRIVE}  NY={cfg.NY}  NX={cfg.NX}  h={cfg.H:g}  "
      f"DP={cfg.DP:.4f} Pa  exact G={G:.4f} Pa/m")
if LOG:
    print(hdr, flush=True)

p_prev = 0.0
for step in range(cfg.NSTEPS):
    t = step * cfg.DT
    pt = cfg.p_inlet(t)
    if DRIVE == "bodyforce":
        pass
    else:
        ns.bcp = make_bcp(pt if cfg.SOLVER == "chorin" else pt - p_prev)
        if cfg.SOLVER == "ipcs_fix":
            # the *full* physical pressure is the traction on the inlet facets
            ns.p_traction.value = pt
        p_prev = pt
    ns.solve_one_step()

    if LOG and (step % max(cfg.NSTEPS // 10, 1) == 0 or step == cfg.NSTEPS - 1):
        pv = vf.sample_p(ns.p_, mesh, probe_x, A)
        extra = ""
        if cfg.SOLVER != "chorin":
            ph = vf.sample_p(ns.phi, mesh, probe_x, A)
            extra = ("   phi=[" + " ".join(f"{v: .3e}" for v in ph) + "]")
        print(f"  {step:>5} {t:8.4f}  " +
              "".join(f"{v: .4e}" for v in pv) + extra, flush=True)

# ---- final comparison -----------------------------------------------------
xs = np.arange(cfg.NX + 1) * cfg.H
pn = vf.sample_p(ns.p_, mesh, xs, A)
p_ex = cfg.DP * (1.0 - xs / cfg.BOX_L)
y = np.arange(cfg.NY + 1) * cfg.H
u_an = G / (2.0 * cfg.MU) * (A**2 - (y - A) ** 2)
un = vf.sample_u(ns.u_, mesh, 0.5 * cfg.BOX_L, y)[:, 0]

print(f"\nfinal: fitted G = {-np.polyfit(xs, pn, 1)[0]:.6g} Pa/m "
      f"(exact {G:.6g})")
print(f"       max |p - p_exact| = {np.max(np.abs(pn - p_ex)):.4e} Pa")
print(f"       u_max num = {np.nanmax(un):.6e}, exact = {u_an.max():.6e}, "
      f"rel = {np.nanmax(un)/u_an.max()-1:.4e}")
print(f"elapsed {time.time()-t0:.1f} s")
