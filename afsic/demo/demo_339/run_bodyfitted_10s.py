#!/usr/bin/env python
"""Run body_fitted to T=10s with Cd/Cl + wake probe monitoring (reference).

用于与 multi_direct_forcing 的 10s 结果对比：判断 St≈0.52/Cd≈2.53 是
mdf 的精度问题，还是本套网格(220×41)+一阶 Chorin 的固有限制。
"""
import os
import numpy as np
from petsc4py import PETSc
from mpi4py import MPI

import dolfinx
from dolfinx.fem import (Function, functionspace, dirichletbc,
                         locate_dofs_topological, form, assemble_scalar)
from dolfinx.mesh import CellType, GhostMode
from basix.ufl import element
from ufl import dot, dx, ds, inner, grad, FacetNormal, as_vector, Measure
from afsic import ChorinSolver
from afsic.common import TurekInlet

comm = MPI.COMM_WORLD
rank = comm.rank
P = dict(Um=1.0, rho=1000.0, mu=1.0, Lx=2.2, Ly=0.41,
         Nx=220, Ny=41, T=10.0, dt=0.001, D=0.1)
BASE = os.path.dirname(os.path.abspath(__file__))
N = int(os.environ.get("STEPS", str(int(P["T"] / P["dt"]))))
out_every = 20

from dolfinx.io import gmsh as gmshio
md = gmshio.read_from_msh(os.path.join(BASE, "body_fitted", "channel_hole.msh"),
                          comm, gdim=2)
mesh, _, facet_tags = md[0], md[1], md[2]
mesh.topology.create_connectivity(1, 2)
fdim = mesh.topology.dim - 1
V = functionspace(mesh, element("Lagrange", mesh.topology.cell_name(), 2, shape=(2,)))
Q = functionspace(mesh, element("Lagrange", mesh.topology.cell_name(), 1))
inlet = TurekInlet(Um=P["Um"], Ly=P["Ly"])
ui = Function(V); ui.interpolate(inlet)
bci = dirichletbc(ui, locate_dofs_topological(V, fdim, facet_tags.find(11)))
u0 = Function(V); u0.x.array[:] = 0.0
bcb = dirichletbc(u0, locate_dofs_topological(V, fdim, facet_tags.find(13)))
bct = dirichletbc(u0, locate_dofs_topological(V, fdim, facet_tags.find(14)))
bcc = dirichletbc(u0, locate_dofs_topological(V, fdim, facet_tags.find(15)))
bcp = dirichletbc(PETSc.ScalarType(0.0),
                  locate_dofs_topological(Q, fdim, facet_tags.find(12)), Q)
sol = ChorinSolver(V, Q, [bci, bcb, bct, bcc], [bcp], P["dt"], P["rho"], P["mu"])

n = FacetNormal(mesh)
dsM = Measure("ds", domain=mesh, subdomain_data=facet_tags)
e_x = as_vector((1.0, 0.0)); e_y = as_vector((0.0, 1.0))
traction = -sol.p_ * n + P["mu"] * dot(grad(sol.u_) + grad(sol.u_).T, n)
form_Fd = form(dot(traction, e_x) * dsM(15))
form_Fl = form(dot(traction, e_y) * dsM(15))

import dolfinx.geometry as dg
tree = dg.bb_tree(mesh, mesh.topology.dim)
probe = np.array([[0.6, 0.2, 0.0]], dtype=np.float64)
cand = dg.compute_collisions_points(tree, probe)
col = dg.compute_colliding_cells(mesh, cand, probe)
pcell = col.links(0)[0]

logf = open(os.path.join(BASE, "_short_run", "bodyfitted_10s.log"), "w")
logf.write("# t Cd Cl uy_wake\n")
if rank == 0:
    print("body_fitted 10s reference: monitoring Cd/Cl/wake uy every "
          f"{out_every} steps", flush=True)
for k in range(N):
    t = k * P["dt"]; inlet.update(t); ui.interpolate(inlet)
    sol.solve_one_step()
    if k % out_every == 0 or k == N - 1:
        fd = comm.allreduce(assemble_scalar(form_Fd), op=MPI.SUM)
        fl = comm.allreduce(assemble_scalar(form_Fl), op=MPI.SUM)
        Cd = -2.0 * fd / (P["rho"] * P["Um"]**2 * P["D"])
        Cl = -2.0 * fl / (P["rho"] * P["Um"]**2 * P["D"])
        vu = np.ravel(sol.u_.eval(probe, np.array([pcell], dtype=np.int32)))
        if rank == 0:
            logf.write(f"{t:.4f} {Cd:.6f} {Cl:.6f} {vu[1]:.6f}\n")
            if k % 500 == 0 or k == N - 1:
                print(f"t={t:.2f}s Cd={Cd:.4f} Cl={Cl:+.4f} uy_wake={vu[1]:+.4f}",
                      flush=True)
logf.close()

# 分析: t>6s 准稳态段的 Cd 均值、Cl 振荡、St
import re
rows = []
for ln in open(os.path.join(BASE, "_short_run", "bodyfitted_10s.log")):
    if ln.startswith("#"): continue
    p = ln.split()
    if len(p) == 4:
        rows.append((float(p[0]), float(p[1]), float(p[2]), float(p[3])))
rows = np.array(rows)
m = rows[:, 0] > 6.0
cd, cl, uy = rows[m, 1], rows[m, 2], rows[m, 3]
if rank == 0 and len(cd) > 0:
    z = uy - uy.mean(); zc = np.where(np.diff(np.sign(z)) != 0)[0]
    per = np.diff(rows[m, 0][zc]); per = per[per > 0.05]
    T = per.mean() if len(per) else np.nan
    St = (1.0 / T) * P["D"] if T == T else np.nan
    print("\n=== body_fitted 10s 参考 (t>6s) ===")
    print(f"  Cd 均值 = {cd.mean():.4f}  (峰峰 {cd.max()-cd.min():.4f})")
    print(f"  Cl 均值 = {cl.mean():+.4f}  峰峰 {cl.max()-cl.min():.4f}")
    print(f"  wake uy 峰峰 = {uy.max()-uy.min():.4f}  周期≈{T:.3f}s  St≈{St:.3f}")
    print(f"  (DFG 2D-3 参考: Cd≈5.57, Cl幅值≈0.0106, St≈0.3)")
