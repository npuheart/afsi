#!/usr/bin/env python
"""Compare direct_forcing vs body_fitted — same params, developed flow.

对两种方法各自推进 N 步（统一参数，越过 t=2s 入口斜坡），
在相同时刻比较：
  - 全场 u_L2 / p_L2 范数
  - 下游中心线上若干探针点的速度
  - 圆柱上游驻点压力（探针）
  - body_fitted: 圆柱表面应力积分得到的 Cd（物理基准）
  - direct_forcing: 其代码自带的 Cd（标记速度代理）+ 体积力积分 Cd

用法：
  conda activate afsi-dolfinx
  python compare_df_bf.py          # 默认 3000 步 (t=3.0s)
  STEPS=2500 python compare_df_bf.py
"""
import os
import sys
import types
import importlib.util

import numpy as np
from petsc4py import PETSc
from mpi4py import MPI

import dolfinx
from dolfinx.fem import (Function, functionspace, dirichletbc,
                         locate_dofs_topological, form, assemble_scalar)
from dolfinx.mesh import CellType, GhostMode, locate_entities, meshtags
from basix.ufl import element
from ufl import (dot, dx, ds, inner, grad, div, FacetNormal, as_vector,
                 SpatialCoordinate, Measure)

from afsic import ChorinSolver, IBMesh, IBInterpolation
from afsic.common import (tag_boundaries, rectangle_boundaries, TurekInlet,
                          MARKER_LEFT, MARKER_RIGHT, MARKER_BOTTOM, MARKER_TOP)
import dolfinx.geometry as dg

comm = MPI.COMM_WORLD
rank = comm.rank
N = int(os.environ.get("STEPS", "3000"))

# 统一参数（SI）
P = dict(Um=1.0, rho=1000.0, mu=1.0, Lx=2.2, Ly=0.41,
         Nx=220, Ny=41, T=10.0, dt=0.001, D=0.1,
         cx=0.2, cy=0.2, r=0.05)

BASE = os.path.dirname(os.path.abspath(__file__))


def load_cfg(sub):
    spec = importlib.util.spec_from_file_location(
        "cfg_" + sub, os.path.join(BASE, sub, "configuration.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.config


# ---------------------------------------------------------------------------
# body_fitted
# ---------------------------------------------------------------------------
def run_body_fitted():
    from dolfinx.io import gmsh as gmshio
    mesh_path = os.path.join(BASE, "body_fitted", "channel_hole.msh")
    md = gmshio.read_from_msh(mesh_path, comm, gdim=2)
    mesh, cell_tags, facet_tags = md[0], md[1], md[2]
    mesh.topology.create_connectivity(1, 2)
    fdim = mesh.topology.dim - 1

    v_cg2 = element("Lagrange", mesh.topology.cell_name(), 2, shape=(2,))
    v_cg1 = element("Lagrange", mesh.topology.cell_name(), 1, shape=(2,))
    s_cg1 = element("Lagrange", mesh.topology.cell_name(), 1)
    V = functionspace(mesh, v_cg2)
    Q = functionspace(mesh, s_cg1)

    inlet = TurekInlet(Um=P["Um"], Ly=P["Ly"])
    ui = Function(V); ui.interpolate(inlet)
    bci = dirichletbc(ui, locate_dofs_topological(V, fdim, facet_tags.find(11)))
    u0 = Function(V); u0.x.array[:] = 0.0
    bcb = dirichletbc(u0, locate_dofs_topological(V, fdim, facet_tags.find(13)))
    bct = dirichletbc(u0, locate_dofs_topological(V, fdim, facet_tags.find(14)))
    bcc = dirichletbc(u0, locate_dofs_topological(V, fdim, facet_tags.find(15)))
    bcp = dirichletbc(PETSc.ScalarType(0.0),
                      locate_dofs_topological(Q, fdim, facet_tags.find(12)), Q)
    solver = ChorinSolver(V, Q, [bci, bcb, bct, bcc], [bcp],
                          P["dt"], P["rho"], P["mu"])
    for _ in range(N):
        t = _ * P["dt"]; inlet.update(t); ui.interpolate(inlet)
        solver.solve_one_step()

    # 圆柱表面应力积分 → Cd (物理基准)
    n = FacetNormal(mesh)
    e_x = as_vector((1.0, 0.0))
    mu, rho, Um, D = P["mu"], P["rho"], P["Um"], P["D"]
    dsM = Measure("ds", domain=mesh, subdomain_data=facet_tags)
    traction = -solver.p_ * n + mu * dot(grad(solver.u_) + grad(solver.u_).T, n)
    fdrag = assemble_scalar(form(dot(traction, e_x) * dsM(15)))
    fdrag = mesh.comm.allreduce(fdrag, op=MPI.SUM)
    # 圆柱所受阻力 = -∫ (σ·n_fluid)·e_x dS（n_fluid 指向圆柱内）
    Cd = -2.0 * fdrag / (rho * Um**2 * D)
    return mesh, solver, dict(Cd=Cd)


# ---------------------------------------------------------------------------
# direct_forcing (delta-kernel 版本, 与 main.py 相同)
# ---------------------------------------------------------------------------
def run_direct_forcing():
    mesh = dolfinx.mesh.create_rectangle(
        comm=comm, points=((0.0, 0.0), (P["Lx"], P["Ly"])),
        n=(P["Nx"], P["Ny"]), cell_type=CellType.quadrilateral,
        ghost_mode=GhostMode.shared_facet)
    ftag = tag_boundaries(mesh, rectangle_boundaries(P["Lx"], P["Ly"]))
    v_cg2 = element("Lagrange", mesh.topology.cell_name(), 2, shape=(2,))
    s_cg1 = element("Lagrange", mesh.topology.cell_name(), 1)
    V = functionspace(mesh, v_cg2)
    Q = functionspace(mesh, s_cg1)
    fdim = mesh.topology.dim - 1

    inlet = TurekInlet(Um=P["Um"], Ly=P["Ly"])
    ui = Function(V); ui.interpolate(inlet)
    bci = dirichletbc(ui, locate_dofs_topological(V, fdim, ftag.find(MARKER_LEFT)))
    u0 = Function(V); u0.x.array[:] = 0.0
    bcb = dirichletbc(u0, locate_dofs_topological(V, fdim, ftag.find(MARKER_BOTTOM)))
    bct = dirichletbc(u0, locate_dofs_topological(V, fdim, ftag.find(MARKER_TOP)))
    bcp = dirichletbc(PETSc.ScalarType(0.0),
                      locate_dofs_topological(Q, fdim, ftag.find(MARKER_RIGHT)), Q)
    solver = ChorinSolver(V, Q, [bci, bcb, bct], [bcp], P["dt"], P["rho"], P["mu"])

    solid_path = os.path.join(BASE, "direct_forcing", "cylinder_solid.xdmf")
    with dolfinx.io.XDMFFile(comm, solid_path, "r") as xdmf:
        structure = xdmf.read_mesh(name="mesh")
        structure.topology.create_connectivity(
            structure.topology.dim, structure.topology.dim - 1)
    v_s = element("Lagrange", structure.topology.cell_name(), 2, shape=(2,))
    Vs = functionspace(structure, v_s)
    solid_coords = Function(Vs)
    solid_coords.interpolate(lambda x: np.array([x[0], x[1]]))

    ibmesh = IBMesh(0.0, P["Lx"], 0.0, P["Ly"], P["Nx"], P["Ny"], 2)
    ib_interp = IBInterpolation(ibmesh)
    coords_bg = Function(V)
    coords_bg.interpolate(lambda x: np.array([x[0], x[1]]))
    ibmesh.build_map(coords_bg._cpp_object)
    ib_interp.evaluate_current_points(solid_coords._cpp_object)

    fluid_one = Function(V)
    one_arr = fluid_one.x.array
    one_arr[0::2] = 1.0; one_arr[1::2] = 1.0
    fluid_one.x.scatter_forward()
    solid_w = Function(Vs)
    ib_interp.fluid_to_solid(fluid_one._cpp_object, solid_w._cpp_object)
    solid_w.x.scatter_forward()
    fluid_alpha = Function(V)
    ib_interp.solid_to_fluid(fluid_alpha._cpp_object, solid_w._cpp_object)
    fluid_alpha.x.scatter_forward()
    alpha_arr = fluid_alpha.x.array
    eps = 1e-6
    bs = V.dofmap.index_map_bs

    solid_vel = Function(Vs)
    solid_force = Function(Vs)
    fluid_force = Function(V)
    rho, dV = P["rho"], (P["Lx"]/P["Nx"])*(P["Ly"]/P["Ny"])
    D, Um = P["D"], P["Um"]
    drag_hist, cd_hist, cd_body_hist = [], [], []

    for _ in range(N):
        t = _ * P["dt"]; inlet.update(t); ui.interpolate(inlet)
        solver.solve_one_step()
        ib_interp.fluid_to_solid(solver.u_._cpp_object, solid_vel._cpp_object)
        solid_vel.x.scatter_forward()
        sv = solid_vel.x.array; sf = solid_force.x.array
        drag_raw, lift_raw = 0.0, 0.0
        for k in range(len(sv)//bs):
            for d in range(bs):
                sf[k*bs+d] = -rho * sv[k*bs+d] / P["dt"]
                if d == 0: drag_raw += sv[k*bs+d]
                else:      lift_raw += sv[k*bs+d]
        solid_force.x.scatter_forward()
        ib_interp.solid_to_fluid(fluid_force._cpp_object, solid_force._cpp_object)
        fluid_force.x.scatter_forward()
        ff = fluid_force.x.array
        for i in range(len(ff)):
            a = alpha_arr[i]
            if a > eps: ff[i] /= a
        fluid_force.x.scatter_forward()
        u_arr = solver.u_.x.array
        for i in range(len(u_arr)):
            if alpha_arr[i] > 1.0: u_arr[i] = 0.0
        solver.u_n.x.array[:] = u_arr[:]
        solver.f.x.array[:] = ff[:]
        solver.f.x.scatter_forward()

        drag = comm.allreduce(drag_raw, op=MPI.SUM) * rho * dV / P["dt"]
        Cd = 2.0 * drag / (rho * Um**2 * D)
        # 体积力积分阻力: F_D = ∫ f_x dV（圆柱区域体积力）
        fv = solver.f.x.array.reshape(-1, bs)
        f_body = np.sum(fv[:, 0]) * dV
        f_body = comm.allreduce(f_body, op=MPI.SUM)
        Cd_body = 2.0 * f_body / (rho * Um**2 * D)
        drag_hist.append((t, drag)); cd_hist.append(Cd); cd_body_hist.append(Cd_body)

    return mesh, solver, dict(Cd_marker=cd_hist[-1],
                              Cd_bodyforce=cd_body_hist[-1],
                              drag_hist=drag_hist)


# ---------------------------------------------------------------------------
# 探针比较
# ---------------------------------------------------------------------------
def probes(mesh, u, p):
    pts = np.array([(0.18, 0.2), (0.25, 0.2), (0.3, 0.2), (0.4, 0.2),
                    (0.5, 0.2), (0.7, 0.2), (1.0, 0.2), (1.5, 0.2)],
                   dtype=np.float64)
    tree = dg.bb_tree(mesh, mesh.topology.dim)
    out = []
    for i in range(len(pts)):
        x0, y0 = pts[i]
        found = False
        for dy in (0.0, 1e-7, -1e-7, 2e-7, -2e-7):
            q = np.array([[x0, y0 + dy, 0.0]], dtype=np.float64)
            cand = dg.compute_collisions_points(tree, q)
            col = dg.compute_colliding_cells(mesh, cand, q)
            links = col.links(0)
            if len(links) > 0:
                cell = links[0]
                vu = np.ravel(u.eval(q, np.array([cell], dtype=np.int32)))
                vp = np.ravel(p.eval(q, np.array([cell], dtype=np.int32)))
                out.append((float(x0), float(vu[0]), float(vu[1]),
                            float(vp[0])))
                found = True
                break
        if not found:
            out.append((float(x0), np.nan, np.nan, np.nan))
    return out


def dump_fields(mesh, u, p, name, t):
    """把最终速度/压力场导出为 XDMF，便于可视化对比。
    (P2 速度需先插值到 P1，与各 main.py 的做法一致)"""
    out_dir = os.path.join(BASE, "_short_run", "compare")
    os.makedirs(out_dir, exist_ok=True)
    v1 = element("Lagrange", mesh.topology.cell_name(), 1, shape=(2,))
    V1 = functionspace(mesh, v1)
    u_io = Function(V1)
    u_io.interpolate(u)
    fv = dolfinx.io.XDMFFile(mesh.comm, os.path.join(out_dir, f"{name}_u.xdmf"), "w")
    fp = dolfinx.io.XDMFFile(mesh.comm, os.path.join(out_dir, f"{name}_p.xdmf"), "w")
    fv.write_mesh(mesh); fp.write_mesh(mesh)
    fv.write_function(u_io, t); fp.write_function(p, t)
    fv.close(); fp.close()


if __name__ == "__main__":
    t_end = N * P["dt"]
    if rank == 0:
        print(f"比较 direct_forcing vs body_fitted，N={N} 步，t_end={t_end:.2f}s\n", flush=True)

    # body_fitted
    if rank == 0:
        print("--- 运行 body_fitted ---", flush=True)
    mesh_b, sol_b, res_b = run_body_fitted()
    if rank == 0:
        print(f"[body_fitted] Cd(表面应力积分) = {res_b['Cd']:.4f}", flush=True)

    # direct_forcing
    if rank == 0:
        print("--- 运行 direct_forcing ---", flush=True)
    mesh_d, sol_d, res_d = run_direct_forcing()
    if rank == 0:
        print(f"[direct_forcing] Cd(标记代理)   = {res_d['Cd_marker']:.4f}", flush=True)
        print(f"[direct_forcing] Cd(体积力积分) = {res_d['Cd_bodyforce']:.4f}", flush=True)

    # 范数
    fu = form(dot(sol_b.u_, sol_b.u_) * dx); fp = form(dot(sol_b.p_, sol_b.p_) * dx)
    uL2_b = comm.allreduce(assemble_scalar(fu), op=MPI.SUM)
    pL2_b = comm.allreduce(assemble_scalar(fp), op=MPI.SUM)
    fu = form(dot(sol_d.u_, sol_d.u_) * dx); fp = form(dot(sol_d.p_, sol_d.p_) * dx)
    uL2_d = comm.allreduce(assemble_scalar(fu), op=MPI.SUM)
    pL2_d = comm.allreduce(assemble_scalar(fp), op=MPI.SUM)

    pb_b = probes(mesh_b, sol_b.u_, sol_b.p_)
    pb_d = probes(mesh_d, sol_d.u_, sol_d.p_)
    dump_fields(mesh_b, sol_b.u_, sol_b.p_, "body_fitted", t_end)
    dump_fields(mesh_d, sol_d.u_, sol_d.p_, "direct_forcing", t_end)

    if rank == 0:
        print("\n" + "=" * 80)
        print(f"流场范数 @ t={t_end:.2f}s")
        print(f"  {'':<18}{'body_fitted':>14}{'direct_forcing':>14}")
        print(f"  {'u_L2':<18}{uL2_b:>14.6f}{uL2_d:>14.6f}")
        print(f"  {'p_L2':<18}{pL2_b:>14.4f}{pL2_d:>14.4f}")
        print("\n下游中心线探针 (y=0.2):  (u_x, u_y, p)")
        print(f"  {'x':>6} | {'body: ux':>10}{'uy':>10}{'p':>10} | "
              f"{'direct: ux':>10}{'uy':>10}{'p':>10}")
        for r_b, r_d in zip(pb_b, pb_d):
            xb, uxb, uyb, pb = r_b
            xd, uxd, uyd, pd = r_d
            print(f"  {xb:>6.2f} | {uxb:>10.5f}{uyb:>10.5f}{pb:>10.3f} | "
                  f"{uxd:>10.5f}{uyd:>10.5f}{pd:>10.3f}")
        print("\n阻力:")
        print(f"  body_fitted    Cd(表面应力)   = {res_b['Cd']:.4f}")
        print(f"  direct_forcing Cd(标记代理)   = {res_d['Cd_marker']:.4f}")
        print(f"  direct_forcing Cd(体积力积分) = {res_d['Cd_bodyforce']:.4f}")
        print(f"  (DFG 2D-3 Re=100 参考: Cd≈5.57, Cl幅值≈0.0106)")
        print("=" * 80)
