"""
Demo 336 集成测试: 方腔驱动圆盘 (Lid-Driven Cavity with Disk)

模拟方腔内流体驱动弹性圆盘运动。
追踪圆盘中心点 (0.5, 0.5) 的运动轨迹。

运行方式: python tests/test_integration.py
"""

import os
import sys
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

import dolfinx
from dolfinx.fem import (Function, functionspace,
                         dirichletbc, locate_dofs_topological)
from dolfinx.mesh import CellType, GhostMode, locate_entities, meshtags
from dolfinx.fem import form, assemble_scalar
from basix.ufl import element
from ufl import (TestFunction, dot, dx, grad, det, inv, ln, inner)

from afsic import ChorinSolver, IBMesh, IBInterpolation

PASS, FAIL = 0, 0


def check(msg, condition):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {msg}")
    else:
        FAIL += 1
        print(f"  [FAIL] {msg}")


def test_lid_driven_disk():
    """
    集成测试: 方腔驱动圆盘 FSI
    
    配置: Nx=32, T=0.5, dt=1/200 → 100 步
    追踪: 圆盘中心点 (0.5, 0.5) 的位移
    """
    comm = MPI.COMM_WORLD
    rank = comm.rank

    # ---- 配置 ----
    config = {
        "T": 0.5, "dt": 1.0 / 200,
        "rho": 1.0, "mu": 0.01,
        "mu_s": 0.1,           # 固体弹性模量
        "Lx": 1.0, "Ly": 1.0,
        "Nx": 32, "Ny": 32,
        "Nl": 10,              # 圆盘网格分辨率
        "velocity_order": 2,
    }
    config["num_steps"] = int(config["T"] / config["dt"])

    # ---- 流体网格 ----
    mesh = dolfinx.mesh.create_rectangle(
        comm=comm, points=((0.0, 0.0), (config["Lx"], config["Ly"])),
        n=(config["Nx"], config["Ny"]),
        cell_type=CellType.quadrilateral, ghost_mode=GhostMode.shared_facet)

    mesh.topology.create_connectivity(1, 2)
    marker_up = 4
    boundaries = [(1, lambda x: np.isclose(x[0], 0)),
                  (2, lambda x: np.isclose(x[0], config["Lx"])),
                  (3, lambda x: np.isclose(x[1], 0)),
                  (4, lambda x: np.isclose(x[1], config["Ly"]))]

    # 固定点 (0,0) 用于压力
    def fixed_pt(x):
        return np.logical_and(np.isclose(x[0], 0.0), np.isclose(x[1], 0.0))
    point_loc = dolfinx.mesh.locate_entities_boundary(mesh, 0, fixed_pt)

    facet_indices, facet_markers = [], []
    fdim = mesh.topology.dim - 1
    for (mk, loc) in boundaries:
        facets = locate_entities(mesh, fdim, loc)
        facet_indices.append(facets)
        facet_markers.append(np.full_like(facets, mk))
    facet_indices = np.hstack(facet_indices).astype(np.int32)
    facet_markers = np.hstack(facet_markers).astype(np.int32)
    sorted_f = np.argsort(facet_indices)
    facet_tag = meshtags(mesh, fdim, facet_indices[sorted_f], facet_markers[sorted_f])

    # ---- 函数空间 ----
    v_cg2 = element("Lagrange", mesh.topology.cell_name(),
                    config["velocity_order"], shape=(mesh.geometry.dim,))
    s_cg1 = element("Lagrange", mesh.topology.cell_name(), 1)
    V = functionspace(mesh, v_cg2)
    Q = functionspace(mesh, s_cg1)

    # ---- 边界条件 ----
    gdim = mesh.geometry.dim

    class UpVelocity:
        def __init__(self, t=0.0):
            self.t = t
        def __call__(self, x):
            values = np.zeros((gdim, x.shape[1]), dtype=PETSc.ScalarType)
            values[0] = 1.0
            return values

    u_up = Function(V)
    up_vel = UpVelocity(0.0)
    u_up.interpolate(up_vel)
    bcu_up = dirichletbc(u_up, locate_dofs_topological(V, fdim, facet_tag.find(marker_up)))

    u0 = np.array((0,) * gdim, dtype=PETSc.ScalarType)
    bcu_left = dirichletbc(u0, locate_dofs_topological(V, fdim, facet_tag.find(1)), V)
    bcu_right = dirichletbc(u0, locate_dofs_topological(V, fdim, facet_tag.find(2)), V)
    bcu_down = dirichletbc(u0, locate_dofs_topological(V, fdim, facet_tag.find(3)), V)
    bcp_pt = dirichletbc(0.0, locate_dofs_topological(Q, 0, point_loc), Q)
    bcu = [bcu_up, bcu_left, bcu_right, bcu_down]
    bcp = [bcp_pt]

    # ---- NS 求解器 ----
    ns_solver = ChorinSolver(V, Q, bcu, bcp, config["dt"], config["rho"], config["mu"])

    # ---- 固体网格 (从文件读取) ----
    home_dir = os.path.expanduser("~")
    mesh_path = f"{home_dir}/afsi-data/336-lid-driven-disk/mesh/circle_{config['Nl']}.xdmf"
    if not os.path.exists(mesh_path):
        print(f"  [SKIP] Mesh file not found: {mesh_path}")
        return

    with dolfinx.io.XDMFFile(comm, mesh_path, "r",
                              encoding=dolfinx.io.XDMFFile.Encoding.HDF5) as file:
        structure = file.read_mesh()

    v_cg2_s = element("Lagrange", structure.topology.cell_name(),
                      config["velocity_order"], shape=(structure.geometry.dim,))
    Vs = functionspace(structure, v_cg2_s)

    solid_coords = Function(Vs)
    solid_velocity = Function(Vs)
    solid_force = Function(Vs)
    solid_coords.interpolate(lambda x: np.array([x[0], x[1]]))

    # ---- 固体本构 (Neo-Hookean) ----
    dVs = TestFunction(Vs)
    mu_s = config["mu_s"]
    FF = grad(solid_coords)
    L_hat = form(-inner(mu_s * (FF - inv(FF).T), grad(dVs)) * dx)
    b1 = dolfinx.fem.petsc.create_vector(Vs)

    # ---- IB 耦合 ----
    ibmesh = IBMesh(0.0, config["Lx"], 0.0, config["Ly"],
                    config["Nx"], config["Ny"], config["velocity_order"])
    ib_interp = IBInterpolation(ibmesh)
    coords_bg = Function(V)
    coords_bg.interpolate(lambda x: np.array([x[0], x[1]]))
    ibmesh.build_map(coords_bg._cpp_object)
    ib_interp.evaluate_current_points(solid_coords._cpp_object)

    # ---- 追踪中心点位移 ----
    center_history = []  # (t, cx, cy)

    # ---- 时间循环 ----
    for step in range(config["num_steps"]):
        t = step * config["dt"]
        up_vel.t = t
        u_up.interpolate(up_vel)

        # 流体求解
        ns_solver.solve_one_step()

        # 流体→固体 插值速度
        ib_interp.fluid_to_solid(ns_solver.u_._cpp_object, solid_velocity._cpp_object)

        # 更新固体位移
        solid_coords.x.array[:] += solid_velocity.x.array[:] * config["dt"]
        solid_coords.x.scatter_forward()

        # 更新 IB 插值点
        ib_interp.evaluate_current_points(solid_coords._cpp_object)

        # 计算固体力
        with b1.localForm() as loc:
            loc.set(0)
        dolfinx.fem.petsc.assemble_vector(b1, L_hat)
        b1.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        with b1.getBuffer() as arr:
            solid_force.x.array[:len(arr)] = arr[:]

        # 固体→流体 扩散力
        ib_interp.solid_to_fluid(ns_solver.f._cpp_object, solid_force._cpp_object)
        ns_solver.f.x.scatter_forward()

        # 追踪中心点
        center_idx = None
        ca = solid_coords.x.array.reshape(-1, 2)
        dist = np.sqrt((ca[:, 0] - 0.5)**2 + (ca[:, 1] - 0.5)**2)
        center_idx = np.argmin(dist)
        cx, cy = ca[center_idx]
        center_history.append((t, cx, cy))

    # ---- 验证 ----
    times = np.array([h[0] for h in center_history])
    cxs = np.array([h[1] for h in center_history])
    cys = np.array([h[2] for h in center_history])

    disp_x = cxs[-1] - cxs[0]
    disp_y = cys[-1] - cys[0]

    print(f"\n  Initial center: ({cxs[0]:.6f}, {cys[0]:.6f})")
    print(f"  Final center:   ({cxs[-1]:.6f}, {cys[-1]:.6f})")
    print(f"  Displacement:   ({disp_x:+.6f}, {disp_y:+.6f})")
    print(f"  max|u_x|: {np.max(np.abs(cxs - 0.5)):.6f}")
    print(f"  max|u_y|: {np.max(np.abs(cys - 0.5)):.6f}")

    # 圆盘应受方腔驱动向右上方运动
    check("Center moves right (dx > 0)", disp_x > 0)
    check("Center moves up (dy > 0)", disp_y > 0)
    check("Motion is bounded (|dx| < 0.5)", abs(disp_x) < 0.5)
    check("Motion is bounded (|dy| < 0.5)", abs(disp_y) < 0.5)
    check("Center starts near (0.5, 0.5)", abs(cxs[0] - 0.5) < 0.01 and abs(cys[0] - 0.5) < 0.01)


if __name__ == "__main__":
    print("=" * 60)
    print("  Demo 336 集成测试: 方腔驱动圆盘")
    print("=" * 60)

    test_lid_driven_disk()

    print("\n" + "=" * 60)
    print(f"  Results: {PASS} passed, {FAIL} failed, {PASS+FAIL} total")
    print("=" * 60)
    sys.exit(0 if FAIL == 0 else 1)
