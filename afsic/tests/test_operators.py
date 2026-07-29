"""
插值算子 S* 与延拓(扩散)算子 S 的独立验证

S* : fluid → solid   (FunctorInterpolate)
      particle.u1 += grid_state.x * wij

S  : solid → fluid    (FunctorSpread)
      grid_state.x += particle.u1 * wij * 1.0 / dx / dy

运行方式: python tests/test_operators.py
"""

import numpy as np
from mpi4py import MPI
import dolfinx
from dolfinx.mesh import CellType, GhostMode
from dolfinx.fem import Function, functionspace
from basix.ufl import element
from afsic import IBMesh, IBInterpolation, IBMesh3D, IBInterpolation3D

ORDER = 2
TOL = 1e-10

PASS, FAIL = 0, 0


def check(msg, condition):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {msg}")
    else:
        FAIL += 1
        print(f"  [FAIL] {msg}")


# =============================================================================
# 2D 公共环境
# =============================================================================

def setup_2d(Nx, Ny):
    ibmesh = IBMesh(0.0, 1.0, 0.0, 1.0, Nx, Ny, ORDER)
    mesh = dolfinx.mesh.create_rectangle(
        comm=MPI.COMM_WORLD, points=((0.0, 0.0), (1.0, 1.0)),
        n=(Nx, Ny), cell_type=CellType.triangle, ghost_mode=GhostMode.shared_facet)
    v_cg2 = element("Lagrange", mesh.topology.cell_name(), 2, shape=(mesh.geometry.dim,))
    V = functionspace(mesh, v_cg2)
    coords = Function(V)
    coords.interpolate(lambda x: np.array([x[0], x[1]]))
    ibmesh.build_map(coords._cpp_object)

    structure = dolfinx.mesh.create_rectangle(
        comm=MPI.COMM_WORLD, points=((0.2, 0.2), (0.8, 0.8)),
        n=(Nx, Ny), cell_type=CellType.triangle, ghost_mode=GhostMode.shared_facet)
    v_cg2_s = element("Lagrange", structure.topology.cell_name(), 2, shape=(structure.geometry.dim,))
    V_solid = functionspace(structure, v_cg2_s)

    ib_interp = IBInterpolation(ibmesh)
    solid_coords = Function(V_solid)
    solid_coords.interpolate(lambda x: np.array([x[0], x[1]]))
    ib_interp.evaluate_current_points(solid_coords._cpp_object)

    return ibmesh, ib_interp, V, V_solid


# =============================================================================
# 插值算子 S* 测试
# =============================================================================

def test_interp_constant():
    """S* 常量场: u=(c1,c2) → S*[u] 处处 = (c1,c2)"""
    print("\n--- S* 常量场插值 ---")
    Nx, Ny = 16, 16
    ibmesh, ib_interp, V, V_solid = setup_2d(Nx, Ny)

    c1, c2 = 3.0, -7.0
    fluid_u = Function(V)
    solid_u = Function(V_solid)
    fluid_u.interpolate(lambda x: np.stack([np.full(x.shape[1], c1),
                                             np.full(x.shape[1], c2)]))
    fluid_u.x.scatter_forward()

    ib_interp.fluid_to_solid(fluid_u._cpp_object, solid_u._cpp_object)
    solid_u.x.scatter_forward()

    ua = solid_u.x.array.reshape(-1, 2)
    err1 = np.max(np.abs(ua[:, 0] - c1))
    err2 = np.max(np.abs(ua[:, 1] - c2))
    max_err = max(err1, err2)

    print(f"  max|S*[u]_x - {c1}| = {err1:.2e}")
    print(f"  max|S*[u]_y - {c2}| = {err2:.2e}")
    check("S* constant → exact", max_err < TOL)


def test_interp_linear():
    """S* 线性场: u(x,y)=(x,y) → S*[u] 在固体点 (X,Y) 处应精确等于 (X,Y)"""
    print("\n--- S* 线性场插值 ---")
    Nx, Ny = 16, 16
    ibmesh, ib_interp, V, V_solid = setup_2d(Nx, Ny)

    fluid_u = Function(V)
    solid_u = Function(V_solid)
    solid_coords = Function(V_solid)
    solid_coords.interpolate(lambda x: np.array([x[0], x[1]]))
    solid_coords.x.scatter_forward()

    fluid_u.interpolate(lambda x: np.array([x[0], x[1]]))
    fluid_u.x.scatter_forward()

    ib_interp.fluid_to_solid(fluid_u._cpp_object, solid_u._cpp_object)
    solid_u.x.scatter_forward()

    # S*[u] 在固体节点应与固体坐标一致
    ua = solid_u.x.array.reshape(-1, 2)
    ca = solid_coords.x.array.reshape(-1, 2)
    err_x = np.max(np.abs(ua[:, 0] - ca[:, 0]))
    err_y = np.max(np.abs(ua[:, 1] - ca[:, 1]))
    max_err = max(err_x, err_y)

    print(f"  max|S*[u]_x - X| = {err_x:.2e}")
    print(f"  max|S*[u]_y - Y| = {err_y:.2e}")
    check("S* linear: u(x,y)=(x,y) → exact at solid nodes", max_err < TOL)


def test_interp_smooth_convergence():
    """S* 光滑场: 网格加密时，插值误差应收敛"""
    print("\n--- S* 光滑场收敛性 ---")
    errors = []
    for N in [8, 16, 32]:
        Nx = Ny = N
        ibmesh, ib_interp, V, V_solid = setup_2d(Nx, Ny)

        fluid_u = Function(V)
        solid_u = Function(V_solid)

        # 光滑场 u = (sin(πx)cos(πy), cos(πx)sin(πy))
        fluid_u.interpolate(lambda x: np.array([np.sin(np.pi*x[0])*np.cos(np.pi*x[1]),
                                                 np.cos(np.pi*x[0])*np.sin(np.pi*x[1])]))
        fluid_u.x.scatter_forward()

        ib_interp.fluid_to_solid(fluid_u._cpp_object, solid_u._cpp_object)
        solid_u.x.scatter_forward()

        # 解析值 (基于固体网格坐标)
        solid_coords = Function(V_solid)
        solid_coords.interpolate(lambda x: np.array([x[0], x[1]]))
        solid_coords.x.scatter_forward()
        ca = solid_coords.x.array.reshape(-1, 2)

        exact_x = np.sin(np.pi*ca[:, 0]) * np.cos(np.pi*ca[:, 1])
        exact_y = np.cos(np.pi*ca[:, 0]) * np.sin(np.pi*ca[:, 1])
        ua = solid_u.x.array.reshape(-1, 2)

        l2_err = np.sqrt(np.mean((ua[:, 0] - exact_x)**2 + (ua[:, 1] - exact_y)**2))
        errors.append((N, l2_err))
        print(f"  N={N:2d}:  L2 error = {l2_err:.4e}")

    # 误差应收敛 (至少网格加倍误差减小)
    ok = errors[0][1] > errors[-1][1] * 0.5
    check("S* convergence (error decreases with N)", ok)


# =============================================================================
# 延拓(扩散)算子 S 测试
# =============================================================================

def test_spread_force_conservation():
    """S 力守恒: Σ_{i,j} S[f]_{i,j}·Δx·Δy = Σ_k f_k"""
    print("\n--- S 力守恒 ---")
    for Nx, Ny in [(8, 8), (16, 16), (32, 32)]:
        ibmesh, ib_interp, V, V_solid = setup_2d(Nx, Ny)

        fluid_f = Function(V)
        fluid_empty = Function(V)  # 清零流体
        solid_f = Function(V_solid)

        # 固体力场
        rng = np.random.RandomState(42)
        solid_f.x.array[:] = rng.randn(solid_f.x.array.shape[0])
        solid_f.x.scatter_forward()

        # 清零流体再扩散
        fluid_f.x.array[:] = 0.0
        fluid_f.x.scatter_forward()
        ib_interp.solid_to_fluid(fluid_f._cpp_object, solid_f._cpp_object)
        fluid_f.x.scatter_forward()

        dx = 1.0 / (ORDER * Nx)
        dy = 1.0 / (ORDER * Ny)

        # 流体侧合力 = Σ F_fluid · ΔxΔy
        fa = fluid_f.x.array.reshape(-1, 2)
        total_fluid = np.array([np.sum(fa[:, 0]) * dx * dy,
                                 np.sum(fa[:, 1]) * dx * dy])

        # 固体侧合力 = Σ f_solid
        sa = solid_f.x.array.reshape(-1, 2)
        total_solid = np.array([np.sum(sa[:, 0]), np.sum(sa[:, 1])])

        rel_err = np.max(np.abs(total_fluid - total_solid)) / max(np.max(np.abs(total_solid)), 1.0)
        print(f"  N={Nx:2d}:  F_fluid=({total_fluid[0]:+.6e},{total_fluid[1]:+.6e})  "
              f"F_solid=({total_solid[0]:+.6e},{total_solid[1]:+.6e})  rel_err={rel_err:.2e}")
        check(f"S force conservation N={Nx}", rel_err < TOL)


def test_spread_conservation_with_const():
    """S 常量力: f=(c1,c2) 处处相等 → 回代对偶性验证 S[f] 的一致性"""
    print("\n--- S 常量力扩散 ---")
    Nx, Ny = 16, 16
    ibmesh, ib_interp, V, V_solid = setup_2d(Nx, Ny)

    fluid_u = Function(V)   # 用作测试场 u(x,y) = (1, 0) 
    fluid_f = Function(V)
    solid_f = Function(V_solid)

    # u = (1.0, 0.0) 常量
    fluid_u.interpolate(lambda x: np.stack([np.ones(x.shape[1]),
                                            np.zeros(x.shape[1])]))
    fluid_u.x.scatter_forward()

    # f = (2.0, 0.0) 常量
    solid_f.interpolate(lambda x: np.stack([np.full(x.shape[1], 2.0),
                                            np.zeros(x.shape[1])]))
    solid_f.x.scatter_forward()

    # S[f]
    ib_interp.solid_to_fluid(fluid_f._cpp_object, solid_f._cpp_object)
    fluid_f.x.scatter_forward()

    # 通过 S* 插值验证: S*[u] = u = (1,0)，然后 Σ (S*[u])_k · f_k = 2 * N_solid
    solid_u = Function(V_solid)
    ib_interp.fluid_to_solid(fluid_u._cpp_object, solid_u._cpp_object)
    solid_u.x.scatter_forward()

    # LHS: Σ S*[u] · f = 2.0 * N_solid_node (每个节点 x 分量)
    sa = solid_f.x.array.reshape(-1, 2)
    ua_s = solid_u.x.array.reshape(-1, 2)
    lhs = np.sum(ua_s[:, 0] * sa[:, 0] + ua_s[:, 1] * sa[:, 1])

    # RHS: Σ u · S[f] · dx·dy = Σ 1 * S[f]_x · dx·dy
    dx = 1.0 / (ORDER * Nx)
    dy = 1.0 / (ORDER * Ny)
    fa = fluid_f.x.array.reshape(-1, 2)
    rhs = np.sum(fa[:, 0]) * dx * dy  # u_y = 0, 所以只有 x 分量

    rel_err = abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1.0)
    print(f"  LHS={lhs:.8e}  RHS={rhs:.8e}  rel_err={rel_err:.2e}")
    check("S constant force duality check", rel_err < TOL)


# =============================================================================
# 3D 测试
# =============================================================================

def setup_3d(Nx, Ny, Nz):
    ibmesh = IBMesh3D(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, Nx, Ny, Nz, ORDER)
    mesh = dolfinx.mesh.create_box(
        comm=MPI.COMM_WORLD, points=((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        n=(Nx, Ny, Nz), cell_type=CellType.hexahedron, ghost_mode=GhostMode.shared_facet)
    v_cg2 = element("Lagrange", mesh.topology.cell_name(), 2, shape=(mesh.geometry.dim,))
    V = functionspace(mesh, v_cg2)
    coords = Function(V)
    coords.interpolate(lambda x: np.array([x[0], x[1], x[2]]))
    ibmesh.build_map(coords._cpp_object)

    structure = dolfinx.mesh.create_box(
        comm=MPI.COMM_WORLD, points=((0.2, 0.2, 0.2), (0.8, 0.8, 0.8)),
        n=(Nx, Ny, Nz), cell_type=CellType.hexahedron, ghost_mode=GhostMode.shared_facet)
    v_cg2_s = element("Lagrange", structure.topology.cell_name(), 2, shape=(structure.geometry.dim,))
    V_solid = functionspace(structure, v_cg2_s)

    ib_interp = IBInterpolation3D(ibmesh)
    solid_coords = Function(V_solid)
    solid_coords.interpolate(lambda x: np.array([x[0], x[1], x[2]]))
    ib_interp.evaluate_current_points(solid_coords._cpp_object)

    return ibmesh, ib_interp, V, V_solid


def test_interp_constant_3d():
    """S* 3D 常量场"""
    print("\n--- S* 3D 常量场 ---")
    Nx, Ny, Nz = 8, 8, 8
    ibmesh, ib_interp, V, V_solid = setup_3d(Nx, Ny, Nz)

    c1, c2, c3 = 2.0, -3.0, 5.0
    fluid_u = Function(V)
    solid_u = Function(V_solid)
    fluid_u.interpolate(lambda x: np.stack([np.full(x.shape[1], c1),
                                            np.full(x.shape[1], c2),
                                            np.full(x.shape[1], c3)]))
    fluid_u.x.scatter_forward()
    ib_interp.fluid_to_solid(fluid_u._cpp_object, solid_u._cpp_object)
    solid_u.x.scatter_forward()

    ua = solid_u.x.array.reshape(-1, 3)
    max_err = max(np.max(np.abs(ua[:, 0] - c1)),
                  np.max(np.abs(ua[:, 1] - c2)),
                  np.max(np.abs(ua[:, 2] - c3)))
    print(f"  max error = {max_err:.2e}")
    check("S* 3D constant → exact", max_err < TOL)


def test_spread_force_conservation_3d():
    """S 3D 力守恒"""
    print("\n--- S 3D 力守恒 ---")
    Nx, Ny, Nz = 8, 8, 8
    ibmesh, ib_interp, V, V_solid = setup_3d(Nx, Ny, Nz)

    fluid_f = Function(V)
    solid_f = Function(V_solid)

    rng = np.random.RandomState(123)
    solid_f.x.array[:] = rng.randn(solid_f.x.array.shape[0])
    solid_f.x.scatter_forward()
    fluid_f.x.array[:] = 0.0
    fluid_f.x.scatter_forward()
    ib_interp.solid_to_fluid(fluid_f._cpp_object, solid_f._cpp_object)
    fluid_f.x.scatter_forward()

    dx = 1.0 / (ORDER * Nx); dy = 1.0 / (ORDER * Ny); dz = 1.0 / (ORDER * Nz)
    fa = fluid_f.x.array.reshape(-1, 3)
    total_fluid = np.array([np.sum(fa[:, 0]) * dx * dy * dz,
                             np.sum(fa[:, 1]) * dx * dy * dz,
                             np.sum(fa[:, 2]) * dx * dy * dz])
    sa = solid_f.x.array.reshape(-1, 3)
    total_solid = np.array([np.sum(sa[:, 0]), np.sum(sa[:, 1]), np.sum(sa[:, 2])])

    rel_err = np.max(np.abs(total_fluid - total_solid)) / max(np.max(np.abs(total_solid)), 1.0)
    print(f"  F_fluid=({total_fluid[0]:+.6e},{total_fluid[1]:+.6e},{total_fluid[2]:+.6e})")
    print(f"  F_solid=({total_solid[0]:+.6e},{total_solid[1]:+.6e},{total_solid[2]:+.6e})")
    print(f"  rel_err={rel_err:.2e}")
    check("S 3D force conservation", rel_err < TOL)


# =============================================================================
# 主入口
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  插值算子 S* 与延拓算子 S  独立验证")
    print("=" * 60)

    # --- 插值算子 S* ---
    print("\n" + "-" * 40)
    print("  [S*] 插值算子 (fluid → solid)")
    print("-" * 40)
    test_interp_constant()
    test_interp_linear()
    test_interp_smooth_convergence()

    # --- 延拓算子 S ---
    print("\n" + "-" * 40)
    print("  [S] 延拓算子 (solid → fluid)")
    print("-" * 40)
    test_spread_force_conservation()
    test_spread_conservation_with_const()

    # --- 3D ---
    print("\n" + "-" * 40)
    print("  [3D]")
    print("-" * 40)
    test_interp_constant_3d()
    test_spread_force_conservation_3d()

    print("\n" + "=" * 60)
    print(f"  Results: {PASS} passed, {FAIL} failed, {PASS+FAIL} total")
    print("=" * 60)
    exit(0 if FAIL == 0 else 1)
