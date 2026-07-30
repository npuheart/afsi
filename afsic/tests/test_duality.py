"""
Delta 函数积分对偶性验证测试

验证 IB 方法中插值算子 S* 与扩散算子 S 之间的伴随关系：

    (S*[u], f)_Γ  =  (u, S[f])_Ω

即：
    Σ_k (S*[u])_k · f_k  =  Σ_{i,j} u_{i,j} · (S[f])_{i,j} · dx · dy

理论推导 (基于 C++ 实现):
  - S* (FunctorInterpolate):  particle.u1 += grid_state.x * wij
  - S  (FunctorSpread):       grid_state.x += particle.u1 * wij * 1.0 / dx / dy

  对偶性:  Σ_k U_k·F_k = Σ_{i,j} u_{i,j}·F_{i,j}·dx·dy   ✓ (严格等号)

运行方式: python tests/test_duality.py
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


def report_duality(label, lhs, rhs):
    """报告对偶性结果，智能处理接近零的情况"""
    abs_err = abs(lhs - rhs)
    denom = max(abs(lhs), abs(rhs), 1.0)  # 至少用 1.0 避免除以零
    rel_err = abs_err / denom
    print(f"  [{label}] LHS={lhs:+.8e}  RHS={rhs:+.8e}  |err|={abs_err:.2e}  rel={rel_err:.2e}")
    return abs_err < TOL or rel_err < TOL  # 绝对误差或相对误差满足其一


# =============================================================================
# 2D 测试
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


def ip2d(u, v, dx, dy):
    ua = u.x.array.reshape(-1, 2)
    va = v.x.array.reshape(-1, 2)
    return np.sum(np.sum(ua * va, axis=1)) * dx * dy


def ips(u, v):
    ua = u.x.array.reshape(-1, 2)
    va = v.x.array.reshape(-1, 2)
    return np.sum(np.sum(ua * va, axis=1))


def test_duality_2d(Nx, Ny, fluid_u, solid_f, ib_interp, V, V_solid, label):
    """核心对偶性测试: (S*[u], f)_Γ == (u, S[f])_Ω"""
    fluid_f = Function(V)
    solid_u = Function(V_solid)

    ib_interp.fluid_to_solid(fluid_u._cpp_object, solid_u._cpp_object)
    solid_u.x.scatter_forward()
    ib_interp.solid_to_fluid(fluid_f._cpp_object, solid_f._cpp_object)
    fluid_f.x.scatter_forward()

    dx = 1.0 / (ORDER * Nx)
    dy = 1.0 / (ORDER * Ny)

    lhs = ips(solid_u, solid_f)
    rhs = ip2d(fluid_u, fluid_f, dx, dy)

    ok = report_duality(label, lhs, rhs)
    check(f"Duality 2D {label}", ok)


def test_2d_linear():
    print("\n--- 2D 线性场 ---")
    Nx, Ny = 16, 16
    ibmesh, ib_interp, V, V_solid = setup_2d(Nx, Ny)
    fluid_u = Function(V)
    solid_f = Function(V_solid)
    fluid_u.interpolate(lambda x: np.array([x[0] + 1.0, 2.0 * x[1] + 3.0]))
    fluid_u.x.scatter_forward()
    solid_f.interpolate(lambda x: np.array([np.sin(2 * np.pi * x[0]), np.cos(2 * np.pi * x[1])]))
    solid_f.x.scatter_forward()
    test_duality_2d(Nx, Ny, fluid_u, solid_f, ib_interp, V, V_solid, "linear")


def test_2d_sinusoidal():
    print("\n--- 2D 正弦场 ---")
    Nx, Ny = 16, 16
    ibmesh, ib_interp, V, V_solid = setup_2d(Nx, Ny)
    fluid_u = Function(V)
    solid_f = Function(V_solid)
    fluid_u.interpolate(lambda x: np.array([np.sin(2 * np.pi * x[0]), np.cos(2 * np.pi * x[1])]))
    fluid_u.x.scatter_forward()
    solid_f.interpolate(lambda x: np.array([x[0] * x[1], x[0] + x[1]]))
    solid_f.x.scatter_forward()
    test_duality_2d(Nx, Ny, fluid_u, solid_f, ib_interp, V, V_solid, "sin")


def test_2d_random():
    print("\n--- 2D 随机场 ---")
    Nx, Ny = 16, 16
    ibmesh, ib_interp, V, V_solid = setup_2d(Nx, Ny)
    fluid_u = Function(V)
    solid_f = Function(V_solid)
    rng = np.random.RandomState(42)
    fluid_u.x.array[:] = rng.randn(fluid_u.x.array.shape[0])
    fluid_u.x.scatter_forward()
    solid_f.x.array[:] = rng.randn(solid_f.x.array.shape[0])
    solid_f.x.scatter_forward()
    test_duality_2d(Nx, Ny, fluid_u, solid_f, ib_interp, V, V_solid, "random")


def test_2d_multi_grid():
    print("\n--- 2D 多分辨率 ---")
    for Nx, Ny in [(8, 8), (16, 16), (32, 32)]:
        ibmesh, ib_interp, V, V_solid = setup_2d(Nx, Ny)
        fluid_u = Function(V)
        solid_f = Function(V_solid)
        rng = np.random.RandomState(42)
        fluid_u.x.array[:] = rng.randn(fluid_u.x.array.shape[0])
        fluid_u.x.scatter_forward()
        solid_f.x.array[:] = rng.randn(solid_f.x.array.shape[0])
        solid_f.x.scatter_forward()
        test_duality_2d(Nx, Ny, fluid_u, solid_f, ib_interp, V, V_solid, f"{Nx}x{Ny}")


def test_2d_evaluate():
    """点插值精度: 线性场应精确还原"""
    print("\n--- 2D 点插值精度 ---")
    Nx, Ny = 16, 16
    ibmesh = IBMesh(0.0, 1.0, 0.0, 1.0, Nx, Ny, ORDER)
    mesh = dolfinx.mesh.create_rectangle(
        comm=MPI.COMM_WORLD, points=((0.0, 0.0), (1.0, 1.0)),
        n=(Nx, Ny), cell_type=CellType.triangle, ghost_mode=GhostMode.shared_facet)
    v_cg2 = element("Lagrange", mesh.topology.cell_name(), 2, shape=(mesh.geometry.dim,))
    V = functionspace(mesh, v_cg2)
    coords = Function(V)
    coords.interpolate(lambda x: np.array([x[0], x[1]]))
    ibmesh.build_map(coords._cpp_object)

    # evaluate 在当前接口中只打印，验证不崩溃即可
    ibmesh.evaluate(0.34, 0.67, coords._cpp_object)
    ibmesh.evaluate(0.0, 0.0, coords._cpp_object)
    ibmesh.evaluate(1.0, 1.0, coords._cpp_object)
    check("evaluate() runs without crash", True)


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


def ip3d(u, v, dx, dy, dz):
    ua = u.x.array.reshape(-1, 3)
    va = v.x.array.reshape(-1, 3)
    return np.sum(np.sum(ua * va, axis=1)) * dx * dy * dz


def ips3d(u, v):
    ua = u.x.array.reshape(-1, 3)
    va = v.x.array.reshape(-1, 3)
    return np.sum(np.sum(ua * va, axis=1))


def test_3d_random():
    print("\n--- 3D 随机场 ---")
    Nx, Ny, Nz = 8, 8, 8
    ibmesh, ib_interp, V, V_solid = setup_3d(Nx, Ny, Nz)
    fluid_u = Function(V); fluid_f = Function(V)
    solid_u = Function(V_solid); solid_f = Function(V_solid)

    rng = np.random.RandomState(123)
    fluid_u.x.array[:] = rng.randn(fluid_u.x.array.shape[0])
    fluid_u.x.scatter_forward()
    solid_f.x.array[:] = rng.randn(solid_f.x.array.shape[0])
    solid_f.x.scatter_forward()

    ib_interp.fluid_to_solid(fluid_u._cpp_object, solid_u._cpp_object)
    solid_u.x.scatter_forward()
    ib_interp.solid_to_fluid(fluid_f._cpp_object, solid_f._cpp_object)
    fluid_f.x.scatter_forward()

    dx = 1.0 / (ORDER * Nx); dy = 1.0 / (ORDER * Ny); dz = 1.0 / (ORDER * Nz)
    lhs = ips3d(solid_u, solid_f)
    rhs = ip3d(fluid_u, fluid_f, dx, dy, dz)
    ok = report_duality("3D random", lhs, rhs)
    check("Duality 3D random", ok)


def test_3d_sinusoidal():
    print("\n--- 3D 正弦场 ---")
    Nx, Ny, Nz = 8, 8, 8
    ibmesh, ib_interp, V, V_solid = setup_3d(Nx, Ny, Nz)
    fluid_u = Function(V); fluid_f = Function(V)
    solid_u = Function(V_solid); solid_f = Function(V_solid)

    fluid_u.interpolate(lambda x: np.array([
        np.sin(2 * np.pi * x[0]) * np.cos(2 * np.pi * x[1]),
        np.cos(2 * np.pi * x[0]) * np.sin(2 * np.pi * x[2]),
        np.sin(2 * np.pi * x[1]) * np.cos(2 * np.pi * x[2]),
    ]))
    fluid_u.x.scatter_forward()
    solid_f.interpolate(lambda x: np.array([x[0] + 2 * x[1], x[1] - x[2], x[0] * x[2]]))
    solid_f.x.scatter_forward()

    ib_interp.fluid_to_solid(fluid_u._cpp_object, solid_u._cpp_object)
    solid_u.x.scatter_forward()
    ib_interp.solid_to_fluid(fluid_f._cpp_object, solid_f._cpp_object)
    fluid_f.x.scatter_forward()

    dx = 1.0 / (ORDER * Nx); dy = 1.0 / (ORDER * Ny); dz = 1.0 / (ORDER * Nz)
    lhs = ips3d(solid_u, solid_f)
    rhs = ip3d(fluid_u, fluid_f, dx, dy, dz)
    ok = report_duality("3D sin", lhs, rhs)
    check("Duality 3D sin", ok)


# =============================================================================
# 主入口
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  Delta 函数积分对偶性验证")
    print("  (S*[u], f)_Γ  =  (u, S[f])_Ω")
    print("=" * 60)

    test_2d_linear()
    test_2d_sinusoidal()
    test_2d_random()
    test_2d_multi_grid()
    test_2d_evaluate()
    test_3d_random()
    test_3d_sinusoidal()

    print("\n" + "=" * 60)
    print(f"  Results: {PASS} passed, {FAIL} failed, {PASS+FAIL} total")
    print("=" * 60)

    exit(0 if FAIL == 0 else 1)
