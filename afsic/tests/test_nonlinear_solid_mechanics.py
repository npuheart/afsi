from mpi4py import MPI

from dolfinx import log, default_scalar_type
from dolfinx.fem.petsc import NonlinearProblem
import numpy as np
import ufl

from dolfinx import fem, mesh

# https://fenicsproject.discourse.group/t/dolfinx-seems-much-slower-than-dolfin-in-solving-nonlinear-mechanics/17750/5
def test_bar_bending():
    ########################################################################
    ## 1. 定义网格与函数空间
    ########################################################################
    L = 20.0
    domain = mesh.create_box(
        MPI.COMM_WORLD,
        [[0.0, 0.0, 0.0], [L, 1, 1]],
        [40, 10, 10],
        mesh.CellType.tetrahedron,
    )

    ########################################################################
    ## 2. 定义边界
    ########################################################################
    def left(x):
        return np.isclose(x[0], 0)

    def right(x):
        return np.isclose(x[0], L)

    V = fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim,)))
    fdim = domain.topology.dim - 1
    left_facets = mesh.locate_entities_boundary(domain, fdim, left)
    right_facets = mesh.locate_entities_boundary(domain, fdim, right)
    marked_facets = np.hstack([left_facets, right_facets])
    marked_values = np.hstack([np.full_like(left_facets, 1), np.full_like(right_facets, 2)])
    sorted_facets = np.argsort(marked_facets)
    facet_tag = mesh.meshtags(
        domain, fdim, marked_facets[sorted_facets], marked_values[sorted_facets]
    )
    u_bc = np.array((0,) * domain.geometry.dim, dtype=default_scalar_type)
    left_dofs = fem.locate_dofs_topological(V, facet_tag.dim, facet_tag.find(1))
    bcs = [fem.dirichletbc(u_bc, left_dofs, V)]

    ########################################################################
    ## 3. 定义本构关系
    ########################################################################
    B = fem.Constant(domain, default_scalar_type((0, 0, 0)))
    T = fem.Constant(domain, default_scalar_type((0, 0, 0)))

    v = ufl.TestFunction(V)
    disp = fem.Function(V)

    d = len(disp)
    I = ufl.variable(ufl.Identity(d))
    F = ufl.variable(I + ufl.grad(disp))
    C = ufl.variable(F.T * F)
    Ic = ufl.variable(ufl.tr(C))
    J = ufl.variable(ufl.det(F))

    E = default_scalar_type(1.0e4)
    nu = default_scalar_type(0.3)
    mu = fem.Constant(domain, E / (2 * (1 + nu)))
    lmbda = fem.Constant(domain, E * nu / ((1 + nu) * (1 - 2 * nu)))
    psi = (mu / 2) * (Ic - 3) - mu * ufl.ln(J) + (lmbda / 2) * (ufl.ln(J)) ** 2
    P = ufl.diff(psi, F)

    ########################################################################
    ## 4. 定义弱形式
    ########################################################################
    metadata = {"quadrature_degree": 5}
    ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tag, metadata=metadata)
    dx = ufl.Measure("dx", domain=domain, metadata=metadata)

    cffi_options = ["-Ofast", "-march=native"]
    jit_options = {
        "cffi_extra_compile_args": cffi_options,
        "cffi_libraries": ["m"],
    }
    F = ufl.inner(ufl.grad(v), P) * dx - ufl.inner(v, B) * dx - ufl.inner(v, T) * ds(2)

    ########################################################################
    ## 5. 定义非线性求解器
    ########################################################################
    petsc_options = {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        "snes_type": "newtonls",
        "snes_linesearch_type": "none",
        "snes_atol": 1e-7,
        "snes_rtol": 1e-6,
        "snes_monitor": None,
    }
    problem = NonlinearProblem(
        F,
        disp,
        bcs=bcs,
        petsc_options_prefix="bar_bending",
        petsc_options=petsc_options,
        jit_options=jit_options,
    )

    ########################################################################
    ## 6. 求解并验证结果
    ########################################################################
    from dolfinx.geometry import bb_tree, compute_colliding_cells, compute_collisions_points

    tval0 = -1.5
    tree = bb_tree(domain, domain.geometry.dim)
    x0 = np.array([L / 2, 0.5, 0.5], dtype=default_scalar_type)
    cell_candidates = compute_collisions_points(tree, x0)
    cell = compute_colliding_cells(domain, cell_candidates, x0).array
    first_cell = cell[0]

    expected = np.array([
        [-0.07666793,  0.01112278, -1.0976045],
        [-0.27184167,  0.02014685, -2.05597763],
    ])

    for n in range(1, 3):
        T.value[2] = n * tval0
        _, converged_reason, num_iterations = problem.solve()
        u = disp.eval(x0, first_cell)[:3]
        np.testing.assert_allclose(u, expected[n - 1], rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    log.set_log_level(log.LogLevel.INFO)
    test_bar_bending()
    print("All assertions passed.")