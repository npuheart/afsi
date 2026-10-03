"""Verification of afsic.euler.PeskinRK2Solver (periodic Peskin/IB2d two-stage scheme).

Taylor–Green vortex on the periodic unit box (exact NS solution):
    u = ( sin(2 pi x) cos(2 pi y), -cos(2 pi x) sin(2 pi y) ) exp(-8 pi^2 nu t)
    p = rho/4 ( cos(4 pi x) + cos(4 pi y) ) exp(-16 pi^2 nu t)
"""
import numpy as np
import pytest
from mpi4py import MPI

dolfinx = pytest.importorskip("dolfinx")
from dolfinx import fem  # noqa: E402
from dolfinx.mesh import CellType, create_rectangle  # noqa: E402
import ufl  # noqa: E402

from afsic.euler.PeskinRK2Solver import PeskinRK2Solver, periodic_master_map  # noqa: E402

pytestmark = pytest.mark.skipif(MPI.COMM_WORLD.size > 1, reason="serial solver")


def taylor_green_error(n, dt, T=0.1, rho=1.0, mu=0.01, grad_div=0.0):
    nu = mu / rho
    mesh = create_rectangle(MPI.COMM_WORLD, ((0.0, 0.0), (1.0, 1.0)), (n, n),
                            cell_type=CellType.quadrilateral)
    s = PeskinRK2Solver(mesh, (0.0, 1.0, 0.0, 1.0), dt, rho, mu, grad_div=grad_div)

    def u_ex(t):
        return lambda x: np.vstack((np.sin(2 * np.pi * x[0]) * np.cos(2 * np.pi * x[1]),
                                    -np.cos(2 * np.pi * x[0]) * np.sin(2 * np.pi * x[1]))) \
            * np.exp(-8 * np.pi ** 2 * nu * t)

    s.u_n.interpolate(u_ex(0.0))
    steps = int(round(T / dt))
    for _ in range(steps):
        s.solve_one_step()
    ue = fem.Function(s.V)
    ue.interpolate(u_ex(steps * dt))
    e = fem.assemble_scalar(fem.form(ufl.inner(s.u_ - ue, s.u_ - ue) * ufl.dx)) ** 0.5
    nrm = fem.assemble_scalar(fem.form(ufl.inner(ue, ue) * ufl.dx)) ** 0.5
    div = fem.assemble_scalar(fem.form(ufl.div(s.u_) ** 2 * ufl.dx)) ** 0.5
    s.cleanup()
    return e / nrm, div


def test_periodic_map_corners():
    mesh = create_rectangle(MPI.COMM_WORLD, ((0.0, 0.0), (1.0, 1.0)), (4, 4),
                            cell_type=CellType.quadrilateral)
    Q = fem.functionspace(mesh, ("Lagrange", 1))
    m = periodic_master_map(Q, (0.0, 1.0, 0.0, 1.0))
    X = Q.tabulate_dof_coordinates()[:, :2]
    d = np.abs(X[m] - X)
    assert np.allclose(np.minimum(d, 1.0 - d), 0.0)     # images differ by a period
    assert not np.any(np.isclose(X[m][:, 0], 1.0) | np.isclose(X[m][:, 1], 1.0))


def test_taylor_green_accuracy():
    e8, _ = taylor_green_error(8, 1e-3)
    e16, _ = taylor_green_error(16, 1e-3)
    assert e16 < 5e-3
    assert np.log2(e8 / e16) > 2.5          # Q2 velocity


def test_taylor_green_time_order():
    e1, _ = taylor_green_error(24, 4e-2, T=0.2, mu=0.05)
    e2, _ = taylor_green_error(24, 2e-2, T=0.2, mu=0.05)
    assert np.log2(e1 / e2) > 1.7          # formally second order in time


if __name__ == "__main__":
    for n in (8, 16, 32):
        print(n, taylor_green_error(n, 1e-3))
    for dt in (4e-2, 2e-2, 1e-2):
        print(dt, taylor_green_error(16, dt, T=0.2, mu=0.05))
