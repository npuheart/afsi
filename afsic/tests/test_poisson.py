from mpi4py import MPI
from petsc4py.PETSc import ScalarType  # type: ignore

import numpy as np

import ufl
from dolfinx import fem, mesh
from afsic import define_poisson_problem


def solve_poisson(n=(32, 16)):
    msh = mesh.create_rectangle(
        comm=MPI.COMM_WORLD,
        points=((0.0, 0.0), (2.0, 1.0)),
        n=n,
        cell_type=mesh.CellType.triangle,
    )
    V = fem.functionspace(msh, ("Lagrange", 1))

    tdim = msh.topology.dim
    fdim = tdim - 1
    facets = mesh.locate_entities_boundary(
        msh,
        dim=fdim,
        marker=lambda x: np.isclose(x[0], 0.0) | np.isclose(x[0], 2.0),
    )
    dofs = fem.locate_dofs_topological(V=V, entity_dim=fdim, entities=facets)
    bc = fem.dirichletbc(value=ScalarType(0), dofs=dofs, V=V)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(msh)
    f = 10 * ufl.exp(-((x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2) / 0.02)
    g = ufl.sin(5 * x[0])

    problem = define_poisson_problem(u, v, f, g, ufl.dx, ufl.ds, bc)
    return problem.solve()


def test_poisson_solves():
    uh = solve_poisson()
    print(sum(uh.x.array[:]))
    # TODO: The result should be approximately 83.42890213751603
    assert isinstance(uh, fem.Function)

if __name__ == "__main__":
    test_poisson_solves()
