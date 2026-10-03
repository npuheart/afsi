"""First IB2d comparison case for the AFSI solver (demo_444).

Reproduces the standard IB2d example ``Example_Standard_Rubberband /
Rubberband_with_Springs`` with the AFSI solvers and runs it in two ways:

``SHAPE=fiber`` (default)
    The IB2d structure itself: 64 Lagrangian points on an ellipse with
    semi-axes (0.2, 0.4) around (1/2, 1/2), closed by 64 zero-rest-length
    springs of stiffness k = 2.5e4, force density f_i = k (X_{i+1} +
    X_{i-1} - 2 X_i) (IB2d's spring law), loads L_i = f_i * DS (IB2d's
    constant-ds convention, see below).

``SHAPE=thick``
    The same problem as a continuum band with thickness: a circular annulus
    (stress-free reference, outer radius sqrt(0.2*0.4) = 0.28284, wall
    thickness T_WALL) prestrained by the affine map that carries the circle
    into the same initial ellipse (area preserving), neo-Hookean solid with
    the shear modulus calibrated to IB2d's oscillation period (MU_S = 750,
    volumetric penalty 10*MU_S).  The stiff band needs DT = 1e-4: at
    DT = 1e-3 the explicit fluid-structure coupling is unstable (see the
    MU_S note below).

``FLUID=rt`` (default)
    The divergence-conforming RT/DG solver with the nodal coupling
    (RTNodalCoupling, E / E^T, no kernel).  Its velocity is exactly
    divergence-free, so the marker-sampled velocity keeps the enclosed area
    to O(0.1 %) over the IB2d window - matching IB2d's own conservation.

``FLUID=chorin``
    The P2/P1 projection background with the four-point IB kernel (the
    demos 441/442/443 pipeline).  Kept for comparison: its discrete
    velocity is divergence-free only against P1 pressure test functions,
    and the sub-cell divergence sampled at the markers leaks the enclosed
    area roughly 100x faster (documented on the demo page).

``FLUID=ipcs``
    Same mesh/kernel/coupling as ``chorin`` but the incremental
    pressure-correction solver (CN viscous, incremental projection).
    Probing tool for the leak question: IPCS rearranges the projection
    but keeps the same Q2/Q1 pair, so it probes whether the leak is a
    Chorin-specific artifact or a property of the velocity/pressure pair.

Both fluids share the IB2d parameters (rho = 1, mu = 0.01, dt = 1e-3) and a
fluid resolution N = 16 whose velocity nodes (33 x 33, spacing 1/32) sit on
IB2d's 32^2 Eulerian grid.  The thick runs need dt = 1e-4 (stiff-solid
stability of the explicit coupling); everything else is identical.
Difference in setup: IB2d uses a periodic box, here the box has no-slip
walls.

Run (from this directory):

    conda activate afsi-dolfinx
    python main.py                         # fiber, RT, T=1.5 s (1500 steps)
    SHAPE=thick python main.py             # thick annulus, RT
    FLUID=chorin python main.py            # projection background instead
    FLUID=ipcs python main.py              # incremental pressure correction
    T_END=0.1 python main.py               # quick

Environment variables
---------------------
SHAPE     fiber | thick                                        [fiber]
FLUID     rt | chorin | ipcs                                   [rt]
N         fluid cells per direction (lattice = 2N+1 points)    [16]
LX, LY    box size (validation knob: moves the no-slip walls)  [1.0]
T_END     final time [s]                                       [1.5]
DT        time step [s]                                        [1e-3]
OUT_EVERY metric/dump stride in steps                          [20]
M_MEM     fiber markers                                        [64]
K_SPRING  IB2d spring stiffness                                [2.5e4]
DS        spreading weight for the fiber loads (see below)     [1/64, IB2d ds]
MU_F      fluid viscosity (validation knob)                    [0.01]
M         thick: angular cells                                 [96]
K_RAD     thick: radial cells (wall layers)                    [2]
T_WALL    thick: wall thickness                                [0.05]
MU_S      thick: neo-Hookean shear modulus  (calibrated)        [750]
KAPPA_STAB thick: volumetric penalty (= 10 mu_s)               [7500]
RT_SOLVER rt: 'direct' or 'block' linear solver                [direct]
RT_DEGREE rt: Basix RT polynomial degree (>=2)                 [2]
RT_PENALTY rt: SIP penalty / degree^2 (default 20)             [20]
XSTEP     fiber-rt: 'euler' | 'midpoint' (IB2d half-step)      [euler]
XFER      fiber-rt: 'nodal' | 'kernel' (IB2d 4-pt delta)       [nodal]
TAG       label for output files             [<shape>_N<N>[_<fluid>]]
"""
import os

from mpi4py import MPI
from petsc4py import PETSc

import numpy as np

import dolfinx
from dolfinx import log
from dolfinx.fem import (Function, functionspace, dirichletbc,
                         locate_dofs_topological, form, assemble_scalar,
                         Constant, Expression)
from dolfinx.mesh import CellType, GhostMode, locate_entities, create_mesh
from basix.ufl import element
from scipy.sparse import coo_matrix
from dolfinx.fem.petsc import create_vector, assemble_vector

from ufl import (Measure, TestFunction, SpatialCoordinate, as_vector,
                 dot, dx, inner, grad, det, div as ufl_div)

from afsic import (ChorinSolver, IPCSSolver, IBMesh, IBInterpolation,
                   RTFluidSolver, RTNodalCoupling)
from materials import NeoHookean

comm = MPI.COMM_WORLD
rank = comm.rank


def _phi4(r):
    """Peskin 4-point delta function (scalar phi, without the 1/dx factor)."""
    r = np.abs(r)
    out = np.zeros_like(r)
    lo = r < 1.0
    hi = (r >= 1.0) & (r < 2.0)
    out[lo] = (3.0 - 2.0 * r[lo]
               + np.sqrt(1.0 + 4.0 * r[lo] - 4.0 * r[lo] ** 2)) / 8.0
    out[hi] = (5.0 - 2.0 * r[hi]
               - np.sqrt(-7.0 + 12.0 * r[hi] - 4.0 * r[hi] ** 2)) / 8.0
    return out


class KernelCoupling:
    """IB2d-convention transfer for the RT path: Peskin's 4-point regularized
    delta evaluated between the markers and the (2N+1) velocity lattice
    (spacing 1/(2N), aligned with IB2d's Eulerian grid), remapped into the RT
    space through the nodal coupling of that lattice.

    ``interpolate`` and ``spread`` are exact adjoints and reproduce IB2d's
    operators: v_m = sum_j phi4 phi4 u(x_j) (partition of unity), and the
    adjoint spread.  Only interior lattice nodes 1..2N-1 are used.
    """

    def __init__(self, V, lx, ly, n):
        self.n = int(n)
        self.hx = lx / (2 * self.n)
        self.hy = ly / (2 * self.n)
        idx = np.arange(1, 2 * self.n)
        gx = idx * self.hx
        gy = idx * self.hy
        lat = np.array([[x, y] for y in gy for x in gx])
        self.n_lat = len(lat)
        self.lc = RTNodalCoupling(V)
        self.E_lat = self.lc.update(lat)
        self.offsets = np.array([-1, 0, 1, 2])
        self.E_ker = None
        self.points = None

    def update(self, positions):
        if hasattr(positions, "x"):
            xy = positions.x.array.reshape(-1, 2)
        else:
            xy = np.asarray(positions, dtype=np.float64)[:, :2]
        n_m = len(xy)
        i0 = np.floor(xy[:, 0] / self.hx).astype(int)
        j0 = np.floor(xy[:, 1] / self.hy).astype(int)
        ii = i0[:, None] + self.offsets[None, :]      # (n_m, 4) lattice cols
        jj = j0[:, None] + self.offsets[None, :]      # (n_m, 4) lattice rows
        px = _phi4(xy[:, 0:1] / self.hx - ii)         # (n_m, 4)
        py = _phi4(xy[:, 1:2] / self.hy - jj)         # (n_m, 4)
        w = px[:, :, None] * py[:, None, :]           # (n_m, 4, 4)
        side = 2 * self.n - 1
        cols = (((jj - 1)[:, None, :]) * side
                + ((ii - 1)[:, :, None]))             # (n_m, 4, 4)
        ok = (((ii >= 1) & (ii <= side))[:, :, None]
              & ((jj >= 1) & (jj <= side))[:, None, :])
        mm = np.repeat(np.arange(n_m), 16)[ok.reshape(-1)]
        cc = cols.reshape(-1)[ok.reshape(-1)]
        vv = w.reshape(-1)[ok.reshape(-1)]
        rows = np.concatenate([2 * mm, 2 * mm + 1])
        cols2 = np.concatenate([2 * cc, 2 * cc + 1])
        vals2 = np.concatenate([vv, vv])
        W = coo_matrix((vals2, (rows, cols2)),
                       shape=(2 * n_m, 2 * self.n_lat)).tocsr()
        self.E_ker = (W @ self.E_lat).tocsr()
        self.points = xy
        return self.E_ker

    def interpolate(self, velocity, out=None):
        coeff = (velocity.x.array if hasattr(velocity, "x")
                 else np.asarray(velocity))
        return np.asarray(self.E_ker @ coeff)

    def spread(self, loads, out=None):
        if hasattr(loads, "getArray"):
            loads = loads.getArray(readonly=True)
        return np.asarray(self.E_ker.T @ np.asarray(loads))


# ---------------------------------------------------------------------------
# Configuration (IB2d rubberband example values)
# ---------------------------------------------------------------------------
SHAPE = os.environ.get("SHAPE", "fiber")
FLUID = os.environ.get("FLUID", "rt")
assert SHAPE in ("fiber", "thick")
assert FLUID in ("rt", "chorin", "ipcs")

N = int(os.environ.get("N", "16"))
DT = float(os.environ.get("DT", "1e-3"))
T_END = float(os.environ.get("T_END", "1.5"))
OUT_EVERY = int(os.environ.get("OUT_EVERY", "20"))
STEPS = int(round(T_END / DT))

LX = float(os.environ.get("LX", "1.0"))
LY = float(os.environ.get("LY", "1.0"))
CX, CY = 0.5, 0.5
RHO_F = 1.0
MU_F = float(os.environ.get("MU_F", "0.01"))

RMAX = 0.2          # ellipse semi-axis in x (IB2d: printed as rmax)
RMIN = 0.4          # ellipse semi-axis in y (IB2d: printed as rmin)
M_MEM = int(os.environ.get("M_MEM", "64"))
K_SPRING = float(os.environ.get("K_SPRING", "2.5e4"))

M_ANG = int(os.environ.get("M", "96"))     # thick: angular cells
K_RAD = int(os.environ.get("K_RAD", "2"))  # thick: radial layers
T_WALL = float(os.environ.get("T_WALL", "0.05"))
# neo-Hookean shear modulus / volumetric penalty, calibrated so the thick
# band rings with the IB2d rubberband: the original mu_s = 40 gives a 1.52-s
# period (8x slower than IB2d); with mu_s = 750 (kappa = 10 mu_s) the first
# swing lands on IB2d's (a_x peak 0.106 s / 0.382 vs IB2d 0.10 / 0.382;
# a_y 0.218 / 0.362 vs IB2d 0.22 / 0.363) and the period is 0.233 s vs
# IB2d's ~0.20 s.  NB: the stiff band exceeds the explicit coupling's
# stability limit at dt = 1e-3 (already unstable at mu_s = 160); the thick
# runs need DT = 1e-4 (checked for mu_s = 640/750 at T <= 0.5 s).
MU_S = float(os.environ.get("MU_S", "750"))
KAPPA_STAB = float(os.environ.get("KAPPA_STAB", "7500"))
RT_SOLVER = os.environ.get("RT_SOLVER", "direct")
RT_DEGREE = int(os.environ.get("RT_DEGREE", "2"))
RT_PENALTY = float(os.environ.get("RT_PENALTY", "20"))
XSTEP = os.environ.get("XSTEP", "euler")
XFER = os.environ.get("XFER", "nodal")
assert XSTEP in ("euler", "midpoint")
assert XFER in ("nodal", "kernel")

# stress-free outer radius chosen so the affine map gives the IB2d ellipse
R_OUT = np.sqrt(RMAX * RMIN)
SX = RMAX / R_OUT    # initial x scale of the affine prestrain (= 1/sqrt(2))
SY = RMIN / R_OUT    # initial y scale (= sqrt(2))

# IB2d spreading weight.  Their Lagrangian spring output is a force DENSITY:
# it is multiplied by the constant ds = min(Lx/(2 Nx_ib2d), Ly/(2 Ny_ib2d))
# before spreading (IBM_Driver.m:203, "Peskin constant ds";
# please_Find_Lagrangian_Forces_On_Eulerian_grid.m multiplies fx,fy by ds
# ahead of the 4-pt kernel spread).  With the nominal AFSI lattice N = 16
# sized to IB2d's grid (Nx_ib2d = 2 N), ds = 1/(4*16) = 1/64.  It is IB2d's
# Lagrangian spacing: a FIXED constant, it does not follow N (an earlier
# 1/(4 N) default silently halved the fiber force in refined-mesh runs).
#
# DS = 1/(4 N) reproduces the IB2d dynamics (probe 2026-10-02): marker
# semi-axis oscillation period 0.201 s vs IB2d's 0.190 s, first/second a_x
# peaks at 0.106/0.302 s vs IB2d's 0.10/0.32 s.  An earlier note claimed
# the running IB2d delivered a further factor 4 less than its documented
# spread; that was a comparison artifact - the max of the spread field's
# x-component (302.7) was matched against the 2-D magnitude of the same
# field (1150.7), which differ by ~4x on this tall ellipse.  Both numbers
# are reproduced exactly by a replica of the documented pipeline.
# DS = 1/(16 N) under-forces the fiber 4x and doubles the period.
# The thick band uses the assembled FEM load, which already carries its
# measure and is unaffected by DS.
DS = float(os.environ.get("DS", str(1.0 / 64.0)))

# output tag: solver suffix appended for non-RT fluids so that the
# chorin/ipcs runs cannot overwrite the RT runs' metrics/snapshots
tag = os.environ.get("TAG", f"{SHAPE}_N{N}"
                     + ("" if FLUID == "rt" else f"_{FLUID}"))
_demo_dir = os.path.dirname(os.path.abspath(__file__))
outdir = os.environ.get("OUTPUT_PATH", os.path.join(_demo_dir, "plot"))
os.makedirs(outdir, exist_ok=True)

if rank == 0:
    print(f"[demo_444:{SHAPE}/{FLUID}] N={N} dt={DT} steps={STEPS} "
          f"rho={RHO_F} mu={MU_F} T_END={T_END}")
    if SHAPE == "fiber":
        print(f"  fiber: {M_MEM} markers, k={K_SPRING}, L_rest=0 (IB2d), "
              f"DS={DS:g} step={XSTEP} xfer={XFER}")
    else:
        print(f"  thick: M={M_ANG} K={K_RAD} wall={T_WALL} R_out={R_OUT:.6g} "
              f"mu_s={MU_S} kappa={KAPPA_STAB}")

# ---------------------------------------------------------------------------
# Fluid mesh (shared) and solver construction
# ---------------------------------------------------------------------------
mesh = dolfinx.mesh.create_rectangle(
    comm=comm, points=((0.0, 0.0), (LX, LY)),
    n=(int(round(N * LX)), int(round(N * LY))),
    cell_type=CellType.quadrilateral, ghost_mode=GhostMode.shared_facet)
mesh.topology.create_connectivity(1, 2)
fdim = mesh.topology.dim - 1
dx_fluid = Measure("dx", domain=mesh)

if FLUID in ("chorin", "ipcs"):
    V = functionspace(mesh, element("Lagrange", mesh.topology.cell_name(), 2,
                                    shape=(mesh.geometry.dim,)))
    Q = functionspace(mesh, element("Lagrange", mesh.topology.cell_name(), 1))

    walls = locate_entities(mesh, fdim,
                            lambda x: np.logical_or(np.logical_or(
                                np.isclose(x[0], 0.0), np.isclose(x[0], LX)),
                                np.logical_or(np.isclose(x[1], 0.0),
                                              np.isclose(x[1], LY))))
    u_zero = np.array((0.0,) * mesh.geometry.dim, dtype=PETSc.ScalarType)
    bcu = [dirichletbc(u_zero, locate_dofs_topological(V, fdim, walls), V)]
    point = dolfinx.mesh.locate_entities_boundary(
        mesh, 0, lambda x: np.logical_and(np.isclose(x[0], 0.0),
                                          np.isclose(x[1], 0.0)))
    bcp = [dirichletbc(PETSc.ScalarType(0.0),
                       locate_dofs_topological(Q, 0, point), Q)]
    # LOAD_MODE=direct (default): b_ib = V_h * f_stored, the exact adjoint
    # of the sampling operator (demo_402/424).  LOAD_MODE=weak keeps the
    # older weak-form term -int f v dx.
    LOAD_MODE = os.environ.get("LOAD_MODE", "direct")
    assert LOAD_MODE in ("direct", "weak")
    _solver_cls = IPCSSolver if FLUID == "ipcs" else ChorinSolver
    solver = _solver_cls(V, Q, bcu, bcp, DT, RHO_F, MU_F,
                         ib_body_force=(LOAD_MODE == "weak"))
    if LOAD_MODE == "direct":
        b_direct = create_vector(V)
        solver.ib_load = b_direct
        # lattice cell area (the IB grid is the (2N+1) P2-node lattice)
        V_h = (LX / (2 * N)) * (LY / (2 * N))
    omega_expr = Expression(solver.u_[1].dx(0) - solver.u_[0].dx(1),
                            Q.element.interpolation_points)
    omega = Function(Q, name="omega")
else:
    facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
    solver = RTFluidSolver(mesh, DT, rho=RHO_F, mu=MU_F, degree=RT_DEGREE,
                           convection=os.environ.get("RT_CONVECTION",
                                                    "1") != "0",
                           dirichlet_facets=facets,
                           penalty=RT_PENALTY * RT_DEGREE ** 2,
                           linear_solver=RT_SOLVER)
    V, Q = solver.V, solver.Q
    if XFER == "kernel":
        transfer = KernelCoupling(V, LX, LY, N)
    else:
        transfer = RTNodalCoupling(V)
    omega = None

# ---------------------------------------------------------------------------
# Immersed structure
# ---------------------------------------------------------------------------
solid_coords = None
solid_force = None
solid_velocity = None
solid_coords_io = None
vert_dof = None
tracked_rings = []
X = None                 # fiber markers (RT path)
load_rt = None           # RT spread load (fiber path)

if SHAPE == "fiber":
    theta = 2.0 * np.pi * np.arange(M_MEM) / M_MEM
    # IB2d vertex file: x = 1/2 + rmax cos(theta), y = 1/2 + rmin sin(theta)
    pts = np.column_stack([CX + RMAX * np.cos(theta),
                           CY + RMIN * np.sin(theta)])
    tracked_rings = [np.arange(M_MEM)]

    if FLUID == "rt":
        X = pts.copy()
        transfer.update(X)
        load_rt = transfer.spread(np.zeros(2 * M_MEM))

        def spring_load(xy):
            """IB2d vertex force density x ds (assembled nodal load)."""
            return K_SPRING * DS * (np.roll(xy, -1, axis=0)
                                    + np.roll(xy, 1, axis=0) - 2.0 * xy)

        if XSTEP == "euler":
            def advance():
                global load_rt
                solver.solve_one_step(load_rt)
                v = transfer.interpolate(solver.u_).reshape(-1, 2)
                X[:] = X + DT * v
                transfer.update(X)
                load_rt = transfer.spread(spring_load(X).ravel())
                return v
        else:
            def advance():
                # IB2d midpoint scheme: force and interpolation stencil at
                # x^{n+1/2}, move from x^n with u^{n+1}
                transfer.update(X)
                v_old = transfer.interpolate(solver.u_).reshape(-1, 2)
                x_half = X + 0.5 * DT * v_old
                transfer.update(x_half)
                load = transfer.spread(spring_load(x_half).ravel())
                solver.solve_one_step(load)
                v = transfer.interpolate(solver.u_).reshape(-1, 2)
                X[:] = X + DT * v
                return v
    else:
        cells = np.column_stack([np.arange(M_MEM),
                                 (np.arange(M_MEM) + 1) % M_MEM]).astype(np.int64)
        coord_element = element("Lagrange", "interval", 1, shape=(2,))
        structure = create_mesh(comm, cells, coord_element, pts)

        v_cg1 = element("Lagrange", "interval", 1, shape=(2,))
        Vs = functionspace(structure, v_cg1)
        Vs_io = Vs
        assert Vs.dofmap.index_map_bs == 2

        solid_coords = Function(Vs, name="solid_coords")
        solid_force = Function(Vs, name="solid_force")
        solid_coords_io = Function(Vs_io, name="solid_coords_io")
        solid_velocity = Function(Vs, name="solid_velocity")

        solid_coords.interpolate(lambda x: np.array([x[0], x[1]]))
        # NOTE: the P1 dof order is a permutation of the input point order;
        # the marker convention is recovered through coordinate matching
        # (vert_dof, built below) so the spring loop follows the IB2d order.

elif SHAPE == "thick":
    # polar quad annulus; reference (stress-free) configuration is the circle
    R_IN = R_OUT - T_WALL
    pts = np.zeros((M_ANG * (K_RAD + 1), 2), dtype=np.float64)
    for i in range(M_ANG):
        th = 2.0 * np.pi * i / M_ANG
        for k in range(K_RAD + 1):
            r = R_IN + (R_OUT - R_IN) * k / K_RAD
            pts[i * (K_RAD + 1) + k] = (CX + r * np.cos(th),
                                        CY + r * np.sin(th))
    cells_vtk = np.zeros((M_ANG * K_RAD, 4), dtype=np.int64)
    c = 0
    for i in range(M_ANG):
        ip = (i + 1) % M_ANG
        for k in range(K_RAD):
            v_ik = i * (K_RAD + 1) + k
            v_ipk = ip * (K_RAD + 1) + k
            cells_vtk[c] = [v_ik, v_ipk, v_ipk + 1, v_ik + 1]
            c += 1
    perm = np.asarray(dolfinx.cpp.io.perm_vtk(CellType.quadrilateral, 4),
                      dtype=np.int64)
    coord_element = element("Lagrange", "quadrilateral", 1, shape=(2,))
    structure = create_mesh(comm, cells_vtk[:, perm], coord_element, pts)
    assert structure.geometry.dim == 2

    v_cg2 = element("Lagrange", structure.topology.cell_name(), 2, shape=(2,))
    v_cg1 = element("Lagrange", structure.topology.cell_name(), 1, shape=(2,))
    Vs = functionspace(structure, v_cg2)
    Vs_io = functionspace(structure, v_cg1)
    assert Vs.dofmap.index_map_bs == 2

    solid_coords = Function(Vs, name="solid_coords")
    solid_force = Function(Vs, name="solid_force")
    solid_coords_io = Function(Vs_io, name="solid_coords_io")
    solid_velocity = Function(Vs, name="solid_velocity")

    # prestrain: stress-free circle -> the same initial ellipse as the fiber
    solid_coords.interpolate(
        lambda x: np.array([CX + (x[0] - CX) * SX, CY + (x[1] - CY) * SY]))

    dVs = TestFunction(Vs)
    material = NeoHookean(mu_s=MU_S, lambda_s=KAPPA_STAB, model="flory")
    PK1 = material.first_piola_kirchhoff_stress_v1(structure, solid_coords)
    L_hat = form(-inner(PK1, grad(dVs)) * dx)
    b1 = create_vector(Vs)

    VJ = functionspace(structure, element("Lagrange",
                                          structure.topology.cell_name(), 1))
    J_func = Function(VJ, name="J")
    J_expr = Expression(det(grad(solid_coords)), VJ.element.interpolation_points)

    # reference rings tracked for metrics (indices into the vertex list)
    tracked_rings = [np.arange(M_ANG) * (K_RAD + 1),              # inner ring
                     np.arange(M_ANG) * (K_RAD + 1) + K_RAD]      # outer ring

# vertex dofs of the io space, matched by coordinates (robust to reordering)
if SHAPE == "thick" or (SHAPE == "fiber" and FLUID != "rt"):
    dof_rows = Vs_io.tabulate_dof_coordinates()[:, :2]
    match_targets = np.vstack([pts[r] for r in tracked_rings])
    vert_dof = []
    for c0 in match_targets:
        j = int(np.argmin(np.linalg.norm(dof_rows - c0, axis=1)))
        assert np.linalg.norm(dof_rows[j] - c0) < 1e-10 * (1.0 + np.linalg.norm(c0))
        vert_dof.append(j)
    vert_dof = np.array(vert_dof)
    assert len(np.unique(vert_dof)) == len(vert_dof), \
        "dof matching is not injective"
    if SHAPE == "fiber":
        assert np.allclose(solid_coords.x.array.reshape(-1, 2)[vert_dof], pts,
                           atol=1e-12), "marker order does not recover the IB2d list"

def tracked_xy():
    if SHAPE == "fiber" and FLUID == "rt":
        return X.copy()
    solid_coords_io.interpolate(solid_coords)
    return solid_coords_io.x.array.reshape(-1, 2)[vert_dof]

def polygon_area(xy):
    x, y = xy[:, 0], xy[:, 1]
    xp, yp = np.roll(x, -1), np.roll(y, -1)
    return 0.5 * abs(np.sum(x * yp - xp * y))

def polygon_extents(xy):
    cx, cy = xy[:, 0].mean(), xy[:, 1].mean()
    return (float(np.max(np.abs(xy[:, 0] - cx))),
            float(np.max(np.abs(xy[:, 1] - cy))))

# initial metrics + regression guards --------------------------------------
xy0 = tracked_xy()
if SHAPE == "fiber":
    A0 = polygon_area(xy0)
    A_poly = 0.5 * M_MEM * np.sin(2 * np.pi / M_MEM) * RMAX * RMIN
    assert abs(A0 - A_poly) < 1e-12, f"initial area {A0} vs {A_poly}"
    ax0, ay0 = polygon_extents(xy0)
    assert abs(ax0 - RMAX) < 1e-12 and abs(ay0 - RMIN) < 1e-12
else:
    A0 = polygon_area(xy0[M_ANG:2 * M_ANG])           # outer ring
    ax0, ay0 = polygon_extents(xy0[M_ANG:2 * M_ANG])
    # outer ring starts exactly on the IB2d ellipse
    assert abs(ax0 - RMAX) < 1e-10 and abs(ay0 - RMIN) < 1e-10

# ---------------------------------------------------------------------------
# Coupling machinery (chorin: lattice + kernel; rt: nodal coupling set up
# in the structure branch above / below)
# ---------------------------------------------------------------------------
if FLUID != "rt":
    ibmesh = IBMesh(0.0, LX, 0.0, LY, N, N, 2)
    ib_interpolation = IBInterpolation(ibmesh)
    coords_bg = Function(V)
    coords_bg.interpolate(lambda x: np.array([x[0], x[1]]))
    ibmesh.build_map(coords_bg._cpp_object)
    ib_interpolation.evaluate_current_points(solid_coords._cpp_object)
else:
    if SHAPE == "thick":
        transfer.update(solid_coords)

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
csv_path = os.path.join(outdir, f"metrics_{tag}.csv")
with open(csv_path, "w") as fh:
    fh.write("step,t,area,ax,ay,max_u,kinetic"
             + (",minJ" if SHAPE == "thick" else "") + "\n")

series = {"t": [], "xy": [], "area": [], "ax": [], "ay": []}

log.set_log_level(log.LogLevel.WARNING)

def metrics(step, t):
    xy = tracked_xy()
    n_r = len(tracked_rings[0])
    if SHAPE == "fiber":
        area = polygon_area(xy[:n_r])
        ax, ay = polygon_extents(xy[:n_r])
    else:
        area = polygon_area(xy[n_r:2 * n_r])          # outer ring
        ax, ay = polygon_extents(xy[n_r:2 * n_r])
    umax = float(np.max(np.abs(solver.u_.x.array)))
    ke = 0.5 * RHO_F * float(assemble_scalar(
        form(dot(solver.u_, solver.u_) * dx_fluid)))
    row = [step, t, area, ax, ay, umax, ke]
    if SHAPE == "thick":
        J_func.interpolate(J_expr)
        row.append(float(np.min(J_func.x.array)))
    with open(csv_path, "a") as fh:
        fh.write(",".join((str(int(row[0])),)
                          + tuple(f"{v:.10g}" for v in row[1:])) + "\n")
    series["t"].append(t)
    series["xy"].append(xy.copy())
    series["area"].append(area)
    series["ax"].append(ax)
    series["ay"].append(ay)
    if rank == 0:
        extra = f" minJ={row[7]:.4f}" if SHAPE == "thick" else ""
        print(f"  step {step:6d} t={t:7.4f} area={area:.8f} "
              f"ax={ax:.5f} ay={ay:.5f} |u|={umax:.3e} KE={ke:.3e}{extra}")
    return row

# ---------------------------------------------------------------------------
# Coupling closures for the remaining paths
# ---------------------------------------------------------------------------
if SHAPE == "fiber" and FLUID != "rt":
    def spring_force():
        """IB2d force density x ds in marker order, written in dof order."""
        full = solid_coords.x.array.reshape(-1, 2)   # dof order
        xy = full[vert_dof]                          # marker (IB2d) order
        force = (K_SPRING * DS
                 * (np.roll(xy, -1, axis=0) + np.roll(xy, 1, axis=0)
                    - 2.0 * xy))
        out = np.zeros_like(full)
        out[vert_dof] = force
        solid_force.x.array[:] = out.ravel()

    def advance():
        solver.solve_one_step()
        ib_interpolation.fluid_to_solid(solver.u_._cpp_object,
                                        solid_velocity._cpp_object)
        solid_coords.x.array[:] += solid_velocity.x.array[:] * DT
        solid_coords.x.scatter_forward()
        ib_interpolation.evaluate_current_points(solid_coords._cpp_object)
        spring_force()
        ib_interpolation.solid_to_fluid(solver.f._cpp_object,
                                        solid_force._cpp_object)
        solver.f.x.scatter_forward()
        if LOAD_MODE == "direct":
            solver.f.x.petsc_vec.copy(result=b_direct)
            b_direct.scale(V_h)
        return solid_velocity.x.array.reshape(-1, 2)[vert_dof]

elif SHAPE == "thick" and FLUID != "rt":
    def advance():
        solver.solve_one_step()
        ib_interpolation.fluid_to_solid(solver.u_._cpp_object,
                                        solid_velocity._cpp_object)
        solid_coords.x.array[:] += solid_velocity.x.array[:] * DT
        solid_coords.x.scatter_forward()
        ib_interpolation.evaluate_current_points(solid_coords._cpp_object)
        with b1.localForm() as loc:
            loc.set(0)
        assemble_vector(b1, L_hat)
        b1.ghostUpdate(addv=PETSc.InsertMode.ADD,
                       mode=PETSc.ScatterMode.REVERSE)
        with b1.getBuffer() as arr:
            solid_force.x.array[: len(arr)] = arr[:]
        ib_interpolation.solid_to_fluid(solver.f._cpp_object,
                                        solid_force._cpp_object)
        solver.f.x.scatter_forward()
        if LOAD_MODE == "direct":
            solver.f.x.petsc_vec.copy(result=b_direct)
            b_direct.scale(V_h)

elif SHAPE == "thick" and FLUID == "rt":
    def advance():
        with b1.localForm() as loc:
            loc.set(0)
        assemble_vector(b1, L_hat)
        b1.ghostUpdate(addv=PETSc.InsertMode.ADD,
                       mode=PETSc.ScatterMode.REVERSE)
        transfer.update(solid_coords)
        load = transfer.spread(b1)
        solver.solve_one_step(load)
        transfer.interpolate(solver.u_, out=solid_velocity)
        solid_coords.x.array[:] += solid_velocity.x.array[:] * DT
        solid_coords.x.scatter_forward()

# ---------------------------------------------------------------------------
# Time loop (explicit coupling, same order as demos 441/442)
# ---------------------------------------------------------------------------
for step in range(STEPS + 1):
    t = step * DT
    if step > 0:
        advance()
    if step % OUT_EVERY == 0 or step == STEPS:
        metrics(step, t)

# ---------------------------------------------------------------------------
# Final snapshot
# ---------------------------------------------------------------------------
fields = dict(
    t_series=np.array(series["t"]),
    xy_series=np.array(series["xy"]),
    area_series=np.array(series["area"]),
    ax_series=np.array(series["ax"]),
    ay_series=np.array(series["ay"]),
    meta=np.array([N, DT, T_END, RMAX, RMIN, K_SPRING, M_MEM, A0]),
)
if FLUID != "rt":
    omega.interpolate(omega_expr)
    fields.update(
        fluid_x=V.tabulate_dof_coordinates()[:, :2],
        fluid_u=solver.u_.x.array.reshape(-1, mesh.geometry.dim),
        p_x=Q.tabulate_dof_coordinates()[:, :2], p=solver.p_.x.array,
        omega_x=Q.tabulate_dof_coordinates()[:, :2], omega=omega.x.array,
    )
else:
    fields.update(fluid_u=solver.u_.x.array, p=solver.p_.x.array)

np.savez_compressed(os.path.join(outdir, f"snapshot_{tag}.npz"), **fields)
if rank == 0:
    print(f"[demo_444:{SHAPE}/{FLUID}] done: {STEPS} steps, N={N}; "
          f"csv={csv_path}")
