"""Efficiency comparison: demo_422 (monolithic IBFE + MUMPS) vs
demo_336 (Chorin operator-split + HYPRE) on the same 64x64 lid-driven
cavity with an immersed elastic disk.

Both core loops are run in-process with the SAME physics (mu=0.01, mu_s=0.1,
disk R=0.2 at (0.6,0.5)).  Per-step wall time excludes setup/JIT.  The two
schemes differ fundamentally:
  * demo_336: explicit operator-split Chorin, dt=1/200 (CFL-limited)
  * demo_422: implicit monolithic Newton 3x3, dt=0.01 (backward Euler)
so we report BOTH per-step time and wall time per simulated second.
"""
import os
import sys
import time
import numpy as np

os.environ.setdefault("NX", "64")
os.environ.setdefault("NY", "64")
os.environ.setdefault("SOLID_H", "0.0125")
os.environ.setdefault("STEPS", str(int(os.environ.get("NSTEPS", "5"))))
NSTEPS = int(os.environ.get("NSTEPS", "5"))

from petsc4py import PETSc
from mpi4py import MPI

# demo_336 dependencies (afsic C++ extension)
G = "/home/Pengfei.Ma@glasgow.ac.uk/afsi"
sys.path.insert(0, f"{G}/afsic/src")
from afsic import ChorinSolver, IBMesh, IBInterpolation  # noqa: E402

import dolfinx
from dolfinx.fem import (Function, functionspace, dirichletbc,
                         locate_dofs_topological, form, assemble_scalar)
from dolfinx.mesh import CellType, GhostMode, locate_entities, meshtags
from basix.ufl import element
from ufl import (Identity, TestFunction, TrialFunction, inv, det, ln,
                 div, dot, dx, grad, inner)
from dolfinx.fem.petsc import create_vector, assemble_vector

results = {}


def header(s):
    print(f"\n{'=' * 68}\n{s}\n{'=' * 68}")


# ===========================================================================
# Part A: demo_336 (Chorin operator-split) core loop, no swanlab / XDMF
# ===========================================================================
def run_demo336():
    header("Part A: demo_336  Chorin operator-split (afsic)  64x64 quad")
    config = {"nssolver": "chorinsolver", "dt": 1.0 / 200.0, "rho": 1.0,
              "Lx": 1.0, "Ly": 1.0, "Nx": 64, "Ny": 64, "Nl": 20,
              "mu": 0.01, "mu_s": 0.1}

    mesh = dolfinx.mesh.create_rectangle(
        MPI.COMM_WORLD, ((0.0, 0.0), (1.0, 1.0)), (64, 64),
        cell_type=CellType.quadrilateral, ghost_mode=GhostMode.shared_facet)
    mesh.topology.create_connectivity(1, 2)
    fdim = mesh.topology.dim - 1
    marker_up = 4
    boundaries = [(1, lambda x: np.isclose(x[0], 0)),
                  (2, lambda x: np.isclose(x[0], 1)),
                  (3, lambda x: np.isclose(x[1], 0)),
                  (4, lambda x: np.isclose(x[1], 1))]
    facet_indices, facet_markers = [], []
    for (marker, locator) in boundaries:
        fs = locate_entities(mesh, fdim, locator)
        facet_indices.append(fs)
        facet_markers.append(np.full_like(fs, marker))
    facet_indices = np.hstack(facet_indices).astype(np.int32)
    facet_markers = np.hstack(facet_markers).astype(np.int32)
    tag = meshtags(mesh, fdim, facet_indices[np.argsort(facet_indices)],
                   facet_markers[np.argsort(facet_indices)])

    v_cg2 = element("Lagrange", mesh.topology.cell_name(), 2,
                    shape=(mesh.geometry.dim,))
    s_cg1 = element("Lagrange", mesh.topology.cell_name(), 1)
    V = functionspace(mesh, v_cg2)
    Q = functionspace(mesh, s_cg1)

    class UpVelocity:
        def __init__(self, t): self.t = t
        def __call__(self, x):
            v = np.zeros((2, x.shape[1]), dtype=PETSc.ScalarType)
            v[0] = 1.0
            return v

    u_up = Function(V)
    upv = UpVelocity(0.0)
    u_up.interpolate(upv)
    bcu_up = dirichletbc(u_up, locate_dofs_topological(V, fdim, tag.find(marker_up)))
    u_nonslip = np.zeros(2, dtype=PETSc.ScalarType)
    bcu = [bcu_up,
           dirichletbc(u_nonslip, locate_dofs_topological(V, fdim, tag.find(1)), V),
           dirichletbc(u_nonslip, locate_dofs_topological(V, fdim, tag.find(2)), V),
           dirichletbc(u_nonslip, locate_dofs_topological(V, fdim, tag.find(3)), V)]
    point_loc = dolfinx.mesh.locate_entities_boundary(mesh, 0,
                                                      lambda x: np.isclose(x[0], 0) & np.isclose(x[1], 0))
    bcp = [dirichletbc(0.0, locate_dofs_topological(Q, 0, point_loc), Q)]

    ns = ChorinSolver(V, Q, bcu, bcp, config["dt"], config["rho"], config["mu"])

    home = os.path.expanduser("~")
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD,
                             f"{home}/afsi-data/336-lid-driven-disk/mesh/circle_20.xdmf",
                             "r", encoding=dolfinx.io.XDMFFile.Encoding.HDF5) as f:
        structure = f.read_mesh()
    Vs = functionspace(structure, element("Lagrange", structure.topology.cell_name(),
                                          2, shape=(structure.geometry.dim,)))
    solid_coords = Function(Vs, name="solid_coords")
    solid_force = Function(Vs, name="solid_force")
    solid_velocity = Function(Vs, name="solid_velocity")

    mu_s = config["mu_s"]
    dVs = TestFunction(Vs)
    FF = grad(solid_coords)
    L_hat = form(-inner(mu_s * (FF - inv(FF).T), grad(dVs)) * dx)
    b1 = create_vector(Vs)  # dolfinx 0.10: create_vector takes a space, not a form

    ibmesh = IBMesh(0.0, 1.0, 0.0, 1.0, 64, 64, 2)
    ib_interp = IBInterpolation(ibmesh)
    coords_bg = Function(V)
    coords_bg.interpolate(lambda x: np.array([x[0], x[1]]))
    solid_coords.interpolate(lambda x: np.array([x[0], x[1]]))
    ibmesh.build_map(coords_bg._cpp_object)
    ib_interp.evaluate_current_points(solid_coords._cpp_object)

    nu = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
    np_ = Q.dofmap.index_map.size_local
    ns_ = Vs.dofmap.index_map.size_local * Vs.dofmap.index_map_bs

    # ---- timing ----
    t_step = []
    for _ in range(NSTEPS):
        upv.t = 0.0
        u_up.interpolate(upv)
        t0 = time.perf_counter()
        ns.solve_one_step()
        ib_interp.fluid_to_solid(ns.u_._cpp_object, solid_velocity._cpp_object)
        solid_coords.x.array[:] += solid_velocity.x.array[:] * config["dt"]
        solid_coords.x.scatter_forward()
        ib_interp.evaluate_current_points(solid_coords._cpp_object)
        b1.set(0)  # zero the vector (petsc4py: Vec.set, no localForm().set)
        assemble_vector(b1, L_hat)
        b1.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        with b1.getBuffer() as arr:
            solid_force.x.array[:len(arr)] = arr[:]
        ib_interp.solid_to_fluid(ns.f._cpp_object, solid_force._cpp_object)
        ns.f.x.scatter_forward()
        t_step.append(time.perf_counter() - t0)

    per = float(np.mean(t_step[1:])) if NSTEPS > 1 else float(t_step[0])
    dt = config["dt"]
    results["336"] = dict(dofs=(nu, np_, ns_), per_step=per, dt=dt,
                          per_sec=per / dt)
    print(f"  fluid dofs: velocity {nu} + pressure {np_} = {nu + np_}; "
          f"solid {ns_}")
    print(f"  dt = {dt}   steps = {NSTEPS}")
    print(f"  per-step: {per:.4f} s   ({NSTEPS / per:.1f} steps/s)")
    print(f"  per simulated second: {per / dt:.2f} s wall / 1 s sim")


# ===========================================================================
# Part B: demo_422 (monolithic IBFE + MUMPS)
# ===========================================================================
def run_demo422():
    header("Part B: demo_422  monolithic IBFE + MUMPS  64x64 tri")
    from main import make_config, ImmersedFEM

    cfg = make_config()
    cfg["Nx"] = cfg["Ny"] = 64
    cfg["num_steps"] = NSTEPS
    cfg["output_path"] = None
    s = ImmersedFEM(cfg)

    nu, np_, ns_ = s.n_u, s.n_p, s.n_s
    dt = cfg["dt"]
    s.X[:s.n_u] = s._initial_velocity()

    t_step = []
    for _ in range(NSTEPS):
        t0 = time.perf_counter()
        s.solve_monolithic()
        t_step.append(time.perf_counter() - t0)

    per = float(np.mean(t_step[1:])) if NSTEPS > 1 else float(t_step[0])
    results["422"] = dict(dofs=(nu, np_, ns_), per_step=per, dt=dt,
                          per_sec=per / dt)
    print(f"  fluid dofs: velocity {nu} + pressure {np_} = {nu + np_}; "
          f"solid {ns_}")
    print(f"  dt = {dt}   steps = {NSTEPS}")
    print(f"  per-step: {per:.4f} s   ({NSTEPS / per:.1f} steps/s)")
    print(f"  per simulated second: {per / dt:.2f} s wall / 1 s sim")


# ===========================================================================
if __name__ == "__main__":
    run_demo336()
    run_demo422()

    header("SUMMARY  (64x64, same physics)")
    a, b = results["336"], results["422"]
    print(f"  fluid dofs      : 336={sum(a['dofs'][:2])}   422={sum(b['dofs'][:2])}"
          f"   (solid 336={a['dofs'][2]} / 422={b['dofs'][2]})")
    print(f"  dt              : 336={a['dt']}   422={b['dt']}")
    print(f"  per-step (s)    : 336={a['per_step']:.4f}   422={b['per_step']:.4f}"
          f"   ratio 422/336 = {b['per_step'] / a['per_step']:.1f}x")
    print(f"  wall/1s-sim (s) : 336={a['per_sec']:.2f}   422={b['per_sec']:.2f}"
          f"   ratio 422/336 = {b['per_sec'] / a['per_sec']:.1f}x")
    print("\n  note: dt differs by scheme (explicit CFL vs implicit); wall time "
          "per simulated\n  second is the fair efficiency metric.")
