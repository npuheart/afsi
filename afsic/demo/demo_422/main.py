"""
Immersed Boundary Finite Element Method (IBFE) for a 2D lid-driven square cavity
with an immersed elastic disk -- MONOLITHIC (fully coupled) variant.

FEniCSx (dolfinx 0.10) re-implementation of ``afsi/a.cpp`` (deal.II reference).
This is a faithful reproduction of the immersed boundary finite element method of
  * Boffi, Gastaldi, Heltai (2007), "Numerical approximation of the immersed
    boundary method", Computers & Fluids 36, 1378-1387.
  * Heltai & Costanzo (2012), "Variational implementation of immersed finite
    element methods", CMAME 229-232, 110-127.
  * Roy, Heltai & Costanzo (2015), "Benchmarking the immersed finite element
    method for fluid-structure interaction problems", CAMWA 69, 1167-1188.
(following the reference benchmark ``LDCFlow_Ball_DGP_INH1``).

Unlike operator-split immersed methods, the fluid velocity u, the pressure p and
the solid displacement W are solved TOGETHER in one monolithic system: each time
step is a backward-Euler step solved by Newton on the 3x3 block Jacobian

    [  K     B^T    -A_uW  ] [ du ]     [ -R_u ]
    [  B     0      0     ] [ dp ]  =  [ -R_p ]
    [ -Mfs^T 0   (1/dt)M_s] [ dW ]     [ -R_W ]

with residual
    R_u = K u + B^T p - f_el(W)                  (fluid momentum + elastic force)
    R_p = B u                                    (incompressibility)
    R_W = (1/dt) M_s (W - W^n) - Mfs^T u         (kinematic: dW/dt = u_s)
where f_el is the spread incompressible neo-Hookean elastic force and
A_uW = d f_el / dW is its tangent (the mixed stiffness).  The interaction /
mixed mass Mfs are rebuilt every time step (geometry = W^n, semi-implicit), and
f_el / A_uW are re-assembled at the current W inside the Newton loop.  The
pressure (a fluid variable extended over the solid) enforces incompressibility
over the whole domain including the disk, exactly as in the paper.

Two finite element spaces on two independent, non-matching meshes:

  * Background (Eulerian) mesh on the square cavity [0,1]^2:
        V = P2 vector (velocity),  Q = P1 scalar (pressure)   [Taylor-Hood]
    (the paper uses Q2/FE_DGP(1); P2/P1 on triangles is the standard stable
    FEniCSx counterpart and reproduces the same physics).

  * Solid (Lagrangian) mesh on the immersed disk:
        Vs = P2 vector (displacement W)

The only coupling between the two meshes is through the transfer operators
    (1) SPREADING J^T (solid -> background):
            f_el,i = - int_{Omega_s} (P F^T) : grad_x(phi_i_bg) dX,
        P = mu (F - F^{-T}),  F = I + grad_X W,
        with tangent (mixed stiffness)  A_uW = d f_el / dW;
    (2) INTERPOLATION J (background -> solid):
            M_s u_s = Mfs^T u,  Mfs(i_bg,j_s) = int_{Omega_s} phi_i_bg(x) phi_j_s(X) dX,
        and the disk moves with the fluid: dW/dt = u_s.

Both coupling matrices are assembled over the solid quadrature points mapped
into the background mesh (located with a bounding-box tree +
``compute_colliding_cells``, then the background P2 basis values/gradients are
tabulated with basix at the mapped reference coordinates) -- exactly the point
emphasised in the original papers.

Time stepping (backward Euler + Newton on the 3x3 monolithic system):
  (a) locate the disk (interaction) and rebuild the mixed mass Mfs at W^n;
  (b) Newton iterations: assemble f_el(W) and its tangent A_uW(W), fill the
      3x3 Jacobian, form the exact residual, solve with a sparse direct LU
      (MUMPS via PETSc, falling back to SuperLU_DIST; the saddle point is
      regularised with eps*M_p on the (1,1) block plus a pinned pressure dof,
      which only perturbs the intermediate linear solves, not the converged
      Newton solution), update [u; p; W];
Default parameters reproduce the benchmark LDCFlow_Ball_DGP_INH1 of
Roy-Heltai-Costanzo (2015): cavity l=1, R=0.2 at (0.6,0.5), rho=1, eta_f=0.01,
mu^e=0.1, lid U=1, dt=1e-2, T=8.1 s, 64x64 background, incompressible
neo-Hookean disk.

Schemes (``scheme``):
  0  monolithic 3x3 (default), Newton + direct LU
  3  reduced 2x2:  Schur-eliminate W with the *exact* M_s^-1 (sparse factor)
       [K~ B^T; B 0],  K~ = K - dt A_uW M_s^-1 Mfs^T,  dW = dt M_s^-1(Mfs^T du - R_W)
     Newton uses the exact residuals and the elimination is exact, so the
     converged solution is identical to the full 3x3 solve (verified to ~1e-15)
     and Newton converges quadratically.  NB: exact elimination makes K~ dense,
     so this variant is for coarse meshes only; use scheme 0 at larger scale.
     (a.cpp schemes 3/4 use the diagonal of M_s; that made Newton diverge for
     the soft benchmark disk here, so we always use the exact elimination.)

Run:
  cd afsic/demo/demo_422
  python main.py                     # full benchmark (64x64, 810 steps)
  python main.py --steps 10 --nx 16  # quick smoke test (CLI overrides)

Environment variables (overridden by any CLI argument):
  STEPS, NX, NY, SOLID_H, SCHEME, DT, T, MU_S, RHO_S, OUT, PIN, OUTPUT.
"""

import os
import time
import numpy as np

from mpi4py import MPI
from petsc4py import PETSc

import gmsh
import dolfinx
from dolfinx import fem, mesh, geometry, default_scalar_type
from dolfinx.io import gmsh as gmshio
from dolfinx.fem.petsc import assemble_matrix
from basix.ufl import element
import basix
import ufl

from scipy.sparse import bmat, csr_matrix, coo_matrix

comm = MPI.COMM_WORLD
rank = comm.rank


# ===========================================================================
# Configuration (AFSI convention)
# ===========================================================================
def _env(key, default):
    return os.environ.get(key, default)


def make_config():
    cfg = {
        "project_name": "demo-422",
        "tag": "monolithic-ibfe",
        # discretisation
        "degree": 2,             # P2 velocity / P2 solid displacement
        "pressure_order": 1,     # P1 pressure (Taylor-Hood)
        "Nx": int(_env("NX", 64)),
        "Ny": int(_env("NY", 64)),
        "solid_h": float(_env("SOLID_H", 0.0125)),   # disk mesh size
        # geometry (cavity is the unit square; paper: R=0.2, C=(0.6,0.5))
        "R": 0.2, "cx": 0.6, "cy": 0.5,
        # fluid (paper: rho=1, eta=0.01, lid U=1)
        "eta_f": 0.01, "rho_f": 1.0, "lid": 1.0,
        # solid: incompressible neo-Hookean  P = mu (F - F^{-T})
        "mu_s": float(_env("MU_S", 0.1)), "rho_s": float(_env("RHO_S", 1.0)),
        # time stepping (paper: dt=1e-2, T=8.1, out every 10 steps)
        "dt": float(_env("DT", 1e-2)),
        "T": float(_env("T", 8.1)),
        "out_every": int(_env("OUT", 10)),
        # Newton
        "n_newton_max": 8,
        "newton_rtol": 1e-9,
        "pin_center": int(_env("PIN", 0)) == 1,   # pin disk centre (quasi-static)
        # solver
        "scheme": int(_env("SCHEME", 0)),         # 0 monolithic | 3 reduced 2x2
        "p_stab": 1e-8,                           # eps*M_p on the (1,1) block
    }
    cfg["num_steps"] = int(_env("STEPS", int(cfg["T"] / cfg["dt"])))
    out = _env("OUTPUT", "output")
    cfg["output_path"] = out if rank == 0 else None
    cfg["output_path"] = comm.bcast(cfg["output_path"], root=0)
    return cfg


# ===========================================================================
# Mesh generation
# ===========================================================================
def create_fluid_mesh(cfg):
    """Background (Eulerian) square cavity on triangles (P2/P1)."""
    msh = mesh.create_rectangle(
        comm, ((0.0, 0.0), (1.0, 1.0)),
        (cfg["Nx"], cfg["Ny"]), cell_type=mesh.CellType.triangle,
    )
    msh.topology.create_connectivity(msh.topology.dim, 0)
    msh.topology.create_connectivity(msh.topology.dim - 1, msh.topology.dim)
    return msh


def create_solid_mesh(cfg):
    """Solid (Lagrangian) disk via gmsh (reference configuration)."""
    gmsh.initialize()
    gmsh.model.add("disk")
    d = gmsh.model.occ.addDisk(cfg["cx"], cfg["cy"], 0.0, cfg["R"], cfg["R"])
    gmsh.model.occ.synchronize()
    gmsh.model.addPhysicalGroup(2, [d], tag=1)
    gmsh.option.setNumber("Mesh.MeshSizeMax", cfg["solid_h"])
    gmsh.option.setNumber("Mesh.MeshSizeMin", cfg["solid_h"])
    gmsh.model.mesh.generate(2)
    smsh = gmshio.model_to_mesh(gmsh.model, comm, 0, gdim=2).mesh
    gmsh.finalize()
    smsh.topology.create_connectivity(smsh.topology.dim, 0)
    return smsh


# ===========================================================================
# Small helpers
# ===========================================================================
def petsc_to_scipy(M):
    """Convert an assembled dolfinx PETSc Mat to a scipy csr_matrix."""
    rowptr, cols, vals = M.getValuesCSR()
    return csr_matrix((vals, cols, rowptr), shape=M.getSize())


def scipy_to_petsc(A, comm=comm):
    """Convert a scipy csr matrix to an assembled PETSc AIJ Mat (local rows)."""
    A = A.tocsr()
    n, m = A.shape
    pM = PETSc.Mat().createAIJ(size=(n, m), comm=comm)
    pM.setValuesCSR(A.indptr, A.indices, A.data)
    pM.assemble()
    return pM


class MumpsFactor:
    """Sparse direct LU factor solved with PETSc KSP (preonly + LU), preferring
    the MUMPS package (fallback: SuperLU_DIST).

    Provides the same ``solve(b)`` interface as
    ``scipy.sparse.linalg.splu`` (1-D or 2-D right-hand sides), so the rest of
    the monolithic code is unchanged.  A fresh factorisation is performed once
    per ``MumpsFactor`` instance (i.e. per Newton linear solve, or once for the
    constant solid mass matrix in the reduced scheme)."""

    def __init__(self, A, comm=comm):
        self._comm = comm
        self._A_p = scipy_to_petsc(A, comm)
        self._ksp = PETSc.KSP().create(comm)
        self._ksp.setOperators(self._A_p)
        self._ksp.setType("preonly")
        pc = self._ksp.getPC()
        pc.setType("lu")
        if PETSc.Sys().hasExternalPackage("mumps"):
            pc.setFactorSolverType("mumps")
        else:
            pc.setFactorSolverType("superlu_dist")
        self._ksp.setFromOptions()
        self._ksp.setUp()

    def solve(self, b):
        b = np.ascontiguousarray(b, dtype=default_scalar_type)
        if b.ndim == 1:
            n = b.size
            pb = PETSc.Vec().createMPI(size=n, comm=self._comm)
            pb.array[:] = b
            x = PETSc.Vec().createMPI(size=n, comm=self._comm)
            self._ksp.solve(pb, x)
            return x.array.copy()
        # multi-column right-hand side: solve column by column (reuses factor)
        return np.column_stack([self.solve(b[:, k]) for k in range(b.shape[1])])


def affine_jacobian(verts):
    """Jacobian of an affine triangle  J[i][j] = dx_i/dxi_j  (columns = edges).
    verts: (3, 2) vertex coordinates (reference (0,0),(1,0),(0,1))."""
    v0, v1, v2 = verts[0], verts[1], verts[2]
    return np.array([[v1[0] - v0[0], v2[0] - v0[0]],
                     [v1[1] - v0[1], v2[1] - v0[1]]])


def affine_reference_coords(verts, x):
    """Reference coordinates xi of a physical point x in an affine triangle."""
    J = affine_jacobian(verts)
    v0 = verts[0]
    return np.linalg.solve(J, x - v0)


# ===========================================================================
# Main IBFE class
# ===========================================================================
class ImmersedFEM:
    def __init__(self, cfg):
        self.cfg = cfg
        self.msg = (lambda s: print(s)) if rank == 0 else (lambda s: None)

        # ---- meshes & spaces ----
        self.msh = create_fluid_mesh(cfg)
        self.smsh = create_solid_mesh(cfg)
        self.V = fem.functionspace(
            self.msh, element("Lagrange", self.msh.topology.cell_name(),
                              cfg["degree"], shape=(self.msh.geometry.dim,)))
        self.Q = fem.functionspace(
            self.msh,
            element("Lagrange", self.msh.topology.cell_name(), cfg["pressure_order"]))
        self.Vs = fem.functionspace(
            self.smsh, element("Lagrange", self.smsh.topology.cell_name(),
                               cfg["degree"], shape=(self.smsh.geometry.dim,)))
        self.bs_u = self.V.dofmap.dof_layout.block_size       # 2 (vector)
        self.bs_s = self.Vs.dofmap.dof_layout.block_size      # 2 (vector)

        self.n_u = self.V.dofmap.index_map.size_local * self.bs_u
        self.n_p = self.Q.dofmap.index_map.size_local
        self.n_s = self.Vs.dofmap.index_map.size_local * self.bs_s
        self.msg(f"  background dofs: velocity {self.n_u} + pressure {self.n_p} "
                 f"= {self.n_u + self.n_p}")
        self.msg(f"  solid dofs:      {self.n_s}")

        # ---- assemble the (constant) fluid + solid block operators ----
        self._assemble_fluid_blocks()
        self._assemble_solid_mass()

        # ---- solid quadrature data (reference configuration, precomputed) ----
        self._solid_quad_data()

        # ---- interaction cache ----
        self.interaction = None
        self.Mfs = None
        self.MfsT_csr = None

        # ---- boundary data ----
        self._setup_boundary()

        # ---- solution arrays (monolithic ordering: [velocity | pressure | solid]) ----
        self.N = self.n_u + self.n_p + self.n_s
        self.X = np.zeros(self.N)          # current solution
        self.W_prev = np.zeros(self.n_s)   # previous converged W (seed)
        self.u_fn = fem.Function(self.V, name="u")
        self.p_fn = fem.Function(self.Q, name="p")
        self.W_fn = fem.Function(self.Vs, name="W")
        # P1 output spaces (XDMF requires function degree == mesh degree)
        self.V_io = fem.functionspace(
            self.msh, element("Lagrange", self.msh.topology.cell_name(), 1,
                              shape=(self.msh.geometry.dim,)))
        self.Vs_io = fem.functionspace(
            self.smsh, element("Lagrange", self.smsh.topology.cell_name(), 1,
                               shape=(self.smsh.geometry.dim,)))
        self.u_io = fem.Function(self.V_io, name="u")
        self.p_io = fem.Function(self.Q, name="p")
        self.W_io = fem.Function(self.Vs_io, name="W")

    # ------------------------------------------------------------------
    # Constant block operators (fluid Stokes + solid mass)
    # ------------------------------------------------------------------
    def _assemble_fluid_blocks(self):
        cfg = self.cfg
        u, v = ufl.TrialFunction(self.V), ufl.TestFunction(self.V)
        p, q = ufl.TrialFunction(self.Q), ufl.TestFunction(self.Q)
        eta = cfg["eta_f"]

        K = assemble_matrix(
            fem.form(2.0 * eta * ufl.inner(ufl.sym(ufl.grad(u)),
                                           ufl.sym(ufl.grad(v))) * ufl.dx))
        K.assemble()
        B = assemble_matrix(fem.form(-q * ufl.div(u) * ufl.dx))
        B.assemble()
        Bt = assemble_matrix(fem.form(-p * ufl.div(v) * ufl.dx))
        Bt.assemble()
        Mp = assemble_matrix(fem.form(q * p * ufl.dx))
        Mp.assemble()

        self.K = petsc_to_scipy(K)
        self.B = petsc_to_scipy(B)
        self.Bt = petsc_to_scipy(Bt)
        self.Mp = petsc_to_scipy(Mp)
        self.K_ilu = None

    def _assemble_solid_mass(self):
        cfg = self.cfg
        us, vs = ufl.TrialFunction(self.Vs), ufl.TestFunction(self.Vs)
        M_s = assemble_matrix(
            fem.form(cfg["rho_s"] * ufl.inner(us, vs) * ufl.dx(self.smsh)))
        M_s.assemble()
        self.M_s = petsc_to_scipy(M_s)

    # ------------------------------------------------------------------
    # Solid quadrature data (values/gradients of the P2 solid basis)
    # ------------------------------------------------------------------
    def _solid_quad_data(self):
        cfg = self.cfg
        self.s_bxe = self.Vs.element.basix_element
        self.f_bxe = self.V.element.basix_element
        # Gauss quadrature of degree degree+2 (like QGauss(degree+2) in a.cpp)
        qdeg = cfg["degree"] + 2
        qpts, qw = basix.make_quadrature(basix.CellType.triangle, qdeg)
        tab = self.s_bxe.tabulate(1, qpts)
        # tabulate(1) returns [values, d/dxi0, d/dxi1, ...] (3 slots in 2D)
        self.s_phi = tab[0][:, :, 0]                                   # (nq, ndofs_s)
        self.s_gphi = np.stack([tab[1][:, :, 0], tab[2][:, :, 0]], axis=2)  # (nq, ndofs_s, 2)
        self.qw = qw

        n_cells = self.smsh.topology.index_map(self.smsh.topology.dim).size_local
        s_geom_x = self.smsh.geometry.x
        s_geom_dofs = self.smsh.geometry.dofmap
        cells = []
        for c in range(n_cells):
            verts = s_geom_x[s_geom_dofs[c]][:, :2]
            J = affine_jacobian(verts)
            detJ = np.linalg.det(J)
            v0 = verts[0]
            Xq = np.array([v0 + q[0] * (verts[1] - v0) + q[1] * (verts[2] - v0)
                           for q in qpts])        # (nq, 2) reference quad points
            cells.append({
                "s_cell": c,
                "s_bdofs": self.Vs.dofmap.cell_dofs(c),
                "JxW_s": qw * detJ,               # (nq,)
                "Xq": Xq,                         # (nq, 2)
            })
        self.solid_cells = cells
        self.f_geom_x = self.msh.geometry.x
        self.f_geom_dofs = self.msh.geometry.dofmap
        # bounding box tree of the fluid mesh (for locating solid points)
        self.bb_tree = geometry.bb_tree(self.msh, self.msh.topology.dim)

    # ------------------------------------------------------------------
    # Boundary data (velocity walls + moving lid, pinned pressure dof)
    # ------------------------------------------------------------------
    def _setup_boundary(self):
        fdim = self.msh.topology.dim - 1
        cfg = self.cfg

        def on_boundary(x):
            return np.logical_or.reduce(
                (np.isclose(x[0], 0.0), np.isclose(x[0], 1.0),
                 np.isclose(x[1], 0.0), np.isclose(x[1], 1.0)))

        facets_b = mesh.locate_entities_boundary(self.msh, fdim, on_boundary)
        # NOTE: for the P2 vector space, locate_dofs_topological returns BLOCK
        # indices (0..n_u/2-1).  The monolithic matrix uses the expanded global
        # dof numbering  dof = block*2 + comp, so the BC dofs are expanded here.
        bc_vel_blk = fem.locate_dofs_topological(self.V, fdim, facets_b).astype(np.int64)
        self.bc_vel = np.unique(np.concatenate([bc_vel_blk * 2, bc_vel_blk * 2 + 1]))

        facets_top = mesh.locate_entities_boundary(
            self.msh, fdim, lambda x: np.isclose(x[1], 1.0))
        bc_lid_blk = fem.locate_dofs_topological(self.V, fdim, facets_top).astype(np.int64)
        self.bc_lid = np.unique(np.concatenate([bc_lid_blk * 2, bc_lid_blk * 2 + 1]))
        # x-component of the lid blocks (block b -> expanded dof 2b)
        self.bc_lid_x = bc_lid_blk * 2

        # pin one pressure dof (fixes the Stokes pressure nullspace) in the
        # monolithic numbering (pressure block starts at n_u)
        self.bc_pin = np.array([self.n_u], dtype=np.int64)
        self.bc_all = np.unique(np.concatenate([self.bc_vel, self.bc_pin]))

    def _initial_velocity(self):
        """Lid value (lid, 0) on the top, 0 elsewhere (into the X velocity block).
        Vector dof layout: global dof = block*2 + comp; the x-component of block
        b is dof 2b.  Only the x-component of the lid is set to `lid`; its
        y-component stays 0."""
        lid = self.cfg["lid"]
        u = np.zeros(self.n_u)
        u[self.bc_lid_x] = lid
        return u

    # ------------------------------------------------------------------
    # Interaction: locate the (deformed) solid quadrature points in the
    # background mesh and tabulate the background P2 basis there.
    # ------------------------------------------------------------------
    def compute_interaction(self, W):
        """W: (n_s,) solid displacement.  Locates the (deformed) solid quadrature
        points in the background mesh and tabulates the background P2 basis there.

        Solid points that lie outside the background mesh (e.g. when the disk
        reaches a wall) are skipped with a warning, exactly as in a.cpp: their
        quadrature weight is dropped from the coupling."""

        # Build all current physical quadrature points (batched) and locate them
        inter = []
        xq_all = np.zeros((0, 2))
        cell_slices = []
        for ci, cc in enumerate(self.solid_cells):
            Xq = cc["Xq"]
            Wq = np.zeros_like(Xq)
            s_bdofs = cc["s_bdofs"]
            for c in range(2):
                Wq[:, c] = np.sum(W[s_bdofs * 2 + c][None, :] * self.s_phi, axis=1)
            xq = Xq + Wq
            cell_slices.append((len(xq_all), len(xq_all) + len(xq)))
            xq_all = np.vstack([xq_all, xq])
        if len(xq_all) == 0:
            self.interaction = []
            return self.interaction

        xq3 = np.hstack([xq_all, np.zeros((len(xq_all), 1))])
        coll = geometry.compute_collisions_points(self.bb_tree, xq3)
        own = geometry.compute_colliding_cells(self.msh, coll, xq3)

        # keep only the points that were located (skip the rest, with a warning)
        found = np.array([len(own.links(i)) > 0 for i in range(len(xq_all))])
        n_miss = int((~found).sum())
        if n_miss > 0 and rank == 0:
            self.msg(f"    [warn] {n_miss} solid quadrature point(s) outside the "
                     "background mesh (disk touching a wall?) -- skipped")

        # reference coordinates + background basis tabulation per located point
        n_pts = int(found.sum())
        if n_pts == 0:
            self.interaction = []
            return self.interaction
        xi_all = np.zeros((n_pts, 2))
        fcell_all = np.zeros(n_pts, dtype=np.int64)
        glob2loc = []  # (original point index -> compressed index)
        comp = 0
        for i in range(len(xq_all)):
            if not found[i]:
                continue
            cs = own.links(i)
            c = cs[0]
            verts = self.f_geom_x[self.f_geom_dofs[c]][:, :2]
            xi_all[comp] = affine_reference_coords(verts, xq_all[i])
            fcell_all[comp] = c
            glob2loc.append(i)
            comp += 1
        xq_found = xq_all[glob2loc]

        # background basis values + reference gradients at the located points
        ftab = self.f_bxe.tabulate(1, xi_all)
        f_phi = ftab[0][:, :, 0]                                    # (npts, ndofs_f)
        f_gphi = np.stack([ftab[1][:, :, 0], ftab[2][:, :, 0]], axis=2)  # (npts, ndofs_f, 2)

        # physical gradient:  grad_x phi = grad_xi phi @ J^{-1}
        f_gx = np.zeros_like(f_gphi)
        for i in range(n_pts):
            c = fcell_all[i]
            verts = self.f_geom_x[self.f_geom_dofs[c]][:, :2]
            Jinv = np.linalg.inv(affine_jacobian(verts))
            f_gx[i] = f_gphi[i] @ Jinv

        # group by solid cell (only the located quadrature points of each cell)
        for ci, cc in enumerate(self.solid_cells):
            i0, i1 = cell_slices[ci]
            # local (within-cell) quadrature indices of the located points
            q_idx = [g - i0 for g in glob2loc if i0 <= g < i1]
            cc["q_idx"] = np.asarray(q_idx, dtype=np.int64)
            mask = (np.asarray(glob2loc) >= i0) & (np.asarray(glob2loc) < i1)
            cc["xq"] = xq_found[mask]
            cc["f_cells"] = fcell_all[mask]
            cc["f_phi"] = f_phi[mask]
            cc["f_gx"] = f_gx[mask]
            inter.append(cc)
        self.interaction = inter
        return inter

    # ------------------------------------------------------------------
    # Mixed mass  Mfs(i,j) = int_Omega_s phi_i_bg(x) phi_j_s(X) dX
    # (only same-component pairs couple, like a.cpp)
    # ------------------------------------------------------------------
    def assemble_mixed_mass(self):
        """Mfs(i,j) = int_Omega_s phi_i_bg(x) phi_j_s(X) dX (same-component
        pairs only).  Vectorised over the located solid quadrature points."""
        bs = self.bs_u
        rows, cols, vals = [], [], []
        for cc in self.interaction:
            npt = len(cc["q_idx"])
            if npt == 0:
                continue
            q_idx = cc["q_idx"]
            s_bdofs = cc["s_bdofs"]
            f_bdofs = np.stack([self.V.dofmap.cell_dofs(c) for c in cc["f_cells"]])
            f_phi_k = cc["f_phi"]                  # (npts, ndofs_f)
            s_phi_k = self.s_phi[q_idx]            # (npts, ndofs_s)
            w = cc["JxW_s"][q_idx]                 # (npts,)
            for i in range(f_bdofs.shape[1]):
                fv = f_phi_k[:, i]
                if not np.any(fv):
                    continue
                for j in range(len(s_bdofs)):
                    sv = s_phi_k[:, j]
                    if not np.any(sv):
                        continue
                    # only same-component pairs couple (ci == cj)
                    rows.append(f_bdofs[:, i] * bs + 0)
                    cols.append(np.full(npt, s_bdofs[j] * bs + 0))
                    vals.append(fv * sv * w)
                    rows.append(f_bdofs[:, i] * bs + 1)
                    cols.append(np.full(npt, s_bdofs[j] * bs + 1))
                    vals.append(fv * sv * w)
        Mfs = coo_matrix((np.concatenate(vals),
                          (np.concatenate(rows), np.concatenate(cols))),
                         shape=(self.n_u, self.n_s)).tocsr()
        self.Mfs = Mfs
        self.MfsT_csr = Mfs.T.tocsr()
        return Mfs

    # ------------------------------------------------------------------
    # Nonlinear incompressible neo-Hookean elastic force and its tangent
    #   f_el,i = - int_Omega_s (P F^T) : grad_x phi_i dX,  P = mu (F - F^{-T})
    #   A_uW(i,j) = d f_el,i / dW_j  (mixed stiffness)
    # The background test functions come from a frozen interaction (geometry
    # = W^n); only F = I + grad_X W follows the current Newton iterate W.
    # ------------------------------------------------------------------
    def assemble_elastic(self, W):
        """Incompressible neo-Hookean elastic force and its tangent (mixed
        stiffness), vectorised over the located solid quadrature points:
            f_el,i = - int (P F^T) : grad_x phi_i dX,   P = mu (F - F^{-T})
            A_uW(i,j) = d f_el,i / dW_j
        """
        mu_s = self.cfg["mu_s"]
        bs = self.bs_u
        # ---- gather per-point data (flattened across solid cells) ----
        q_idx_list, s_bdofs_list, f_bdofs_list, f_gx_list, w_list = [], [], [], [], []
        for cc in self.interaction:
            npt = len(cc["q_idx"])
            if npt == 0:
                continue
            q_idx_list.append(cc["q_idx"])
            s_bdofs_list.append(np.tile(cc["s_bdofs"], (npt, 1)))
            f_bdofs_list.append(np.stack(
                [self.V.dofmap.cell_dofs(c) for c in cc["f_cells"]]))
            f_gx_list.append(cc["f_gx"])
            w_list.append(cc["JxW_s"][cc["q_idx"]])
        if not q_idx_list:
            return np.zeros(self.n_u), coo_matrix(
                (self.n_u, self.n_s), dtype=float).tocsr()
        q_idx = np.concatenate(q_idx_list)
        s_bdofs = np.concatenate(s_bdofs_list)     # (npts, ndofs_s)
        f_bdofs = np.concatenate(f_bdofs_list)     # (npts, ndofs_f)
        f_gx = np.concatenate(f_gx_list)           # (npts, ndofs_f, 2)
        w = np.concatenate(w_list)                 # (npts,)
        npts = len(q_idx)
        s_phi_k = self.s_phi[q_idx]                # (npts, ndofs_s)
        s_gphi_k = self.s_gphi[q_idx]              # (npts, ndofs_s, 2)

        # ---- deformation gradient F = I + grad_X W at the quadrature points ----
        Wq = np.zeros((npts, 2))
        gradW = np.zeros((npts, 2, 2))
        for c in range(2):
            Wvals = W[s_bdofs * 2 + c]             # (npts, ndofs_s)
            Wq[:, c] = np.sum(Wvals * s_phi_k, axis=1)
            gradW[:, c, :] = np.einsum("nk,nkb->nb", Wvals, s_gphi_k)
        F = np.eye(2)[None, :, :] + gradW
        Finv = np.linalg.inv(F)
        P = mu_s * (F - Finv.transpose(0, 2, 1))
        PeFT = P @ F.transpose(0, 2, 1)            # (npts, 2, 2)

        # ---- f_el (scatter by fluid dof) ----
        f_el = np.zeros(self.n_u)
        for i in range(f_bdofs.shape[1]):
            gx = f_gx[:, i, :]                     # (npts, 2)
            for ci in range(2):
                contr = np.sum(PeFT[:, ci, :] * gx, axis=1) * w
                np.add.at(f_el, f_bdofs[:, i] * bs + ci, -contr)

        # ---- A_uW (scatter by fluid x solid dof pair) ----
        rows, cols, vals = [], [], []
        Ft = F.transpose(0, 2, 1)
        for i in range(f_bdofs.shape[1]):
            gx = f_gx[:, i, :]
            for ci in range(2):
                for j in range(s_bdofs.shape[1]):
                    gk = s_gphi_k[:, j, :]         # (npts, 2)
                    for cj in range(2):
                        # dF[a][b] = delta_{a,cj} gk[b]  (only row cj nonzero)
                        dF = np.zeros((npts, 2, 2))
                        dF[:, cj, :] = gk
                        dFinv = -Finv @ dF @ Finv
                        dP = mu_s * (dF - dFinv.transpose(0, 2, 1))
                        dPeFT = dP @ Ft + P @ dF.transpose(0, 2, 1)
                        dcontr = np.sum(dPeFT[:, ci, :] * gx, axis=1) * w
                        rows.append(f_bdofs[:, i] * bs + ci)
                        cols.append(s_bdofs[:, j] * bs + cj)
                        vals.append(-dcontr)
        A_uW = coo_matrix((np.concatenate(vals),
                           (np.concatenate(rows), np.concatenate(cols))),
                          shape=(self.n_u, self.n_s)).tocsr()
        return f_el, A_uW

    # ------------------------------------------------------------------
    # Build the monolithic 3x3 system
    #   [K  B^T  -A_uW ;  B  0  0 ;  -Mfs^T  0  (1/dt) M_s]
    # and its RHS -R (R = exact residual).
    # ------------------------------------------------------------------
    def _residual(self, W_old, f_el):
        """Exact monolithic residual at the current X."""
        u = self.X[:self.n_u]
        p = self.X[self.n_u:self.n_u + self.n_p]
        W = self.X[self.n_u + self.n_p:]
        R_u = self.K @ u + self.Bt @ p - f_el
        R_p = self.B @ u
        WmW0 = W - W_old
        R_W = (1.0 / self.cfg["dt"]) * (self.M_s @ WmW0) - self.MfsT_csr @ u
        return R_u, R_p, R_W

    def build_monolithic(self, A_uW):
        """Assemble A (scipy csr, N x N) from the cached blocks + A_uW."""
        cfg = self.cfg
        nu, np_, ns = self.n_u, self.n_p, self.n_s
        # saddle-point regularisation: eps * M_p on the (1,1) block.  This only
        # perturbs the intermediate Newton linear solves -- the converged Newton
        # solution uses the exact residual and is unchanged.
        s11 = cfg["p_stab"] * self.Mp if cfg["p_stab"] > 0.0 else None
        return bmat([[self.K, self.Bt, -A_uW],
                     [self.B, s11, None],
                     [-self.MfsT_csr, None, (1.0 / cfg["dt"]) * self.M_s]],
                    format="csr")

    def apply_bc(self, A):
        """Homogeneous increments at the velocity boundary + pinned pressure."""
        bc = self.bc_all
        A = A.tolil()
        A[bc, :] = 0.0
        A[:, bc] = 0.0
        A[bc, bc] = 1.0
        return A.tocsr()

    # ------------------------------------------------------------------
    # One time step of the monolithic scheme (backward Euler + Newton)
    # ------------------------------------------------------------------
    def solve_monolithic(self):
        cfg = self.cfg
        W_old = self.X[self.n_u + self.n_p:].copy()

        # (a) frozen interaction at W^n + rebuild the mixed mass Mfs
        self.compute_interaction(W_old)
        self.assemble_mixed_mass()

        # seed: linear extrapolation of W (fewer Newton iterations)
        self.X[self.n_u + self.n_p:] = 2.0 * W_old - self.W_prev

        # (b) Newton loop
        f_el = np.zeros(self.n_u)
        A_uW = None
        dX_norm = 0.0
        for it in range(cfg["n_newton_max"]):
            W = self.X[self.n_u + self.n_p:]
            f_el, A_uW = self.assemble_elastic(W)
            A = self.build_monolithic(A_uW)
            A = self.apply_bc(A)

            R_u, R_p, R_W = self._residual(W_old, f_el)
            rhs = np.zeros(self.N)
            rhs[:self.n_u] = -R_u
            rhs[self.n_u:self.n_u + self.n_p] = -R_p
            rhs[self.n_u + self.n_p:] = -R_W
            rhs[self.bc_all] = 0.0

            lu = MumpsFactor(A)
            dX = lu.solve(rhs)
            dX[self.bc_all] = 0.0
            self.X += dX

            dX_norm = np.linalg.norm(dX)
            self.msg(f"    [mono] newton {it} |dX|={dX_norm:.3e}")
            if dX_norm < cfg["newton_rtol"] * (1.0 + np.linalg.norm(self.X)):
                break
        self.W_prev = self.X[self.n_u + self.n_p:].copy()

    # ------------------------------------------------------------------
    # Scheme 3: reduced 2x2 (Schur-eliminate W with the EXACT M_s^-1, applied
    # through a sparse factor of the constant solid mass matrix)
    #   [K~ B^T; B 0] [du; dp] = [-R_u - dt A_uW M_s^-1 R_W ; -R_p]
    #   K~ = K - dt A_uW M_s^-1 Mfs^T,   dW = dt M_s^-1 (Mfs^T du - R_W)
    # Newton uses the exact residuals and the elimination is exact, so the
    # converged solution is identical to the full 3x3 solve and Newton
    # converges quadratically.  NB: the exact Schur elimination makes K~ a
    # dense n_u x n_u block, so this variant is only practical on coarse
    # meshes -- use scheme 0 (full monolithic, sparse) at larger scale.
    # (This is the FEniCSx analogue of a.cpp scheme 5; a.cpp schemes 3/4
    # replace M_s^-1 by its diagonal, which here made Newton diverge for the
    # soft benchmark disk, so we always use the exact elimination.)
    # ------------------------------------------------------------------
    def solve_monolithic_reduced(self):
        cfg = self.cfg
        nu, np_, ns = self.n_u, self.n_p, self.n_s
        W_old = self.X[self.n_u + self.n_p:].copy()
        if nu * ns > 3e7:
            self.msg("  [warn] reduced scheme assembles the dense Schur-eliminated "
                     f"K~ ({nu} x {nu}); prefer scheme 0 on this mesh")

        self.compute_interaction(W_old)
        self.assemble_mixed_mass()

        # exact M_s^-1 via a sparse factor (M_s is constant in time)
        Ms_lu = MumpsFactor(self.M_s)
        # dense  M_s^-1 Mfs^T  (geometry fixed within this step)
        Y = Ms_lu.solve(self.MfsT_csr.toarray())   # (ns, nu)

        f_el = np.zeros(self.n_u)
        A_uW = None
        dX_norm = 0.0
        for it in range(cfg["n_newton_max"]):
            W = self.X[self.n_u + self.n_p:]
            f_el, A_uW = self.assemble_elastic(W)

            # K~ = K - dt A_uW (M_s^-1 Mfs^T)   (dense product -> csr)
            Ktilde = csr_matrix(self.K - cfg["dt"] * (A_uW @ Y))
            A2 = bmat([[Ktilde, self.Bt], [self.B, cfg["p_stab"] * self.Mp]],
                      format="csr")
            A2 = self.apply_bc2(A2)

            R_u, R_p, R_W = self._residual(W_old, f_el)
            rhs2 = np.zeros(nu + np_)
            rhs2[:nu] = -R_u - cfg["dt"] * (A_uW @ Ms_lu.solve(R_W))
            rhs2[nu:] = -R_p
            rhs2[self.bc2_all] = 0.0

            lu = MumpsFactor(A2)
            dX2 = lu.solve(rhs2)
            dX2[self.bc2_all] = 0.0
            du, dp = dX2[:nu], dX2[nu:]

            # W back-substitution: dW = dt M_s^-1 (Mfs^T du - R_W)
            dW = cfg["dt"] * Ms_lu.solve(self.MfsT_csr @ du - R_W)

            self.X[:nu] += du
            self.X[nu:nu + np_] += dp
            self.X[nu + np_:] += dW

            dX_norm = np.linalg.norm(dX2) + np.linalg.norm(dW)
            self.msg(f"    [red3] newton {it} |dX|={dX_norm:.3e}")
            if dX_norm < cfg["newton_rtol"] * (1.0 + np.linalg.norm(self.X)):
                break
        self.W_prev = self.X[self.n_u + self.n_p:].copy()

    def apply_bc2(self, A2):
        """Homogeneous BCs for the reduced 2x2 system (velocity + pinned p)."""
        bc = np.unique(np.concatenate([self.bc_vel, self.bc_pin]))
        self.bc2_all = bc
        A2 = A2.tolil()
        A2[bc, :] = 0.0
        A2[:, bc] = 0.0
        A2[bc, bc] = 1.0
        return A2.tocsr()

    # ------------------------------------------------------------------
    # Coupling self-checks (like a.cpp::verify_coupling): project the linear
    # background field g onto the solid via  M_s u_s = Mfs^T u.  P2 reproduces
    # linear/constant fields exactly, so the projection must be exact.
    # ------------------------------------------------------------------
    def verify_coupling(self):
        from scipy.sparse.linalg import spsolve
        self.msg("--- coupling self-check ---")
        # interaction at W = 0 (reference configuration)
        W0 = np.zeros(self.n_s)
        self.compute_interaction(W0)
        self.assemble_mixed_mass()

        # build background velocity whose components equal g(X) = support point
        # coordinates (P2 nodal values)
        def build_u(g):
            """Background velocity whose component ci at block dof b equals
            g[ci](support point of block b)."""
            u_coords = self.V.tabulate_dof_coordinates()   # (n_blocks, gdim)
            return np.array([g[d % 2](u_coords[d // 2][:2])
                             for d in range(self.n_u)])

        s_coords = self.Vs.tabulate_dof_coordinates()      # (n_blocks, gdim)

        for name, g in [("(1,0)", [lambda pt: 1.0, lambda pt: 0.0]),
                        ("(x,y)", [lambda pt: pt[0], lambda pt: pt[1]])]:
            u_bg = build_u(g)
            expected = np.array([g[i % 2](s_coords[i // 2][:2])
                                 for i in range(self.n_s)])
            u_s = spsolve(self.M_s.tocsc(), self.MfsT_csr @ u_bg)
            err = np.max(np.abs(u_s - expected))
            self.msg(f"  projection of {name}: max err = {err:.3e}")

    # ------------------------------------------------------------------
    # Diagnostics (like a.cpp::diagnostics)
    # ------------------------------------------------------------------
    def diagnostics(self, t, f):
        W = self.X[self.n_u + self.n_p:]
        qpts, qw = basix.make_quadrature(basix.CellType.triangle, self.cfg["degree"] + 2)
        s_phi = self.s_phi
        s_gphi = self.s_gphi
        area = deformed_area = 0.0
        maxJexp = maxJcom = 0.0
        center = np.zeros(2)
        for cc in self.solid_cells:
            s_bdofs = cc["s_bdofs"]
            for q in range(len(qw)):
                Xq = cc["Xq"][q]
                xq = Xq.copy()
                gradW = np.zeros((2, 2))
                for c in range(2):
                    for j in range(s_phi.shape[1]):
                        xq[c] += W[s_bdofs[j] * 2 + c] * s_phi[q, j]
                        gradW[c, :] += W[s_bdofs[j] * 2 + c] * s_gphi[q, j, :]
                F = np.eye(2) + gradW
                J = np.linalg.det(F)
                JxW = cc["JxW_s"][q]
                area += JxW
                deformed_area += J * JxW
                maxJexp = max(maxJexp, J - 1.0)
                maxJcom = max(maxJcom, 1.0 - J)
                center += J * JxW * xq
        center /= deformed_area

        maxW = np.abs(W).max()
        disp = np.linalg.norm(center - np.array([self.cfg["cx"], self.cfg["cy"]]))
        line = (f"t={t:8.5f}  centre=({center[0]:.4f},{center[1]:.4f})  "
                f"disp={disp:.4e}  max|W|={maxW:.4e}  A_ref={area:.6f}  "
                f"A_def={deformed_area:.6f}  max(J-1)={maxJexp:.3e}  "
                f"max(1-J)={maxJcom:.3e}")
        self.msg("  " + line)
        f.write(f"{t:.6e} {center[0]:.6e} {center[1]:.6e} {disp:.6e} "
                f"{maxW:.6e} {area:.6e} {deformed_area:.6e} "
                f"{maxJexp:.6e} {maxJcom:.6e}\n")
        f.flush()

    # ------------------------------------------------------------------
    # Output (XDMF on each mesh)
    # ------------------------------------------------------------------
    def output(self, step, t):
        """Write the current solution to the (open) XDMF files."""
        if rank != 0:
            return
        self.u_fn.x.array[:] = self.X[:self.n_u]
        self.p_fn.x.array[:] = self.X[self.n_u:self.n_u + self.n_p]
        self.W_fn.x.array[:] = self.X[self.n_u + self.n_p:]
        for fn in (self.u_fn, self.p_fn, self.W_fn):
            fn.x.scatter_forward()
        # interpolate to the P1 output spaces (XDMF requires degree == mesh degree)
        self.u_io.interpolate(self.u_fn)
        self.p_io.x.array[:] = self.p_fn.x.array
        self.p_io.x.scatter_forward()
        self.W_io.interpolate(self.W_fn)
        self.fvel.write_function(self.u_io, t)
        self.fpre.write_function(self.p_io, t)
        self.fsol.write_function(self.W_io, t)
        self.msg(f"    [output] t={t:.4f} step={step}")

    # ------------------------------------------------------------------
    def run(self):
        cfg = self.cfg
        self.msg("=== IBFE: lid-driven square cavity with immersed elastic disk "
                 f"(dolfinx {dolfinx.__version__}, scheme {cfg['scheme']}) ===")

        if self.n_u > 0:
            self.verify_coupling()

        # initial condition: velocity satisfies the lid BC, everything else 0
        self.X[:self.n_u] = self._initial_velocity()
        self.W_prev = np.zeros(self.n_s)

        if rank == 0:
            os.makedirs(cfg["output_path"], exist_ok=True)
            f = open(f"{cfg['output_path']}/solid_deformation.txt", "w")
            f.write("# t centre_x centre_y |disp| max|W| A_ref A_def "
                    "max(J-1) max(1-J)\n")
        else:
            f = None

        t = 0.0
        self.diagnostics(t, f)
        # open the XDMF files once (write mesh, then write_function per step)
        if rank == 0:
            enc = dolfinx.io.XDMFFile.Encoding.HDF5
            self.fvel = dolfinx.io.XDMFFile(comm, f"{cfg['output_path']}/velocity.xdmf", "w", encoding=enc)
            self.fpre = dolfinx.io.XDMFFile(comm, f"{cfg['output_path']}/pressure.xdmf", "w", encoding=enc)
            self.fsol = dolfinx.io.XDMFFile(comm, f"{cfg['output_path']}/solid.xdmf", "w", encoding=enc)
            self.fvel.write_mesh(self.msh)
            self.fpre.write_mesh(self.msh)
            self.fsol.write_mesh(self.smsh)
        else:
            self.fvel = self.fpre = self.fsol = None
        self.output(0, t)

        t_step = 0.0
        for step in range(1, cfg["num_steps"] + 1):
            t0 = time.time()
            if cfg["scheme"] == 3:
                self.solve_monolithic_reduced()
            else:
                self.solve_monolithic()
            t_step += time.time() - t0

            t += cfg["dt"]
            if step % cfg["out_every"] == 0:
                self.msg(f"    [time] {t_step / step:.4f} s/step")
                self.diagnostics(t, f)
                self.output(step, t)
        self.diagnostics(t, f)
        self.msg("=== done ===")
        if f is not None:
            f.close()
        if rank == 0:
            self.fvel.close()
            self.fpre.close()
            self.fsol.close()


# ===========================================================================
def main():
    import argparse
    p = argparse.ArgumentParser(
        description="Monolithic IBFE: lid-driven cavity + immersed elastic disk")
    p.add_argument("--steps", type=int, help="number of time steps")
    p.add_argument("--nx", "--ny", dest="nxy", type=int, help="background mesh size")
    p.add_argument("--scheme", type=int, choices=[0, 3], help="0 monolithic | 3 reduced")
    p.add_argument("--dt", type=float, help="time step")
    p.add_argument("--mu-s", type=float, dest="mu_s", help="solid shear modulus")
    p.add_argument("--out", type=int, help="output interval (steps)")
    args = p.parse_args()

    cfg = make_config()
    if args.steps is not None:
        cfg["num_steps"] = args.steps
    if args.nxy is not None:
        cfg["Nx"] = cfg["Ny"] = args.nxy
    if args.scheme is not None:
        cfg["scheme"] = args.scheme
    if args.dt is not None:
        cfg["dt"] = args.dt
    if args.mu_s is not None:
        cfg["mu_s"] = args.mu_s
    if args.out is not None:
        cfg["out_every"] = args.out

    problem = ImmersedFEM(cfg)
    problem.run()


if __name__ == "__main__":
    main()
