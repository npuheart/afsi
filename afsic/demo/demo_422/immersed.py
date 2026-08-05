"""Monolithic Immersed Boundary Finite Element (IBFE) solver.

Lid-driven square cavity with an immersed elastic disk -- MONOLITHIC variant
(FEniCSx/dolfinx 0.10 re-implementation of ``afsi/a.cpp``, benchmark
``LDCFlow_Ball_DGP_INH1`` of Roy-Heltai-Costanzo 2015).  Fluid velocity u,
pressure p and solid displacement W are solved together in one implicit 3x3
block system each backward-Euler step, by (quasi-)Newton.  See readme.md.

Run:
  python main.py                     # full benchmark (64x64, 810 steps)
  python main.py --steps 10 --nx 16  # quick smoke test
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

from config import comm, rank, make_config
from meshgen import create_fluid_mesh, create_solid_mesh
from linops import (petsc_to_scipy, scipy_to_petsc, MumpsFactor,
                    affine_jacobian, affine_reference_coords)

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
        self._gather_cache = None   # flattened elastic gather (per interaction)
        # solver caches
        self._lu_cache = None       # frozen monolithic LU factor (direct mode)
        self._A_cache = None        # frozen monolithic Jacobian (gmres mode)
        self._fluid_pre_lu = None   # constant fluid Stokes factor (preconditioner)
        self._Ms_pre_lu = None      # constant solid mass factor (preconditioner)
        self._gmres_P = None        # cached block preconditioner LinearOperator

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
        # precompute the affine Jacobian inverse + v0 for ALL background cells
        # (constant; lets compute_interaction/reference coords be vectorised)
        n_bg = self.msh.topology.index_map(self.msh.topology.dim).size_local
        fverts = self.f_geom_x[self.f_geom_dofs[:n_bg]][:, :, :2]  # (n_bg, 3, 2)
        self.f_v0 = fverts[:, 0].copy()                              # (n_bg, 2)
        e1 = fverts[:, 1] - self.f_v0
        e2 = fverts[:, 2] - self.f_v0
        fJ = np.stack([e1, e2], axis=-1)                             # (n_bg, 2, 2) cols = edges
        self.f_Jinv = np.linalg.inv(fJ)
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
        quadrature weight is dropped from the coupling.

        Vectorised: all solid cells share the same quadrature points, so the
        deformed points, the reference-coordinate mapping and the physical
        gradients are computed in one batched pass (precomputed per-cell
        affine Jacobian inverses)."""

        self._gather_cache = None   # invalidate the cached elastic gather
        nc = len(self.solid_cells)
        if nc == 0:
            self.interaction = []
            return self.interaction
        nq = len(self.qw)
        s_bdofs_all = np.stack([cc["s_bdofs"] for cc in self.solid_cells])  # (nc, ndofs_s)
        Xq_all = np.stack([cc["Xq"] for cc in self.solid_cells])            # (nc, nq, 2)

        # deformed quadrature points  xq = Xq + W(Xq), batched
        W_cell = np.stack([W[s_bdofs_all * 2 + c] for c in range(2)], axis=2)  # (nc, ndofs_s, 2)
        Wq = np.einsum("nkc,qk->nqc", W_cell, self.s_phi)                   # (nc, nq, 2)
        xq_all = (Xq_all + Wq).reshape(nc * nq, 2)

        xq3 = np.hstack([xq_all, np.zeros((len(xq_all), 1))])
        coll = geometry.compute_collisions_points(self.bb_tree, xq3)
        own = geometry.compute_colliding_cells(self.msh, coll, xq3)

        # per located point keep the first colliding (owned) background cell
        n_pts = 0
        fcell_list = []
        idx_map = np.full(nc * nq, -1, dtype=np.int64)   # global pt -> compressed idx
        for i in range(nc * nq):
            cs = own.links(i)
            if len(cs) == 0:
                continue
            fcell_list.append(cs[0])
            idx_map[i] = n_pts
            n_pts += 1
        if n_pts == 0:
            self.interaction = []
            return self.interaction
        n_miss = nc * nq - n_pts
        if n_miss > 0 and rank == 0:
            self.msg(f"    [warn] {n_miss} solid quadrature point(s) outside the "
                     "background mesh (disk touching a wall?) -- skipped")
        fcell_all = np.asarray(fcell_list, dtype=np.int64)                  # (n_pts,)
        sel = np.flatnonzero(idx_map >= 0)                                  # global pt indices found
        xq_found = xq_all[sel]
        fcell_found = fcell_all

        # reference coordinates, vectorised with the precomputed cell Jacobians
        xi_all = np.einsum("pij,pj->pi", self.f_Jinv[fcell_found],
                           xq_found - self.f_v0[fcell_found])              # (n_pts, 2)

        # background basis values + reference gradients at the located points
        ftab = self.f_bxe.tabulate(1, xi_all)
        f_phi = ftab[0][:, :, 0]                                           # (n_pts, ndofs_f)
        f_gphi = np.stack([ftab[1][:, :, 0], ftab[2][:, :, 0]], axis=2)    # (n_pts, ndofs_f, 2)
        # physical gradient:  grad_x phi = grad_xi phi @ J^{-1}
        f_gx = np.einsum("pij,pjk->pik", f_gphi, self.f_Jinv[fcell_found])

        # group by solid cell (uniform nq points per cell)
        inter = []
        for ci, cc in enumerate(self.solid_cells):
            i0, i1 = ci * nq, (ci + 1) * nq
            cm = idx_map[i0:i1]                                            # compressed idx per local pt (-1 = missed)
            m = cm >= 0
            cc["q_idx"] = np.flatnonzero(m).astype(np.int64)               # local quad idx found
            cc["xq"] = xq_all[i0:i1][m]
            cc["f_cells"] = fcell_found[cm[m]]
            cc["f_phi"] = f_phi[cm[m]]
            cc["f_gx"] = f_gx[cm[m]]
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
        rows0, cols0, vals = [], [], []
        for cc in self.interaction:
            npt = len(cc["q_idx"])
            if npt == 0:
                continue
            q_idx = cc["q_idx"]
            s_bdofs = cc["s_bdofs"]
            f_bdofs = np.stack([self.V.dofmap.cell_dofs(c) for c in cc["f_cells"]])
            f_phi_k = cc["f_phi"]                  # (npt, ndofs_f)
            s_phi_k = self.s_phi[q_idx]            # (npt, ndofs_s)
            w = cc["JxW_s"][q_idx]                 # (npt,)
            # per-point weight matrix  V[p,i,j] = f_phi[p,i] * s_phi[p,j] * w[p]
            V = f_phi_k[:, :, None] * s_phi_k[:, None, :] * w[:, None, None]
            I0 = (f_bdofs * bs)[:, :, None]        # (npt, ndofs_f, 1)  block rows
            C0 = (s_bdofs * bs)[None, None, :]     # (1, 1, ndofs_s)    block cols
            rows0.append(np.broadcast_to(I0, V.shape).ravel())
            cols0.append(np.broadcast_to(C0, V.shape).ravel())
            vals.append(V.ravel())
        if not vals:
            Mfs = coo_matrix((self.n_u, self.n_s)).tocsr()
        else:
            r = np.concatenate(rows0)
            c = np.concatenate(cols0)
            v = np.concatenate(vals)
            # only same-component pairs couple (comp 0 and comp 1, same weights)
            r2 = np.concatenate([r, r + 1])
            c2 = np.concatenate([c, c + 1])
            v2 = np.concatenate([v, v])
            Mfs = coo_matrix((v2, (r2, c2)), shape=(self.n_u, self.n_s)).tocsr()
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
    def assemble_elastic(self, W, tangent=True):
        """Incompressible neo-Hookean elastic force and (optionally) its tangent
        (mixed stiffness), vectorised over the located solid quadrature points:
            f_el,i = - int (P F^T) : grad_x phi_i dX,   P = mu (F - F^{-T})
            A_uW(i,j) = d f_el,i / dW_j
        With ``tangent=False`` only the (cheap) force f_el is returned and the
        tangent is skipped -- used by the frozen-Jacobian quasi-Newton, which
        needs the exact force every iteration but the tangent only once per
        step.
        """
        mu_s = self.cfg["mu_s"]
        bs = self.bs_u
        # ---- gather per-point data (flattened across solid cells), cached
        #      per interaction so the per-Newton-iteration force is cheap ----
        g = self._gather_cache
        if g is None:
            q_idx_list, s_bdofs_list = [], []
            f_bdofs_list, f_gx_list, w_list = [], [], []
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
            s_phi_k = self.s_phi[q_idx]                # (npts, ndofs_s)
            s_gphi_k = self.s_gphi[q_idx]              # (npts, ndofs_s, 2)
            g = (q_idx, s_bdofs, f_bdofs, f_gx, w, s_phi_k, s_gphi_k)
            self._gather_cache = g
        q_idx, s_bdofs, f_bdofs, f_gx, w, s_phi_k, s_gphi_k = g
        npts = len(q_idx)

        # ---- deformation gradient F = I + grad_X W at the quadrature points ----
        gradW = np.zeros((npts, 2, 2))
        for c in range(2):
            Wvals = W[s_bdofs * 2 + c]             # (npts, ndofs_s)
            gradW[:, c, :] = np.einsum("nk,nkb->nb", Wvals, s_gphi_k)
        F = np.eye(2)[None, :, :] + gradW
        Finv = np.linalg.inv(F)
        P = mu_s * (F - Finv.transpose(0, 2, 1))
        PeFT = P @ F.transpose(0, 2, 1)            # (npts, 2, 2)

        # ---- f_el (scatter by fluid dof), vectorised over (i, ci) ----
        #   contr[p,i,ci] = sum_a PeFT[p,ci,a] * f_gx[p,i,a] * w[p]
        contr = np.einsum("pca,pia->pic", PeFT, f_gx) * w[:, None, None]
        f_el = np.zeros(self.n_u)
        np.add.at(f_el,
                  (f_bdofs * bs)[:, :, None] + np.arange(2)[None, None, :],
                  -contr)
        if not tangent:
            return f_el, None

        # ---- A_uW (scatter by fluid x solid dof pair) ----
        # ---- A_uW (scatter by fluid x solid dof pair), vectorised with
        #      BLAS matmul.  dF[a][b] = delta_{a,cj} gk[b] has only one nonzero
        #      row, so (j, cj) are flattened into a single jc index.
        ndofs_f = f_bdofs.shape[1]
        ndofs_s = s_bdofs.shape[1]
        Ft = F.transpose(0, 2, 1)
        dF_all = np.zeros((npts, ndofs_s * 2, 2, 2))
        dF_all[:, 0::2, 0, :] = s_gphi_k          # cj=0: row 0 = gk
        dF_all[:, 1::2, 1, :] = s_gphi_k          # cj=1: row 1 = gk
        tmp = np.matmul(Finv[:, None, :, :], dF_all)              # [p,jc,i,b]
        dFinv_all = -np.matmul(tmp, Finv[:, None, :, :])          # [p,jc,i,k]
        dP_all = mu_s * (dF_all - dFinv_all.transpose(0, 1, 3, 2))
        term1 = np.matmul(dP_all, Ft[:, None, :, :])              # [p,jc,m,k]
        term2 = np.matmul(P[:, None, :, :], dF_all.transpose(0, 1, 3, 2))
        dPeFT_all = term1 + term2                                 # [p,jc,m,k]
        flat = dPeFT_all.reshape(npts, ndofs_s * 2 * 2, 2)        # [p,(jc,m),k]
        dcontr = (np.matmul(f_gx, flat.transpose(0, 2, 1))
                  * w[:, None, None])                             # [p,i,(jc,m)]
        dcontr = dcontr.reshape(npts, ndofs_f, ndofs_s, 2, 2)     # [p,i,j,c,m]
        rows = np.broadcast_to((f_bdofs * bs)[:, :, None, None, None]
                               + np.arange(2)[None, None, None, None, :],
                               dcontr.shape).ravel()
        cols = np.broadcast_to((s_bdofs * bs)[:, None, :, None, None]
                               + np.arange(2)[None, None, None, :, None],
                               dcontr.shape).ravel()
        A_uW = coo_matrix((-dcontr.ravel(), (rows, cols)),
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

    @staticmethod
    def _zero_bc(A, bc):
        """Zero the BC rows/columns of a scipy CSR matrix and set diag=1 for
        the BC dofs.  CSR-only (no LIL round-trip, which is O(N^2)-ish and the
        dominant cost at 64x64); bc must be the expanded global dofs."""
        A = A.tocsr().copy()
        is_bc = np.zeros(A.shape[0], dtype=bool)
        is_bc[bc] = True
        # zero entries lying in BC columns (vectorised)
        A.data[is_bc[A.indices]] = 0.0
        # zero BC rows
        indptr = A.indptr
        for i in bc:
            A.data[indptr[i]:indptr[i + 1]] = 0.0
        # set diag = 1 for BC dofs
        for i in bc:
            sl = slice(indptr[i], indptr[i + 1])
            pos = np.flatnonzero(A.indices[sl] == i)
            if pos.size:
                A.data[indptr[i] + pos[0]] = 1.0
            else:
                A[i, i] = 1.0
        return A

    def apply_bc(self, A):
        """Homogeneous increments at the velocity boundary + pinned pressure."""
        return self._zero_bc(A, self.bc_all)

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

        # (b) Newton loop.  Default is a quasi-Newton with a FROZEN Jacobian:
        # the monolithic matrix is factorised once and reused for every
        # correction; only the exact elastic force is re-evaluated each
        # iteration.  This is modified Newton -- it converges to the SAME root
        # as full Newton, so the implicit (stable) solution is unchanged, but it
        # avoids (n_it-1) expensive MUMPS factorisations.  FROZEN=1 factorises
        # once per step; FROZEN=2 (default) reuses the factor across steps and
        # only refactorises on demand (stall / failed convergence), so the
        # factorisation cost is amortised over many steps.  FROZEN=0 restores
        # the original full Newton.
        frozen = cfg["frozen"]
        lu = None
        if frozen == 2 and getattr(self, "_lu_cache", None) is not None:
            lu = self._lu_cache
            self.msg("    [mono] reusing frozen Jacobian factor")
        elif frozen >= 1:
            _, A_uW = self.assemble_elastic(self.X[self.n_u + self.n_p:],
                                            tangent=True)
            lu = MumpsFactor(self.apply_bc(self.build_monolithic(A_uW)))
            self.msg("    [mono] factorised Jacobian (frozen)")
            if frozen == 2:
                self._lu_cache = lu
        else:
            self._lu_cache = None

        dX_prev = None
        refactored = False
        converged = False
        for it in range(cfg["n_newton_max"]):
            W = self.X[self.n_u + self.n_p:]
            if frozen >= 1:
                f_el = self.assemble_elastic(W, tangent=False)[0]
            else:
                f_el, A_uW = self.assemble_elastic(W, tangent=True)
                lu = MumpsFactor(self.apply_bc(self.build_monolithic(A_uW)))

            R_u, R_p, R_W = self._residual(W_old, f_el)
            rhs = np.zeros(self.N)
            rhs[:self.n_u] = -R_u
            rhs[self.n_u:self.n_u + self.n_p] = -R_p
            rhs[self.n_u + self.n_p:] = -R_W
            rhs[self.bc_all] = 0.0

            dX = lu.solve(rhs)
            dX[self.bc_all] = 0.0
            self.X += dX

            dX_norm = np.linalg.norm(dX)
            self.msg(f"    [mono] newton {it} |dX|={dX_norm:.3e}"
                     f"{'  (refactorised)' if refactored else ''}")
            refactored = False
            if dX_norm < cfg["newton_rtol"] * (1.0 + np.linalg.norm(self.X)):
                converged = True
                break
            # adaptive refactorisation on stagnation (frozen mode only)
            if (frozen >= 1 and it > 0 and dX_prev is not None
                    and dX_norm > 0.5 * dX_prev):
                _, A_uW = self.assemble_elastic(
                    self.X[self.n_u + self.n_p:], tangent=True)
                lu = MumpsFactor(self.apply_bc(self.build_monolithic(A_uW)))
                if frozen == 2:
                    self._lu_cache = lu
                refactored = True
            dX_prev = dX_norm

        if not converged and frozen >= 1:
            # the frozen factor stalled/diverged: fall back to a full-Newton
            # restart from the step start with the exact Jacobian (robustness)
            self.msg("    [mono] frozen Newton did not converge -- "
                     "restarting with exact Jacobian")
            self.X[self.n_u + self.n_p:] = W_old
            for it in range(cfg["n_newton_max"]):
                W = self.X[self.n_u + self.n_p:]
                f_el, A_uW = self.assemble_elastic(W, tangent=True)
                lu = MumpsFactor(self.apply_bc(self.build_monolithic(A_uW)))
                R_u, R_p, R_W = self._residual(W_old, f_el)
                rhs = np.zeros(self.N)
                rhs[:self.n_u] = -R_u
                rhs[self.n_u:self.n_u + self.n_p] = -R_p
                rhs[self.n_u + self.n_p:] = -R_W
                rhs[self.bc_all] = 0.0
                dX = lu.solve(rhs)
                dX[self.bc_all] = 0.0
                self.X += dX
                dX_norm = np.linalg.norm(dX)
                self.msg(f"    [mono] restart newton {it} |dX|={dX_norm:.3e}")
                if dX_norm < cfg["newton_rtol"] * (1.0 + np.linalg.norm(self.X)):
                    break
            if frozen == 2:
                self._lu_cache = lu
        self.W_prev = self.X[self.n_u + self.n_p:].copy()

    # ------------------------------------------------------------------
    # Block-preconditioned GMRES linear solver
    #
    # The monolithic Jacobian
    #     A = [[K, Bt, -A_uW], [B, s11, 0], [-Mfs^T, 0, (1/dt) M_s]]
    # differs from the block operator
    #     P = [[K, Bt, 0], [B, s11, 0], [0, 0, (1/dt) M_s]]
    # only in the COUPLING blocks (-A_uW, -Mfs^T), which are low-rank (rank
    # ~ n_s, the solid dofs).  P is CONSTANT (the fluid Stokes block and the
    # solid mass matrix depend only on the meshes), so its two factors are
    # built ONCE and reused as a block preconditioner for GMRES on A.  Each
    # GMRES iteration costs one fluid Stokes backsolve + one M_s backsolve +
    # sparse matrix-vector products.  This is the scalable (large/3D) path:
    # no monolithic factorisation at all -- the low-rank coupling is absorbed
    # by the Krylov space.
    # ------------------------------------------------------------------
    def _build_block_preconditioner(self):
        """Constant block preconditioner P^{-1}: factors of the fluid Stokes
        block (with BCs) and the solid mass matrix, built once."""
        if getattr(self, "_fluid_pre_lu", None) is None:
            cfg = self.cfg
            s11 = cfg["p_stab"] * self.Mp if cfg["p_stab"] > 0.0 else None
            F = bmat([[self.K, self.Bt], [self.B, s11]], format="csr")
            F = self.apply_bc(F)                 # velocity BCs + pressure pin
            self._fluid_pre_lu = MumpsFactor(F)
            self._Ms_pre_lu = MumpsFactor(self.M_s)
            self.msg("    [gmres] built block preconditioner "
                     f"(fluid {F.shape[0]}x{F.shape[0]} + M_s {self.n_s}x{self.n_s})")
        return self._fluid_pre_lu, self._Ms_pre_lu

    def _block_gmres_solve(self, A, rhs, rtol=1e-8, maxiter=300):
        """Solve A x = rhs with GMRES + the constant block preconditioner."""
        from scipy.sparse.linalg import LinearOperator, gmres
        n_u, n_p, n_s = self.n_u, self.n_p, self.n_s
        fl, ms = self._build_block_preconditioner()
        dt = self.cfg["dt"]
        if getattr(self, "_gmres_P", None) is None:
            def p_inv(r):
                r = np.asarray(r, dtype=float)
                f = fl.solve(np.ascontiguousarray(r[:n_u + n_p]))
                w = dt * ms.solve(np.ascontiguousarray(r[n_u + n_p:]))
                return np.concatenate([f, w])
            self._gmres_P = LinearOperator((self.N, self.N), matvec=p_inv,
                                           dtype=float)
        x, info = gmres(A, rhs, M=self._gmres_P, rtol=rtol, atol=1e-14,
                        maxiter=maxiter, restart=100)
        if info != 0:
            raise RuntimeError(f"GMRES did not converge (info={info})")
        return x

    # ------------------------------------------------------------------
    # One time step, iterative (GMRES + block preconditioner) variant of the
    # monolithic scheme.  The Jacobian matrix A is frozen (cross-step, rebuilt
    # adaptively), and the CONSTANT block preconditioner is reused forever.
    # ------------------------------------------------------------------
    def solve_monolithic_gmres(self):
        cfg = self.cfg
        W_old = self.X[self.n_u + self.n_p:].copy()

        self.compute_interaction(W_old)
        self.assemble_mixed_mass()
        self.X[self.n_u + self.n_p:] = 2.0 * W_old - self.W_prev

        # frozen Jacobian (cross-step): reuse the assembled matrix A, rebuild
        # adaptively on stagnation / failed convergence
        A = getattr(self, "_A_cache", None)
        if A is None:
            _, A_uW = self.assemble_elastic(self.X[self.n_u + self.n_p:],
                                            tangent=True)
            A = self.apply_bc(self.build_monolithic(A_uW))
            self._A_cache = A
            self.msg("    [gmres] assembled frozen Jacobian")

        dX_prev = None
        refactored = False
        converged = False
        for it in range(cfg["n_newton_max"]):
            W = self.X[self.n_u + self.n_p:]
            f_el = self.assemble_elastic(W, tangent=False)[0]

            R_u, R_p, R_W = self._residual(W_old, f_el)
            rhs = np.zeros(self.N)
            rhs[:self.n_u] = -R_u
            rhs[self.n_u:self.n_u + self.n_p] = -R_p
            rhs[self.n_u + self.n_p:] = -R_W
            rhs[self.bc_all] = 0.0

            dX = self._block_gmres_solve(A, rhs)
            dX[self.bc_all] = 0.0
            self.X += dX

            dX_norm = np.linalg.norm(dX)
            self.msg(f"    [gmres] newton {it} |dX|={dX_norm:.3e}"
                     f"{'  (refactorised)' if refactored else ''}")
            refactored = False
            if dX_norm < cfg["newton_rtol"] * (1.0 + np.linalg.norm(self.X)):
                converged = True
                break
            if (it > 0 and dX_prev is not None and dX_norm > 0.5 * dX_prev):
                _, A_uW = self.assemble_elastic(
                    self.X[self.n_u + self.n_p:], tangent=True)
                A = self.apply_bc(self.build_monolithic(A_uW))
                self._A_cache = A
                refactored = True
            dX_prev = dX_norm

        if not converged:
            # full-Newton restart with a fresh Jacobian (robustness)
            self.msg("    [gmres] Newton did not converge -- restarting "
                     "with fresh Jacobian")
            self.X[self.n_u + self.n_p:] = W_old
            for it in range(cfg["n_newton_max"]):
                W = self.X[self.n_u + self.n_p:]
                f_el, A_uW = self.assemble_elastic(W, tangent=True)
                A = self.apply_bc(self.build_monolithic(A_uW))
                self._A_cache = A
                R_u, R_p, R_W = self._residual(W_old, f_el)
                rhs = np.zeros(self.N)
                rhs[:self.n_u] = -R_u
                rhs[self.n_u:self.n_u + self.n_p] = -R_p
                rhs[self.n_u + self.n_p:] = -R_W
                rhs[self.bc_all] = 0.0
                dX = self._block_gmres_solve(A, rhs)
                dX[self.bc_all] = 0.0
                self.X += dX
                dX_norm = np.linalg.norm(dX)
                self.msg(f"    [gmres] restart newton {it} |dX|={dX_norm:.3e}")
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
        return self._zero_bc(A2, bc)

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
            elif cfg["linear_solver"] == "gmres":
                self.solve_monolithic_gmres()
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
