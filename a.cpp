// ---------------------------------------------------------------------------
// ib_cavity_disk_implicit.cpp
//
// Immersed Boundary Finite Element Method (IBFE) for a 2D lid-driven square
// cavity with an immersed elastic disk -- MONOLITHIC (fully coupled) variant.
//
// This is a faithful-but-simplified reproduction of the immersed boundary
// finite element method of
//   * Boffi, Gastaldi, Heltai (2007), "Numerical approximation of the
//     immersed boundary method", Computers & Fluids 36, 1378-1387.
//   * Heltai & Costanzo (2012), "Variational implementation of immersed
//     finite element methods", CMAME 229-232, 110-127.
//   * Roy, Heltai & Costanzo (2015), "Benchmarking the immersed finite
//     element method for fluid-structure interaction problems",
//     CAMWA 69, 1167-1188.
// following the deal.II implementation luca-heltai/fe-ibm (step-46.cc) and
// luca-heltai/ans-ifem.
//
// Unlike ib_cavity_disk.cpp (operator-split), here the fluid velocity u, the
// pressure p and the solid displacement W are solved TOGETHER in one system
// (as in the reference ans-ifem monolithic residual): each time step is a
// backward-Euler step solved by Newton on the 3x3 block Jacobian
//
//   [  K     B^T    -A_uW  ] [ du ]     [ -R_u ]
//   [  B     0      0     ] [ dp ]  =  [ -R_p ]
//   [ -Mfs^T 0   (1/dt)M_s] [ dW ]     [ -R_W ]
//
// with residual
//   R_u = K u + B^T p - f_el(W)                (fluid momentum + elastic force)
//   R_p = B u                                  (incompressibility)
//   R_W = (1/dt) M_s (W - W^n) - Mfs^T u       (kinematic: dW/dt = u_s)
// where f_el is the spread incompressible neo-Hookean elastic force and
// A_uW = d f_el / dW is its tangent (the mixed stiffness).  The interaction /
// mixed mass Mfs are rebuilt every time step (geometry = W^n, semi-implicit),
// and f_el / A_uW are re-assembled at the current W inside the Newton loop.
// The pressure (a fluid variable extended over the solid) enforces the
// incompressibility over the whole domain including the disk, exactly as in
// the paper.
//
// KEY IDEA (the two finite element spaces):
//   * Background (Eulerian) mesh on the square cavity:
//         fluid_fe = FESystem(FE_Q(deg), dim, FE_DGP(deg-1), 1)  (Q2/FE_DGP(1))
//     unknowns (u, p): velocity + pressure (pressure = discontinuous, as in
//     the paper).
//   * Solid (Lagrangian) mesh on the immersed disk:
//         solid_fe = FESystem(FE_Q(deg), dim)                   (Q2 displacement)
//     unknown W: displacement of the disk.
//
// The fluid and solid live on two *independent, non-matching* meshes.  The
// only coupling between them is through the two transfer operators:
//
//   (1) SPREADING  J^T (solid -> background).  The elastic body force of the
//       solid is spread onto the background by taking the L2 inner product of
//       the *solid functions* with the *background test functions*:
//
//           (J^T F, v)_Omega = int_{Omega_s} F(X) . v(X) dX
//
//       For the incompressible neo-Hookean model (INH_0) of the paper the
//       elastic force is computed from the deformation gradient F = I + grad W
//       with first Piola-Kirchhoff stress  P = mu (F - F^{-T}), and spread as
//           f_el,i = - int_{Omega_s} (P F^T) : grad_x(phi_i_bg) dX
//       with tangent (mixed stiffness)
//           A_uW(i,j) = d f_el,i / dW_j
//                    = - int_{Omega_s} d(P F^T)/dW_j : grad_x(phi_i_bg) dX
//       where the background test functions are evaluated at the *solid*
//       quadrature points mapped into the background mesh.
//
//   (2) INTERPOLATION J (background -> solid).  The fluid velocity is
//       restricted to the solid by projecting it with the mixed mass matrix
//
//           M_s u_s = Mfs^T u,   Mfs(i_bg, j_s) =
//           int_{Omega_s} phi_i_bg(X) phi_j_s(X) dX
//
//       and the disk moves with the fluid:  dW/dt = u_s  (kinematic condition).
//
// Both coupling matrices are assembled over the "interaction cells": for each
// solid cell, the solid quadrature points are located inside the background
// mesh with GridTools::find_active_cell_around_point(), and the background
// shape functions are evaluated there.  This is exactly the point emphasised
// in the original papers: the solid functions are inner-producted with the
// background test functions.
//
// Time stepping (backward Euler + Newton on the 3x3 monolithic system):
//   (a) locate the disk (interaction) and rebuild the mixed mass Mfs;
//   (b) Newton iterations: assemble the elastic force f_el(W) and its tangent
//       A_uW(W), fill the 3x3 Jacobian, form the residual, solve with GMRES +
//       block-diagonal preconditioner (multigrid on K, Jacobi on Mp and Ms),
//       update [u; p; W];
//   (c) output.
//
// Default parameters reproduce the benchmark LDCFlow_Ball_DGP_INH1 of
// Roy-Heltai-Costanzo (2015): cavity l=1, R=0.2 at (0.6,0.5), rho=1,
// eta_f=0.01, mu^e=0.1, lid U=1, dt=1e-2, T=8.1 s, 64x64 background,
// FE_DGP(1) pressure, incompressible neo-Hookean disk.
//
// Default parameters reproduce the benchmark LDCFlow_Ball_DGP_INH1 of
// Roy-Heltai-Costanzo (2015): cavity l=1, R=0.2 at (0.6,0.5), rho=1,
// eta_f=0.01, mu^e=0.1, lid U=1, dt=1e-2, T=8.1 s, 64x64 background,
// FE_DGP(1) pressure, incompressible neo-Hookean disk.
//
// ---------------------------------------------------------------------------

#include <deal.II/base/data_out_base.h>
#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_dgp.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/block_linear_operator.h>
#include <deal.II/lac/constrained_linear_operator.h>
#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_ilu.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_constrained_dofs.h>
#include <deal.II/multigrid/mg_matrix.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/mg_transfer.h>
#include <deal.II/multigrid/multigrid.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <vector>

using namespace dealii;

// ===========================================================================
// Problem parameters
// ===========================================================================
struct Parameters
{
  // discretisation  (paper: velocity FE_Q(2), pressure FE_DGP(1))
  unsigned int degree      = 2;
  unsigned int n_cavity_ref = 6; // paper: Fluid refinement = 6 -> 64 x 64
  unsigned int n_solid_ref  = 3; // disk mesh (griffith_ball_ref3, 2626 dofs)

  // geometry (cavity is the unit square [0,1]^2; paper: R=0.2, C=(0.6,0.5))
  double R  = 0.2;
  double cx = 0.6;
  double cy = 0.5;

  // fluid (paper: rho=1, eta=0.01, lid U=1)
  double eta_f  = 0.01;
  double rho_f  = 1.0;
  double lid    = 1.0;

  // solid: incompressible neo-Hookean (INH_0) with first Piola-Kirchhoff
  // stress  P = mu (F - F^{-T})  (paper: Elastic modulus mu^e = 0.1)
  double mu_s   = 0.1;   // shear modulus mu^e
  double lambda_s = 0.4; // unused by the incompressible model (kept for CLI)
  double rho_s  = 1.0;   // solid density (paper: rho = 1)

  // time stepping (paper: Delta t = 1e-2, Final t = 8.1, output every 10 steps)
  double dt             = 1e-2;
  unsigned int n_steps  = 810; // 8.1 s
  unsigned int out_every = 10; // report every 0.1 s

  bool pin_center = false; // free disk (paper: not pinned)

  // ---- solver scheme / preconditioner experiment -------------------
  //   0: 3x3 symmetric block-GS,          GMRES tol 1e-8  (baseline)
  //   1: 3x3 symmetric block-GS,          GMRES tol 1e-6
  //   2: 3x3 symmetric block-GS, tol 1e-6, + linear extrapolation of W
  //      as the Newton seed (fewer Newton iterations)
  //   3: reduced 2x2 (Schur-eliminate W with diagonal M_s), velocity
  //      preconditioner = K multigrid (does NOT cover A_uW), tol 1e-6
  //   4: reduced 2x2 (Schur-eliminate W with diagonal M_s), velocity
  //      preconditioner = ILU(0) of K~ = K - dt A_uW D_s^-1 Mfs^T
  //      (covers the A_uW coupling), tol 1e-6
  int    scheme    = 0;
  double gmres_tol = 1e-8;
};

// ===========================================================================
// Lid velocity: the top wall moves with u = (lid, 0)
// ===========================================================================
template <int dim>
class LidVelocity : public Function<dim>
{
public:
  LidVelocity(const double lid_speed)
    : Function<dim>(dim)
    , speed(lid_speed)
  {}

  double
  value(const Point<dim> &, const unsigned int component) const override
  {
    return (component == 0) ? speed : 0.0;
  }

private:
  const double speed;
};

// ===========================================================================
// Classical geometric multigrid preconditioner for the velocity block
// (vector Laplacian, FESystem(FE_Q(deg), dim)) on the uniformly refined
// cavity mesh.  Built once at start-up and reused every time step.
//
//   level matrices  : vector Laplacian (component-matched, delta_ci,cj) on
//                     each level; Dirichlet (identity) rows on the boundary
//   transfer        : MGTransferPrebuilt (dof embedding from the FE)
//   smoother        : SOR, 2 pre + 2 post sweeps (symmetrised)
//   coarse          : Householder direct solve on the coarsest level
//
// Wrapped in PreconditionMG so it can be used directly as the velocity-block
// preconditioner of the Stokes system.  The dof ordering of this DoFHandler
// (FESystem(FE_Q(deg), dim), not renumbered) matches the velocity block of
// the fluid DoFHandler after its component-wise renumbering (both keep the
// per-cell interleaved ordering of the two velocity components).
// ===========================================================================
template <int dim>
class VelocityMGPreconditioner
{
public:
  VelocityMGPreconditioner(const Triangulation<dim> &tria,
                           const unsigned int        degree)
    : tria(tria)
    , fe(FESystem<dim>(FE_Q<dim>(degree), dim))
    , quad(degree + 1)
    , dh(tria)
  {
    dh.distribute_dofs(fe);
    dh.distribute_mg_dofs();

    // Dirichlet on the whole cavity boundary
    std::set<types::boundary_id> dirichlet_ids;
    for (types::boundary_id b = 0; b < 2 * dim; ++b)
      dirichlet_ids.insert(b);
    mg_constrained_dofs.initialize(dh);
    mg_constrained_dofs.make_zero_boundary_constraints(dh, dirichlet_ids);

    const unsigned int n_levels = tria.n_levels();
    mg_sparsity_patterns.resize(0, n_levels - 1);
    mg_matrices.resize(0, n_levels - 1);
    for (unsigned int level = 0; level < n_levels; ++level)
      {
        DynamicSparsityPattern dsp(dh.n_dofs(level), dh.n_dofs(level));
        MGTools::make_sparsity_pattern(dh, dsp, level);
        mg_sparsity_patterns[level].copy_from(dsp);
        mg_matrices[level].reinit(mg_sparsity_patterns[level]);
      }

    assemble_level_matrices();

    mg_transfer.initialize_constraints(mg_constrained_dofs);
    mg_transfer.build(dh);

    FullMatrix<double> coarse_matrix;
    coarse_matrix.copy_from(mg_matrices[0]);
    mg_coarse.initialize(coarse_matrix);

    mg_smoother.initialize(mg_matrices);
    mg_smoother.set_steps(2);
    mg_smoother.set_symmetric(true);

    mg_matrix = std::make_unique<mg::Matrix<Vector<double>>>(mg_matrices);
    mg = std::make_unique<Multigrid<Vector<double>>>(
      *mg_matrix, mg_coarse, mg_transfer, mg_smoother, mg_smoother);

    preconditioner = std::make_unique<
      PreconditionMG<dim, Vector<double>, MGTransferPrebuilt<Vector<double>>>>(
      dh, *mg, mg_transfer);
  }

  unsigned int
  n_dofs() const
  {
    return dh.n_dofs();
  }

  void
  vmult(Vector<double> &dst, const Vector<double> &src) const
  {
    preconditioner->vmult(dst, src);
  }

private:
  void
  assemble_level_matrices()
  {
    const unsigned int n_levels = tria.n_levels();

    std::vector<AffineConstraints<double>> boundary_constraints(n_levels);
    for (unsigned int level = 0; level < n_levels; ++level)
      {
        boundary_constraints[level].reinit(
          dh.locally_owned_mg_dofs(level),
          DoFTools::extract_locally_relevant_level_dofs(dh, level));
        for (const auto idx : mg_constrained_dofs.get_boundary_indices(level))
          boundary_constraints[level].constrain_dof_to_zero(idx);
        boundary_constraints[level].close();
      }

    FEValues<dim>      fe_values(fe, quad, update_gradients | update_JxW_values);
    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    for (unsigned int level = 0; level < n_levels; ++level)
      for (const auto &cell : dh.mg_cell_iterators_on_level(level))
        {
          fe_values.reinit(cell);
          cell_matrix = 0;
          for (const unsigned int q : fe_values.quadrature_point_indices())
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                  const unsigned int ci = fe.system_to_component_index(i).first;
                  const unsigned int cj = fe.system_to_component_index(j).first;
                  if (ci == cj)
                    cell_matrix(i, j) +=
                      (fe_values.shape_grad(i, q) *
                       fe_values.shape_grad(j, q)) *
                      fe_values.JxW(q);
                }
          cell->get_mg_dof_indices(local_dof_indices);
          boundary_constraints[level].distribute_local_to_global(
            cell_matrix, local_dof_indices, mg_matrices[level]);
        }
  }

  const Triangulation<dim> &tria;
  const FESystem<dim>       fe;
  const QGauss<dim>         quad;
  DoFHandler<dim>           dh;

  MGConstrainedDoFs                               mg_constrained_dofs;
  MGLevelObject<SparsityPattern>                  mg_sparsity_patterns;
  MGLevelObject<SparseMatrix<double>>             mg_matrices;
  MGTransferPrebuilt<Vector<double>>              mg_transfer;
  mg::SmootherRelaxation<PreconditionSOR<SparseMatrix<double>>,
                         Vector<double>>          mg_smoother;
  MGCoarseGridHouseholder<double, Vector<double>> mg_coarse;
  std::unique_ptr<mg::Matrix<Vector<double>>>     mg_matrix;
  std::unique_ptr<Multigrid<Vector<double>>>      mg;
  std::unique_ptr<PreconditionMG<dim,
                                 Vector<double>,
                                 MGTransferPrebuilt<Vector<double>>>>
    preconditioner;
};

// ===========================================================================
// Block-diagonal preconditioner for the Stokes saddle point:
//   P^{-1} = diag( K^{-1}_{approx}, M_p^{-1}_{approx} )
// (velocity block: geometric multigrid V-cycle; pressure block: Jacobi on
//  the pressure mass matrix).  Same idea as tutorial step-22, with the
//  velocity block preconditioned by multigrid.
// ===========================================================================
template <int dim>
class StokesPreconditioner
{
public:
  void
  initialize(const BlockSparseMatrix<double>     &system,
             const SparseMatrix<double>          &pressure_mass,
             const VelocityMGPreconditioner<dim> &velocity_mg)
  {
    velocity_mg_ptr = &velocity_mg;
    Mp_prec.initialize(pressure_mass);
  }

  void
  vmult(BlockVector<double> &dst, const BlockVector<double> &src) const
  {
    velocity_mg_ptr->vmult(dst.block(0), src.block(0));
    Mp_prec.vmult(dst.block(1), src.block(1));
  }

private:
  const VelocityMGPreconditioner<dim>    *velocity_mg_ptr;
  PreconditionJacobi<SparseMatrix<double>> Mp_prec;
};

// ===========================================================================
// Symmetric block-Gauss-Seidel preconditioner for the 3x3 monolithic system
//   [K B^T -A_uW; B 0 0; -Mfs^T 0 (1/dt)M_s],
// with the solid (W) and fluid (u,p) blocks each iterated at their own rate:
//   * forward (lower) sweep:
//       z_u = K_MG^{-1} r_u            (fluid velocity, multigrid, many sweeps)
//       z_p = M_p^{-1} r_p             (pressure)
//       z_W = dt M_s^{-1}( r_W + Mfs^T z_u )   (solid, mass matrix, few iters)
//   * backward (upper) sweep: fold the B^T (p->u) and -A_uW (W->u) couplings:
//       dst_u = K_MG^{-1}( r_u - B^T z_p + A_uW z_W )
// This handles both off-diagonal couplings J_10 = -Mfs^T and J_01 = -A_uW
// (the latter is what the one-sided block-triangular preconditioner missed),
// so the iteration count stays bounded even for stiff solids.
// ===========================================================================
template <int dim>
class MonolithicPreconditioner
{
public:
  void
  initialize(const SparseMatrix<double>          &mixed_mass,      // Mfs
             const SparseMatrix<double>          &pressure_mass,   // Mp
             const SparseMatrix<double>          &solid_mass,      // M_s
             const SparseMatrix<double>          &tangent,         // A_uW
             const SparseMatrix<double>          &velocity_pressure, // B^T
             const double                         dt,
             const VelocityMGPreconditioner<dim> &velocity_mg)
  {
    Mfs            = &mixed_mass;
    M_s            = &solid_mass;
    A_uW           = &tangent;
    Bt             = &velocity_pressure;
    dt_val         = dt;
    velocity_mg_ptr = &velocity_mg;
    Mp_prec.initialize(pressure_mass);
    Ms_prec.initialize(solid_mass);
  }

  void
  vmult(BlockVector<double> &dst, const BlockVector<double> &src) const
  {
    // forward (lower) sweep
    Vector<double> z_u(dst.block(0).size());
    velocity_mg_ptr->vmult(z_u, src.block(0)); // z_u = K_MG^{-1} r_u
    Mp_prec.vmult(dst.block(1), src.block(1)); // z_p = M_p^{-1} r_p
    Vector<double> tmp(dst.block(2).size());
    tmp = src.block(2);
    Mfs->Tvmult_add(tmp, z_u);                 // r_W + Mfs^T z_u
    Vector<double> zW(dst.block(2).size());
    Ms_prec.vmult(zW, tmp);
    zW *= dt_val;                              // z_W = dt M_s^{-1}(r_W + Mfs^T z_u)

    // backward (upper) sweep: re-solve u with the p and W couplings
    Vector<double> rhs_u = src.block(0);
    Vector<double> tmp2(dst.block(0).size());
    Bt->vmult(tmp2, dst.block(1));             // B^T z_p
    rhs_u -= tmp2;
    A_uW->vmult(tmp2, zW);                     // A_uW z_W
    rhs_u += tmp2;
    velocity_mg_ptr->vmult(dst.block(0), rhs_u); // dst_u = K_MG^{-1}(r_u - B^T z_p + A_uW z_W)

    dst.block(2) = zW;
  }

private:
  const SparseMatrix<double>          *Mfs, *M_s, *A_uW, *Bt;
  double                               dt_val;
  const VelocityMGPreconditioner<dim> *velocity_mg_ptr;
  PreconditionJacobi<SparseMatrix<double>> Mp_prec, Ms_prec;
};

// ===========================================================================
// Block-diagonal preconditioner for the reduced 2x2 Stokes-type system
//   [K~ B^T; B 0],   K~ = K - dt A_uW D_s^-1 Mfs^T
// with two choices for the velocity block:
//   scheme 3: geometric multigrid of the *fluid* Laplacian K (built once,
//             does NOT cover the A_uW coupling -> iteration count grows
//             with the solid stiffness mu);
//   scheme 4: ILU(0) of K~ itself (rebuilt every Newton iteration, DOES
//             cover the A_uW coupling -> bounded iterations even for stiff
//             solids).  Pressure block: Jacobi on Mp in both cases.
// ===========================================================================
template <int dim>
class ReducedPreconditioner
{
public:
  void
  initialize(const BlockSparseMatrix<double>     &system,
             const SparseMatrix<double>          &pressure_mass,
             const VelocityMGPreconditioner<dim> &velocity_mg,
             const SparseILU<double>             &ktilde_ilu,
             const int                            scheme)
  {
    velocity_mg_ptr = &velocity_mg;
    Ktilde_ilu_ptr  = &ktilde_ilu;
    scheme_val      = scheme;
    Mp_prec.initialize(pressure_mass);
  }

  void
  vmult(BlockVector<double> &dst, const BlockVector<double> &src) const
  {
    if (scheme_val == 4)
      Ktilde_ilu_ptr->vmult(dst.block(0), src.block(0)); // ILU(K~)
    else
      velocity_mg_ptr->vmult(dst.block(0), src.block(0)); // MG(K)
    Mp_prec.vmult(dst.block(1), src.block(1));
  }

private:
  const VelocityMGPreconditioner<dim>    *velocity_mg_ptr;
  const SparseILU<double>                *Ktilde_ilu_ptr;
  int                                     scheme_val;
  PreconditionJacobi<SparseMatrix<double>> Mp_prec;
};

// ===========================================================================
// Main class
// ===========================================================================
template <int dim>
class ImmersedFEM
{
public:
  ImmersedFEM(const Parameters &prm);

  void run();

private:
  // ---- meshes, elements, dof handlers -------------------------------
  Triangulation<dim> fluid_tria;
  Triangulation<dim> solid_tria;

  FESystem<dim> fluid_fe; // FESystem(FE_Q(deg), dim, FE_DGP(deg-1), 1)
  FESystem<dim> solid_fe; // FESystem(FE_Q(deg), dim)
  DoFHandler<dim> fluid_dh;
  DoFHandler<dim> solid_dh;

  MappingQ1<dim> mapping;

  unsigned int n_u, n_p; // background velocity / pressure dofs
  unsigned int n_s;      // solid dofs

  // ---- matrices -----------------------------------------------------
  SparseMatrix<double> K;    // viscous velocity block     (n_u x n_u)
  SparseMatrix<double> B;    // -divergence block          (n_p x n_u)
  SparseMatrix<double> Mp;   // pressure mass              (n_p x n_p)
  SparseMatrix<double> M_s;  // solid mass                 (n_s x n_s)
  SparseMatrix<double> Mfs;  // mixed mass                 (n_u x n_s)
  SparseMatrix<double> A_uW; // tangent d f_el / dW        (n_u x n_s)

  // 3x3 monolithic block system  [K B^T -A_uW; B 0 0; -Mfs^T 0 (1/dt)M_s]
  BlockSparsityPattern   mono_sparsity;
  BlockSparseMatrix<double> mono_matrix;
  BlockVector<double>    X; // monolithic solution [u_vel; p; W]
  BlockVector<double>    R; // monolithic residual [R_u; R_p; R_W]

  BlockSparsityPattern stokes_sparsity;
  BlockSparseMatrix<double> stokes_matrix; // [K B^T; B 0]

  // ---- reduced (Schur-eliminated) 2x2 system -----------------------
  //   K~ = K - dt A_uW D_s^-1 Mfs^T   (D_s = diag(M_s));  W is eliminated
  //   [K~ B^T; B 0] [du; dp] = [-R_u - dt A_uW D_s^-1 R_W; -R_p]
  SparseMatrix<double> Ktilde; // (n_u x n_u)
  BlockSparsityPattern reduced_sparsity;
  BlockSparseMatrix<double> reduced_matrix;
  SparseILU<double> Ktilde_ilu; // ILU(0) of K~ (covers the A_uW coupling)
  SparseILU<double> M_s_ilu;    // ILU of M_s (exact elimination in scheme 5)
  Vector<double> Ds_inv;        // diag(1/M_s)
  std::vector<std::vector<std::pair<unsigned int, double>>>
    mfs_col; // Mfs column lists (j, Mfs(j,k))
  Vector<double> W_prev; // previous converged W (extrapolation seed)

  // NOTE: deal.II SparseMatrix holds an ObserverPointer to the SparsityPattern
  // passed to reinit(); that SparsityPattern must therefore outlive the
  // matrix.  These are stored as class members for that reason.
  SparsityPattern solid_sparsity;
  SparsityPattern pressure_mass_sparsity;
  SparsityPattern mixed_sparsity;

  // ---- vectors ------------------------------------------------------
  BlockVector<double> u; // background [velocity; pressure]
  Vector<double>      W; // solid displacement

  AffineConstraints<double> center_constraints; // pin the disk centre

  // geometric multigrid preconditioner for the velocity block
  std::unique_ptr<VelocityMGPreconditioner<dim>> velocity_mg;

  // (time, vtu-name) pairs collected for the ParaView .pvd animation records
  // (output() is const, hence mutable)
  mutable std::vector<std::pair<double, std::string>> fluid_output_times;
  mutable std::vector<std::pair<double, std::string>> solid_output_times;

  Parameters prm;

  // ---- methods ------------------------------------------------------
  using InteractionMap =
    std::map<typename DoFHandler<dim>::active_cell_iterator,
             std::vector<std::tuple<Point<dim>,
                                    typename DoFHandler<dim>::active_cell_iterator,
                                    unsigned int>>>;

  void make_grids_and_dofs();
  void build_center_constraints();
  void assemble_stokes();
  void assemble_solid_mass();
  void compute_interaction(const Vector<double> &W,
                           InteractionMap       &interaction) const;
  // mixed mass Mfs over a frozen interaction (geometry)
  void assemble_mixed_mass(const InteractionMap &interaction);
  // nonlinear elastic force f_el(W) and its tangent A_uW = d f_el / dW,
  // using the background test functions of a frozen interaction (geometry)
  void assemble_elastic(const InteractionMap &interaction,
                        const Vector<double>  &W,
                        Vector<double>        &f_el,
                        SparseMatrix<double>  &A_uW);
  void assemble_monolithic(); // fill mono_matrix at the current X
  void assemble_reduced();    // fill reduced_matrix ([K~ B^T; B 0]) at the current X
  void solve_monolithic();    // one Newton-iterated time step (3x3 or reduced)
  void solve_monolithic_reduced(); // one Newton-iterated time step (2x2 reduced)
  void solve_monolithic_reduced_exact(); // scheme 5: exact Schur elimination
  void solve_stokes(const Vector<double> &f_el, const double t);
  void project_velocity_to_solid(Vector<double> &u_s,
                                 const Vector<double> &u_vel) const;
  void output(const unsigned int step, const double t) const;
  void diagnostics(const double t) const;
  void verify_coupling();
};

// ---------------------------------------------------------------------------
template <int dim>
ImmersedFEM<dim>::ImmersedFEM(const Parameters &prm_in)
  : fluid_fe(FE_Q<dim>(prm_in.degree), dim,
             FE_DGP<dim>(prm_in.degree - 1), 1)
  , solid_fe(FE_Q<dim>(prm_in.degree), dim)
  , fluid_dh(fluid_tria)
  , solid_dh(solid_tria)
  , prm(prm_in)
{}

// ---------------------------------------------------------------------------
template <int dim>
void
ImmersedFEM<dim>::make_grids_and_dofs()
{
  // Background (Eulerian) square cavity.  colorize=true gives distinct
  // boundary ids (0=left, 1=right, 2=bottom, 3=top) so the lid can be
  // prescribed independently from the other walls.
  GridGenerator::hyper_cube(fluid_tria, 0.0, 1.0, true);
  fluid_tria.refine_global(prm.n_cavity_ref);

  // Solid (Lagrangian) disk.  hyper_ball attaches a SphericalManifold to the
  // boundary so the circle is represented exactly.
  GridGenerator::hyper_ball(solid_tria, Point<dim>(prm.cx, prm.cy), prm.R);
  solid_tria.refine_global(prm.n_solid_ref);

  fluid_dh.distribute_dofs(fluid_fe);
  solid_dh.distribute_dofs(solid_fe);

  // Renumber the background component-wise: velocity dofs first, pressure
  // after, so that block 0 = velocity and block 1 = pressure.
  std::vector<unsigned int> block_component(dim + 1, 0);
  block_component[dim] = 1;
  DoFRenumbering::component_wise(fluid_dh, block_component);
  const std::vector<types::global_dof_index> dofs_per_block =
    DoFTools::count_dofs_per_fe_block(fluid_dh, block_component);
  n_u = dofs_per_block[0];
  n_p = dofs_per_block[1];
  n_s = solid_dh.n_dofs();

  std::cout << "  background dofs: velocity " << n_u << " + pressure " << n_p
            << " = " << n_u + n_p << "\n";
  std::cout << "  solid dofs:      " << n_s << "\n";

  // Geometric multigrid preconditioner for the velocity block, built once on
  // the uniformly refined cavity mesh.
  velocity_mg = std::make_unique<VelocityMGPreconditioner<dim>>(fluid_tria,
                                                                prm.degree);
  AssertThrow(velocity_mg->n_dofs() == n_u,
              ExcMessage("velocity DoFHandler size mismatch"));

  build_center_constraints();
  assemble_stokes();
  assemble_solid_mass();

  u.reinit(2);
  u.block(0).reinit(n_u);
  u.block(1).reinit(n_p);
  u.collect_sizes();
  W.reinit(n_s);
}

// ---------------------------------------------------------------------------
// Pin the displacement of the disk centre to zero.  With Q2 elements the
// centre is a vertex of the disk mesh, so we simply constrain the two
// displacement dofs sitting at the centre.
// ---------------------------------------------------------------------------
template <int dim>
void
ImmersedFEM<dim>::build_center_constraints()
{
  center_constraints.clear();
  if (!prm.pin_center)
    {
      center_constraints.close();
      return;
    }

  // Pin every solid displacement dof whose support point sits at the disk
  // centre (this holds the disk in place for the quasi-static demo).
  const Point<dim> center(prm.cx, prm.cy);
  std::vector<Point<dim>> support_points(solid_dh.n_dofs());
  DoFTools::map_dofs_to_support_points(mapping, solid_dh, support_points);
  for (unsigned int i = 0; i < solid_dh.n_dofs(); ++i)
    if (support_points[i].distance(center) < 1e-10)
      center_constraints.add_line(i);
  center_constraints.close();
  std::cout << "  disk centre pinned: " << center_constraints.n_constraints()
            << " dofs\n";
}

// ---------------------------------------------------------------------------
// Assemble the background Stokes operator  [ K  B^T ; B  0 ]
//   K(i,j) = 2 eta_f * symgrad(phi_i) : symgrad(phi_j)      (velocity)
//   B(i,j) = - phi_i * div(phi_j)                           (pressure x vel)
// ===========================================================================
template <int dim>
void
ImmersedFEM<dim>::assemble_stokes()
{
  QGauss<dim> quadrature(prm.degree + 1);
  FEValues<dim> fe_values(fluid_fe,
                          quadrature,
                          update_values | update_gradients |
                            update_JxW_values | update_quadrature_points);

  const unsigned int dofs_per_cell = fluid_fe.dofs_per_cell;
  const unsigned int n_q           = quadrature.size();
  FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
  const FEValuesExtractors::Vector velocities(0);
  const FEValuesExtractors::Scalar pressure(dim);

  {
    BlockDynamicSparsityPattern csp(2, 2);
    csp.block(0, 0).reinit(n_u, n_u);
    csp.block(0, 1).reinit(n_u, n_p);
    csp.block(1, 0).reinit(n_p, n_u);
    csp.block(1, 1).reinit(n_p, n_p);
    csp.collect_sizes();

    Table<2, DoFTools::Coupling> coupling(dim + 1, dim + 1);
    for (unsigned int c = 0; c < dim + 1; ++c)
      for (unsigned int d = 0; d < dim + 1; ++d)
        if (c == dim && d == dim)
          coupling[c][d] = DoFTools::none; // pressure-pressure
        else
          coupling[c][d] = DoFTools::always;

    DoFTools::make_sparsity_pattern(fluid_dh,
                                    coupling,
                                    csp,
                                    AffineConstraints<double>(),
                                    false);
    stokes_sparsity.copy_from(csp);
  }
  stokes_matrix.reinit(stokes_sparsity);

  for (const auto &cell : fluid_dh.active_cell_iterators())
    {
      fe_values.reinit(cell);
      cell->get_dof_indices(local_dof_indices);
      local_matrix = 0;

      for (unsigned int q = 0; q < n_q; ++q)
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          {
            const unsigned int ci = fluid_fe.system_to_component_index(i).first;
            for (unsigned int j = 0; j < dofs_per_cell; ++j)
              {
                const unsigned int cj =
                  fluid_fe.system_to_component_index(j).first;
                if (ci < dim && cj < dim)
                  local_matrix(i, j) +=
                    2.0 * prm.eta_f *
                    (fe_values[velocities].symmetric_gradient(i, q) *
                     fe_values[velocities].symmetric_gradient(j, q)) *
                    fe_values.JxW(q);
                else if (ci == dim && cj < dim)
                  local_matrix(i, j) +=
                    -fe_values.shape_value(i, q) *
                    fe_values[velocities].divergence(j, q) * fe_values.JxW(q);
                else if (ci < dim && cj == dim)
                  local_matrix(i, j) +=
                    -fe_values[velocities].divergence(i, q) *
                    fe_values.shape_value(j, q) * fe_values.JxW(q);
              }
          }
      stokes_matrix.add(local_dof_indices, local_matrix);
    }

  // extract the blocks we keep around for preconditioning / output
  K.reinit(stokes_sparsity.block(0, 0));
  K.copy_from(stokes_matrix.block(0, 0));
  B.reinit(stokes_sparsity.block(1, 0));
  B.copy_from(stokes_matrix.block(1, 0));
  // pressure mass (used only as a preconditioner)
  {
    DynamicSparsityPattern dsp(n_p, n_p);
    for (const auto &cell : fluid_dh.active_cell_iterators())
      {
        cell->get_dof_indices(local_dof_indices);
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          if (fluid_fe.system_to_component_index(i).first == dim)
            for (unsigned int j = 0; j < dofs_per_cell; ++j)
              if (fluid_fe.system_to_component_index(j).first == dim)
                {
                  const types::global_dof_index ri = local_dof_indices[i] - n_u;
                  const types::global_dof_index cj = local_dof_indices[j] - n_u;
                  dsp.add(ri, cj);
                }
      }
    pressure_mass_sparsity.copy_from(dsp);
    Mp.reinit(pressure_mass_sparsity);
    for (const auto &cell : fluid_dh.active_cell_iterators())
      {
        fe_values.reinit(cell);
        cell->get_dof_indices(local_dof_indices);
        for (unsigned int q = 0; q < n_q; ++q)
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            if (fluid_fe.system_to_component_index(i).first == dim)
              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                if (fluid_fe.system_to_component_index(j).first == dim)
                  Mp.add(local_dof_indices[i] - n_u,
                         local_dof_indices[j] - n_u,
                         fe_values.shape_value(i, q) *
                           fe_values.shape_value(j, q) * fe_values.JxW(q));
      }
  }
}

// ---------------------------------------------------------------------------
// Solid mass matrix on the disk:  M_s(i,j) = rho_s int_Omega_s phi_i phi_j dX
// ===========================================================================
template <int dim>
void
ImmersedFEM<dim>::assemble_solid_mass()
{
  QGauss<dim> quadrature(prm.degree + 2);
  FEValues<dim> fe_values(solid_fe,
                          quadrature,
                          update_values | update_JxW_values);
  const unsigned int dofs_per_cell = solid_fe.dofs_per_cell;
  const unsigned int n_q           = quadrature.size();
  FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  DynamicSparsityPattern dsp(n_s, n_s);
  for (const auto &cell : solid_dh.active_cell_iterators())
    {
      cell->get_dof_indices(local_dof_indices);
      for (unsigned int i = 0; i < dofs_per_cell; ++i)
        for (unsigned int j = 0; j < dofs_per_cell; ++j)
          dsp.add(local_dof_indices[i], local_dof_indices[j]);
    }
  solid_sparsity.copy_from(dsp);
  M_s.reinit(solid_sparsity);

  for (const auto &cell : solid_dh.active_cell_iterators())
    {
      fe_values.reinit(cell);
      cell->get_dof_indices(local_dof_indices);
      local_matrix = 0;
      for (unsigned int q = 0; q < n_q; ++q)
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          {
            const unsigned int ci =
              solid_fe.system_to_component_index(i).first;
            for (unsigned int j = 0; j < dofs_per_cell; ++j)
              {
                const unsigned int cj =
                  solid_fe.system_to_component_index(j).first;
                // vector basis functions: only same-component pairs couple
                // (delta_{ci,cj}); without this the mass matrix is singular
                if (ci == cj)
                  local_matrix(i, j) +=
                    prm.rho_s * fe_values.shape_value(i, q) *
                    fe_values.shape_value(j, q) * fe_values.JxW(q);
              }
          }
      M_s.add(local_dof_indices, local_matrix);
    }
}

// ===========================================================================
// Locate, for each solid quadrature point, the background cell that contains
// it (and its reference coordinates inside that cell).  This is the geometry
// machinery behind the coupling: every solid quadrature point is mapped into
// the background mesh, where the background test functions are evaluated.
// ===========================================================================
template <int dim>
void
ImmersedFEM<dim>::compute_interaction(
  const Vector<double> &W,
  std::map<typename DoFHandler<dim>::active_cell_iterator,
           std::vector<std::tuple<Point<dim>,
                                  typename DoFHandler<dim>::active_cell_iterator,
                                  unsigned int>>> &interaction) const
{
  interaction.clear();

  using CellIt = typename DoFHandler<dim>::active_cell_iterator;

  QGauss<dim> quadrature(prm.degree + 2);
  FEValues<dim> fe_values(solid_fe,
                          quadrature,
                          update_values | update_JxW_values |
                            update_quadrature_points);
  const unsigned int n_q = quadrature.size();
  std::vector<types::global_dof_index> s_dofs(solid_fe.dofs_per_cell);
  std::vector<Tensor<1, dim>> W_q(n_q);
  const Point<dim> center(prm.cx, prm.cy);

  for (const auto &cell : solid_dh.active_cell_iterators())
    {
      fe_values.reinit(cell);
      cell->get_dof_indices(s_dofs);

      // displacement evaluated at the solid quadrature points
      for (unsigned int q = 0; q < n_q; ++q)
        W_q[q] = 0.0;
      for (unsigned int i = 0; i < solid_fe.dofs_per_cell; ++i)
        {
          const unsigned int c = solid_fe.system_to_component_index(i).first;
          for (unsigned int q = 0; q < n_q; ++q)
            W_q[q][c] += W[s_dofs[i]] * fe_values.shape_value(i, q);
        }

      for (unsigned int q = 0; q < n_q; ++q)
        {
          // current (mapped) solid quadrature point
          Point<dim> xq = fe_values.quadrature_point(q) + W_q[q];

          bool found = false;
          for (unsigned int attempt = 0; attempt < 8 && !found; ++attempt)
            {
              Point<dim> p = xq;
              if (attempt > 0)
                {
                  // nudge toward the disk centre to escape a cell boundary /
                  // vertex
                  const double eps = 1e-10 * std::pow(2.0, attempt);
                  p += (center - xq) * eps;
                }
              try
                {
                  const auto cell_ref =
                    GridTools::find_active_cell_around_point(mapping,
                                                             fluid_dh,
                                                             p);
                  if (cell_ref.first != fluid_dh.end())
                    {
                      interaction[cell_ref.first].push_back(
                        std::make_tuple(cell_ref.second, cell, q));
                      found = true;
                    }
                }
              catch (...)
                {}
            }
          if (!found)
            std::cerr << "WARNING: solid point " << xq
                      << " not located in the background mesh.\n";
        }
    }
}

// ===========================================================================
// Assemble the two coupling matrices (the heart of the method):
//
//   Mfs(i_bg, j_s) = int_Omega_s  phi_i_bg(x)   * phi_j_s(X)  dX
//   Kfs(i_bg, j_s) = int_Omega_s  sigma(phi_j_s) : grad(phi_i_bg) dX
//
// i.e. the SOLID functions are inner-producted with the BACKGROUND test
// functions, both evaluated at the solid quadrature points mapped into the
// background mesh.
// ===========================================================================
template <int dim>
void
ImmersedFEM<dim>::assemble_mixed_mass(const InteractionMap &interaction)
{
  // sparsity (velocity-solid mixed coupling) -- shared by Mfs and A_uW
  {
    DynamicSparsityPattern dsp(n_u, n_s);
    std::vector<types::global_dof_index> f_dofs(fluid_fe.dofs_per_cell);
    std::vector<types::global_dof_index> s_dofs(solid_fe.dofs_per_cell);
    for (const auto &pr : interaction)
      {
        const auto bg_cell = pr.first;
        bg_cell->get_dof_indices(f_dofs);
        for (const auto &ip : pr.second)
          {
            const auto s_cell = std::get<1>(ip);
            s_cell->get_dof_indices(s_dofs);
            for (unsigned int i = 0; i < fluid_fe.dofs_per_cell; ++i)
              {
                if (fluid_fe.system_to_component_index(i).first >= dim)
                  continue; // pressure dofs are not coupled
                for (unsigned int j = 0; j < solid_fe.dofs_per_cell; ++j)
                  dsp.add(f_dofs[i], s_dofs[j]);
              }
          }
      }
    mixed_sparsity.copy_from(dsp);
  }
  Mfs.reinit(mixed_sparsity);
  A_uW.reinit(mixed_sparsity);

  // mixed mass  Mfs(i,j) = int_Omega_s phi_i_bg(x) . phi_j_s(X) dX
  QGauss<dim> quadrature(prm.degree + 2);
  FEValues<dim> solid_fe_values(solid_fe,
                                quadrature,
                                update_values | update_JxW_values);
  std::vector<types::global_dof_index> s_dofs(solid_fe.dofs_per_cell);
  std::vector<types::global_dof_index> f_dofs(fluid_fe.dofs_per_cell);

  for (const auto &pr : interaction)
    {
      const auto bg_cell = pr.first;
      bg_cell->get_dof_indices(f_dofs);

      std::vector<Point<dim>> points(pr.second.size());
      std::vector<double>     weights(pr.second.size(), 1.0);
      for (unsigned int k = 0; k < pr.second.size(); ++k)
        points[k] = std::get<0>(pr.second[k]);
      Quadrature<dim> bg_quad(points, weights);

      FEValues<dim> bg_fe_values(fluid_fe, bg_quad, update_values);
      bg_fe_values.reinit(bg_cell);

      for (unsigned int k = 0; k < pr.second.size(); ++k)
        {
          const auto       s_cell = std::get<1>(pr.second[k]);
          const unsigned int sq    = std::get<2>(pr.second[k]);
          solid_fe_values.reinit(s_cell);
          s_cell->get_dof_indices(s_dofs);
          const double JxW = solid_fe_values.JxW(sq);
          for (unsigned int i = 0; i < fluid_fe.dofs_per_cell; ++i)
            {
              const unsigned int ci =
                fluid_fe.system_to_component_index(i).first;
              if (ci >= dim)
                continue;
              for (unsigned int j = 0; j < solid_fe.dofs_per_cell; ++j)
                {
                  const unsigned int cj =
                    solid_fe.system_to_component_index(j).first;
                  if (ci == cj)
                    Mfs.add(f_dofs[i],
                            s_dofs[j],
                            bg_fe_values.shape_value(i, k) *
                              solid_fe_values.shape_value(j, sq) * JxW);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Nonlinear incompressible neo-Hookean elastic force and its tangent
// (mixed stiffness):
//     f_el,i = - int_Omega_s (P F^T) : grad_x phi_i dX,   P = mu (F - F^{-T})
//     A_uW(i,j) = d f_el,i / dW_j
//              = - int_Omega_s d(P F^T)/dW_j : grad_x phi_i dX
// The background test functions are taken from a *frozen* interaction (the
// geometry is W^n, semi-implicit); only F = I + grad_X W follows the current
// Newton iterate W.
// ===========================================================================
template <int dim>
void
ImmersedFEM<dim>::assemble_elastic(const InteractionMap &interaction,
                                   const Vector<double>  &W,
                                   Vector<double>        &f_el,
                                   SparseMatrix<double>  &A_uW)
{
  f_el = 0.0;
  A_uW = 0.0;

  QGauss<dim> quadrature(prm.degree + 2);
  FEValues<dim> solid_fe_values(solid_fe,
                                quadrature,
                                update_values | update_gradients |
                                  update_JxW_values);
  std::vector<types::global_dof_index> s_dofs(solid_fe.dofs_per_cell);
  std::vector<types::global_dof_index> f_dofs(fluid_fe.dofs_per_cell);

  for (const auto &pr : interaction)
    {
      const auto bg_cell = pr.first;
      bg_cell->get_dof_indices(f_dofs);

      std::vector<Point<dim>> points(pr.second.size());
      std::vector<double>     weights(pr.second.size(), 1.0);
      for (unsigned int k = 0; k < pr.second.size(); ++k)
        points[k] = std::get<0>(pr.second[k]);
      Quadrature<dim> bg_quad(points, weights);

      FEValues<dim> bg_fe_values(fluid_fe, bg_quad, update_gradients);
      bg_fe_values.reinit(bg_cell);

      for (unsigned int k = 0; k < pr.second.size(); ++k)
        {
          const auto       s_cell = std::get<1>(pr.second[k]);
          const unsigned int sq    = std::get<2>(pr.second[k]);
          solid_fe_values.reinit(s_cell);
          s_cell->get_dof_indices(s_dofs);

          // deformation gradient  F = I + grad_X W
          Tensor<2, dim> gradW;
          gradW = 0.0;
          for (unsigned int j = 0; j < solid_fe.dofs_per_cell; ++j)
            {
              const unsigned int    cj =
                solid_fe.system_to_component_index(j).first;
              const double          wj = W[s_dofs[j]];
              const Tensor<1, dim> &gs = solid_fe_values.shape_grad(j, sq);
              for (unsigned int b = 0; b < dim; ++b)
                gradW[cj][b] += wj * gs[b];
            }
          Tensor<2, dim> F;
          F = 0.0;
          for (unsigned int a = 0; a < dim; ++a)
            F[a][a] = 1.0;
          F += gradW;

          // incompressible neo-Hookean (INH_0):  P = mu (F - F^{-T})
          const Tensor<2, dim> Finv = invert(F);
          const Tensor<2, dim> P    = prm.mu_s * (F - transpose(Finv));
          const Tensor<2, dim> PeFT = P * transpose(F);
          const double         JxW  = solid_fe_values.JxW(sq);

          for (unsigned int i = 0; i < fluid_fe.dofs_per_cell; ++i)
            {
              const unsigned int    ci =
                fluid_fe.system_to_component_index(i).first;
              if (ci >= dim)
                continue;
              const Tensor<1, dim> &gb = bg_fe_values.shape_grad(i, k);

              // elastic force on the fluid:  f_el,i -= (P F^T) : grad_x phi_i
              double contr = 0.0;
              for (unsigned int b = 0; b < dim; ++b)
                contr += PeFT[ci][b] * gb[b];
              f_el[f_dofs[i]] -= contr * JxW;

              // tangent  A_uW(i,j) = d f_el,i / dW_j   (mixed stiffness)
              for (unsigned int j = 0; j < solid_fe.dofs_per_cell; ++j)
                {
                  const unsigned int    cj =
                    solid_fe.system_to_component_index(j).first;
                  const Tensor<1, dim> &gk = solid_fe_values.shape_grad(j, sq);

                  // dF[a][b] = delta_{a,cj} gk[b]
                  Tensor<2, dim> dF;
                  dF = 0.0;
                  for (unsigned int b = 0; b < dim; ++b)
                    dF[cj][b] = gk[b];

                  // d(F^{-1}) = -F^{-1} dF F^{-1};  d(F^{-T}) = transpose
                  const Tensor<2, dim> dFinv = -Finv * dF * Finv;
                  Tensor<2, dim>       dFinvT;
                  for (unsigned int a = 0; a < dim; ++a)
                    for (unsigned int b = 0; b < dim; ++b)
                      dFinvT[a][b] = dFinv[b][a];

                  // dP = mu (dF - dF^{-T});  d(P F^T) = dP F^T + P dF^T
                  const Tensor<2, dim> dP    = prm.mu_s * (dF - dFinvT);
                  const Tensor<2, dim> dPeFT =
                    dP * transpose(F) + P * transpose(dF);

                  double dcontr = 0.0;
                  for (unsigned int b = 0; b < dim; ++b)
                    dcontr += dPeFT[ci][b] * gb[b];
                  A_uW.add(f_dofs[i], s_dofs[j], -dcontr * JxW);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Fill the 3x3 monolithic Jacobian
//   [  K     B^T    -A_uW  ]
//   [  B     0      0     ]
//   [ -Mfs^T 0   (1/dt)M_s ]
// from the current A_uW / Mfs.
// ===========================================================================
template <int dim>
void
ImmersedFEM<dim>::assemble_monolithic()
{
  mono_matrix = 0.0;
  mono_matrix.block(0, 0).copy_from(K);                 // K
  mono_matrix.block(0, 1).copy_from(stokes_matrix.block(0, 1)); // B^T
  for (unsigned int i = 0; i < n_u; ++i)
    for (auto it = A_uW.begin(i); it != A_uW.end(i); ++it)
      mono_matrix.block(0, 2).set(i, it->column(), -it->value()); // -A_uW
  mono_matrix.block(1, 0).copy_from(B);                 // B
  for (unsigned int i = 0; i < n_u; ++i)
    for (auto it = Mfs.begin(i); it != Mfs.end(i); ++it)
      mono_matrix.block(2, 0).set(it->column(), i, -it->value()); // -Mfs^T
  for (unsigned int i = 0; i < n_s; ++i)
    for (auto it = M_s.begin(i); it != M_s.end(i); ++it)
      mono_matrix.block(2, 2).set(i,
                                  it->column(),
                                  (1.0 / prm.dt) * it->value()); // (1/dt)M_s
}

// ---------------------------------------------------------------------------
// One time step of the monolithic (fully coupled) scheme, backward Euler +
// Newton on the 3x3 system:
//   R_u = K u + B^T p - f_el(W)
//   R_p = B u
//   R_W = (1/dt) M_s (W - W^n) - Mfs^T u
// The interaction / mixed mass are frozen at W^n (semi-implicit geometry);
// f_el and its tangent A_uW are re-assembled at the current W.
// ===========================================================================
template <int dim>
void
ImmersedFEM<dim>::solve_monolithic()
{
  if (prm.scheme >= 3)
    {
      solve_monolithic_reduced();
      return;
    }

  // X = [u_vel; p; W] is the current solution (W = W^n on entry)
  const Vector<double> W_old = X.block(2);

  // scheme 2: seed the Newton iteration with a linear extrapolation of W
  if (prm.scheme == 2)
    {
      X.block(2) = W_old;
      X.block(2) += W_old;
      X.block(2) -= W_prev;
    }

  // --- frozen interaction at W^n --------------------------------------
  InteractionMap interaction;
  compute_interaction(W_old, interaction);
  assemble_mixed_mass(interaction);

  // --- 3x3 block sparsity (rebuild: mixed pattern follows the disk) ---
  {
    BlockDynamicSparsityPattern dsp(3, 3);
    dsp.block(0, 0).reinit(n_u, n_u);
    dsp.block(0, 1).reinit(n_u, n_p);
    dsp.block(0, 2).reinit(n_u, n_s);
    dsp.block(1, 0).reinit(n_p, n_u);
    dsp.block(1, 1).reinit(n_p, n_p);
    dsp.block(1, 2).reinit(n_p, n_s);
    dsp.block(2, 0).reinit(n_s, n_u);
    dsp.block(2, 1).reinit(n_s, n_p);
    dsp.block(2, 2).reinit(n_s, n_s);
    dsp.collect_sizes();
    const auto copy_pattern = [](DynamicSparsityPattern      &dst,
                                 const SparsityPattern &src) {
      for (unsigned int i = 0; i < src.n_rows(); ++i)
        for (auto it = src.begin(i); it != src.end(i); ++it)
          dst.add(i, it->column());
    };
    copy_pattern(dsp.block(0, 0), stokes_sparsity.block(0, 0));
    copy_pattern(dsp.block(0, 1), stokes_sparsity.block(0, 1));
    copy_pattern(dsp.block(0, 2), mixed_sparsity);
    copy_pattern(dsp.block(1, 0), stokes_sparsity.block(1, 0));
    copy_pattern(dsp.block(2, 2), solid_sparsity);
    for (unsigned int i = 0; i < n_u; ++i)
      for (auto it = mixed_sparsity.begin(i); it != mixed_sparsity.end(i); ++it)
        dsp.block(2, 0).add(it->column(), i); // transpose of mixed_sparsity
    mono_sparsity.copy_from(dsp);
    mono_matrix.reinit(mono_sparsity);
  }

  // --- boundary dofs (velocity walls + lid, pinned pressure): the Newton
  //     increments are zero there (X already holds the boundary values) ----
  std::map<types::global_dof_index, double> bc;
  {
    const Functions::ZeroFunction<dim> zero(dim);
    const LidVelocity<dim>             lid(prm.lid);
    VectorTools::interpolate_boundary_values(mapping, fluid_dh, 0, zero, bc);
    VectorTools::interpolate_boundary_values(mapping, fluid_dh, 1, zero, bc);
    VectorTools::interpolate_boundary_values(mapping, fluid_dh, 2, zero, bc);
    VectorTools::interpolate_boundary_values(mapping, fluid_dh, 3, lid, bc);
    bc[n_u] = 0.0; // pin one pressure dof (Stokes pressure nullspace)
    for (auto &p : bc)
      p.second = 0.0; // increments, not values
  }

  // --- Newton loop ----------------------------------------------------
  Vector<double> f_el(n_u);
  const unsigned int max_newton = 8;
  for (unsigned int it = 0; it < max_newton; ++it)
    {
      assemble_elastic(interaction, X.block(2), f_el, A_uW);
      assemble_monolithic();

      // residual
      R = 0.0;
      K.vmult_add(R.block(0), X.block(0));                     // K u
      stokes_matrix.block(0, 1).vmult_add(R.block(0),
                                          X.block(1));         // B^T p
      R.block(0) -= f_el;                                      // - f_el
      B.vmult(R.block(1), X.block(0));                         // B u
      Vector<double> WmW0 = X.block(2);
      WmW0 -= W_old;
      M_s.vmult(R.block(2), WmW0);                             // (1/dt)M_s(W-W^n)
      R.block(2) *= 1.0 / prm.dt;
      Vector<double> mfsTu(n_s);
      Mfs.Tvmult(mfsTu, X.block(0));                           // Mfs^T u
      R.block(2) -= mfsTu;

      // solve  mono_matrix * dX = -R  (with zero increment at the boundary)
      BlockVector<double> rhs(3);
      rhs.block(0).reinit(n_u);
      rhs.block(1).reinit(n_p);
      rhs.block(2).reinit(n_s);
      rhs.collect_sizes();
      rhs = R;
      rhs *= -1.0;

      BlockVector<double> dX(3);
      dX.block(0).reinit(n_u);
      dX.block(1).reinit(n_p);
      dX.block(2).reinit(n_s);
      dX.collect_sizes();
      dX = 0.0;

      MatrixTools::apply_boundary_values(bc, mono_matrix, dX, rhs,
                                         /*eliminate_columns=*/false);

      MonolithicPreconditioner<dim> preconditioner;
      preconditioner.initialize(Mfs, Mp, M_s, A_uW, stokes_matrix.block(0, 1),
                                prm.dt, *velocity_mg);

      SolverControl control(20000, prm.gmres_tol * rhs.l2_norm());
      SolverGMRES<BlockVector<double>>::AdditionalData data(400, false, true);
      SolverGMRES<BlockVector<double>> solver(control, data);
      solver.solve(mono_matrix, dX, rhs, preconditioner);

      X += dX;

      std::cout << "    [mono] newton " << it << " gmres=" << control.last_step()
                << " |dX|=" << std::scientific << dX.l2_norm() << "\n";
      if (dX.l2_norm() < 1e-9 * (1.0 + X.l2_norm()))
        break;
    }
}

// ---------------------------------------------------------------------------
// Assemble the reduced 2x2 Schur system
//   [K~ B^T; B 0],   K~ = K - dt A_uW D_s^-1 Mfs^T,  D_s = diag(M_s)
// at the current X (A_uW from the last Newton iteration).  The solid block is
// eliminated exactly with the *diagonal* approximation of M_s^{-1} (the solid
// mass matrix is strongly diagonally dominant); the outer Newton loop still
// uses the exact residuals, so the converged solution is unchanged.
// ===========================================================================
template <int dim>
void
ImmersedFEM<dim>::assemble_reduced()
{
  // diagonal inverse of M_s
  Ds_inv.reinit(n_s);
  for (unsigned int i = 0; i < n_s; ++i)
    Ds_inv[i] = 1.0 / M_s.diag_element(i);

  // column lists of Mfs: mfs_col[k] = { (j, Mfs(j,k)) }
  mfs_col.assign(n_s, {});
  for (unsigned int j = 0; j < n_u; ++j)
    for (auto it = Mfs.begin(j); it != Mfs.end(j); ++it)
      mfs_col[it->column()].emplace_back(j, it->value());

  // --- 2x2 sparsity: K~ pattern = K  ∪  (A_uW * D_s^-1 Mfs^T) ----------
  {
    BlockDynamicSparsityPattern dsp(2, 2);
    dsp.block(0, 0).reinit(n_u, n_u);
    dsp.block(0, 1).reinit(n_u, n_p);
    dsp.block(1, 0).reinit(n_p, n_u);
    dsp.block(1, 1).reinit(n_p, n_p);
    dsp.collect_sizes();
    for (unsigned int i = 0; i < n_u; ++i)
      for (auto it = K.begin(i); it != K.end(i); ++it)
        dsp.block(0, 0).add(i, it->column());
    for (unsigned int i = 0; i < n_u; ++i)
      for (auto it = A_uW.begin(i); it != A_uW.end(i); ++it)
        for (const auto &jp : mfs_col[it->column()])
          dsp.block(0, 0).add(i, jp.first);
    for (unsigned int i = 0; i < n_u; ++i)
      for (auto it = stokes_matrix.block(0, 1).begin(i);
           it != stokes_matrix.block(0, 1).end(i);
           ++it)
        dsp.block(0, 1).add(i, it->column());
    for (unsigned int i = 0; i < n_p; ++i)
      for (auto it = stokes_matrix.block(1, 0).begin(i);
           it != stokes_matrix.block(1, 0).end(i);
           ++it)
        dsp.block(1, 0).add(i, it->column());
    reduced_sparsity.copy_from(dsp);
    reduced_matrix.reinit(reduced_sparsity);
  }

  // --- values: K~ = K - dt A_uW D_s^-1 Mfs^T --------------------------
  // NOTE: cannot use SparseMatrix::copy_from(K): that overload requires the
  // *same* SparsityPattern object (it copies by memory order, with the Assert
  // disabled in Release).  b0's pattern differs from K's (it also contains the
  // A_uW*Mfs^T fill-in), so we copy entry-by-entry with set().
  reduced_matrix = 0.0;
  for (unsigned int i = 0; i < n_u; ++i)
    for (auto it = K.begin(i); it != K.end(i); ++it)
      reduced_matrix.block(0, 0).set(i, it->column(), it->value());
  for (unsigned int i = 0; i < n_u; ++i)
    for (auto it = stokes_matrix.block(0, 1).begin(i);
         it != stokes_matrix.block(0, 1).end(i);
         ++it)
      reduced_matrix.block(0, 1).set(i, it->column(), it->value());
  for (unsigned int i = 0; i < n_p; ++i)
    for (auto it = stokes_matrix.block(1, 0).begin(i);
         it != stokes_matrix.block(1, 0).end(i);
         ++it)
      reduced_matrix.block(1, 0).set(i, it->column(), it->value());
  for (unsigned int i = 0; i < n_u; ++i)
    for (auto it = A_uW.begin(i); it != A_uW.end(i); ++it)
      {
        const unsigned int k   = it->column();
        const double       aik = it->value();
        if (aik == 0.0)
          continue;
        const double scale = -prm.dt * aik * Ds_inv[k];
        for (const auto &jp : mfs_col[k])
          reduced_matrix.block(0, 0).add(i, jp.first, scale * jp.second);
      }

  // cache a copy for the ILU(K~) preconditioner (scheme 4)
  Ktilde.reinit(reduced_sparsity.block(0, 0));
  Ktilde.copy_from(reduced_matrix.block(0, 0));
}

// ---------------------------------------------------------------------------
// One time step of the *reduced* monolithic scheme:  Schur-eliminate W,
//   [K~ B^T; B 0] [du; dp] = [-R_u - dt A_uW D_s^-1 R_W; -R_p],
// solve with GMRES + ReducedPreconditioner, then recover
//   dW = dt D_s^-1 (Mfs^T du - R_W).
// The Newton residuals R_u, R_p, R_W use the *exact* M_s, so the converged
// solution is identical to the full 3x3 solve; only the linear system inside
// each Newton step is reduced.
// ===========================================================================
template <int dim>
void
ImmersedFEM<dim>::solve_monolithic_reduced()
{
  if (prm.scheme == 5)
    {
      solve_monolithic_reduced_exact();
      return;
    }

  // X = [u_vel; p; W] is the current solution (W = W^n on entry)
  const Vector<double> W_old = X.block(2);

  // --- frozen interaction at W^n --------------------------------------
  InteractionMap interaction;
  compute_interaction(W_old, interaction);
  assemble_mixed_mass(interaction);

  // --- boundary dofs (velocity walls + lid, pinned pressure) ----------
  std::map<types::global_dof_index, double> bc;
  {
    const Functions::ZeroFunction<dim> zero(dim);
    const LidVelocity<dim>             lid(prm.lid);
    VectorTools::interpolate_boundary_values(mapping, fluid_dh, 0, zero, bc);
    VectorTools::interpolate_boundary_values(mapping, fluid_dh, 1, zero, bc);
    VectorTools::interpolate_boundary_values(mapping, fluid_dh, 2, zero, bc);
    VectorTools::interpolate_boundary_values(mapping, fluid_dh, 3, lid, bc);
    bc[n_u] = 0.0; // pin one pressure dof
    for (auto &p : bc)
      p.second = 0.0;
  }

  // --- Newton loop ----------------------------------------------------
  Vector<double> f_el(n_u);
  const unsigned int max_newton = 8;
  for (unsigned int it = 0; it < max_newton; ++it)
    {
      assemble_elastic(interaction, X.block(2), f_el, A_uW);
      assemble_reduced();

      // exact residuals
      R = 0.0;
      K.vmult_add(R.block(0), X.block(0));               // K u
      stokes_matrix.block(0, 1).vmult_add(R.block(0),
                                          X.block(1));   // B^T p
      R.block(0) -= f_el;                                // - f_el
      B.vmult(R.block(1), X.block(0));                   // B u
      Vector<double> WmW0 = X.block(2);
      WmW0 -= W_old;
      M_s.vmult(R.block(2), WmW0);
      R.block(2) *= 1.0 / prm.dt;
      Vector<double> mfsTu(n_s);
      Mfs.Tvmult(mfsTu, X.block(0));
      R.block(2) -= mfsTu;

      // reduced RHS:  -R_u - dt A_uW D_s^-1 R_W ; -R_p
      BlockVector<double> rhs(2);
      rhs.block(0).reinit(n_u);
      rhs.block(1).reinit(n_p);
      rhs.collect_sizes();
      rhs.block(1) = R.block(1);
      rhs.block(1) *= -1.0;
      Vector<double> tmpW(n_s);
      for (unsigned int i = 0; i < n_s; ++i)
        tmpW[i] = Ds_inv[i] * R.block(2)[i];
      tmpW *= prm.dt;
      Vector<double> AuWDinvRW(n_u);
      A_uW.vmult(AuWDinvRW, tmpW);
      rhs.block(0) = R.block(0);
      rhs.block(0) *= -1.0;
      rhs.block(0) -= AuWDinvRW;

      BlockVector<double> dX2(2);
      dX2.block(0).reinit(n_u);
      dX2.block(1).reinit(n_p);
      dX2.collect_sizes();
      dX2 = 0.0;
      MatrixTools::apply_boundary_values(bc, reduced_matrix, dX2, rhs,
                                         /*eliminate_columns=*/false);

      // preconditioner (scheme 3: MG(K); scheme 4: ILU(K~))
      if (prm.scheme == 4)
        Ktilde_ilu.initialize(Ktilde);
      ReducedPreconditioner<dim> preconditioner;
      preconditioner.initialize(reduced_matrix,
                                Mp,
                                *velocity_mg,
                                Ktilde_ilu,
                                prm.scheme);

      SolverControl control(20000, prm.gmres_tol * rhs.l2_norm());
      SolverGMRES<BlockVector<double>>::AdditionalData data(400, false, true);
      SolverGMRES<BlockVector<double>> solver(control, data);
      solver.solve(reduced_matrix, dX2, rhs, preconditioner);

      // W back-substitution:  dW = dt D_s^-1 (Mfs^T du - R_W)
      Vector<double> dW(n_s);
      Mfs.Tvmult(dW, dX2.block(0));
      dW -= R.block(2);
      for (unsigned int i = 0; i < n_s; ++i)
        dW[i] *= Ds_inv[i] * prm.dt;

      X.block(0) += dX2.block(0);
      X.block(1) += dX2.block(1);
      X.block(2) += dW;

      const double dX_norm =
        std::sqrt(dX2.block(0).l2_norm() * dX2.block(0).l2_norm() +
                  dX2.block(1).l2_norm() * dX2.block(1).l2_norm() +
                  dW.l2_norm() * dW.l2_norm());
      std::cout << "    [red] newton " << it << " gmres=" << control.last_step()
                << " |dX|=" << std::scientific << dX_norm << "\n";
      if (dX_norm < 1e-9 * (1.0 + X.l2_norm()))
        break;
    }
}

// ---------------------------------------------------------------------------
// Scheme 5: *exact* Schur elimination of W.  The solid mass M_s^{-1} is applied
// with an ILU (M_s is a well-conditioned mass matrix, so the ILU is essentially
// exact), and K~ = K - dt A_uW M_s^-1 Mfs^T is kept as an implicit linear
// operator (it cannot be assembled explicitly since M_s^-1 is implicit).
// The 2x2 system [K~ B^T; B 0] is solved with GMRES + a constrained
// block-diagonal preconditioner (multigrid on K, Jacobi on Mp); homogeneous
// boundary conditions are imposed via AffineConstraints through
// constrained_linear_operator().  Because the elimination is exact, Newton
// converges quadratically (like the 3x3 solve), unlike schemes 3/4 where the
// diagonal M_s^{-1} approximation limits Newton to linear convergence.
// ===========================================================================
template <int dim>
void
ImmersedFEM<dim>::solve_monolithic_reduced_exact()
{
  // X = [u_vel; p; W] is the current solution (W = W^n on entry)
  const Vector<double> W_old = X.block(2);

  // --- frozen interaction at W^n --------------------------------------
  InteractionMap interaction;
  compute_interaction(W_old, interaction);
  assemble_mixed_mass(interaction);

  // ILU of M_s (constant over time) for the exact Schur elimination
  M_s_ilu.initialize(M_s);

  // seed the Newton iteration with a linear extrapolation of W (fewer iters)
  X.block(2) = W_old;
  X.block(2) += W_old;
  X.block(2) -= W_prev;

  // homogeneous constraints: velocity walls + lid, pinned pressure
  AffineConstraints<double> constraints;
  {
    const Functions::ZeroFunction<dim> zero(dim);
    const LidVelocity<dim>             lid(prm.lid);
    std::map<types::global_dof_index, double> bv;
    VectorTools::interpolate_boundary_values(mapping, fluid_dh, 0, zero, bv);
    VectorTools::interpolate_boundary_values(mapping, fluid_dh, 1, zero, bv);
    VectorTools::interpolate_boundary_values(mapping, fluid_dh, 2, zero, bv);
    VectorTools::interpolate_boundary_values(mapping, fluid_dh, 3, lid, bv);
    bv[n_u] = 0.0; // pin one pressure dof
    for (const auto &p : bv)
      constraints.add_constraint(p.first, {}, 0.0);
    constraints.close();
  }

  // Operator K~ = K - dt A_uW M_s^-1 Mfs^T (M_s^-1 = ILU, implicit).  Built
  // ONCE: linear_operator(A_uW) captures a *reference* to A_uW, so the lambda
  // picks up the re-assembled A_uW automatically in every Newton iteration.
  const auto Op_K     = linear_operator(K);
  const auto Op_Bt    = linear_operator(stokes_matrix.block(0, 1));
  const auto Op_B     = linear_operator(B);
  const auto Op_MfsT  = transpose_operator(linear_operator(Mfs));
  const auto Op_MsInv = linear_operator(M_s_ilu);
  const auto Op_A     = linear_operator(A_uW);
  const auto Op_Ktilde = Op_K - prm.dt * Op_A * Op_MsInv * Op_MfsT;
  const auto Op_null  = null_operator(Op_Ktilde);
  std::array<std::array<LinearOperator<Vector<double>, Vector<double>>, 2>, 2>
    blocks;
  blocks[0][0] = Op_Ktilde;
  blocks[0][1] = Op_Bt;
  blocks[1][0] = Op_B;
  blocks[1][1] = Op_null;
  const auto A2   = block_operator<2, 2>(blocks);
  const auto A2_c = constrained_linear_operator(constraints, A2);

  Vector<double> f_el(n_u);
  const unsigned int max_newton = 8;
  for (unsigned int it = 0; it < max_newton; ++it)
    {
      assemble_elastic(interaction, X.block(2), f_el, A_uW);

      // exact residuals
      R = 0.0;
      K.vmult_add(R.block(0), X.block(0));               // K u
      stokes_matrix.block(0, 1).vmult_add(R.block(0),
                                          X.block(1));   // B^T p
      R.block(0) -= f_el;                                // - f_el
      B.vmult(R.block(1), X.block(0));                   // B u
      Vector<double> WmW0 = X.block(2);
      WmW0 -= W_old;
      M_s.vmult(R.block(2), WmW0);
      R.block(2) *= 1.0 / prm.dt;
      Vector<double> mfsTu(n_s);
      Mfs.Tvmult(mfsTu, X.block(0));
      R.block(2) -= mfsTu;

      // reduced RHS:  -R_u - dt A_uW M_s^-1 R_W ; -R_p  (exact M_s^-1 = ILU)
      BlockVector<double> rhs2(2);
      rhs2.block(0).reinit(n_u);
      rhs2.block(1).reinit(n_p);
      rhs2.collect_sizes();
      rhs2.block(1) = R.block(1);
      rhs2.block(1) *= -1.0;
      Vector<double> tmpW(n_s);
      M_s_ilu.vmult(tmpW, R.block(2));
      tmpW *= prm.dt;
      Vector<double> AuWDinvRW(n_u);
      A_uW.vmult(AuWDinvRW, tmpW);
      rhs2.block(0) = R.block(0);
      rhs2.block(0) *= -1.0;
      rhs2.block(0) -= AuWDinvRW;

      const auto rhs2_c = constrained_right_hand_side(constraints, A2, rhs2);

      // preconditioner: block-diag (MG(K) | Mp).  We use the *unconstrained*
      // operator here: GMRES solves the constrained system A2_c, and feeding
      // it the plain block-diagonal preconditioner is the standard practical
      // combination (constraining the MG preconditioner via
      // constrained_linear_operator() crashes inside PreconditionMG here).
      StokesPreconditioner<dim> base_prec;
      base_prec.initialize(stokes_matrix, Mp, *velocity_mg);
      LinearOperator<BlockVector<double>, BlockVector<double>> Op_prec;
      Op_prec.vmult = [&base_prec](BlockVector<double>       &v,
                                   const BlockVector<double> &u) {
        base_prec.vmult(v, u);
        v.block(1)(0) = 0.0; // keep the pinned pressure dof (global n_u) clean
      };

      BlockVector<double> dX2(2);
      dX2.block(0).reinit(n_u);
      dX2.block(1).reinit(n_p);
      dX2.collect_sizes();
      dX2 = 0.0;

      SolverControl control(20000, prm.gmres_tol * rhs2.l2_norm());
      SolverGMRES<BlockVector<double>>::AdditionalData data(400, false, true);
      SolverGMRES<BlockVector<double>> solver(control, data);
      solver.solve(A2_c, dX2, rhs2_c, Op_prec);

      // W back-substitution (exact):  dW = dt M_s^-1 (Mfs^T du - R_W)
      Vector<double> dW(n_s);
      Mfs.Tvmult(dW, dX2.block(0));
      dW -= R.block(2);
      Vector<double> dW2(n_s);
      M_s_ilu.vmult(dW2, dW);
      dW2 *= prm.dt;

      X.block(0) += dX2.block(0);
      X.block(1) += dX2.block(1);
      X.block(2) += dW2;

      const double dX_norm =
        std::sqrt(dX2.block(0).l2_norm() * dX2.block(0).l2_norm() +
                  dX2.block(1).l2_norm() * dX2.block(1).l2_norm() +
                  dW2.l2_norm() * dW2.l2_norm());
      std::cout << "    [red5] newton " << it
                << " gmres=" << control.last_step()
                << " |dX|=" << std::scientific << dX_norm << "\n";
      if (dX_norm < 1e-9 * (1.0 + X.l2_norm()))
        break;
    }
}

// ---------------------------------------------------------------------------
// Solve the Stokes system  [K B^T; B 0] (u,p) = (f_el, 0) with lid BCs and a
// pinned pressure dof.
// ===========================================================================
template <int dim>
void
ImmersedFEM<dim>::solve_stokes(const Vector<double> &f_el, const double t)
{
  (void)t;
  BlockVector<double> rhs(2);
  rhs.block(0).reinit(n_u);
  rhs.block(1).reinit(n_p);
  rhs.collect_sizes();
  rhs.block(0) = f_el;
  rhs.block(1) = 0.0;

  // boundary conditions: no-slip walls + moving lid
  const Functions::ZeroFunction<dim> zero(dim);
  const LidVelocity<dim>             lid(prm.lid);
  std::map<types::global_dof_index, double> boundary_values;
  VectorTools::interpolate_boundary_values(mapping, fluid_dh, 0, zero,
                                           boundary_values);
  VectorTools::interpolate_boundary_values(mapping, fluid_dh, 1, zero,
                                           boundary_values);
  VectorTools::interpolate_boundary_values(mapping, fluid_dh, 2, zero,
                                           boundary_values);
  VectorTools::interpolate_boundary_values(mapping, fluid_dh, 3, lid,
                                           boundary_values);
  // pin one pressure dof (fixes the Stokes pressure nullspace)
  boundary_values[n_u] = 0.0;

  u = 0.0;
  // NB: eliminate_columns = false.  We reuse the same stokes_matrix object for
  // every time step, and deal.II's apply_boundary_values() is only safe to
  // call repeatedly on the same matrix if it does NOT eliminate columns (the
  // column elimination modifies the RHS using the *original* matrix, which is
  // lost after the first call).
  MatrixTools::apply_boundary_values(boundary_values,
                                     stokes_matrix,
                                     u,
                                     rhs,
                                     /*eliminate_columns=*/false);

  StokesPreconditioner<dim> preconditioner;
  preconditioner.initialize(stokes_matrix, Mp, *velocity_mg);

  SolverControl control(20000, 1e-10 * rhs.l2_norm());
  SolverGMRES<BlockVector<double>>::AdditionalData data(400, false, true);
  SolverGMRES<BlockVector<double>> solver(control, data);
  solver.solve(stokes_matrix, u, rhs, preconditioner);

  std::cout << "    [gmres] iters=" << control.last_step()
            << " |r|=" << std::scientific << control.last_value() << "\n";
  if (control.last_step() >= control.max_steps())
    std::cout << "    !!! GMRES did not converge\n";
}

// ---------------------------------------------------------------------------
// Interpolate the fluid velocity onto the solid by Galerkin projection with
// the mixed mass matrix:
//     M_s u_s = Mfs^T u_vel
// ===========================================================================
template <int dim>
void
ImmersedFEM<dim>::project_velocity_to_solid(Vector<double>       &u_s,
                                            const Vector<double> &u_vel) const
{
  Vector<double> rhs(n_s);
  Mfs.Tvmult(rhs, u_vel);

  u_s = 0.0;
  SolverControl control(10000, 1e-13 * rhs.l2_norm());
  SolverCG<Vector<double>> solver(control);
  PreconditionJacobi<SparseMatrix<double>> preconditioner;
  preconditioner.initialize(M_s);
  solver.solve(M_s, u_s, rhs, preconditioner);

  center_constraints.distribute(u_s);
}

// ---------------------------------------------------------------------------
template <int dim>
void
ImmersedFEM<dim>::output(const unsigned int step, const double t) const
{
  // fluid
  {
    DataOut<dim> data_out;
    data_out.attach_dof_handler(fluid_dh);
    std::vector<std::string> names(dim + 1, "u");
    names[dim] = "p";
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      interpretation(dim + 1,
                     DataComponentInterpretation::component_is_part_of_vector);
    interpretation[dim] = DataComponentInterpretation::component_is_scalar;
    data_out.add_data_vector(u, names, DataOut<dim>::type_dof_data,
                             interpretation);
    data_out.build_patches();
    const std::string filename =
      "ib-fluid-" + std::to_string(step) + ".vtu";
    std::ofstream out(filename);
    data_out.write_vtu(out);
    fluid_output_times.emplace_back(t, filename);
  }
  // solid
  {
    DataOut<dim> data_out;
    data_out.attach_dof_handler(solid_dh);
    std::vector<std::string> names(dim, "W");
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      interpretation(dim,
                     DataComponentInterpretation::component_is_part_of_vector);
    data_out.add_data_vector(W, names, DataOut<dim>::type_dof_data,
                             interpretation);
    data_out.build_patches();
    const std::string filename =
      "ib-solid-" + std::to_string(step) + ".vtu";
    std::ofstream out(filename);
    data_out.write_vtu(out);
    solid_output_times.emplace_back(t, filename);
  }

  // ParaView master files: one .pvd per field that references all .vtu files
  // tagged with their time, so the sequence plays back as an animation.
  {
    std::ofstream pvd("ib-fluid.pvd");
    DataOutBase::write_pvd_record(pvd, fluid_output_times);
  }
  {
    std::ofstream pvd("ib-solid.pvd");
    DataOutBase::write_pvd_record(pvd, solid_output_times);
  }

  std::cout << "    [output] t = " << t << " step " << step << "\n";
}

// ---------------------------------------------------------------------------
// Report the disk centre, area and velocity diagnostics (like the *_global.gpl
// output of the reference code).
// ===========================================================================
template <int dim>
void
ImmersedFEM<dim>::diagnostics(const double t) const
{
  QGauss<dim> quadrature(prm.degree + 2);
  FEValues<dim> fe_values(solid_fe,
                          quadrature,
                          update_JxW_values | update_quadrature_points |
                            update_values | update_gradients);
  const unsigned int n_q = quadrature.size();
  std::vector<types::global_dof_index> s_dofs(solid_fe.dofs_per_cell);
  double                     area = 0.0;       // reference area (constant)
  double                     deformed_area = 0.0; // int det(F) dX (physical)
  double                     maxJ1 = 0.0;      // max |det(F) - 1|
  double                     maxJexp = 0.0;    // max (det(F)-1)  (expansion)
  double                     maxJcom = 0.0;    // max (1-det(F))  (compression)
  Point<dim>                 center;
  for (const auto &cell : solid_dh.active_cell_iterators())
    {
      fe_values.reinit(cell);
      cell->get_dof_indices(s_dofs);
      for (unsigned int q = 0; q < n_q; ++q)
        {
          // current position = reference + displacement
          Point<dim> xq = fe_values.quadrature_point(q);
          for (unsigned int i = 0; i < solid_fe.dofs_per_cell; ++i)
            {
              const unsigned int c =
                solid_fe.system_to_component_index(i).first;
              xq[c] += W[s_dofs[i]] * fe_values.shape_value(i, q);
            }
          // deformation gradient F = I + grad_X W (incompressibility check)
          Tensor<2, dim> gradW;
          gradW = 0.0;
          for (unsigned int i = 0; i < solid_fe.dofs_per_cell; ++i)
            {
              const unsigned int    c = solid_fe.system_to_component_index(i).first;
              const double          w = W[s_dofs[i]];
              const Tensor<1, dim> &gs = fe_values.shape_grad(i, q);
              for (unsigned int b = 0; b < dim; ++b)
                gradW[c][b] += w * gs[b];
            }
          Tensor<2, dim> F;
          F = 0.0;
          for (unsigned int a = 0; a < dim; ++a)
            F[a][a] = 1.0;
          F += gradW;
          const double J = determinant(F);

          area += fe_values.JxW(q);
          deformed_area += J * fe_values.JxW(q);
          maxJ1 = std::max(maxJ1, std::abs(J - 1.0));
          maxJexp = std::max(maxJexp, J - 1.0);
          maxJcom = std::max(maxJcom, 1.0 - J);
          // centroid in the *current* configuration:  int_{B_t} x dx
          //   = int_{B_0} x(X) J(X) dX, i.e. weight by det(F)
          center += J * fe_values.JxW(q) * xq;
        }
    }
  center /= deformed_area; // current-configuration centroid

  // component-wise maximum displacement of the solid
  double maxWx = 0.0, maxWy = 0.0, maxW = 0.0;
  std::vector<types::global_dof_index> s_dofs2(solid_fe.dofs_per_cell);
  for (const auto &cell : solid_dh.active_cell_iterators())
    {
      cell->get_dof_indices(s_dofs2);
      for (unsigned int i = 0; i < solid_fe.dofs_per_cell; ++i)
        {
          const unsigned int c = solid_fe.system_to_component_index(i).first;
          const double        w = std::abs(W[s_dofs2[i]]);
          if (c == 0)
            maxWx = std::max(maxWx, w);
          else
            maxWy = std::max(maxWy, w);
          maxW = std::max(maxW, w);
        }
    }

  // displacement of the disk centroid from its initial position
  const Point<dim> init(prm.cx, prm.cy);
  const double disp_x = center[0] - init[0];
  const double disp_y = center[1] - init[1];
  const double disp   = center.distance(init);

  std::cout << "  t=" << std::setw(8) << std::fixed << std::setprecision(5)
            << t << "  centre=(" << std::setprecision(4) << center[0] << ","
            << center[1] << ")  disp=(" << disp_x << "," << disp_y << ") |"
            << disp << "|  max|Wx|=" << maxWx << "  max|Wy|=" << maxWy
            << "  max|W|=" << maxW << "  A_ref=" << std::setprecision(6)
            << area << "  A_def=" << deformed_area
            << "  max(J-1)=" << std::scientific << maxJexp
            << "  max(1-J)=" << maxJcom << "\n";

  // append a compact per-time solid deformation report
  std::ofstream f("solid_deformation.txt", std::ios::app);
  f << std::scientific << std::setprecision(6) << t << " " << center[0] << " "
    << center[1] << " " << disp_x << " " << disp_y << " " << disp << " "
    << maxWx << " " << maxWy << " " << maxW << " " << area << " "
    << deformed_area << " " << maxJexp << " " << maxJcom << "\n";
}

// ---------------------------------------------------------------------------
// Self-check of the two-space coupling: project the linear background field
// g(x,y) = (x, y) onto the solid.  Q2 elements reproduce linear fields
// exactly, so the projected solid vector should equal g at every solid dof.
// ===========================================================================
// ---------------------------------------------------------------------------
template <int dim>
void
ImmersedFEM<dim>::verify_coupling()
{
  InteractionMap interaction;
  compute_interaction(W, interaction); // W == 0 here (reference configuration)
  assemble_mixed_mass(interaction);

  // helper: build a background velocity vector whose components equal a given
  // scalar field (or a constant on the x-component).  Both meshes use Q2,
  // which reproduces linear fields exactly under the Galerkin projection, so
  // these self-checks verify that the mixed operators are assembled correctly.
  const auto build_g_vel = [&](const std::vector<Point<dim>> &sp,
                               const bool constant_x_only) {
    Vector<double> g_vel(n_u);
    std::vector<types::global_dof_index> f_dofs(fluid_fe.dofs_per_cell);
    for (const auto &cell : fluid_dh.active_cell_iterators())
      {
        cell->get_dof_indices(f_dofs);
        for (unsigned int i = 0; i < fluid_fe.dofs_per_cell; ++i)
          {
            const unsigned int c =
              fluid_fe.system_to_component_index(i).first;
            if (c >= dim)
              continue;
            if (constant_x_only)
              g_vel[f_dofs[i]] = (c == 0 ? 1.0 : 0.0);
            else
              g_vel[f_dofs[i]] = sp[f_dofs[i]][c];
          }
      }
    return g_vel;
  };
  const auto build_expected = [&](const std::vector<Point<dim>> &sp,
                                  const bool constant_x_only) {
    Vector<double> expected(n_s);
    std::vector<types::global_dof_index> s_dofs(solid_fe.dofs_per_cell);
    for (const auto &cell : solid_dh.active_cell_iterators())
      {
        cell->get_dof_indices(s_dofs);
        for (unsigned int j = 0; j < solid_fe.dofs_per_cell; ++j)
          {
            const unsigned int c =
              solid_fe.system_to_component_index(j).first;
            if (constant_x_only)
              expected[s_dofs[j]] = (c == 0 ? 1.0 : 0.0);
            else
              expected[s_dofs[j]] = sp[s_dofs[j]][c];
          }
      }
    return expected;
  };

  std::vector<Point<dim>> fsp(fluid_dh.n_dofs()), ssp(solid_dh.n_dofs());
  // NB: the pressure is FE_DGP, which has no support points, so
  // DoFTools::map_dofs_to_support_points() cannot be used on the fluid.  Fill
  // the velocity support points manually (only those are needed here).
  {
    std::vector<types::global_dof_index> f_dofs(fluid_fe.dofs_per_cell);
    for (const auto &cell : fluid_dh.active_cell_iterators())
      {
        cell->get_dof_indices(f_dofs);
        for (unsigned int i = 0; i < fluid_fe.dofs_per_cell; ++i)
          if (fluid_fe.system_to_component_index(i).first < dim)
            fsp[f_dofs[i]] =
              mapping.transform_unit_to_real_cell(cell,
                                                  fluid_fe.unit_support_point(i));
      }
  }
  DoFTools::map_dofs_to_support_points(mapping, solid_dh, ssp);

  {
    // check 1: constant field g = (1,0) must be reproduced exactly.
    // NB: the disk-centre dofs are intentionally pinned to zero, so we apply
    // the same pinning to the reference field before comparing.
    Vector<double> g_vel = build_g_vel(fsp, true);
    Vector<double> expected = build_expected(ssp, true);
    Vector<double> u_s(n_s);
    project_velocity_to_solid(u_s, g_vel);
    center_constraints.distribute(expected);
    double max_err = 0.0;
    for (unsigned int i = 0; i < n_s; ++i)
      max_err = std::max(max_err, std::abs(u_s[i] - expected[i]));
    std::cout << "  coupling self-check: projection of constant (1,0), max err "
              << std::scientific << max_err << "\n";
  }
  {
    // check 2: linear field g(x,y) = (x,y) must be reproduced exactly
    Vector<double> g_vel = build_g_vel(fsp, false);
    Vector<double> expected = build_expected(ssp, false);
    Vector<double> u_s(n_s);
    project_velocity_to_solid(u_s, g_vel);
    center_constraints.distribute(expected);
    double max_err = 0.0;
    for (unsigned int i = 0; i < n_s; ++i)
      max_err = std::max(max_err, std::abs(u_s[i] - expected[i]));
    std::cout
      << "  coupling self-check: projection of (x,y) onto solid, max err "
      << std::scientific << max_err << "\n";
  }
}

// ---------------------------------------------------------------------------
template <int dim>
void
ImmersedFEM<dim>::run()
{
  std::cout << "=== IBFE: lid-driven square cavity with immersed elastic disk "
               "(dim=" << dim << ") ===\n";
  make_grids_and_dofs();

  std::cout << "--- coupling self-check ---\n";
  verify_coupling();

  std::cout << "--- time stepping (monolithic IBFE, backward Euler + Newton) ---\n";
  {
    // per-time solid deformation report (t centre disp_x disp_y |disp| max|Wx| max|Wy| max|W| area)
    std::ofstream f("solid_deformation.txt", std::ios::trunc);
    f << "# t centre_x centre_y disp_x disp_y |disp| max|Wx| max|Wy| max|W| A_ref A_def max(J-1) max(1-J)\n";
  }
  double t = 0.0;
  output(0, t);
  diagnostics(t);

  // monolithic unknowns  X = [u_vel; p; W]
  X.reinit(3);
  X.block(0).reinit(n_u);
  X.block(1).reinit(n_p);
  X.block(2).reinit(n_s);
  X.collect_sizes();
  X = 0.0;
  R.reinit(3);
  R.block(0).reinit(n_u);
  R.block(1).reinit(n_p);
  R.block(2).reinit(n_s);
  R.collect_sizes();

  // previous converged W (extrapolation seed for scheme 2)
  W_prev.reinit(n_s);
  W_prev = 0.0;

  // initial velocity boundary values (lid + walls) live in X
  {
    const Functions::ZeroFunction<dim> zero(dim);
    const LidVelocity<dim>             lid(prm.lid);
    std::map<types::global_dof_index, double> bv;
    VectorTools::interpolate_boundary_values(mapping, fluid_dh, 0, zero, bv);
    VectorTools::interpolate_boundary_values(mapping, fluid_dh, 1, zero, bv);
    VectorTools::interpolate_boundary_values(mapping, fluid_dh, 2, zero, bv);
    VectorTools::interpolate_boundary_values(mapping, fluid_dh, 3, lid, bv);
    for (const auto &p : bv)
      X.block(0)(p.first) = p.second;
  }

  double t_step = 0.;
  for (unsigned int step = 1; step <= prm.n_steps; ++step)
    {
      auto t0 = std::chrono::steady_clock::now();
      solve_monolithic();
      t_step += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      W_prev = X.block(2); // last converged W (extrapolation seed)

      // copy the monolithic solution into the output fields
      u.block(0) = X.block(0);
      u.block(1) = X.block(1);
      W = X.block(2);

      t += prm.dt;
      if (step % prm.out_every == 0)
        {
          std::cout << "    [time] " << t_step / step << " s/step\n";
          diagnostics(t);
          output(step, t);
        }
    }
  diagnostics(t);
  std::cout << "=== done ===\n";
}

// ===========================================================================
int
main(int argc, char *argv[])
{
  try
    {
      Parameters prm;

      // optional command line overrides (also used for the benchmark of
      // Roy-Heltai-Costanzo 2015, "Disk entrained in a lid-driven cavity"):
      //   ib_cavity_disk_implicit [n_steps] [dt] [mu_s] [out_every]
      //                          [n_cavity_ref] [n_solid_ref] [lambda_s] [rho_s]
      //                          [R] [cx] [cy] [eta_f] [pin_center] [lid] [scheme]
      //   scheme: 0 baseline 3x3 sGS; 1 tol 1e-6; 2 tol 1e-6 + W extrapolation;
      //           3 reduced 2x2 (MG(K)); 4 reduced 2x2 (ILU(K~))
      if (argc > 1)
        prm.n_steps = std::atoi(argv[1]);
      if (argc > 2)
        prm.dt = std::atof(argv[2]);
      if (argc > 3)
        prm.mu_s = std::atof(argv[3]);
      if (argc > 4)
        prm.out_every = std::atoi(argv[4]);
      if (argc > 5)
        prm.n_cavity_ref = std::atoi(argv[5]);
      if (argc > 6)
        prm.n_solid_ref = std::atoi(argv[6]);
      if (argc > 7)
        prm.lambda_s = std::atof(argv[7]);
      if (argc > 8)
        prm.rho_s = std::atof(argv[8]);
      if (argc > 9)
        prm.R = std::atof(argv[9]);
      if (argc > 10)
        prm.cx = std::atof(argv[10]);
      if (argc > 11)
        prm.cy = std::atof(argv[11]);
      if (argc > 12)
        prm.eta_f = std::atof(argv[12]);
      if (argc > 13)
        prm.pin_center = (std::atoi(argv[13]) != 0);
      if (argc > 14)
        prm.lid = std::atof(argv[14]);
      if (argc > 15)
        prm.scheme = std::atoi(argv[15]);
      prm.gmres_tol = (prm.scheme == 0) ? 1e-8 : 1e-6;

      ImmersedFEM<2> problem(prm);
      problem.run();
    }
  catch (const std::exception &exc)
    {
      std::cerr << "Exception: " << exc.what() << "\n"
                << "Aborting!\n";
      return 1;
    }
  return 0;
}
