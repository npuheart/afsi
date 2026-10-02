"""Offline single-rank immersed FSI comparison: RT nodal vs existing IB4.

Same nodal solid load assembly and explicit coordinate update as demo_402.
Default: a pre-stretched elastic box in a closed unit fluid box.
--case turek loads demo_402's existing cylinder/tail solid mesh (CGS units).
No cloud logging, email or external data upload is performed.
"""
import argparse
import csv
import json
from pathlib import Path
import time
import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem, mesh, io
from dolfinx.fem.petsc import assemble_vector
from afsic import RTFluidSolver, RTNodalCoupling, ChorinSolver, IBMesh, IBInterpolation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=['rt', 'ib'], default='rt')
    parser.add_argument('--case', choices=['elastic-box', 'turek'], default='elastic-box')
    parser.add_argument('--nx', type=int, default=16)
    parser.add_argument('--ny', type=int)
    parser.add_argument('--solid-n', type=int, default=6)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--dt', type=float)
    parser.add_argument('--convection', action='store_true')
    parser.add_argument('--out', type=Path, default=Path('rt-results'))
    parser.add_argument('--inlet-speed', type=float, default=200.)
    parser.add_argument('--ramp-time', type=float, default=2.)
    parser.add_argument('--kappa-hat', type=float, default=0.1)
    args = parser.parse_args()
    if MPI.COMM_WORLD.size != 1:
        parser.error('This prototype supports a single MPI rank only')
    if args.nx < 2 or args.steps < 1 or args.solid_n < 1 or args.ramp_time <= 0:
        parser.error('Invalid mesh, step count or ramp-time')
    is_turek = args.case == 'turek'
    solid_path = Path(__file__).resolve().parent.parent / 'demo_402' / 'turek_mesh.xdmf'
    if is_turek and (not solid_path.exists() or not solid_path.with_suffix('.h5').exists()):
        parser.error('Generate the Turek mesh first: cd demo/demo_402 && python generate_mesh.py')
    Lx, Ly = (246., 41.) if is_turek else (1., 1.)
    ny = args.ny or (max(4, round(args.nx / 6)) if is_turek else args.nx)
    dt = args.dt or (5e-5 if is_turek else 1e-3)
    if dt <= 0 or ny < 2:
        parser.error('dt must be positive and ny >= 2')
    rho, mu = 1., (10. if is_turek else 0.1)
    fluid = mesh.create_rectangle(MPI.COMM_WORLD, ((0., 0.), (Lx, Ly)),
                                  (args.nx, ny), cell_type=mesh.CellType.quadrilateral,
                                  ghost_mode=mesh.GhostMode.shared_facet)
    fdim = fluid.topology.dim - 1
    fluid.topology.create_connectivity(fdim, fluid.topology.dim)
    def boundary(x):
        return np.isclose(x[0], 0) | np.isclose(x[1], 0) | np.isclose(x[1], Ly)
    facets = (mesh.locate_entities_boundary(fluid, fdim, boundary) if is_turek
              else mesh.exterior_facet_indices(fluid.topology))
    if args.method == 'rt':
        solver = RTFluidSolver(fluid, dt, rho, mu, degree=2,
                               convection=args.convection, dirichlet_facets=facets)
        V, Q = solver.V, solver.Q
        transfer = RTNodalCoupling(V)
    else:
        # Set the existing solver's module switch explicitly for this run.
        import importlib
        chorin_module = importlib.import_module('afsic.euler.ChorinSolver')
        chorin_module._NO_CONVECTION = not args.convection
        V = fem.functionspace(fluid, ('Lagrange', 2, (2,)))
        Q = fem.functionspace(fluid, ('Lagrange', 1))
        u_D = fem.Function(V)
        bcu = [fem.dirichletbc(u_D, fem.locate_dofs_topological(V, fdim, facets))]
        if is_turek:
            outfacets = mesh.locate_entities_boundary(fluid, fdim, lambda x: np.isclose(x[0], Lx))
            pdofs = fem.locate_dofs_topological(Q, fdim, outfacets)
        else:
            pdofs = fem.locate_dofs_geometrical(Q, lambda x: np.isclose(x[0], 0) & np.isclose(x[1], 0))
        bcp = [fem.dirichletbc(PETSc.ScalarType(0), pdofs, Q)]
        solver = ChorinSolver(V, Q, bcu, bcp, dt, rho, mu, ib_body_force=False)
        solver.u_D = u_D
        # Tight tolerances for diagnostic comparisons, without changing library defaults.
        for ksp in (solver.solver1, solver.solver2, solver.solver3):
            ksp.setTolerances(rtol=1e-11, atol=1e-13)
            ksp.setErrorIfNotConverged(True)
        lattice = IBMesh(0., Lx, 0., Ly, args.nx, ny, 2)
        transfer = IBInterpolation(lattice)
        coords = fem.Function(V)
        coords.interpolate(lambda x: np.vstack((x[0], x[1])))
        lattice.build_map(coords._cpp_object)
        dV = (Lx / (2 * args.nx)) * (Ly / (2 * ny))
    if is_turek:
        with io.XDMFFile(MPI.COMM_WORLD, str(solid_path), 'r') as xdmf:
            solid = xdmf.read_mesh(name='mesh')
            solid.topology.create_connectivity(solid.topology.dim, solid.topology.dim - 1)
            cells = xdmf.read_meshtags(solid, name='cell_tags')
        dx_s = ufl.Measure('dx', domain=solid, subdomain_data=cells)
        mu_s, lam_s = 1.e7, 8.e7
    else:
        solid = mesh.create_rectangle(MPI.COMM_WORLD, ((0.3, 0.3), (0.7, 0.7)),
                                     (args.solid_n, args.solid_n), cell_type=mesh.CellType.triangle)
        dx_s = ufl.Measure('dx', domain=solid)
        mu_s, lam_s = 1., 10.
    Vs = fem.functionspace(solid, ('Lagrange', 2, (2,)))
    chi, U, L_function = (fem.Function(Vs) for _ in range(3))
    reference = Vs.tabulate_dof_coordinates()[:, :2].copy()
    if is_turek:
        chi.interpolate(lambda x: np.vstack((x[0], x[1])))
    else:
        chi.interpolate(lambda x: np.vstack((0.5 + 1.1 * (x[0] - 0.5),
                                              0.5 + (x[1] - 0.5) / 1.1)))
    v_s = ufl.TestFunction(Vs)
    F = ufl.grad(chi)
    strain = 0.5 * (F.T * F - ufl.Identity(2))
    P = F * (lam_s * ufl.tr(strain) * ufl.Identity(2) + 2 * mu_s * strain)
    # Identical Saint Venant--Kirchhoff weak nodal force principle to demo_402.
    load_expr = -ufl.inner(P, ufl.grad(v_s)) * dx_s
    if is_turek:
        beta = args.kappa_hat * rho * (Lx / args.nx) / dt**2
        load_expr -= beta * ufl.inner(chi - ufl.SpatialCoordinate(solid), v_s) * dx_s(1)
    load_form = fem.form(load_expr)
    volume_form = fem.form(ufl.det(F) * dx_s)
    refvolume = fem.assemble_scalar(fem.form(fem.Constant(solid, PETSc.ScalarType(1.)) * dx_s))
    initial_volume = fem.assemble_scalar(volume_form)
    initial_positions = chi.x.array.reshape(-1,2).copy()
    div_form = fem.form(ufl.div(solver.u_)**2 * ufl.dx)
    kinetic_form = fem.form(0.5 * rho * ufl.inner(solver.u_, solver.u_) * ufl.dx)
    # P2 triangle deformation gradients: sample vertices, edge midpoints and centroid.
    J_expr = fem.Expression(ufl.det(F), np.array([[0.,0.],[1.,0.],[0.,1.],
                                                [.5,0.],[0.,.5],[.5,.5],[1/3,1/3]]))
    solid_cells = np.arange(solid.topology.index_map(solid.topology.dim).size_local, dtype=np.int32)
    args.out.mkdir(parents=True, exist_ok=True)
    history = []
    start = time.perf_counter()
    b_ib = solver.u_.x.petsc_vec.duplicate()
    for step in range(args.steps):
        t = (step + 1) * dt
        if is_turek:
            ramp = .5 * (1 - np.cos(np.pi * min(t / args.ramp_time, 1.)))
            solver.u_D.interpolate(lambda x: np.vstack((
                6 * args.inlet_speed * ramp * x[1] * (Ly - x[1]) / Ly**2,
                np.zeros(x.shape[1]))))
        # Both operations use chi_n. Assemble -> spread -> solve -> sample -> update.
        load = assemble_vector(load_form)
        load.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        if args.method == 'rt':
            transfer.update(chi)
            transfer.spread(load, b_ib)
        else:
            transfer.evaluate_current_points(chi._cpp_object)
            L_function.x.array[:] = load.getArray(readonly=True)
            transfer.solid_to_fluid(solver.f._cpp_object, L_function._cpp_object)
            b_ib.getArray()[:] = dV * solver.f.x.array
        solver.ib_load = b_ib
        solver.solve_one_step()
        if args.method == 'rt':
            transfer.interpolate(solver.u_, U)
        else:
            transfer.fluid_to_solid(solver.u_._cpp_object, U._cpp_object)
        fluid_power = float(solver.u_.x.array @ b_ib.getArray(readonly=True))
        solid_power = float(U.x.array @ load.getArray(readonly=True))
        power_error = abs(fluid_power - solid_power) / max(1., abs(fluid_power), abs(solid_power))
        chi.x.array[:] += dt * U.x.array
        volume = fem.assemble_scalar(volume_form)
        min_j = float(np.min(J_expr.eval(solid, solid_cells)))
        row = dict(step=step+1, time=t, div_l2=float(np.sqrt(max(0., fem.assemble_scalar(div_form)))),
                   power_relative_error=power_error, volume=volume,
                   volume_drift=(volume-initial_volume)/initial_volume,
                   mean_J=volume/refvolume, sampled_min_J=min_j,
                   kinetic_energy=float(fem.assemble_scalar(kinetic_form)),
                   max_solid_speed=float(np.max(np.linalg.norm(U.x.array.reshape(-1,2), axis=1))),
                   elapsed=time.perf_counter()-start)
        if not all(np.isfinite(value) for value in row.values()) or min_j <= 0:
            raise RuntimeError(f'Nonfinite/inverted solid at step {step+1}: {row}')
        history.append(row)
        load.destroy()
    with (args.out / 'history.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    np.savez(args.out / 'final_state.npz', solid_reference=reference,
             solid_positions=chi.x.array.reshape(-1,2), solid_velocity=U.x.array.reshape(-1,2),
             fluid_coefficients=solver.u_.x.array, pressure_coefficients=solver.p_.x.array)
    summary = dict(method=args.method, case=args.case, nx=args.nx, ny=ny, dt=dt,
                   steps=args.steps, convection=args.convection,
                   inlet_speed=args.inlet_speed, ramp_time=args.ramp_time, kappa_hat=args.kappa_hat,
                   solid_n=args.solid_n, fluid_velocity_dofs=solver.u_.x.array.size,
                   fluid_pressure_dofs=solver.p_.x.array.size,
                   max_div_l2=max(r['div_l2'] for r in history),
                   max_power_error=max(r['power_relative_error'] for r in history),
                   final_volume_drift=history[-1]['volume_drift'],
                   min_sampled_J=min(r['sampled_min_J'] for r in history),
                   max_solid_displacement=float(np.max(np.linalg.norm(chi.x.array.reshape(-1,2)-reference,axis=1))),
                   max_displacement_since_start=float(np.max(np.linalg.norm(chi.x.array.reshape(-1,2)-initial_positions,axis=1))),
                   runtime_seconds=time.perf_counter()-start)
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    b_ib.destroy()


if __name__ == '__main__':
    main()
