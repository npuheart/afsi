"""Bounded 2D three-phase examples: supported plate or free elastic water entry.

Run with the afsi-dolfinx environment. See README.md before interpreting results.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
import shutil

import numpy as np
from mpi4py import MPI
from dolfinx import mesh

from afsic.multiphase import MACGrid, PhaseProperties, ElasticSolid2D, ThreePhaseSimulation
from afsic.multiphase.water_entry import create_water_entry, entry_diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("dam-break", "water-entry"), default="dam-break")
    parser.add_argument("--nx", type=int)
    parser.add_argument("--ny", type=int, default=64)
    parser.add_argument("--dt", type=float, default=1e-4, help="maximum step; reduced automatically by stability estimates")
    parser.add_argument("--time", type=float)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--max-seconds", type=float, default=50.)
    parser.add_argument("--output-every", type=int, default=20)
    parser.add_argument("--output", type=Path, default=Path("results/three_phase"))
    parser.add_argument("--shear", type=float)
    parser.add_argument("--bulk", type=float)
    parser.add_argument("--anchor-stiffness", type=float)
    parser.add_argument("--rho-gas", type=float, default=1.2)
    parser.add_argument("--rho-solid", type=float)
    parser.add_argument("--water-level", type=float, default=.35)
    parser.add_argument("--release-gap", type=float, default=.11)
    parser.add_argument("--snapshots", action="store_true", help="save time-indexed states at each output interval")
    parser.add_argument("--kernel", choices=("bspline", "peskin4"), default="bspline")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    dropping = args.case == "water-entry"
    for name, value in dict(nx=64 if dropping else 96, time=.25 if dropping else .15,
                            shear=2000. if dropping else 2e4, bulk=20000. if dropping else 1e5,
                            rho_solid=1200. if dropping else 1100.,
                            anchor_stiffness=0. if dropping else 1e7).items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    if MPI.COMM_WORLD.size != 1:
        parser.error("This initial backend is serial; do not launch with mpirun -n > 1")
    if (not np.isfinite([args.dt, args.time, args.max_seconds]).all()
            or min(args.dt, args.time, args.max_seconds, args.max_steps, args.output_every) <= 0):
        parser.error("Time, dt, execution limits and output interval must be positive")
    if dropping:
        if args.anchor_stiffness != 0.:
            parser.error("--anchor-stiffness is only for dam-break; water-entry always has zero support")
        sim, case_info = create_water_entry(args.nx, args.ny, args.shear, args.bulk,
            args.rho_gas, args.rho_solid, args.water_level, args.release_gap, args.kernel)
        grid, solid = sim.grid, sim.solid
    else:
        grid = MACGrid(args.nx, args.ny, 1.2, .8)
        properties = PhaseProperties(rho_gas=args.rho_gas, rho_solid=args.rho_solid)
        h = min(grid.dx, grid.dy)
        nx_s, ny_s = max(2, int(np.ceil(.06/(.5*h)))), max(4, int(np.ceil(.35/(.5*h))))
        domain = mesh.create_rectangle(MPI.COMM_SELF, [[.62, .05], [.68, .40]], [nx_s, ny_s], cell_type=mesh.CellType.triangle)
        solid = ElasticSolid2D(domain, args.shear, args.bulk, properties.rho_solid,
                               anchor=lambda x: x[1] < .09,
                               anchor_stiffness=args.anchor_stiffness, anchor_damping=1e4)
        x, y = grid.coordinates()
        sim = ThreePhaseSimulation(grid, solid, np.maximum(x-.58, y-.55), properties, kernel=args.kernel)
        case_info = dict(lx=grid.lx, ly=grid.ly, phases=vars(properties), solid_cells=[nx_s, ny_s])
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    parameters = {k: v for k, v in vars(args).items() if k not in ("output", "resume", "max_steps", "max_seconds", "output_every")}
    parameters.update(case_info)
    import afsic.multiphase as backend
    sources = [Path(__file__)] + sorted(Path(backend.__file__).parent.glob("*.py"))
    parameters["source_sha256"] = hashlib.sha256(b"".join(p.read_bytes() for p in sources)).hexdigest()
    metadata, checkpoint, history = out/"parameters.json", out/"checkpoint.npz", out/"metrics.csv"
    nominal_entry_time = -1.
    if args.resume:
        if not metadata.exists() or not checkpoint.exists() or json.loads(metadata.read_text()) != parameters:
            raise ValueError("Resume requires matching parameters, source, and checkpoint")
        with np.load(checkpoint, allow_pickle=False) as data:
            if not np.array_equal(data["solid_reference"], solid.reference):
                raise ValueError("Checkpoint solid mesh differs")
            for key in ("u", "v", "p", "rho", "mu"):
                getattr(sim.fluid, key)[:] = data[key]
            sim.phi[:], sim.psi[:] = data["phi"], data["psi"]
            solid.positions[:] = data["solid_positions"]
            solid.velocity.x.array[:] = data["solid_velocity"].ravel()
            sim.fluid.time, sim.step_number = float(data["time"]), int(data["step"])
            sim.liquid_volume0, sim.mass0 = float(data["liquid_volume0"]), float(data["mass0"])
            sim.last_volume_shift = float(data["last_volume_shift"])
            nominal_entry_time = float(data["nominal_entry_time"])
        sim.coupling.update(solid.positions)
        # A previous process may have written metrics after its last checkpoint.
        # Keep only committed steps; the checkpoint's own row is emitted below.
        if history.exists():
            with history.open() as stream:
                reader = csv.DictReader(stream)
                names = reader.fieldnames
                kept = [row for row in reader if int(row["step"]) < sim.step_number]
            with history.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=names)
                writer.writeheader()
                writer.writerows(kept)
    else:
        if metadata.exists() or checkpoint.exists() or history.exists():
            raise ValueError("Output already exists; use --resume or a new directory")
        metadata.write_text(json.dumps(parameters, indent=2)+"\n")

    def save():
        values = {k: getattr(sim.fluid, k) for k in ("u", "v", "p", "rho", "mu")}
        values.update(phi=sim.phi, psi=sim.psi, solid_positions=solid.positions,
                      solid_reference=solid.reference, solid_cells=solid.cells,
                      solid_velocity=solid.velocity.x.array.reshape(-1, 2),
                      time=sim.fluid.time, step=sim.step_number,
                      liquid_volume0=sim.liquid_volume0, mass0=sim.mass0,
                      last_volume_shift=sim.last_volume_shift, nominal_entry_time=nominal_entry_time)
        temporary = out/"checkpoint.tmp"
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **values)
        temporary.replace(checkpoint)
        if args.snapshots:
            shutil.copy2(checkpoint, out/f"snapshot_{sim.step_number:08d}.npz")

    def diagnostics(row=None):
        nonlocal nominal_entry_time
        row = sim.diagnostics() if row is None else row
        if dropping:
            row.update(entry_diagnostics(sim))
            if nominal_entry_time < 0 and row["nominal_waterline_gap"] <= 0:
                nominal_entry_time = sim.fluid.time
            row["nominal_entry_time"] = nominal_entry_time
        return row

    def log(row):
        exists = history.exists()
        with history.open("a", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            if not exists:
                writer.writeheader()
            writer.writerow(row)
        print(f"step={row['step']} t={row['time']:.6g} mass_err={row['mass_relative_error']:.2e} "
              f"div={row['divergence_max']:.2e} Jmin={row['solid_J_min']:.6g} "
              f"rho_mismatch={row['density_geometry_relative_l1']:.3g}", flush=True)

    log(diagnostics())
    save()
    started = time.monotonic()
    reason = "max_steps"
    last_logged = sim.step_number
    try:
        for _ in range(args.max_steps):
            remaining = args.time-sim.fluid.time
            if remaining <= 1e-14*max(1., args.time):
                reason = "complete"
                break
            dt = min(args.dt, sim.stable_dt(), remaining)
            row = diagnostics(sim.step(dt))
            if sim.step_number % args.output_every == 0:
                log(row)
                save()
                last_logged = sim.step_number
            if time.monotonic()-started >= args.max_seconds:
                reason = "max_seconds"
                break
        if sim.fluid.time >= args.time-1e-14*max(1., args.time):
            reason = "complete"
        if sim.step_number != last_logged:
            log(diagnostics())
        save()
        (out/"status.json").write_text(json.dumps(dict(step=sim.step_number, time=sim.fluid.time,
            complete=reason == "complete", stop_reason=reason, nominal_entry_time=nominal_entry_time,
            loop_seconds=time.monotonic()-started), indent=2)+"\n")
    except Exception as exc:
        # Keep the last good atomic checkpoint; never save partially failed state.
        (out/"status.json").write_text(json.dumps(dict(complete=False, stop_reason="error", error=str(exc)), indent=2)+"\n")
        raise
    finally:
        solid.close()


if __name__ == "__main__":
    main()
