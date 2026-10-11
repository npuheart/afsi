"""Continue a water-entry checkpoint to a later time in a separate directory.

The original parameter and source checks in main.py remain active. Only the
requested end time changes; provenance records the original metadata.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--time', type=float, required=True)
    args = parser.parse_args()
    source, out = args.source.resolve(), args.output.resolve()
    params = json.loads((source/'parameters.json').read_text())
    if params['case'] != 'water-entry':
        parser.error('Requires a water-entry checkpoint')
    if not np.isfinite(args.time) or args.time <= params['time']:
        parser.error('New end time must exceed the previous end time')
    if out.exists():
        parser.error('Use a new output directory to preserve existing results')
    with np.load(source/'checkpoint.npz', allow_pickle=False) as state:
        start_time = float(state['time'])
    shutil.copytree(source, out)
    (out/'extension.json').write_text(json.dumps(dict(source=str(source),
        start_time=start_time, original_parameters=params, target_time=args.time), indent=2)+'\n')
    params['time'] = args.time
    (out/'parameters.json').write_text(json.dumps(params, indent=2)+'\n')
    (out/'status.json').write_text(json.dumps(dict(complete=False,
        time=start_time, stop_reason='extension_pending'), indent=2)+'\n')
    # Remove stale plots copied from the shorter run.
    for name in ('water_entry_sequence.png', 'water_entry_history.png', 'water_entry.gif'):
        (out/name).unlink(missing_ok=True)
    demo = Path(__file__).resolve().parent
    command = [sys.executable, str(demo/'main.py'), '--resume', '--output', str(out),
               '--max-steps', '100000', '--max-seconds', '45', '--output-every', '400']
    for key in ('case','nx','ny','dt','time','shear','bulk','anchor_stiffness',
                'rho_gas','rho_solid','water_level','release_gap','kernel'):
        command += ['--'+key.replace('_','-'), str(params[key])]
    if params['snapshots']:
        command.append('--snapshots')
    margin = 3*max(params['lx']/params['nx'], params['ly']/params['ny'])
    for chunk in range(1000):
        with np.load(out/'checkpoint.npz', allow_pickle=False) as state:
            positions = state['solid_positions']
            previous = float(state['time'])
            gap = min(positions[:,0].min(), positions[:,1].min(),
                      params['lx']-positions[:,0].max(), params['ly']-positions[:,1].max())
        if gap <= margin:
            status = dict(complete=False, time=previous, stop_reason='wall_proximity',
                          minimum_wall_gap=float(gap), guard_margin=margin,
                          explanation='Solid-wall contact is not implemented; checkpoint retained.')
            (out/'status.json').write_text(json.dumps(status, indent=2)+'\n')
            print(json.dumps(status), flush=True)
            break
        print(f'CHUNK {chunk+1}: starting from t={previous:.8f}', flush=True)
        result = subprocess.run(command)
        if result.returncode:
            raise SystemExit(result.returncode)
        status = json.loads((out/'status.json').read_text())
        print(json.dumps(status), flush=True)
        if status['complete']:
            break
        if status['time'] <= previous:
            raise RuntimeError('Continuation made no progress')
    else:
        raise RuntimeError('Continuation chunk limit reached')
    if params['snapshots']:
        subprocess.run([sys.executable, str(demo/'plot_entry.py'), str(out), '--gif'], check=True)


if __name__ == '__main__':
    main()
