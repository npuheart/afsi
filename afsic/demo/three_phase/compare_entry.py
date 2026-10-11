"""Compare completed water-entry runs with the same physics on two grids."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def read_run(directory):
    params=json.loads((directory/'parameters.json').read_text())
    status=json.loads((directory/'status.json').read_text())
    if params.get('case') != 'water-entry' or not status.get('complete'):
        raise ValueError(f'{directory}: expected a completed water-entry run')
    with (directory/'metrics.csv').open() as stream:
        rows=list(csv.DictReader(stream))
    data={key:np.array([float(row[key]) for row in rows]) for key in rows[0]}
    return params,status,data


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('coarse',type=Path)
    parser.add_argument('fine',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    runs=[read_run(path) for path in (args.coarse,args.fine)]
    keys=['lx','ly','time','dt','shear','bulk','phases','water_level','release_gap','solid_bounds',
          'kernel','anchor_stiffness','anchor_damping','initial_velocity','source_sha256']
    for key in keys:
        if runs[0][0][key] != runs[1][0][key]:
            raise ValueError(f'Cannot isolate refinement: {key} differs')
    if not (runs[1][0]['nx'] > runs[0][0]['nx'] and runs[1][0]['ny'] > runs[0][0]['ny']):
        raise ValueError('The second grid must be finer in both directions')
    args.output.mkdir(parents=True,exist_ok=True)
    labels=[f"{p['nx']} x {p['ny']}" for p,_,_ in runs]
    fig,axes=plt.subplots(2,3,figsize=(13,7),constrained_layout=True)
    for label,(params,status,data) in zip(labels,runs):
        t=data['time']
        axes[0,0].plot(t,data['solid_com_y'],label=label)
        axes[0,1].plot(t,data['solid_com_vy'],label=label)
        axes[0,2].plot(t,data['solid_bottom_y'],label=label)
        axes[1,0].plot(t,data['solid_J_min'],label=label+' min')
        axes[1,0].plot(t,data['solid_J_max'],'--',label=label+' max')
        axes[1,1].plot(t,100*data['solid_area_relative_error'],label=label)
        axes[1,2].plot(t,100*data['density_geometry_relative_l1'],label=label)
    axes[0,2].axhline(runs[0][0]['water_level'],ls=':',color='gray',label='Initial waterline')
    for ax,title in zip(axes.flat,['COM height (m)','COM vertical velocity (m/s)',
                                  'Lowest solid node (m)','Element Jacobian J',
                                  'Solid total area change (%)','Density/geometry L1 (%)']):
        ax.set(xlabel='Time (s)',ylabel=title)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle('Water-entry refinement: same physics, finer fluid/solid grids and adaptive time steps')
    fig.savefig(args.output/'grid_comparison.png',dpi=180)
    plt.close(fig)
    summary={}
    for label,(params,status,data) in zip(labels,runs):
        summary[label]=dict(nx=params['nx'],ny=params['ny'],solid_cells=params['solid_cells'],
            steps=status['step'],nominal_entry_time=status['nominal_entry_time'],
            final={key:float(values[-1]) for key,values in data.items()},
            sampled_J_min=float(data['solid_J_min'].min()),
            sampled_J_max=float(data['solid_J_max'].max()))
    (args.output/'comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
    specs=[('Steps',lambda p,s,d:s['step']),
           ('Fluid cell width (m)',lambda p,s,d:p['lx']/p['nx']),
           ('Solid triangles',lambda p,s,d:2*np.prod(p['solid_cells'])),
           ('Initial-waterline crossing (s)',lambda p,s,d:s['nominal_entry_time']),
           ('Final COM height (m)',lambda p,s,d:d['solid_com_y'][-1]),
           ('Final COM vertical velocity (m/s)',lambda p,s,d:d['solid_com_vy'][-1]),
           ('Final minimum J',lambda p,s,d:d['solid_J_min'][-1]),
           ('Final maximum J',lambda p,s,d:d['solid_J_max'][-1]),
           ('Final solid area change (%)',lambda p,s,d:100*d['solid_area_relative_error'][-1]),
           ('Final density/geometry L1 (%)',lambda p,s,d:100*d['density_geometry_relative_l1'][-1]),
           ('Final relative mass error',lambda p,s,d:d['mass_relative_error'][-1]),
           ('Final maximum divergence (1/s)',lambda p,s,d:d['divergence_max'][-1])]
    lines=['# Water-entry grid refinement','',f'| Metric | {labels[0]} | {labels[1]} |',
           '| --- | ---: | ---: |']
    for title,evaluate in specs:
        numbers=[evaluate(*run) for run in runs]
        lines.append('| '+title+' | '+' | '.join(f'{value:.8g}' for value in numbers)+' |')
    lines += ['', 'Both runs reach the same final time with identical physical parameters and solver source.',
              'Fluid and solid meshes are refined together. The automatic stability limit also reduces',
              'the time step, and the interface width remains 1.5 cells (its physical width decreases).',
              'This is a combined resolution comparison, not an isolated spatial or temporal convergence test.',
              'Two grids do not establish a convergence order. The crossing time refers to the initial',
              'waterline, not exact physical contact. Jacobian errors and density/geometry mismatch must',
              'be assessed alongside global conservation; a completed run is not quantitative validation.', '']
    (args.output/'comparison.md').write_text('\n'.join(lines))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
