"""Compare AV2D tuning histories; each input label is paired with a directory."""
import argparse
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--run',nargs=2,action='append',metavar=('LABEL','DIRECTORY'),required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
for label,path in a.run:
    with (Path(path)/'metrics.csv').open() as f:rows=list(csv.DictReader(f))
    rows=sorted({int(r['step']):r for r in rows}.values(),key=lambda r:int(r['step']))
    d={k:np.array([float(r[k]) for r in rows]) for k in rows[0]}
    t=d['time']*1000
    axes[0,0].semilogy(t,np.maximum(d['max_speed'],1e-8),label=label)
    imbalance=np.abs(d['Q_in_cm2_s']-d['Q_out_cm2_s'])/np.maximum.reduce([np.abs(d['Q_in_cm2_s']),np.abs(d['Q_out_cm2_s']),np.full_like(t,1e-14)])
    axes[0,1].semilogy(t,np.maximum(imbalance,1e-16),label=label)
    axes[1,0].plot(t,d['Jmin1'],label=label)
    axes[1,1].plot(t,d['displacement0']*10,label=label)
for ax,y in zip(axes.flat,['Peak fluid speed (cm/s)','Relative inlet/outlet imbalance','Minimum aorta J','Maximum leaflet displacement (mm)']):
    ax.set(xlabel='Time (ms)',ylabel=y);ax.legend(fontsize=8);ax.grid(alpha=.2)
axes[1,0].axhline(0,color='gray',ls=':')
fig.suptitle('AV2D numerical tuning: initial 50 ms; physical parameters unchanged')
a.output.mkdir(exist_ok=True,parents=True)
fig.savefig(a.output/'tuning_comparison.png',dpi=180)
