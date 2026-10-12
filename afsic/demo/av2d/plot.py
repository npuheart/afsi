"""Plot AV2D pilot geometry, prescribed pressures and measured diagnostics."""
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation,PillowWriter
from matplotlib.collections import PolyCollection
import numpy as np

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('output',type=Path)
p.add_argument('--gif',action='store_true')
a=p.parse_args()
paths=sorted(a.output.glob('snapshot_*.npz'))
states=[]
for path in paths:
    with np.load(path) as d:states.append({k:d[k].copy() for k in ['time','x0','x1','cells0','cells1','ref0','ref1']})
if not states:raise ValueError('No saved snapshots')

def panel(ax,state):
    ax.clear()
    for i,color in [(1,'#aaaaaa'),(0,'#ca6633')]:
        x=state[f'x{i}'].reshape(-1,2)-3.84
        ax.add_collection(PolyCollection(x[state[f'cells{i}']],facecolor=color,edgecolor='none'))
    ax.set(xlim=(-3.84,3.84),ylim=(-3.84,3.84),aspect='equal',xlabel='x (cm)',ylabel='y (cm)',title=f"t = {float(state['time']):.5f} s")
fig,axes=plt.subplots(1,3,figsize=(10,4),constrained_layout=True)
for ax,idx in zip(axes,[0,len(states)//2,len(states)-1]):panel(ax,states[idx])
fig.suptitle('AV2D in AFSI: prescribed-pressure pilot (not a validated reproduction)')
fig.savefig(a.output/'sequence.png',dpi=160);plt.close(fig)
with (a.output/'metrics.csv').open() as stream:rows=list(csv.DictReader(stream))
# Restart emits the checkpoint row again; retain the last row for each step.
rows=sorted({int(r['step']):r for r in rows}.values(),key=lambda r:int(r['step']))
d={k:np.array([float(r[k]) for r in rows]) for k in rows[0]}
fig,axs=plt.subplots(2,2,figsize=(10,7),constrained_layout=True)
for i in [0,1]:
    axs[0,0].plot(d['time'],d[f'Jmin{i}'],label=f'Part {i} min J')
    axs[0,0].plot(d['time'],d[f'Jmax{i}'],'--',label=f'Part {i} max J')
    axs[0,1].plot(d['time'],d[f'displacement{i}'],label=f'Part {i}')
for name in ['Q_in_cm2_s','Q_out_cm2_s']:axs[1,0].plot(d['time'],d[name],label=name)
if 'leaflet_gap_cm' in d:axs[1,1].plot(d['time'],d['leaflet_gap_cm'],label='Minimum node gap')
for ax,label in zip(axs.flat,['Jacobian','Max displacement (cm)','2D flow (cm2/s)','Leaflet gap (cm)']):
    ax.set(xlabel='Time (s)',ylabel=label);ax.legend(fontsize=8);ax.grid(alpha=.2)
fig.savefig(a.output/'history.png',dpi=160);plt.close(fig)
if a.gif:
    fig,ax=plt.subplots(figsize=(5,5),constrained_layout=True)
    anim=FuncAnimation(fig,lambda i:panel(ax,states[i]),frames=len(states),interval=120)
    anim.save(a.output/'motion.gif',writer=PillowWriter(fps=8));plt.close(fig)
