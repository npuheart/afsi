"""Run an AV2D pressure-replay cycle, then plot its saved trajectory."""
import argparse
from pathlib import Path
import subprocess
import sys
import math
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--source',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--resume',action='store_true')
p.add_argument('--convection',choices=['explicit','implicit'],default='explicit')
p.add_argument('--advection-refresh',type=int,default=20)
p.add_argument('--nx',type=int,default=120)
p.add_argument('--solid-refinement',type=int,choices=[0,1,2],default=0)
p.add_argument('--dt',type=float,default=5e-6)
p.add_argument('--time',type=float,default=.8512)
a=p.parse_args()
demo=Path(__file__).resolve().parent
cmd=[sys.executable,'-u',str(demo/'main.py'),'--source',str(a.source),'--output',str(a.output),'--nx',str(a.nx),'--dt',str(a.dt),'--time',str(a.time),'--convection',a.convection,'--advection-refresh',str(a.advection_refresh),'--solid-refinement',str(a.solid_refinement),'--max-steps',str(math.ceil(a.time/a.dt)+1),'--solver','coupled','--output-every','200','--snapshot-every','2000','--log-every','200']
if a.resume:cmd.append('--resume')
r=subprocess.run(cmd)
if (a.output/'checkpoint.npz').exists():
    plot=subprocess.run([sys.executable,str(demo/'plot.py'),str(a.output),'--gif'])
    if plot.returncode:print('Plotting failed; numerical status/checkpoints remain available',flush=True)
sys.exit(r.returncode)
