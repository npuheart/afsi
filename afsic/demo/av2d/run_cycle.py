"""Run the AV2D pilot and plot its saved trajectory on completion or failure."""
import argparse
from pathlib import Path
import subprocess
import sys
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--source',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--resume',action='store_true')
a=p.parse_args()
demo=Path(__file__).resolve().parent
cmd=[sys.executable,'-u',str(demo/'main.py'),'--source',str(a.source),'--output',str(a.output),'--nx','120','--dt','0.000005','--time','0.8512','--output-every','1000']
if a.resume:cmd.append('--resume')
r=subprocess.run(cmd)
if (a.output/'checkpoint.npz').exists():
    plot=subprocess.run([sys.executable,str(demo/'plot.py'),str(a.output),'--gif'])
    if plot.returncode:print('Plotting failed; numerical status/checkpoints remain available',flush=True)
sys.exit(r.returncode)
