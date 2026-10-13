"""Check continuous versus checkpoint/restart integration for the coupled solver."""
import argparse
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('source',type=Path)
p.add_argument('--convection',choices=['explicit','implicit'],default='explicit')
a=p.parse_args()
with tempfile.TemporaryDirectory(prefix='av2d-restart-') as directory:
    root=Path(directory)
    base=[sys.executable,str(Path(__file__).with_name('main.py')),'--source',str(a.source),'--nx','16','--time','.0002','--output-every','19','--log-every','19','--convection',a.convection]
    for folder,steps,resume in [('continuous',40,False),('split',19,False),('split',21,True)]:
        cmd=base+['--output',str(root/folder),'--max-steps',str(steps)]+(['--resume'] if resume else [])
        subprocess.run(cmd,check=True,stdout=subprocess.DEVNULL)
    with np.load(root/'continuous/checkpoint.npz') as x,np.load(root/'split/checkpoint.npz') as y:
        for key in x.files:
            if key in ['u','advector']:np.testing.assert_allclose(x[key],y[key],atol=1e-12,rtol=1e-10)
            elif key=='p':np.testing.assert_allclose(x[key],y[key],atol=1e-8,rtol=1e-10)
            else:np.testing.assert_array_equal(x[key],y[key])
        print('Restart check passed; max velocity difference:',np.max(np.abs(x['u']-y['u'])),'max pressure difference:',np.max(np.abs(x['p']-y['p'])))
