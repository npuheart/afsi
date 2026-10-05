#!/bin/bash
# batch.sh DEMO:NX:DT:TEND:PD ...
PY=/Users/pengfei.ma/miniconda3/envs/afsi-dolfinx/bin/python
L=/Users/pengfei.ma/afsi/afsic/demo/ib2d/demo_560/_probe/logs
mkdir -p $L
for spec in "$@"; do
  IFS=: read DEMO NX DT TEND PD <<< "$spec"
  cd /Users/pengfei.ma/afsi/afsic/demo/ib2d/$DEMO || { echo "MISSING $DEMO"; continue; }
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
  echo "[$(date +%H:%M:%S)] $DEMO ref NX=$NX DT=$DT T=$TEND PD=$PD START"
  { time $PY -u run_reference.py --nx $NX --dt $DT --tend $TEND --print-dump $PD ; } > $L/$DEMO.ref.log 2>&1
  RC=$?; FIN=$($PY -c "
import numpy as np,sys
try:
    d=np.load('ib2d_reference.npz')
    print('ref frames=%d finite=%s umax=%.3e' % (d['u'].shape[0], np.all(np.isfinite(d['u'])), np.abs(d['u']).max()))
except Exception as e: print('ref LOAD FAIL', e)")
  echo "[$(date +%H:%M:%S)] $DEMO ref rc=$RC $FIN"
  { time NX=$NX DT=$DT TFINAL=$TEND PRINT_DUMP=$PD $PY -u main.py ; } > $L/$DEMO.afsi.log 2>&1
  RC=$?; FIN=$($PY -c "
import numpy as np
try:
    d=np.load('plot/afsi_result.npz')
    print('afsi frames=%d finite=%s umax=%.3e' % (d['u'].shape[0], np.all(np.isfinite(d['u'])), np.abs(d['u']).max()))
except Exception as e: print('afsi LOAD FAIL', e)")
  echo "[$(date +%H:%M:%S)] $DEMO afsi rc=$RC $FIN"
  $PY compare.py > $L/$DEMO.compare.log 2>&1
  echo "[$(date +%H:%M:%S)] $DEMO compare rc=$?"
done
echo "[$(date +%H:%M:%S)] BATCH DONE"
