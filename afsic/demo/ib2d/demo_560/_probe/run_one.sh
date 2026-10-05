#!/bin/bash
# usage: run_one.sh DEMO NX DT TEND PD
set -u
PY=/Users/pengfei.ma/miniconda3/envs/afsi-dolfinx/bin/python
DEMO=$1; NX=$2; DT=$3; TEND=$4; PD=$5
L=/Users/pengfei.ma/afsi/afsic/demo/ib2d/demo_560/_probe/logs
mkdir -p $L
cd /Users/pengfei.ma/afsi/afsic/demo/ib2d/$DEMO || exit 9
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "[$(date +%H:%M:%S)] $DEMO ref NX=$NX DT=$DT T=$TEND PD=$PD start"
{ time $PY -u run_reference.py --nx $NX --dt $DT --tend $TEND --print-dump $PD ; } \
    > $L/$DEMO.ref.log 2>&1
echo "[$(date +%H:%M:%S)] $DEMO afsi start"
{ time NX=$NX DT=$DT TFINAL=$TEND PRINT_DUMP=$PD $PY -u main.py ; } \
    > $L/$DEMO.afsi.log 2>&1
echo "[$(date +%H:%M:%S)] $DEMO compare start"
$PY compare.py > $L/$DEMO.compare.log 2>&1
echo "[$(date +%H:%M:%S)] $DEMO done rc=$?"
