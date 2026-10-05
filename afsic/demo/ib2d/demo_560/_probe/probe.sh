PY=/Users/pengfei.ma/miniconda3/envs/afsi-dolfinx/bin/python
cd /Users/pengfei.ma/afsi/afsic/demo/ib2d/demo_560
for DT in 2e-5 5e-5 1e-4; do
  N=$($PY -c "n=0;t=0.0;dt=$DT
while n<150: t+=dt; n+=1
print(n)")
  ${PY} run_reference.py --nx 48 --dt $DT --tend 1e-1 --print-dump 1000 --out _probe/ref_$DT.npz > _probe/log_$DT.txt 2>&1
  $PY - "$DT" <<'PYEOF'
import numpy as np, sys
dt = sys.argv[1]
d = np.load("_probe/ref_%s.npz" % dt, allow_pickle=True)
u = d["u"]; X = d["X"]
print("dt=%s frames=%d  u finite=%s  max|u|=%s  X finite=%s" % (
    dt, u.shape[0], np.all(np.isfinite(u)),
    np.nanmax(np.abs(u)) if not np.all(np.isfinite(u)) else np.abs(u).max(),
    np.all(np.isfinite(X))))
PYEOF
done
echo PROBE_DONE
