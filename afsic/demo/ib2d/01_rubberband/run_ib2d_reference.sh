#!/usr/bin/env bash
# Run the original IB2d "Rubberband_with_Springs" example and convert its
# output to ib2d_reference.npz (consumed by compare.py).
#
#   ./run_ib2d_reference.sh                 # uses MATLAB if found, else Octave
#   ENGINE=octave ./run_ib2d_reference.sh
#   MATLAB=/Applications/MATLAB_R2025b.app/bin/matlab ./run_ib2d_reference.sh
#
# The IB2d sources are untouched: the example is copied to ./ib2d_run and, for
# Octave only, a patched copy of IBM_Blackbox is used (see octave_compat.py).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../../.." && pwd)"
IB2D="${IB2D:-$REPO/third_party/ib2d}"
EXAMPLE="$IB2D/matIB2d/Examples/Example_Standard_Rubberband/Rubberband_with_Springs"
RUN="$HERE/ib2d_run"

if [ ! -d "$IB2D" ]; then
  git clone https://github.com/nickabattista/ib2d "$IB2D"
fi

rm -rf "$RUN" && mkdir -p "$RUN"
cp "$EXAMPLE"/input2d "$EXAMPLE"/rubberband.vertex "$EXAMPLE"/rubberband.spring "$EXAMPLE"/main2d.m "$RUN"/
# headless: no MATLAB figures
sed -i.bak 's/^plot_Matlab = 1/plot_Matlab = 0/' "$RUN/input2d" && rm -f "$RUN/input2d.bak"

ENGINE="${ENGINE:-}"
MATLAB="${MATLAB:-$(command -v matlab || true)}"
if [ -z "$ENGINE" ]; then
  if [ -n "$MATLAB" ]; then ENGINE=matlab; else ENGINE=octave; fi
fi

cd "$RUN"
if [ "$ENGINE" = matlab ]; then
  BLACKBOX="$IB2D/matIB2d/IBM_Blackbox"
  "$MATLAB" -batch "addpath('$BLACKBOX'); main2d"
else
  BLACKBOX="$RUN/IBM_Blackbox_octave"
  python3 "$HERE/octave_compat.py" "$IB2D/matIB2d/IBM_Blackbox" "$BLACKBOX"
  octave --no-gui --quiet --eval "addpath('$BLACKBOX'); main2d"
fi

cd "$HERE"
python3 ib2d_reference.py "$RUN/viz_IB2d" "$HERE/ib2d_reference.npz"
