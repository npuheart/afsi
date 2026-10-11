# Numerical verification: experimental three-phase backend

Date: 2026-10-11. Environment: macOS arm64, Python 3.12, DOLFINx 0.10.0,
NumPy 2.5.3, SciPy 1.18.1. Serial execution, `AFSI_OFFLINE=1`.
All runs in the table use the final default composite B-spline kernel.

## Automated checks

`python -m pytest afsic/tests/test_multiphase.py -q`: **14 passed**.

Coverage: pure-phase material limits; signed distance orientation/translation;
planar-interface reinitialization and liquid volume outside the solid;
variable-density projection at ratios 1 and 1000; air/water hydrostatic balance;
constant-velocity preservation under consistent mass/momentum transport;
closed-tank mass conservation and positivity; viscous dissipation;
discrete coupling power, constant reproduction and force balance for both
kernels; continuous divergence of the composite B-spline interpolant;
solid rigid-motion invariance and finite-difference energy/load agreement;
and an actually deforming, coupled three-phase step.

## Complete simulation checks

Numbers below are **final-state diagnostics**, not temporal maxima or convergence
rates. Volume error is the geometric liquid volume after the explicit global
correction. Total density mass is independently conserved by transport.

| Case | Steps | Time (s) | Relative mass error | Liquid area error (m²) | Max divergence (1/s) | Min solid J | Density/geometry relative L1 | Complete |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 32 x 24 smoke | 20 | 0.002 | -3.295e-16 | 0.000e+00 | 2.678e-12 | 0.999967 | 0.09% | True |
| 48 x 32 coarse impact | 1860 | 0.15 | 6.614e-16 | -5.551e-17 | 2.555e-12 | 0.679406 | 10.34% | True |
| 96 x 64 default | 3719 | 0.15 | 8.285e-16 | 0.000e+00 | 2.028e-11 | 0.658522 | 9.16% | True |

The 48 x 32 plate is under-resolved. Even on the default grid, cellwise solid
volume error and density/geometry mismatch remain material limitations.
A positive J only shows that elements have not inverted; it does **not** show
that local incompressibility or displacement accuracy is satisfactory.
The two meshes are not a convergence study. No pressure/displacement comparison
with the paper's experiments has been performed. These are executable baseline
checks, not validation of the published high-order multiphase TS-IFED method.

## Restart and execution limits

A 20-step run at 32 x 24, dt=1e-4, T=.002 was compared with a 7-step run plus
13 resumed steps. Every array/scalar in `checkpoint.npz` was **bitwise identical**.
A resume attempt with changed shear modulus was rejected; the original checkpoint
SHA-256 remained unchanged. The default-grid run also exercised repeated 45-second
execution limits and real checkpoint reloads.

From the repository root, the run commands were:

```bash
conda activate afsi-dolfinx
export AFSI_OFFLINE=1
python -m pytest afsic/tests/test_multiphase.py -q
python afsic/demo/three_phase/main.py --nx 32 --ny 24 --time .002 \
  --dt .0001 --max-steps 30 --output /tmp/three-phase-smoke
python afsic/demo/three_phase/main.py --nx 48 --ny 32 --time .15 \
  --max-steps 2200 --max-seconds 45 --output-every 100 --output /tmp/three-phase-coarse
python afsic/demo/three_phase/main.py --time .15 --max-steps 5000 \
  --max-seconds 45 --output-every 200 --output /tmp/three-phase-default
# Repeat with --resume until status.json says complete=true:
python afsic/demo/three_phase/main.py --time .15 --max-steps 5000 \
  --max-seconds 45 --output-every 200 --output /tmp/three-phase-default --resume
python afsic/demo/three_phase/plot.py /tmp/three-phase-default
```

Generated checkpoints, CSV histories, figures and parameter/source hashes are
kept locally under this demo's ignored `results/validation/` directory. They are
not part of the Git commit; the scripts and this verification record are tracked.
