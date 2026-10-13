# AV2D tuning — 2026-10-13

Physical geometry, material constants, support stiffness, pressure waveform and CGS units were retained.
The fluid discretization and open-boundary treatment were changed; this is a numerical stability study.

| Experiment | Outcome / time (s) |
| --- | --- |
| Chorin 120 | Failed at 0.004105 |
| Chorin 120, half dt | Failed at 0.004035 |
| Chorin + original-style feedback | Failed at 0.00161 |
| IPCS + feedback | Failed at 0.00047 |
| Coupled 60, no backflow term | Failed at 0.012705 |
| Coupled 60, stabilized | Completed requested interval at 0.05 |
| Coupled 120, stabilized | Completed requested interval at 0.05 |
| Coupled 60, stabilized, half dt | Completed requested interval at 0.01 |

The original failure is in the aortic wall at step 821. Peak velocity occurs at the upper pressure-opening edge.
Halving dt does not remove the failure. Feedback alone makes the failure earlier; adding it is not the adopted fix.

## Adopted change

`coupled.py` solves the mixed Q2/Q1 velocity–pressure system together, applies prescribed pressure as traction,
uses backward-Euler viscosity, explicit skew convection, and an energy-dissipating term on inward traction-boundary flow.
The linear matrix is factored with MUMPS and reused. The extra boundary term is a numerical open-boundary model;
it is not the original IBAMR FeedbackForcer. Existing AFSI library solvers are unchanged.

## Completed 120×120 run to 0.05 s

| Quantity | Final value |
| --- | ---: |
| Steps | 10000 |
| Leaflet J | [0.9723296, 1.021106] |
| Aorta J | [0.9459024, 1.042021] |
| Maximum leaflet displacement (mm) | 3.155285 |
| Peak speed (cm/s) | 549.9341 |
| Inlet / outlet flow (cm²/s, +y positive) | -319.714736 / -319.714736 |
| Relative flow imbalance | 1.51e-14 |

## Checks

- Uniform pressure (100 dyn/cm²) on both traction ends: peak velocity ≈ 5e-17 cm/s; recovered pressure agrees with the prescribed constant.
- Continuous 40 versus 20+20 resumed steps: solid coordinates identical; velocity/pressure agree to roundoff (about 1e-16 / 3e-12).
- Coarse-grid dt-halving at 0.01 s: outlet-flow difference 0.0219%.
- Both stabilized grids completed 10,000 steps to 0.05 s without element inversion.

## Limits and continuation

The coarse 60×60 aorta has final minimum J≈0.711, versus ≈0.946 on 120×120. Local volume error and
resolution sensitivity remain important; global flow balance does not establish local accuracy. Peak jet speeds are large,
and valve contact, sealing/leakage, pressure/flow accuracy and a whole-cycle solution are not validated.

The new 120×120 full-cycle run targets 0.8512 s in `results/cycle_120_coupled/`.
It is a restarted attempt, not a completed heartbeat. Check `status.json` for current progress and `complete=true`.
It saves checkpoints every 200 steps, snapshots every 2000, and includes PID/UTC heartbeat and exact failure-step diagnostics.
Based on the 0.05 s test, allow roughly 1–2 hours; runtime and long-time stability may change.

All generated data and experimental logs are local, ignored by Git. Successful test outputs are
`results/tuning/backflow120/` and `results/tuning/backflow60/`; comparison plot is
`results/tuning/comparison/tuning_comparison.png`.

## Reproduce

```bash
AFSI_OFFLINE=1 python afsic/demo/av2d/main.py \
  --source /Users/pengfei/Documents/GitHub/WholeLVHeartModelling/AV2D \
  --output /tmp/av2d-stabilized-test --nx 120 --dt 5e-6 --time .05 --solver coupled
python afsic/demo/av2d/plot.py /tmp/av2d-stabilized-test --gif
```

## Streamlines, structural refinement and late-time convection test

The original coupled full-cycle attempt ultimately failed at step 112624, t=0.563120 s:
aorta Jmin=-2.483. At its last checkpoint (0.563000 s), peak speed was 3839.52 cm/s
at (x,y)=(0.768,1.696) cm. Aorta displacement was only 0.00450 cm.
Thus the observed inversion alone does not establish insufficient structural node density as its cause.

Every saved-frame animation now includes masked instantaneous streamlines and a fixed-scale speed background.
The old failed trajectory remains in `results/cycle_120_coupled/` with its updated `motion.gif`.

New controls from exactly that last checkpoint, unchanged original solid meshes and dt=2.5e-6:

| Convection | Result |
| --- | --- |
| Explicit | Failed at 0.563195 s, aorta Jmin=-101.183 |
| Frozen implicit Oseen, refreshed every 20 steps | Completed diagnostic target 0.565 s |

At 0.565 s the implicit diagnostic had peak speed 969.686 cm/s, leaflet J=[0.971412,1.025860],
aorta J=[0.730165,1.211903]. This demonstrates passage through the old failure time, not long-time validity.
The inherited checkpoint already contains large oscillations; it is not a clean full-cycle solution.
Output folders: `results/tuning/late_explicit120_halfdt/`, `results/tuning/late_implicit120/`.

Uniform solid refinement halves typical boundary MFAC from about 1 to 0.5 on the 120 Q2 grid.
Initial boundary maxima: leaflets 0.604358, aorta 0.563279. Reference J and near-zero force checks passed;
leaflet root side-set facets doubled from 10 to 20. A refined explicit half-dt smoke test completed 0.01 s
(4000 steps), leaflet J=[0.994305,1.005599], aorta J=[0.980823,1.029972].
Output: `results/tuning/refined120_halfdt_smoke/`.

Implicit uniform-pressure equilibrium passed (max velocity 3.28e-17 cm/s). Continuous 40 versus 19+21
restart steps passed, with max velocity/pressure differences 1.44e-16 / 2.05e-12; the test splits inside
an advector refresh interval and checks its saved state as well.

A fresh full-cycle experiment starts from t=0 in `results/cycle_120_refined_implicit/`:
fluid 120², solids uniformly refined once, dt=2.5e-6, implicit convection, refresh interval 20.
It requires 340480 steps. This version refactors a mixed matrix every 20 steps, so allow several hours;
check `status.json` rather than assuming completion. The launcher now derives its maximum step count
from the target time, avoiding the old 200000-step cap at half dt. This is a candidate configuration,
not an established converged or impermeable valve solution.
