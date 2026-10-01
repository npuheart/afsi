# demo_441 — Cook's membrane (FEM fluid + IB₄ coupling)

Cook's membrane benchmark in the modified 13 cm x 13 cm domain of Li et al.
2025 (arXiv:2412.15408), following the configuration of Wells et al. 2023.

Differences from the paper: the fluid is solved with the finite-element
(Chorin projection, P2/P1) background of AFSI and the coupling uses only the
4-point (IB₄) kernel.  The paper's comparison additionally shows IB/BS/CBS
kernels on a finite-difference fluid solver.

## Files

| File | Description |
|---|---|
| `materials.py` | neo-Hookean models: `flory` (modified invariants + volumetric energy, the paper's stabilised form) and `standard` (compressible form, default) |
| `main.py` | runs the immersed-boundary FSI and writes `plot/metrics_*.csv` + `plot/snapshot_*.npz` |
| `plot_geometry.py` | site figure of the setup |
| `readme.md` | this file |

## Run

```bash
conda activate afsi-dolfinx
python main.py                                # M=8, paper protocol TL=20/TF=50

M=16 MFAC=1 python main.py                    # finer Lagrangian mesh
M=8  MFAC=1 TL=2 TF=20 DT_FACTOR=0.002 BETA_MULT=10 python main.py
```

Environment variables: `M` (elements per side of the longest structural edge,
default 8), `MFAC` (mesh factor, default 1.0), `N` (fluid cells per direction,
default `ceil(M*MFAC*10/6.5)`), `TL`, `TF` (load ramp / final time, defaults
20/50 s), `DT_FACTOR` (`dt = factor * h`, default 0.001), `BETA_MULT`
(multiplier on the penaltly `0.125 h / dt` of the paper, default 1.0), `MODEL`
(`standard`/`flory`), `TAG` (output label).

## Notes on the implementation

* The clamp is a penalty tether to the reference position with
  `beta = BETA_MULT * 0.125 * h / dt`; because the coupling transmits the
  tether reaction to the fluid, `BETA_MULT = 10` gave a slip of ~0.05 cm
  (vs. ~0.3 cm with the paper-scale 1.0) and is used for the reported runs.
* The spread force entering the fluid equals `F_ext + F_elastic`, so the
  external traction contributes with `+` to the weak form (verified by sign
  experiments: the traction must ADD, the penalty SUBTRACT).
* Probes must be indexed in the array order of `solid_coords.x.array`
  (`ref.reshape(-1, 2)`), not in the row order of
  `tabulate_dof_coordinates()` (cell-first-visit).
* The paper protocol (TL=20 s, TF=50 s) with `BETA_MULT=10` gives
  `Delta_Y -> 0.625 cm` (M=8, N=13) and `0.655 cm` (M=16, N=25), inside the
  paper's plateau band 0.60–0.68 cm (M=32, their figure).
* Bulk-modulus calibration check (`KAPPA_MULT`, see demo_443): with `10K` the
  volume is pinned to the reference's range (`J in [0.99, 1.02]` vs their
  `[0.966, 1.010]`) but the corner displacement rises to 0.745/0.750 cm
  (M=8/16, mesh-converged) — above the band. The raw constant leaves the
  volume looser and lands inside the band. For Cook's the mechanism is only
  half the story (see demo_443 vs this case); both results are documented.
  All values are CGS — the compression case required the opposite volume
  adjustment, so no unit factor (e.g. Pa vs dyn/cm2) is involved.
* A fast protocol (TL=2 s, TF=20 s, same `dt`) reproduces the plateau at
  M=8 (0.626 cm) but leaves the membrane ringing at M=16 (still 0.80 cm at
  t=20 s, relaxing toward 0.655 cm); use the paper protocol for settled values
  beyond M=8.
