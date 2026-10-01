# demo_442 — Pressurized circular membrane at equilibrium

A closed elastic membrane of radius `R = 1/4` centred at `(1/2, 1/2)`, started
in its circular equilibrium configuration in a still fluid.  The membrane force
density is `F = kappa d^2X/ds^2` (tension weak form `kappa |X_s|^2 / 2` with
`kappa = 1`).  Any change of the enclosed area is a discretisation error of the
coupling; the spurious vorticity (ideally zero) probes the force-spreading
operator.  Case of Li et al. 2025 (arXiv:2412.15408) after Gruninger & Griffith
2024.

Differences from the paper: the fluid is solved with the finite-element
(Chorin projection, P2/P1) background of AFSI and the coupling uses only the
4-point (IB₄) kernel; the paper runs this case in a periodic unit square,
while here the box has no-slip walls (the membrane stays 0.25 away from them).

## Files

| File | Description |
|---|---|
| `main.py` | runs the case, writes `plot/metrics_*.csv` + `plot/snapshot_*.npz` |
| `plot_fields.py` | spurious-vorticity / velocity maps (site figures) |
| `plot_geometry.py` | site figure of the setup |
| `readme.md` | this file |

## Run

```bash
conda activate afsi-dolfinx
python main.py                        # N=128, MFAC=0.5, t to 1 s (paper settings)

N=128 MFAC=1.0 python main.py         # mesh-factor sweep
T_END=0.05 OUT_EVERY=1 python main.py # very short diagnostic
```

Environment variables: `N` (fluid cells per direction, default 128), `MFAC`
(marker spacing / h, default 0.5), `T_END` (default 1.0), `KAPPA`, `R`,
`OUT_EVERY` (metric stride), `TAG`.

## Results (N=128, MFAC=0.5, dt=h/8, t to 1 s)

* Area conservation: `|A - A0|/A0 = 1.3e-2` at `t = 1 s`, growing linearly.
  The pressure jump establishes immediately at `p_i - p_o = 3.965`
  (equilibrium value `kappa/R = 4.0`, i.e. within 1 %).
* Spurious velocity: `max |u| = 4.6e-3`, `max |omega| = 0.63`; the vorticity
  forms cell-scale patches around the membrane rather than a smooth ring.
* The steady creep of the markers is the residual imbalance between the
  spread tension force and the fluid's reaction: with the FE fluid background
  (no grid-scale dissipation) the net radial flux does not cancel, unlike in
  the paper's periodic spectral setting where `|dA|/A ~ 1e-6` for IB₄.
* The enclosed area is measured from the P1 membrane nodes remapped to arc
  order; the paper uses 10 000 passive tracers for the same purpose.
