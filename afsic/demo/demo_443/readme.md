# demo_443 — Compression test (rectangular block)

Quasi-static plane-strain compression of a 20 x 10 cm neo-Hookean block
immersed centrally in a 40 x 40 cm fluid box, after Wells et al. 2023 as used
in Li et al. 2025 (arXiv:2412.15408).  A downward traction of 200 dyn/cm acts
on the central 10 cm of the top surface; the bottom has zero vertical
displacement and the top zero horizontal displacement (both by penalties); all
other boundaries are stress free.  Probe: vertical displacement at the centre
of the top surface (20, 25).

Differences from the paper: the fluid is solved with the finite-element
(Chorin projection, P2/P1) background of AFSI and the coupling uses only the
4-point (IB4) kernel, on a fluid grid refined by a factor two relative to the
paper's `N = ceil(M*MFAC)` so the IB4 support is resolved.

## Files

| File | Description |
|---|---|
| `materials.py` | neo-Hookean models (see demo_441) |
| `main.py` | runs the case, writes `plot/metrics_*.csv` + `plot/snapshot_*.npz` |
| `plot_snapshot.py` | deformed block coloured by `J` + fluid speed (diagnostic figure) |
| `plot_fields.py` | two-panel site figure (J + fluid speed) |
| `plot_geometry.py` | site figure of the setup |
| `readme.md` | this file |

## Run

```bash
conda activate afsi-dolfinx
# reported run: calibrated bulk (see "Incompressibility" below), marker spacing 2h
M=8  N=32 DT_FACTOR=0.002 BETA_MULT=10 TL=5 TF=100 KAPPA_MULT=10 python main.py
# stable coarse companion with the raw paper bulk:
M=32 N=32 DT_FACTOR=0.002 BETA_MULT=10 TL=5 TF=50  KAPPA_MULT=1  python main.py
```

Environment variables: `M` (elements along the 20 cm edge, default 16), `N`
(fluid cells, default `ceil(2 M MFAC)`), `MFAC`, `TL`, `TF`, `DT_FACTOR`,
`BETA_MULT` (constraint penalty multiplier; the paper's `kappa_S =
2.5*(2.5 h/dt)` is `BETA_MULT=1`), `KAPPA_MULT` (material bulk modulus
multiplier, see below), `MODEL`, `TAG`.

## Incompressibility: why KAPPA_MULT matters

The reference keeps `J = 1` through its discrete divergence-free coupling and
calls the material's volumetric energy "technically redundant" — its displaced
results (about -4.05 cm) are therefore effectively incompressible, with a
Jacobian range of [0.892, 1.021] at M=32.  The FE-coupled variant here leaks a
little volume (the same leak that drives the demo_442 creep), so the
*volumetric stiffness of the material* is what pins `J`.  Calibration runs at
M=8, N=32:

| `KAPPA_MULT` | dY (t=25 s) | J range |
| --- | --- | --- |
| 1 (paper constant, 374.2) | -4.80 | [0.68, 1.09] |
| 10 (used in the reported runs) | **-4.09** | [0.88, 1.03] |
| 100 | -3.87 | [0.96, 1.01] |
| 1000 | locks: element inverts at t~2.5 s, then a frozen state | inverted |

The window is two-sided: too soft and the coupling's volume leak softens the
response; too stiff and the discrete system locks at the punch corner (element
inversion followed by a completely frozen state — all diagnostics constant,
fluid velocity identically zero).  The same lock appears at M>=16 with the
markers at h or below.  The paper does not discuss a locking limit; it quotes
the penalisation's time-step restriction and pressure-response effects.

`KAPPA_MULT = 10` reproduces both the reference displacement (-4.03..-4.09)
and its Jacobian range; the settled run at M=8, TF=100 gives
`dY = -4.10 cm`, `J in [0.881, 1.031]`.

## Other implementation notes and traps

* The zero-horizontal-displacement condition applies to the ENTIRE top
  boundary, including the loaded central 10 cm; applying it only outside the
  loaded patch lets the central top slide sideways and produces an unphysical
  folded/punched equilibrium (dY ~ -6.2 cm and elements crushed to J ~ 0.1).
* Stability window with the stiff bulk: with `KAPPA_MULT = 10` only marker
  spacings of roughly `2h` or more remain stable — markers at `h` or below
  blow up around `t ~ 10 s` (at both `DT_FACTOR = 0.002` and `0.001`), even
  though the same configurations are stable at the raw bulk.  The reported
  run therefore uses `M = 8, N = 32` (marker spacing `2h`).
* `MODEL=flory` (modified invariants + volumetric energy, the paper's
  stabilised form) does not by itself fix the offset: at the raw bulk it
  gives `dY = -4.84 cm` with `J` down to `0.54` — soft and volume-leaking.
  The reported runs use `MODEL=standard` with the calibrated bulk modulus.
* The intermediate Lagrangian resolution (`M = 16`) is pathological here at
  both `N = 32` and `N = 40` (elements invert, `J -> 0`), at the raw bulk and
  at the calibrated one.  Displacement-only diagnostics do not reveal the
  degradation — always check the J range.
