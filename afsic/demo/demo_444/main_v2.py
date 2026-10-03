"""demo_444 v2 preset: RT fiber with the reduced-dissipation configuration.

Outcome of the isolation study (2026-10-03) that compared the RT fiber against
IB2d and tried every switch of main.py one at a time (tau = swing-decay time
constant over [0.1, 0.7] s; A(0.8) = enclosed-area drift at t = 0.8 s):

    variant               tau (s)   A(0.8 s)    verdict
    --------------------  --------  ----------  ----------------------------
    base (N=16, RT2)      0.313     -5.3%       published configuration
    midpoint time scheme  0.295     -4.6%       IB2d half-step: no help
    kernel 4-pt transfer  0.274     -86%        fiber collapses: unusable
    N=32 (DS fixed)       0.337     -44%        only +8%, area leak blows up
    box 2x2 (walls 4x)    0.344     -5.5%       +10% but wrong box geometry
    dt/2                  0.319     -5.0%       +2%
    SIP penalty x1/4..x8  0.288-0.327  -       <= +4%, non-monotone
    RT degree 3           0.372     -26.6%      +19%  <- the real lever
    RT degree 3 + pen x8  0.377     -9.8%       +20%, leak matches IB2d
    IB2d reference        0.701     -8.8%

Rationale for the two changes (everything else stays at the main.py defaults:
euler staggering, nodal coupling, convection on, DT=1e-3, N=16, 1x1 no-slip
box):

* RT_DEGREE=3 -- the rocking fiber's Stokes layer is ~0.018 (nu/omega)^(1/2)
  while the N=16 cell is 0.0625; the richer velocity space represents the
  fiber-scale flow better and numerically damps it less (tau 0.313 -> 0.372).
* RT_PENALTY=160 (x8 the default SIP interior-penalty weight) -- pulls the
  enclosed-area drift at 0.8 s from -26.6% down to -9.8%, i.e. the IB2d level
  (-8.8%), at a small cost in tau (0.372 -> 0.377 net with both).

Known trade-off (deliberately documented): the oscillation period shortens
0.196 -> 0.181 s (-6%; IB2d ~0.20 s), and the period already drifts low in the
degree-3 space.  The first-swing peak is unchanged (0.365 @ 0.10 s).

Usage:  python main_v2.py                     # fiber RT, T=1.5, tag fiber_v2_N16
        N=16 T_END=0.8 python main_v2.py      # quick check
Every main.py environment variable still applies; TAG defaults to
<shape>_v2_N<N> so v2 outputs never overwrite the published runs.
"""
import os
import runpy

os.environ.setdefault("RT_DEGREE", "3")
os.environ.setdefault("RT_PENALTY", "160")

_n = os.environ.get("N", "16")
_shape = os.environ.get("SHAPE", "fiber")
_fluid = os.environ.get("FLUID", "rt")
os.environ.setdefault("TAG", f"{_shape}_v2_N{_n}"
                      + ("" if _fluid == "rt" else f"_{_fluid}"))

runpy.run_path(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "main.py"),
    run_name="__main__")
