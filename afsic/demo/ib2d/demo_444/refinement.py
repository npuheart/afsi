"""Grid-refinement comparison AFSI vs IB2d (volume conservation and band motion).

    python refinement.py out.png  N32:afsi32.npz:ib2d32.npz  N64:afsi64.npz:ib2d64.npz ...

Each case is ``label:afsi_result.npz:ib2d_reference.npz`` (cases generated with
``make_rubberband.py``, i.e. the same physical band tension at every Nx).
"""
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def area(X):
    return 0.5 * abs(np.sum(X[:, 0] * np.roll(X[:, 1], -1) - np.roll(X[:, 0], -1) * X[:, 1]))


def main(out, cases):
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    colors = plt.cm.viridis(np.linspace(0.0, 0.8, len(cases)))
    print(f"{'case':>6} {'area loss AFSI':>15} {'area loss IB2d':>15} {'max|dX| t<=0.5':>15} {'max|dX|':>9}")
    for (label, fa, fr), c in zip(cases, colors):
        A, R = np.load(fa), np.load(fr)
        tA, tR = A["t"], R["t"]
        aA = np.array([area(X) for X in A["X"]])
        aR = np.array([area(X) for X in R["X"]])
        ax[0].plot(tR, R["X"][:, 0, 0], "-", color=c, lw=1.0, label=f"IB2d {label}")
        ax[0].plot(tA, A["X"][:, 0, 0], "--", color=c, lw=1.6, label=f"AFSI {label}")
        ax[1].plot(tR, aR / aR[0], "-", color=c, lw=1.0, label=f"IB2d {label}")
        ax[1].plot(tA, aA / aA[0], "--", color=c, lw=1.6, label=f"AFSI {label}")
        common = [(i, int(np.argmin(abs(tR - t)))) for i, t in enumerate(tA) if np.min(abs(tR - t)) < 1e-8]
        dX = np.array([np.linalg.norm(A["X"][i] - R["X"][j], axis=1).max() for i, j in common])
        tc = tA[[i for i, _ in common]]
        print(f"{label:>6} {100 * (1 - aA[-1] / aA[0]):14.2f}% {100 * (1 - aR[-1] / aR[0]):14.2f}% "
              f"{dX[tc <= 0.5 + 1e-9].max():15.3e} {dX.max():9.3e}")
    ax[0].set_xlabel("t")
    ax[0].set_ylabel("x of Lagrangian point 1")
    ax[0].legend(fontsize=7, ncol=2)
    ax[1].set_xlabel("t")
    ax[1].set_ylabel("enclosed area / initial area")
    ax[1].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1], [tuple(a.split(":")) for a in sys.argv[2:]])
