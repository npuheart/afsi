"""Build a "relaxed" copy of this example's input with the pre-tension removed.

**Upstream data inconsistency.**  ``viv.d_spring`` declares two damped springs
(``kappa = 1.5e4``, ``b = 50``, ``L0 = 0.0667``) whose endpoints are
**0.1576 m and 0.1581 m apart** in ``viv.vertex`` -- 2.4x their resting length.
At ``t = 0`` the pair therefore carries a static pre-tension of
``kappa (L - L0) ~ 1.4e3`` on just four of 2388 markers, which pyIB2d absorbs
but AFSI cannot (it blows up within ~20 steps; see ``readme.md``).

This script writes ``relaxed_input/`` (git-ignored) where only that one column
changes -- ``L0`` is set to the initial distance of each damped spring, so the
structure starts unstressed exactly like the other 678 springs and 452 beams
(whose ``t = 0`` force is verified to be zero)::

    python make_relaxed_input.py
    python run_reference.py --src relaxed_input --nx 256 --tend 0.05 --print-dump 100
    IB2D_EXAMPLE=relaxed_input NX=256 TFINAL=0.05 PRINT_DUMP=100 python main.py
    python compare.py

Everything else (``input2d``, ``viv.vertex/spring/beam/target``) is copied
verbatim; ``ib2d_input/`` keeps the upstream files untouched.
"""
import os
import shutil

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "ib2d_input")
OUT = os.path.join(HERE, "relaxed_input")


def main():
    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    os.makedirs(OUT)
    for fn in sorted(os.listdir(SRC)):
        p = os.path.join(SRC, fn)
        if os.path.isfile(p) and not fn.endswith((".m", ".py")):
            shutil.copy(p, os.path.join(OUT, fn))
    for fn in ("please_Compute_External_Forcing.py", "update_Springs.py"):
        if os.path.exists(os.path.join(SRC, fn)):
            shutil.copy(os.path.join(SRC, fn), os.path.join(OUT, fn))

    X = np.loadtxt(os.path.join(SRC, "viv.vertex"), skiprows=1)
    d = np.loadtxt(os.path.join(SRC, "viv.d_spring"), skiprows=1)
    for row in d:
        a, b = int(row[0]) - 1, int(row[1]) - 1          # .d_spring is 1-based
        L = float(np.hypot(*(X[b] - X[a])))
        print(f"damped spring {a}-{b}: L0 {row[3]:.6g} -> {L:.6g} "
              f"(pre-tension {row[2] * (L - row[3]):.6g})")
        row[3] = L
    with open(os.path.join(OUT, "viv.d_spring"), "w") as fh:
        fh.write(f"{d.shape[0]}\n")
        np.savetxt(fh, d, fmt="%.16e")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
