"""IB2d's Lagrangian **coagulation** model (cell aggregation by bond springs).

This is a Python port of the black-box MATLAB implementation
``matIB2d/IBM_Blackbox/give_Me_Coagulation_Force_Densities.m`` (which is called
from ``please_Find_Lagrangian_Forces_On_Eulerian_grid.m`` when
``coagulation = 1``).  The pyIB2d module of the same name is **not** a usable
port of it (see the notes at the bottom of this file); the model below follows
the MATLAB one, with the repairs that are needed to run it from Python:

* identifiers are converted to Python's 0-based indexing (``index_base`` = 1
  for the ``matIB2d`` examples, whose files number the points from 1);
* ``please_Remove_Bonds`` is completed.  The shipped MATLAB file is *truncated*
  in the middle of that function (it ends after ``N = length(aggregate_list);``
  with no ``return``), so the MATLAB code cannot survive a bond fracture at all.
  The natural intent -- delete the fractured bond rows -- is implemented here.

Model (identical to the MATLAB source)
-------------------------------------
The ``<name>.coagulation`` file holds

    line 1     start_id          first Lagrangian index of the cell block
    line 2     threshold_radius  bond if the two *cell centres* are closer
    line 3     bond_strength     k of the bond spring
    line 4     fracture_force    bond-rupture threshold
    line 5..   n_pts of cell 1, cell 2, ...   (the cells are the FIRST
                                              markers of the vertex file)

Every step, at the half-step marker positions:

1. ``cell centre`` = midpoint of marker ``lag1`` and the diametrically opposed
   marker ``lag1 + n_pts/2`` of the same cell (``please_Find_Cell_Centers``);
2. every *unbonded* pair of cells whose centre distance ``r < threshold_radius``
   gets a new bond (``please_Find_Distances_Between_Cells``);
3. the bonded markers are the marker of cell *i* closest to cell *j*'s centre and
   vice versa; the bond's resting length is their current distance
   (``please_Find_Closest_Lag_Pts_To_Cell_Centers``);
4. the bond acts as a linear spring ``alpha = 1`` with the IB2d minimum-image
   fold on its ``dx``/``dy``::

       sF = 0.5 (alpha + 1) k (L - L_r)^alpha * d_hat
          = k (L - L_r) d_hat                       (alpha = 1)

   applied ``+sF`` to the master marker and ``-sF`` to the slave marker;
5. a bond created *before* this step is removed when its spring force fails the
   rupture test ``sF_x^2 + sF_y^2 < fracture_force`` -- note that IB2d compares
   the **squared** force with the threshold (upstream unit quirk, reproduced).
   Newly created bonds are never tested in the step they are created.

Upstream bugs / quirks reproduced or documented
-----------------------------------------------
* the rupture test is on ``|f|^2``, not ``|f|`` (step 5);
* cell centres, centre distances and the closest-marker search never apply a
  periodic minimum image, so a cell straddling the boundary is mis-centred and
  bonds across the boundary are never formed -- only the bond *spring* wraps
  ``dx``/``dy`` (this is exactly what the ``Example_Testing_Thru_Boundaries``
  family probes);
* the MATLAB file's ``please_Remove_Bonds`` is truncated (see above);
* the MATLAB file's header comments mislabel the ``.coagulation`` rows
  ("row 3: threshold force") and its reader's comments claim the cell counts
  start at row 4 (they start at row 5).

The pyIB2d ``give_Me_Coagulation_Force_Densities.py`` is dead code (nothing in
``IBM_Driver.py`` imports it) *and* broken: ``please_Find_Cell_Centers`` uses a
float index (``lag1_ind + 0.5*n``) on a numpy array, the closest-point search
reads columns 1 and 2 of a two-column ``connects`` array, the
"already bonded" test ``if np.where(checkVec == j):`` is always true (a
non-empty tuple is truthy) so no second bond is ever created, ``aggregate_list
!= (0,)`` is applied to an ``int`` placeholder, ``please_Remove_Bonds`` reshapes
with ``N/5`` (float) and the returned name ``remove_bonds`` does not match
``remove_Bonds``.
"""
import numpy as np


def read_coagulation(path):
    """Parse an IB2d ``<name>.coagulation`` file (one value per line)."""
    vals = np.atleast_1d(np.loadtxt(path))
    assert vals.size >= 5, f"malformed coagulation file {path}"
    return dict(start=int(round(vals[0])), threshold=float(vals[1]),
                bond_k=float(vals[2]), fracture=float(vals[3]),
                counts=vals[4:].astype(int))


class Coagulation:
    """IB2d's coagulation state machine + bond-spring force densities.

    Attributes
    ----------
    aggregate : (n_bonds, 5) float array
        columns ``cell_i, cell_j, lag_i, lag_j, RL`` of the active bonds
        (the MATLAB ``aggregate_list``).
    n_formed, n_fractured : int
        cumulative counters (diagnostics only).
    """

    def __init__(self, spec, Lx, Ly, index_base=1):
        self.start = int(spec["start"]) - int(index_base)
        self.threshold = float(spec["threshold"])
        self.bond_k = float(spec["bond_k"])
        self.fracture = float(spec["fracture"])
        self.counts = np.asarray(spec["counts"], dtype=int)
        assert self.start >= 0 and self.counts.sum() > 0
        self.Lx, self.Ly = float(Lx), float(Ly)
        self.alpha = 1
        # 0-based first marker of every cell; the cells are consecutive
        self.first = self.start + np.concatenate(
            ([0], np.cumsum(self.counts)))[:-1]
        self.aggregate = np.zeros((0, 5))
        self.n_formed = 0
        self.n_fractured = 0

    # ------------------------------------------------------------------ #
    def centers(self, X):
        """Cell centres: midpoint of marker ``lag1`` and ``lag1 + n/2``."""
        i1 = self.first
        i2 = self.first + self.counts // 2
        return 0.5 * (X[i1] + X[i2])

    def _unbonded_close_pairs(self, X):
        """New ``(cell_i, cell_j)`` pairs closer than ``threshold_radius``."""
        C = self.centers(X)
        n = len(self.counts)
        new = []
        for i in range(n):
            for j in range(i + 1, n):
                if np.hypot(C[i, 0] - C[j, 0],
                            C[i, 1] - C[j, 1]) < self.threshold:
                    if self.aggregate.shape[0]:
                        rows = np.nonzero(self.aggregate[:, 0] == i)[0]
                        if rows.size and np.any(self.aggregate[rows, 1] == j):
                            continue
                    new.append((i, j))
        return new, C

    def _bond_from_pair(self, X, C, i, j):
        """Closest markers of the pair + the bond's resting length."""
        def closest(cell, other_center):
            ids = self.first[cell] + np.arange(self.counts[cell])
            d = np.hypot(X[ids, 0] - other_center[0],
                         X[ids, 1] - other_center[1])
            return int(ids[int(np.argmin(d))])       # ties: first minimum
        lag_i = closest(i, C[j])
        lag_j = closest(j, C[i])
        rl = float(np.hypot(X[lag_i, 0] - X[lag_j, 0],
                            X[lag_i, 1] - X[lag_j, 1]))
        return lag_i, lag_j, rl

    def forces(self, X):
        """Bond force densities at the marker positions ``X`` (Nb, 2).

        Advances the bond state exactly like one call of the MATLAB
        ``give_Me_Coagulation_Force_Densities``.
        """
        Nb = X.shape[0]
        fx = np.zeros(Nb)
        fy = np.zeros(Nb)

        n_old = self.aggregate.shape[0]
        new, C = self._unbonded_close_pairs(X)
        if new:
            rows = np.array([self._bond_from_pair(X, C, i, j)
                             for i, j in new])
            rows = np.column_stack([np.array(new, float), rows])
            self.aggregate = np.vstack([self.aggregate, rows])
            self.n_formed += len(new)

        remove = []
        for b in range(self.aggregate.shape[0]):
            _, _, mi, si, rl = self.aggregate[b]
            mi, si = int(mi), int(si)
            dx = X[si, 0] - X[mi, 0]
            dy = X[si, 1] - X[mi, 1]
            if abs(dx) > self.Lx / 2:
                dx = np.sign(dx) * (self.Lx - np.sign(dx) * dx)
            if abs(dy) > self.Ly / 2:
                dy = np.sign(dy) * (self.Ly - np.sign(dy) * dy)
            L = np.hypot(dx, dy)
            fac = 0.5 * (self.alpha + 1) * self.bond_k \
                * (L - rl) ** self.alpha / L
            sFx, sFy = fac * dx, fac * dy
            if b < n_old and sFx ** 2 + sFy ** 2 >= self.fracture:
                remove.append(b)                     # ruptured before this step
                continue
            fx[mi] += sFx
            fy[mi] += sFy
            fx[si] -= sFx
            fy[si] -= sFy

        if remove:
            self.aggregate = np.delete(self.aggregate, remove, axis=0)
            self.n_fractured += len(remove)
        return fx, fy

    # ------------------------------------------------------------------ #
    def snapshot(self):
        """Serialisable bond state (for the reference/AFSI dumps)."""
        return np.array(self.aggregate, copy=True)

    def bond_lengths(self, X):
        """Current marker-marker distance of every active bond."""
        if not self.aggregate.shape[0]:
            return np.zeros(0)
        mi = self.aggregate[:, 2].astype(int)
        si = self.aggregate[:, 3].astype(int)
        return np.hypot(X[si, 0] - X[mi, 0], X[si, 1] - X[mi, 1])


def analytic_self_test():
    """Independent closed-form check of the bond model on two synthetic cells.

    Cell A = 8 markers on a radius-0.1 circle at (1, 0); cell B = 8 markers on a
    radius-0.1 circle at (1.3, 0); 1-based identifiers (``index_base = 1``).
    The pair is inside the threshold, so exactly one bond must form between the
    facing markers (A's +x marker and B's -x marker), its resting length must be
    ``0.3 - 0.2 = 0.1`` and the spring force must be equal and opposite on the
    two markers (``sum(f) = 0``).  Stretching the bond by 0.05 must then give
    ``|f| = k |L - L_r|``.

    Returns ``(ok, detail)``.
    """
    n = 8
    th = np.arange(n) * 2 * np.pi / n
    A = np.column_stack([1.0 + 0.1 * np.cos(th), 0.1 * np.sin(th)])
    B = np.column_stack([1.3 + 0.1 * np.cos(th), 0.1 * np.sin(th)])
    X = np.vstack([A, B])
    spec = dict(start=1, threshold=0.4, bond_k=250.0, fracture=1e12,
                counts=np.array([n, n]))
    coag = Coagulation(spec, Lx=10.0, Ly=10.0, index_base=1)
    fx, fy = coag.forces(X)
    ok = coag.aggregate.shape == (1, 5)
    ok = ok and int(coag.aggregate[0, 2]) == 0
    ok = ok and int(coag.aggregate[0, 3]) == n + n // 2
    ok = ok and abs(coag.aggregate[0, 4] - 0.1) < 1e-12
    # L == L_r initially -> (essentially) no force, and equal + opposite anyway
    ok = ok and abs(np.hypot(fx, fy).max()) < 1e-9
    ok = ok and abs(fx.sum()) < 1e-12 and abs(fy.sum()) < 1e-12

    X2 = X.copy()
    X2[n + n // 2, 0] -= 0.05                    # stretch the bond
    fx2, fy2 = coag.forces(X2)
    L = np.hypot(X2[n + n // 2, 0] - X2[0, 0], X2[n + n // 2, 1] - X2[0, 1])
    ok = ok and abs(np.hypot(fx2[0], fy2[0]) - 250.0 * abs(L - 0.1)) < 1e-9
    ok = ok and abs(fx2.sum()) < 1e-9 and abs(fy2.sum()) < 1e-9
    detail = dict(bonds=int(coag.aggregate.shape[0]),
                  RL=float(coag.aggregate[0, 4]),
                  force_max=float(np.hypot(fx, fy).max()),
                  k_dL=float(np.hypot(fx2[0], fy2[0])),
                  k_abs_L_minus_RL=float(250.0 * abs(L - 0.1)),
                  sum_f=float(fx2.sum()))
    return bool(ok), detail


if __name__ == "__main__":
    ok, detail = analytic_self_test()
    print("analytic self-test:", "PASS" if ok else "FAIL", detail)
