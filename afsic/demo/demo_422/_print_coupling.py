"""Print / visualize the coupling matrix  C = A_s M_ww^{-1} M_wu = dt A_uW M_s^-1 Mfs^T.

Claims to verify:
  M_s^{-1}      : fully dense (ns x ns)
  A_uW, Mfs^T   : sparse (support near the disk)
  C             : DENSE WITHIN the interaction band (fluid dofs near the disk),
                  zero elsewhere  -- so "band-dense", not globally dense.
Prints densities, a numeric sub-block, and spy plots (PNG).
"""
import os
os.environ["NX"] = "16"; os.environ["NY"] = "16"
os.environ["SOLID_H"] = "0.05"
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import make_config
from immersed import ImmersedFEM

cfg = make_config()
ib = ImmersedFEM(cfg)
ib.compute_interaction(np.zeros(ib.n_s))
ib.assemble_mixed_mass()
nu, np_, ns = ib.n_u, ib.n_p, ib.n_s
dt = cfg["dt"]

Ms = ib.M_s.toarray()
MfsT = ib.MfsT_csr.toarray()
W = 0.01 * np.sin(2.0 * np.pi * np.arange(ns) / ns)
_, A_uW = ib.assemble_elastic(W, tangent=True)
A_uW = A_uW.toarray()
Minv = np.linalg.inv(Ms)                    # exact M_s^{-1} (dense)

# C = A_s M_ww^{-1} M_wu = (-A_uW)(dt M_s^{-1})(-Mfs^T) = dt A_uW M_s^{-1} Mfs^T
C = dt * (A_uW @ Minv @ MfsT)


def nnz_frac(M):
    return np.count_nonzero(np.abs(M) > 0) / M.size


print(f"nu={nu} ns={ns}")
print(f"  A_uW  ({nu}x{ns}): nnz={np.count_nonzero(A_uW):>8}  "
      f"density={nnz_frac(A_uW)*100:6.2f}%")
print(f"  Mfs^T ({ns}x{nu}): nnz={np.count_nonzero(MfsT):>8}  "
      f"density={nnz_frac(MfsT)*100:6.2f}%")
print(f"  M_s^-1({ns}x{ns}): nnz={np.count_nonzero(Minv):>8}  "
      f"density={nnz_frac(Minv)*100:6.2f}%   (should be ~100% = fully dense)")
print(f"  C=dt A_uW M_s^-1 Mfs^T ({nu}x{nu}): nnz={np.count_nonzero(C):>8}  "
      f"density={nnz_frac(C)*100:6.2f}%")

# nonzero rows/cols of C -> the interaction band
nz_rows = np.nonzero(np.abs(C).sum(axis=1) > 0)[0]
nz_cols = np.nonzero(np.abs(C).sum(axis=0) > 0)[0]
print(f"  C nonzero rows: {len(nz_rows)}  cols: {len(nz_cols)}  "
      f"(band size ~ {len(nz_rows)}x{len(nz_cols)})")
print(f"  within-band fill: "
      f"{np.count_nonzero(C[np.ix_(nz_rows, nz_cols)]) / (len(nz_rows)*len(nz_cols)) * 100:.1f}%"
      f"  (should be ~100% = dense inside the band)")

# numeric sub-block of C (a few rows/cols of the band)
r0 = nz_rows[:8]; c0 = nz_cols[:8]
print("\nC numeric sub-block (8 band rows x 8 band cols):")
np.set_printoptions(precision=2, suppress=True, linewidth=120)
print(C[np.ix_(r0, c0)])

# spy plots
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
for ax, (name, M) in zip(axs.ravel(), [
        ("A_uW (sparse, nu x ns)", A_uW),
        ("Mfs^T (sparse, ns x nu)", MfsT),
        ("M_s^-1 (FULLY dense, ns x ns)", Minv),
        ("C = dt A_uW M_s^-1 Mfs^T (band-dense, nu x nu)", C)]):
    ax.spy(np.abs(M) > 0, markersize=0.2)
    ax.set_title(f"{name}\nnnz={np.count_nonzero(M)}/{M.size} "
                 f"({nnz_frac(M)*100:.2f}%)")
plt.tight_layout()
plt.savefig("_plot_coupling_C.png", dpi=120)
print("\nsaved _plot_coupling_C.png ;  band rows/cols:", len(nz_rows), len(nz_cols))
print("DONE")
