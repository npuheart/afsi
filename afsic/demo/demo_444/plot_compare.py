"""Seven-curve comparison for demo_444: IB2d vs the six AFSI runs
({fiber, thick} x {RT, Chorin, IPCS}, identical parameters).

Reads
-----
- AFSI outputs: plot/snapshot_*.npz written by main.py for the tags
  fiber_N16, thick_N16 (RT), *_chorin and *_ipcs (projection variants).
- IB2d outputs: the Rubberband_with_Springs example directory, parsed from
  hier_IB2d_data/fMag.*.vtk (Lagrangian points, TIME field) and
  viz_IB2d/uMag.*.vtk (Eulerian speed).

Writes (into OUT, default ./figures):
- demo444-area.png      area(t), log area error, a_x(t), a_y(t) - 7 curves each
- demo444-shapes.png    shapes at selected times (IB2d + the six AFSI runs)
- demo444-velocity.png  max marker speed |dX/dt|(t) - 7 curves

Run from this directory:

    IB2D_DIR=<path to Rubberband_with_Springs> python plot_compare.py
"""
import os
import re
import glob

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
IB2D_DIR = os.environ.get("IB2D_DIR")
if not IB2D_DIR:
    raise SystemExit(
        "Set IB2D_DIR to the IB2d example directory (the one that contains "
        "input2d, e.g. .../IB2d/matIB2d/Examples/Example_Standard_Rubberband/"
        "Rubberband_with_Springs) and re-run.")
OUT = os.environ.get("OUT", os.path.join(HERE, "figures"))
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------------------
# IB2d series
# ---------------------------------------------------------------------------
def ib2d_points(path):
    txt = open(path).read()
    t = float(re.search(r'TIME 1 1 double\n\s*([\d.eE+-]+)', txt).group(1))
    n = int(re.search(r'POINTS (\d+) float', txt).group(1))
    body = txt.split(f'POINTS {n} float', 1)[1]
    nums = np.array(body.split()[:3 * n], dtype=float)
    return t, nums.reshape(n, 3)[:, :2]

def ib2d_scalar(path):
    txt = open(path).read()
    t = float(re.search(r'TIME 1 1 double\n\s*([\d.eE+-]+)', txt).group(1))
    body = txt.split('LOOKUP_TABLE default', 1)[1]
    return t, np.array(body.split(), dtype=float)

def polygon_area(xy):
    x, y = xy[:, 0], xy[:, 1]
    return 0.5 * abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))

ib_t, ib_xy, ib_area, ib_ax, ib_ay = [], [], [], [], []
for f in sorted(glob.glob(os.path.join(IB2D_DIR, 'hier_IB2d_data/fMag.*.vtk'))):
    t, xy = ib2d_points(f)
    ib_t.append(t); ib_xy.append(xy)
    ib_area.append(polygon_area(xy))
    ib_ax.append(np.abs(xy[:, 0] - 0.5).max())
    ib_ay.append(np.abs(xy[:, 1] - 0.5).max())
ib_t = np.array(ib_t); ib_area = np.array(ib_area)
ib_ax = np.array(ib_ax); ib_ay = np.array(ib_ay)

ib_u = [(0.0, 0.0)]
for f in sorted(glob.glob(os.path.join(IB2D_DIR, 'viz_IB2d/uMag.*.vtk'))):
    t, v = ib2d_scalar(f)
    ib_u.append((t, v.max()))
ib_u = np.array(ib_u)

# ---------------------------------------------------------------------------
# AFSI series: 2 shapes x 3 solvers, identical parameters (only FLUID differs)
# ---------------------------------------------------------------------------
def load_afsi(tag):
    path = os.path.join(HERE, 'plot', f'snapshot_{tag}.npz')
    return np.load(path) if os.path.exists(path) else None

M_ANG = 96
RUNS = [
    ('fiber (RT)', 'fiber_N16', '#c62828', '-'),
    ('fiber (Chorin)', 'fiber_N16_chorin', '#ef6c00', '--'),
    ('fiber (IPCS)', 'fiber_N16_ipcs', '#ad1457', '-.'),
    ('thick (RT)', 'thick_N16', '#1565c0', '-'),
    ('thick (Chorin)', 'thick_N16_chorin', '#00838f', '--'),
    ('thick (IPCS)', 'thick_N16_ipcs', '#4527a0', '-.'),
]
DATA = {tag: load_afsi(tag) for _, tag, _, _ in RUNS}
for tag, d in DATA.items():
    if d is None:
        print('WARNING: missing snapshot for', tag)

# ---------------------------------------------------------------------------
# Figure 1: area, relative error, a_x, a_y - seven curves each
# (IB2d + the six AFSI runs {fiber, thick} x {RT, Chorin, IPCS})
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 4, figsize=(19, 4.4))
A0 = np.pi * 0.2 * 0.4
green = '#2e7d32'


def afsi_curves(ax, key, yform=None):
    for name, tag, col, ls in RUNS:
        d = DATA[tag]
        if d is None:
            continue
        y = d[key]
        if yform is not None:
            y = yform(y)
        ax.plot(d['t_series'], y, ls, lw=1.5, color=col, label=name)


ax = axes[0]
ax.plot(ib_t, ib_area, 'o-', ms=2.2, lw=1.2, color=green, label='IB2d')
afsi_curves(ax, 'area_series')
ax.axhline(A0, color='0.7', ls=':', lw=1)
ax.set_xlabel('t [s]'); ax.set_ylabel('enclosed area')
ax.set_title('enclosed area (linear)')
ax.set_ylim(0.0, 0.27)
ax.grid(True, alpha=0.25)

ax = axes[1]
ax.plot(ib_t, np.abs(ib_area / A0 - 1.0) + 1e-8, 'o-', ms=2.2, lw=1.2,
        color=green, label='IB2d')
afsi_curves(ax, 'area_series', yform=lambda a: np.abs(a / A0 - 1.0) + 1e-8)
ax.set_yscale('log')
ax.set_xlabel('t [s]'); ax.set_ylabel(r'relative area error $|A/A_0-1|$')
ax.set_title('flux through the band (log)')
ax.grid(True, which='both', alpha=0.25)

for k, (key, lab) in enumerate((('ax_series', '$a_x$'),
                                ('ay_series', '$a_y$')), start=2):
    ax = axes[k]
    ax.plot(ib_t, ib_ax if key == 'ax_series' else ib_ay, 'o-', ms=2.2,
            lw=1.2, color=green, label='IB2d')
    afsi_curves(ax, key)
    ax.axhline(0.2, color='0.7', ls=':', lw=1)
    ax.axhline(0.4, color='0.7', ls=':', lw=1)
    ax.set_xlabel('t [s]'); ax.set_ylabel('semi-axis [m]')
    ax.set_title('semi-axis ' + lab)
    ax.set_ylim(0.15, 0.45)
    ax.grid(True, alpha=0.25)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=7, fontsize=8.5,
           frameon=False, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout()
fig.subplots_adjust(bottom=0.16)
fig.savefig(os.path.join(OUT, 'demo444-area.png'), dpi=160)
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 2: shapes at selected times - IB2d + all six AFSI runs
# ---------------------------------------------------------------------------
times = [0.1, 0.2, 0.4, 0.8, 1.5]
theta = 2 * np.pi * np.arange(64) / 64
ell0 = np.column_stack([0.5 + 0.2 * np.cos(theta), 0.5 + 0.4 * np.sin(theta)])

rows = [('IB2d', None, green)]
for name, tag, col, ls in RUNS:
    rows.append((name, DATA[tag], col))
fig, axes = plt.subplots(len(rows), len(times),
                         figsize=(2.9 * len(times), 1.85 * len(rows)),
                         sharex=True, sharey=True)
for r, (name, data, col) in enumerate(rows):
    for c, tt in enumerate(times):
        ax = axes[r, c]
        ax.plot(ell0[:, 0], ell0[:, 1], ':', color='0.65', lw=1)
        if data is None:
            i = int(np.argmin(np.abs(ib_t - tt)))
            xy = ib_xy[i]
            ax.set_title(f'IB2d  t = {ib_t[i]:.2f} s', fontsize=8)
        else:
            i = int(np.argmin(np.abs(data['t_series'] - tt)))
            ring = data['xy_series'][i]
            xy = ring[M_ANG:2 * M_ANG] if name.startswith('thick') else ring
            ax.set_title(f'{name}  t = {data["t_series"][i]:.2f} s',
                         fontsize=8)
        ax.plot(np.r_[xy[:, 0], xy[0, 0]], np.r_[xy[:, 1], xy[0, 1]], '-',
                lw=1.3, color=col)
        ax.set_aspect('equal')
        ax.set_xlim(0.05, 0.95); ax.set_ylim(0.05, 0.95)
        ax.set_xticks([0.2, 0.5, 0.8]); ax.set_yticks([0.2, 0.5, 0.8])
    axes[r, 0].set_ylabel(name, fontsize=8)
fig.suptitle('rubberband shapes (dotted: initial ellipse)', fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'demo444-shapes.png'), dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 3: max marker speed (finite difference of the marker paths; a
# kinematic quantity that is comparable across codes - unlike raw velocity
# DOF norms, which scale with the discretisation).  IB2d + six AFSI runs.
# ---------------------------------------------------------------------------
def marker_speed(t, xy):
    v = np.linalg.norm(np.diff(xy, axis=0), axis=-1) / np.diff(t)[:, None]
    return t[1:], v.max(axis=1)


fig, ax = plt.subplots(figsize=(8.6, 4.0))
tb, vb = marker_speed(ib_t, np.asarray(ib_xy))
ax.plot(tb, vb, 'o-', ms=2.5, lw=1.3, color=green, label='IB2d')
for name, tag, col, ls in RUNS:
    d = DATA[tag]
    if d is None:
        continue
    ring = slice(M_ANG, 2 * M_ANG) if name.startswith('thick') else slice(None)
    tt, vv = marker_speed(d['t_series'], d['xy_series'][:, ring])
    ax.plot(tt, vv, ls, lw=1.6, color=col, label=name)
ax.set_xlabel('t [s]'); ax.set_ylabel('max marker speed [m/s]')
ax.set_title('structure speed |dX/dt|')
ax.set_ylim(0.0, 5.0)
ax.grid(True, alpha=0.25)
ax.legend(fontsize=8, frameon=False, ncol=2)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'demo444-velocity.png'), dpi=160)
plt.close(fig)

print(f"figures written to {OUT}")
print("key numbers (t = 0.2 / t = 1.5):")
i2 = int(np.argmin(np.abs(ib_t - 0.2))); i15 = len(ib_t) - 1
print(f"  {'IB2d':16s} a_x(0.2)={ib_ax[i2]:.4f} a_y(0.2)={ib_ay[i2]:.4f} "
      f"area(1.5)={ib_area[i15]:.5f} ({100*(ib_area[i15]/A0-1):+.2f}%)")
for name, tag, col, ls in RUNS:
    d = DATA[tag]
    if d is None:
        continue
    i2 = int(np.argmin(np.abs(d['t_series'] - 0.2)))
    i15 = int(np.argmin(np.abs(d['t_series'] - 1.5)))
    print(f"  {name:16s} a_x(0.2)={d['ax_series'][i2]:.4f} "
          f"a_y(0.2)={d['ay_series'][i2]:.4f} "
          f"area(1.5)={d['area_series'][i15]:.5f} "
          f"({100*(d['area_series'][i15]/A0-1):+.2f}%)")
