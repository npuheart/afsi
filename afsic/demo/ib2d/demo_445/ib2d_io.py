"""Readers for IB2d input files (``input2d``, ``*.vertex``, ``*.spring``,
``*.target``, ``*.nonInv_beam``).

Only the parts needed by the AFSI ports of the IB2d examples are implemented.
"""
import os
import re

import numpy as np


def read_input2d(path):
    """Parse an IB2d ``input2d`` file into a flat ``{name: value}`` dict.

    Numbers are converted to float/int, the structure name is kept as string.
    """
    params = {}
    with open(path) as fh:
        for line in fh:
            line = line.split("%", 1)[0].strip()
            m = re.match(r"^([A-Za-z0-9_]+)\s*=\s*(.+?);?\s*$", line)
            if not m:
                continue
            key, val = m.group(1), m.group(2).strip().rstrip(";").strip()
            if val.startswith('"') or val.startswith("'"):
                params[key] = val.strip("\"'")
                continue
            try:
                f = float(val)
                params[key] = int(f) if f.is_integer() and "." not in val and "e" not in val.lower() else f
            except ValueError:
                params[key] = val
    return params


def read_vertex(path):
    """``.vertex`` file -> (Nb, 2) array of Lagrangian point coordinates."""
    with open(path) as fh:
        n = int(float(fh.readline().split()[0]))
        xy = np.loadtxt(fh, max_rows=n)
    return np.atleast_2d(xy)[:, :2]


def read_spring(path):
    """``.spring`` file -> (leader, follower) 0-based int array and (k, L_rest,
alpha)."""
    rows = []
    with open(path) as fh:
        n = int(float(fh.readline().split()[0]))
        for line in fh:
            v = line.split()
            if v:
                rows.append([float(s) for s in v])
            if len(rows) == n:
                break
    conn = np.array([[int(r[0]) - 1, int(r[1]) - 1] for r in rows], dtype=np.int64)
    k = np.array([r[2] for r in rows])
    L = np.array([r[3] for r in rows])
    alpha = np.array([r[4] if len(r) > 4 else 1.0 for r in rows])
    return conn, k, L, alpha


def read_target(path):
    """``.target`` file -> (ids (0-based), stiffness).

    IB2d stores only the point index and the stiffness; the target positions
    are the points' *initial* coordinates (filled in by the driver), i.e. the
    targets are tethered to where they started.
    """
    with open(path) as fh:
        n = int(float(fh.readline().split()[0]))
        rows = np.loadtxt(fh, max_rows=n)
    rows = np.atleast_2d(rows)
    assert rows.shape[1] >= 2, "unexpected .target layout"
    return rows[:, 0].astype(np.int64) - 1, rows[:, 1].astype(float).copy()


def read_noninv_beam(path):
    """``.nonInv_beam`` file -> dict with 0-based (p1, p2, p3), kb and the
    reference second difference C = (p1 + p3 - 2 p2) stored at generation time.
    ``p2`` is the *middle* node (the one that feels 2x the force)."""
    with open(path) as fh:
        n = int(float(fh.readline().split()[0]))
        rows = np.loadtxt(fh, max_rows=n)
    rows = np.atleast_2d(rows)
    assert rows.shape[1] == 6, "unexpected .nonInv_beam layout"
    return dict(p1=rows[:, 0].astype(np.int64) - 1,
                p2=rows[:, 1].astype(np.int64) - 1,
                p3=rows[:, 2].astype(np.int64) - 1,
                kb=rows[:, 3].astype(float).copy(),
                C=np.column_stack([rows[:, 4], rows[:, 5]]).astype(float))


def load_example(example_dir):
    """Read ``input2d`` and all structure files of an IB2d example directory.

    Keys: ``params``, ``name``, ``X``, ``springs`` (if any), ``targets`` (if
    ``target_pts`` is set), ``beams`` (if ``nonInvariant_beams`` is set).
    """
    params = read_input2d(os.path.join(example_dir, "input2d"))
    name = params.get("string_name", "rubberband")
    data = {"params": params, "name": name,
            "X": read_vertex(os.path.join(example_dir, f"{name}.vertex"))}
    spring_file = os.path.join(example_dir, f"{name}.spring")
    if params.get("springs", 0) and os.path.exists(spring_file):
        data["springs"] = read_spring(spring_file)
    target_file = os.path.join(example_dir, f"{name}.target")
    if params.get("target_pts", 0) and os.path.exists(target_file):
        data["targets"] = read_target(target_file)
    beam_file = os.path.join(example_dir, f"{name}.nonInv_beam")
    if params.get("nonInvariant_beams", 0) and os.path.exists(beam_file):
        data["beams"] = read_noninv_beam(beam_file)
    return data
