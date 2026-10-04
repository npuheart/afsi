"""Readers for IB2d input files of the *pyIB2d* examples (``input2d``,
``*.vertex``, ``*.spring``, ``*.d_spring``, ``*.beam``, ``*.nonInv_beam``,
``*.target``, ``*.muscle``).

Unlike the older demos (demo_444/445), the input files shipped with the
pyIB2d examples number the Lagrangian points from 0 (the MATLAB examples use
1-based indices), so everything here is read as-is in 0-based convention.

Only the parts needed by the AFSI ports of the IB2d examples are implemented.
"""
import os
import re

import numpy as np


def read_input2d(path):
    """Parse an IB2d ``input2d`` file into a flat ``{name: value}`` dict."""
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
                params[key] = (int(f) if f.is_integer() and "." not in val
                               and "e" not in val.lower() else f)
            except ValueError:
                params[key] = val
    return params


def _read_rows(path):
    """Generic ``n <rows>`` IB2d file -> (n, rows) with rows an (n, m) array."""
    with open(path) as fh:
        n = int(float(fh.readline().split()[0]))
        rows = np.atleast_2d(np.loadtxt(fh, max_rows=n))
    return rows


def read_vertex(path):
    """``.vertex`` file -> (Nb, 2) array of Lagrangian point coordinates."""
    return _read_rows(path)[:, :2]


def read_spring(path):
    """``.spring`` -> dict(conn (0-based), k, L, alpha)."""
    rows = _read_rows(path)
    alpha = rows[:, 4] if rows.shape[1] > 4 else np.ones(len(rows))
    return dict(conn=rows[:, :2].astype(np.int64), k=rows[:, 2].copy(),
                L=rows[:, 3].copy(), alpha=np.asarray(alpha, float))


def read_damped_spring(path):
    """``.d_spring`` -> dict(conn, k, L, b) (5 columns, IB2d
    ``read_Damped_Spring_Points``)."""
    rows = _read_rows(path)
    assert rows.shape[1] >= 5, "unexpected .d_spring layout"
    return dict(conn=rows[:, :2].astype(np.int64), k=rows[:, 2].copy(),
                L=rows[:, 3].copy(), b=rows[:, 4].copy())


def read_beam(path):
    """``.beam`` (invariant torsional springs) -> dict(p1, p2, p3, kb, C).

    ``p2`` is the middle node.  Column 5 (``C``) is the reference cross
    product stored at generation time (0 for a straight/curved-at-rest beam
    set up by the examples).
    """
    rows = _read_rows(path)
    assert rows.shape[1] == 5, "unexpected .beam layout"
    return dict(p1=rows[:, 0].astype(np.int64), p2=rows[:, 1].astype(np.int64),
                p3=rows[:, 2].astype(np.int64), kb=rows[:, 3].copy(),
                C=rows[:, 4].copy())


def read_noninv_beam(path):
    """``.nonInv_beam`` -> dict(p1, p2, p3, kb, C=(p1+p3-2p2) at rest)."""
    rows = _read_rows(path)
    assert rows.shape[1] == 6, "unexpected .nonInv_beam layout"
    return dict(p1=rows[:, 0].astype(np.int64), p2=rows[:, 1].astype(np.int64),
                p3=rows[:, 2].astype(np.int64), kb=rows[:, 3].copy(),
                C=np.column_stack([rows[:, 4], rows[:, 5]]).astype(float))


def read_target(path):
    """``.target`` -> dict(ids (0-based), k); anchors are the initial
    positions (filled in by the driver)."""
    rows = _read_rows(path)
    return dict(ids=rows[:, 0].astype(np.int64), k=rows[:, 1].copy())


def read_muscle(path):
    """``.muscle`` (Hill force-velocity + length-tension model) ->
    dict(conn, LFO, SK, a, b, Fmax) -- IB2d ``read_Muscle_Points`` columns:
    p1, p2, length at max tension, muscle constant, Hill a, Hill b, F-max."""
    rows = _read_rows(path)
    assert rows.shape[1] == 7, "unexpected .muscle layout"
    return dict(conn=rows[:, :2].astype(np.int64), LFO=rows[:, 2].copy(),
                SK=rows[:, 3].copy(), a=rows[:, 4].copy(), b=rows[:, 5].copy(),
                Fmax=rows[:, 6].copy())


def read_mass(path):
    """``.mass`` -> dict(ids, k, M) -- IB2d ``read_Mass_Points`` columns:
    Lagrangian id, "mass-spring" stiffness, mass value.

    In IB2d a mass point is a *ghost* particle: the marker (which moves with
    the fluid) is coupled to the ghost by a spring ``k``; the ghost itself
    obeys ``M dv/dt = -F + M g`` and does not move with the fluid."""
    rows = _read_rows(path)
    assert rows.shape[1] == 3, "unexpected .mass layout"
    return dict(ids=rows[:, 0].astype(np.int64), k=rows[:, 1].copy(),
                M=rows[:, 2].copy())


def load_example(example_dir):
    """Read ``input2d`` and every structure file switched on in it.

    Returns a dict with ``params``, ``name``, ``X`` and -- depending on the
    ``input2d`` flags -- ``springs``, ``d_springs``, ``targets``, ``beams``
    (invariant), ``noninv_beams``, ``muscles``.
    """
    params = read_input2d(os.path.join(example_dir, "input2d"))
    name = params.get("string_name", "structure")
    data = {"params": params, "name": name,
            "X": read_vertex(os.path.join(example_dir, f"{name}.vertex"))}
    opts = (
        ("springs", "springs", read_spring),
        ("d_springs", "damped_springs", read_damped_spring),
        ("targets", "target_pts", read_target),
        ("beams", "beams", read_beam),
        ("noninv_beams", "nonInvariant_beams", read_noninv_beam),
        ("muscles", "FV_LT_muscle", read_muscle),
        ("mass", "mass_pts", read_mass),
    )
    for key, flag, reader in opts:
        fname = os.path.join(example_dir, f"{name}.{key_extension(key)}")
        if params.get(flag, 0) and os.path.exists(fname):
            data[key] = reader(fname)
    return data


def key_extension(key):
    """Structure key -> IB2d file extension."""
    return {"springs": "spring", "d_springs": "d_spring", "targets": "target",
            "beams": "beam", "noninv_beams": "nonInv_beam",
            "muscles": "muscle", "mass": "mass"}[key]
