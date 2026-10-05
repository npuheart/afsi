"""Readers for IB2d input files (``input2d``, ``*.vertex``, ``*.spring``,
``*.d_spring``, ``*.beam``, ``*.nonInv_beam``, ``*.target``, ``*.muscle``,
``*.mass``, ``*.tracer``).

The ``matIB2d`` examples number their Lagrangian points from 1 (MATLAB
indexing) while the ``pyIB2d`` examples number them from 0, and the python
driver indexes ``xLag`` with the identifiers it reads from the files.  The
identifiers are therefore shifted by ``-index_base`` in :func:`load_example`
(``index_base=0`` for the pyIB2d examples, the default, and
``index_base=1`` for the matIB2d ones); the files themselves are kept
verbatim.

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


def read_concentration(path, kdiff=None):
    """``.concentration`` -> dict(k, C0): the ``(Ny, Nx)`` initial
    concentration field plus its diffusion coefficient.

    Two upstream layouts exist: the ``pyIB2d`` driver reads the coefficient
    from the *first line* (``IBM_Driver.read_In_Concentration_Info``) while the
    ``matIB2d`` examples store the bare ``(Ny, Nx)`` field and take ``kDiff``
    from ``input2d``.  The layout is detected from the first line; ``kdiff``
    supplies the coefficient for the bare-field form."""
    with open(path) as fh:
        first = fh.readline().split()
    if len(first) == 1:
        return dict(k=float(first[0]), C0=np.loadtxt(path, skiprows=1))
    assert kdiff is not None, \
        f"{path}: bare concentration field but no kDiff in input2d"
    return dict(k=float(kdiff), C0=np.loadtxt(path))


def read_tracer(path):
    """``.tracer`` -> (Nt, 2) initial tracer positions (passive particles
    advected by the fluid, no forces)."""
    return _read_rows(path)[:, :2]


_ID_FIELDS = {"springs": ["conn"], "d_springs": ["conn"], "targets": ["ids"],
              "beams": ["p1", "p2", "p3"],
              "noninv_beams": ["p1", "p2", "p3"],
              "muscles": ["conn"], "mass": ["ids"]}


def load_example(example_dir, index_base=0):
    """Read ``input2d`` and every structure file switched on in it.

    ``index_base`` is the value of the first identifier in the files (1 for
    the MATLAB examples, 0 for the pyIB2d ones); all identifiers are returned
    0-based, i.e. shifted by ``-index_base``.

    Returns a dict with ``params``, ``name``, ``X`` and -- depending on the
    ``input2d`` flags -- ``springs``, ``d_springs``, ``targets``, ``beams``
    (invariant), ``noninv_beams``, ``muscles``, ``mass``, ``tracers``, ``concentration``.
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
        ("tracers", "tracers", read_tracer),
        ("concentration", "concentration", read_concentration),
    )
    for key, flag, reader in opts:
        fname = os.path.join(example_dir, f"{name}.{key_extension(key)}")
        if params.get(flag, 0) and os.path.exists(fname):
            data[key] = (reader(fname, params.get("kDiff"))
                         if key == "concentration" else reader(fname))
    if index_base != 0:
        shift = -int(index_base)
        for key, fields in _ID_FIELDS.items():
            if key not in data:
                continue
            for f in fields:
                data[key][f] = data[key][f] + shift
            ids = np.concatenate([np.asarray(data[key][f]).ravel()
                                  for f in fields])
            assert ids.min() >= 0, f"{key}: identifiers below 0 after shifting"
            assert ids.max() < len(data["X"]), \
                f"{key}: identifiers outside the vertex file after shifting"
    return data


def key_extension(key):
    """Structure key -> IB2d file extension."""
    return {"springs": "spring", "d_springs": "d_spring", "targets": "target",
            "beams": "beam", "noninv_beams": "nonInv_beam",
            "muscles": "muscle", "mass": "mass", "tracers": "tracer",
            "concentration": "concentration"}[key]
