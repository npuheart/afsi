"""Readers for IB2d input files (``input2d``, ``*.vertex``, ``*.spring``,
``*.d_spring``, ``*.beam``, ``*.nonInv_beam``, ``*.target``, ``*.muscle``,
``*.mass``, ``*.porous``, ``*.poroelastic``, ``*.tracer``, ``*.concentration``,
``*.brinkman``, ``*.user_force``).

The ``matIB2d`` examples number their Lagrangian points from 1 (MATLAB
indexing) while the ``pyIB2d`` examples number them from 0, and the python
driver indexes ``xLag`` with the identifiers it reads from the files.  The
identifiers are therefore shifted by ``-index_base`` in :func:`load_example`
(0 for the pyIB2d examples, 1 for the matIB2d ones); the files themselves are
kept verbatim.  :func:`detect_index_base` guesses the convention from the
example's own structure files so that one template serves both families.

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
    """``.spring`` -> dict(conn, k, L, alpha); ``alpha`` is IB2d's degree of
    non-linearity (column 5, default 1 = linear)."""
    rows = _read_rows(path)
    alpha = rows[:, 4] if rows.shape[1] > 4 else np.ones(len(rows))
    return dict(conn=rows[:, :2].astype(np.int64), k=rows[:, 2].copy(),
                L=rows[:, 3].copy(), alpha=np.asarray(alpha, float))


def read_damped_spring(path):
    """``.d_spring`` -> dict(conn, k, L, b) (IB2d ``read_Damped_Spring_Points``
    columns: master, slave, kappa, resting length, damping coefficient)."""
    rows = _read_rows(path)
    assert rows.shape[1] >= 5, "unexpected .d_spring layout"
    return dict(conn=rows[:, :2].astype(np.int64), k=rows[:, 2].copy(),
                L=rows[:, 3].copy(), b=rows[:, 4].copy())


def read_beam(path):
    """``.beam`` (invariant torsional springs) -> dict(p1, p2, p3, kb, C).

    ``p2`` is the middle node.  Column 5 (``C``) is the reference cross
    product stored at generation time (0 for a straight beam).
    """
    rows = _read_rows(path)
    assert rows.shape[1] == 5, "unexpected .beam layout"
    return dict(p1=rows[:, 0].astype(np.int64), p2=rows[:, 1].astype(np.int64),
                p3=rows[:, 2].astype(np.int64), kb=rows[:, 3].copy(),
                C=rows[:, 4].copy())


def read_noninv_beam(path):
    """``.nonInv_beam`` -> dict(p1, p2, p3, kb, C) with ``C`` the reference
    vector ``(p1 + p3 - 2 p2)`` at rest (two columns of the file)."""
    rows = _read_rows(path)
    assert rows.shape[1] == 6, "unexpected .nonInv_beam layout"
    return dict(p1=rows[:, 0].astype(np.int64), p2=rows[:, 1].astype(np.int64),
                p3=rows[:, 2].astype(np.int64), kb=rows[:, 3].copy(),
                C=np.column_stack([rows[:, 4], rows[:, 5]]).astype(float))


def read_target(path):
    """``.target`` -> dict(ids, k); the anchors are the initial positions of
    the flagged markers (filled in by the driver / the port)."""
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
    obeys ``M dv/dt = -F + M g`` and does not move with the fluid.
    (The mass mechanism is *not* implemented by the template.)"""
    rows = _read_rows(path)
    assert rows.shape[1] == 3, "unexpected .mass layout"
    return dict(ids=rows[:, 0].astype(np.int64), k=rows[:, 1].copy(),
                M=rows[:, 2].copy())


def read_porous(path):
    """``.porous`` -> dict(ids, kappa, stencil) -- IB2d ``read_Porous_Points``
    columns: Lagrangian id, porosity coefficient, stencil flag
    (-2..2 = which one-sided/central derivative the point uses along the
    porous curve)."""
    rows = _read_rows(path)
    assert rows.shape[1] == 3, "unexpected .porous layout"
    return dict(ids=rows[:, 0].astype(np.int64), kappa=rows[:, 1].copy(),
                stencil=rows[:, 2].astype(np.int64))


def read_poroelastic(path):
    """``.poroelastic`` -> dict(ids, c) -- IB2d ``read_PoroElastic_Points``:
    Lagrangian id + coefficient ``c``; each flagged point additionally moves by
    ``dt * F_spring / (mu c)`` (with F_spring the *spring-only* force at X_h)."""
    rows = _read_rows(path)
    assert rows.shape[1] == 2, "unexpected .poroelastic layout"
    return dict(ids=rows[:, 0].astype(np.int64), c=rows[:, 1].copy())


def read_tracer(path):
    """``.tracer`` -> (Nt, 2) initial tracer positions (passive particles
    advected by the fluid, no force feedback)."""
    return _read_rows(path)[:, :2]


def read_concentration(path):
    """``.concentration`` -> dict(k, C0): the first line is the diffusion
    coefficient, the rest the (Ny, Nx) initial concentration field
    (IB2d ``read_In_Concentration_Info``)."""
    with open(path) as fh:
        k = float(fh.readline().strip())
    return dict(k=k, C0=np.loadtxt(path, skiprows=1))


def read_brinkman(path):
    """``.brinkman`` -> dict(k, Brink): permeability coefficient followed by the
    (Ny, Nx) resistance map (IB2d ``read_In_Brinkman_Info``; the map is used
    exactly as stored, ``F -= mu * Brink * u^{n}``)."""
    with open(path) as fh:
        k = float(fh.readline().strip())
        brink = np.loadtxt(fh)
    return dict(k=k, Brink=brink)


def read_user_force(path):
    """``.user_force`` -> the raw transposed array IB2d hands to the user's
    ``give_Me_General_User_Defined_Force_Densities`` (``np.loadtxt(unpack=True)``:
    shape ``(n_columns, n_rows)``).

    The layout is completely free, so the array is copied verbatim -- a ported
    user module receives exactly what pyIB2d would pass it.
    """
    with open(path) as fh:
        fh.readline()                     # first line: number of rows
        return np.loadtxt(fh, unpack=True)


_ID_FIELDS = {"springs": ["conn"], "d_springs": ["conn"],
              "targets": ["ids"], "beams": ["p1", "p2", "p3"],
              "noninv_beams": ["p1", "p2", "p3"],
              "muscles": ["conn"], "mass": ["ids"], "porous": ["ids"],
              "poroelastic": ["ids"]}

# number of leading columns holding Lagrangian identifiers, per file extension
_ID_COLS = {"spring": 2, "d_spring": 2, "beam": 3, "nonInv_beam": 3,
            "target": 1, "muscle": 2, "mass": 1, "porous": 1, "poroelastic": 1}


def detect_index_base(example_dir):
    """Guess whether the structure files number their points from 0 or 1.

    Reads the identifiers of the first structure file present and returns
    ``0`` when they start at 0, ``1`` otherwise.  ``.user_force`` is checked
    last because its layout is free (the user module, not IB2d, interprets it).
    """
    params = read_input2d(os.path.join(example_dir, "input2d"))
    name = params.get("string_name", "structure")
    for ext in ("spring", "beam", "nonInv_beam", "target", "muscle", "mass",
                "d_spring", "porous", "poroelastic", "user_force"):
        path = os.path.join(example_dir, f"{name}.{ext}")
        if os.path.exists(path):
            if ext == "user_force":
                with open(path) as fh:
                    fh.readline()
                    rows = np.atleast_2d(np.loadtxt(fh))
            else:
                rows = _read_rows(path)
            return 0 if rows[:, 0].min() == 0 else 1
    return 0


def load_example(example_dir, index_base=None):
    """Read ``input2d`` and every structure file switched on in it.

    ``index_base`` is the value of the first identifier in the files (1 for the
    MATLAB examples, 0 for the pyIB2d ones); all identifiers are returned
    0-based, i.e. shifted by ``-index_base``.  ``None`` (default) auto-detects
    it with :func:`detect_index_base`.

    Returns a dict with ``params``, ``name``, ``X`` and -- depending on the
    ``input2d`` flags -- ``springs``, ``d_springs``, ``targets``, ``beams``
    (invariant), ``noninv_beams``, ``muscles``, ``mass``, ``porous``,
    ``poroelastic``, ``tracers``, ``concentration``, ``brinkman``,
    ``user_force``.
    """
    params = read_input2d(os.path.join(example_dir, "input2d"))
    name = params.get("string_name", "structure")
    if index_base is None:
        index_base = detect_index_base(example_dir)
    data = {"params": params, "name": name, "index_base": int(index_base),
            "X": read_vertex(os.path.join(example_dir, f"{name}.vertex"))}
    opts = (
        ("springs", "springs", read_spring),
        ("d_springs", "damped_springs", read_damped_spring),
        ("targets", "target_pts", read_target),
        ("beams", "beams", read_beam),
        ("noninv_beams", "nonInvariant_beams", read_noninv_beam),
        ("muscles", "FV_LT_muscle", read_muscle),
        ("mass", "mass_pts", read_mass),
        ("porous", "porous_media", read_porous),
        ("poroelastic", "poroelastic", read_poroelastic),
        ("tracers", "tracers", read_tracer),
        ("concentration", "concentration", read_concentration),
        ("brinkman", "brinkman", read_brinkman),
        ("user_force", "user_force_model", read_user_force),
    )
    for key, flag, reader in opts:
        fname = os.path.join(example_dir, f"{name}.{key_extension(key)}")
        if params.get(flag, 0) and os.path.exists(fname):
            data[key] = reader(fname)
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
            "muscles": "muscle", "mass": "mass", "porous": "porous",
            "poroelastic": "poroelastic", "tracers": "tracer",
            "concentration": "concentration", "brinkman": "brinkman",
            "user_force": "user_force"}[key]
