import os, re
ROOT = "/Users/pengfei.ma/afsi/afsic/demo/ib2d"
UP = "/Users/pengfei.ma/afsi/third_party/ib2d/matIB2d/Examples/Example_Jellyfish_Swimming"

HEAD = '''"""Line-by-line port of the example's ``update_Springs.m``
(``Example_Jellyfish_Swimming/{sub}``).

pyIB2d's driver imports this module from the run directory (once per step, when
``update_springs`` is on) and calls

    springs_info = update_Springs(dt, current_time, xLag, yLag, springs_info)

with ``springs_info`` the (N_springs, 5) array ``[leader, follower, k, resting
length, alpha]`` and the *full-step* marker positions ``xLag``/``yLag``.

{desc}

MATLAB's 1-based indexing is translated to 0-based here; the identifiers
(1..N) are rebased by the reader (``ib2d_io.load_example`` auto-detects the
1-based numbering of this matIB2d example).
"""
import numpy as np
'''

FILES = {}

FILES[640] = HEAD.format(
    sub="Hoover_Jellyfish",
    desc="""The .m file contracts the 39 springs of the bell margin (MATLAB rows
``319:357``) every step,

    RL_i = |cos(freq t pi)|,   freq = 2  ->  period 1 s,

``|cos|`` gives two contractions per period, so the bell (this is the Hoovers'
model, 512-resolution geometry) pulses twice per second.""") + '''
FREQ = 2.0        # contraction frequency: RL = |cos(freq t pi)|
FIRST_ROW = 319   # MATLAB springs_info(319:357, 4)
LAST_ROW = 357


def update_Springs(dt, current_time, xLag, yLag, springs):
    springs[FIRST_ROW - 1:LAST_ROW, 3] = \\
        abs(np.cos(FREQ * current_time * np.pi))
    return springs
'''

FILES[641] = HEAD.format(
    sub="Tentacle_Jelly",
    desc="""The .m file contracts the 39 tentacle springs (MATLAB rows
``starting_muscle:ending_muscle`` = ``318+1 : 318+39``) every step,

    RL_i = |cos(freq t pi)|,   freq = 0.8  ->  period 1.25 s,

so the tentacles beat 1.6 times per second.  The example additionally solves a
poroelastic displacement field (``poroelastic = 1``).""") + '''
FREQ = 0.8        # contraction frequency: RL = |cos(freq t pi)|
FIRST_ROW = 318 + 1        # MATLAB starting_muscle
LAST_ROW = 318 + 39        # MATLAB ending_muscle


def update_Springs(dt, current_time, xLag, yLag, springs):
    springs[FIRST_ROW - 1:LAST_ROW, 3] = \\
        abs(np.cos(FREQ * current_time * np.pi))
    return springs
'''

FILES[643] = HEAD.format(
    sub="Failed_Attempts/Jellyfish_Springs_Only",
    desc="""This is one of the authors' own ``Failed_Attempts/``: only the springs
are active (no beams, no target points), so the bell has no bending stiffness.
The .m file drives the 25 springs of the bell margin (MATLAB rows ``107:131``)
with the hand-tabulated resting lengths ``distVec``,

    RL_i = distVec_i * (1 - 0.75 sin(4 pi t)),

which is *not* an absolute value: the resting lengths alternate between 0.25x
and 1.75x of ``distVec`` twice per second.""") + '''
# hand-tabulated (MATLAB distVec, rows 107:131)
DIST_VEC = np.array([
    8.660254e-02, 8.829346e-02, 8.985511e-02, 9.129139e-02, 9.260716e-02,
    9.380641e-02, 9.489209e-02, 9.586726e-02, 9.673485e-02, 9.749671e-02,
    9.815495e-02, 9.871098e-02, 9.916637e-02, 9.952229e-02, 9.977934e-02,
    9.993819e-02, 9.999924e-02, 9.996260e-02, 9.982821e-02, 9.959568e-02,
    9.926457e-02, 9.883399e-02, 9.830305e-02, 9.767028e-02, 9.693457e-02])
FIRST_ROW = 107                       # MATLAB springs_info(107+(i-1), 4)
DIST = 8.649829e-02                   # MATLAB dist (tabulated reference length)


def update_Springs(dt, current_time, xLag, yLag, springs):
    rl = DIST_VEC * (1.0 - 0.75 * np.sin(4.0 * np.pi * current_time))
    springs[FIRST_ROW - 1:FIRST_ROW - 1 + DIST_VEC.size, 3] = rl
    return springs
'''

FILES[644] = HEAD.format(
    sub="Failed_Attempts/Jellyfish_With_Beams/Forward_Swimmer",
    desc="""This is one of the authors' own ``Failed_Attempts/`` (the folder is
called "Forward_Swimmer").  The bell carries *invariant* beams (``jelly.beam``,
``beams = 1``) and the .m file drives only the very last spring,

    RL(end) = dist * (1 - 0.5 |sin(10 pi t)|),   dist = 0.1250077,

so only a single spring is actuated -- 10 contractions per second (the
sub-grid actuation is why the authors filed this run as a failed attempt).""") + '''
DIST = 1.250077e-01    # MATLAB dist (distance between ends of jellyfish bell)


def update_Springs(dt, current_time, xLag, yLag, springs):
    springs[-1, 3] = DIST * (1.0 - 0.5 * abs(np.sin(10.0 * np.pi * current_time)))
    return springs
'''

FILES[645] = HEAD.format(
    sub="Failed_Attempts/Jellyfish_With_Beams/Backwards_Swimmer_for_LOLz",
    desc="""This is one of the authors' own ``Failed_Attempts/`` (the folder name
says it all: "Backwards_Swimmer_for_LOLz").  The bell carries *invariant* beams
(``jelly.beam``, ``beams = 1``) and the .m file drives only the very last
spring,

    RL(end) = dist * |cos(10 pi t)|,   dist = 0.08649829,

i.e. a single spring beating 20 times per second.""") + '''
DIST = 8.649829e-02    # MATLAB dist (distance between ends of jellyfish bell)


def update_Springs(dt, current_time, xLag, yLag, springs):
    springs[-1, 3] = DIST * abs(np.cos(10.0 * np.pi * current_time))
    return springs
'''

FILES[646] = HEAD.format(
    sub="Failed_Attempts/Jellyfish_IBAMR_Convert",
    desc="""This is one of the authors' own ``Failed_Attempts/``: a conversion of an
IBAMR jellyfish setup.  The .m file contracts the springs of the bell margin
(MATLAB rows ``319:end``) every step with a *full* sine (note the ``2*pi``, the
other jellyfish examples use ``pi`` and ``|cos|``),

    RL_i = |sin(freq t 2 pi)|,   freq = 0.875  ->  period 1.143 s,

so the bell contracts once per period.  The shipped ``input2d`` uses a 128x1024
box (16 long, 2 wide) with ``dt = 1e-5`` and ``Tfinal = 12``.""") + '''
FREQ = 0.875      # contraction frequency: RL = |sin(freq t 2 pi)|
FIRST_ROW = 319   # MATLAB springs_info(319:end, 4)


def update_Springs(dt, current_time, xLag, yLag, springs):
    springs[FIRST_ROW - 1:, 3] = \\
        abs(np.sin(FREQ * current_time * 2.0 * np.pi))
    return springs
'''

# ---- demo_642: RL interpolation, table parsed from the .m -------------------
src = open(os.path.join(UP, "Failed_Attempts/Jellyfish_Spring_RL_Interpolation",
                        "update_Springs.m")).read()
tbl = re.search(r"RL = \[(.*?)\];", src, re.S).group(1)
rows = [[float(x) for x in line.split()] for line in tbl.strip().splitlines()]
assert all(len(r) == 2 for r in rows) and len(rows) == 41, (len(rows), rows[:2])
def fmt(vals, per=3):
    out, chunk = [], []
    for v in vals:
        chunk.append("%.15e" % v)
        if len(chunk) == per:
            out.append(", ".join(chunk)); chunk = []
    if chunk:
        out.append(", ".join(chunk))
    return ",\n    ".join(out)
FILES[642] = HEAD.format(
    sub="Failed_Attempts/Jellyfish_Spring_RL_Interpolation",
    desc="""This is one of the authors' own ``Failed_Attempts/``.  It is the only
jellyfish variant with ``Lx = Ly = 1`` (a small square box).  The .m file
linearly interpolates every resting length from a "phase 1" to a "phase 2"
value over 0.2 s (``rate = 5``, i.e. ``RL = RL1 + 5 t (RL2 - RL1)``): the first
``Nbody = 106`` springs of the body use the Lagrangian spacings
``ds1 = 3.90625e-3`` -> ``ds2 = 3.342919e-3`` and the remaining springs use the
hand-tabulated 41x2 table below (MATLAB ``give_Resting_Lengths()``; only the
first rows are used, this geometry has just 17 springs past ``Nbody``).""") + '''
N_BODY = 106          # MATLAB Nbody: springs 1..106 use the ds interpolation
DS1 = 3.906250e-03    # MATLAB ds1 (phase-1 Lagrangian spacing)
DS2 = 3.342919e-03    # MATLAB ds2 (phase-2 Lagrangian spacing)
RATE = 5.0            # MATLAB 5*t: phases are 0.2 s apart
# MATLAB give_Resting_Lengths(): column 1 = phase 1, column 2 = phase 2
RL_PHASE1 = np.array([
    %s])
RL_PHASE2 = np.array([
    %s])


def update_Springs(dt, current_time, xLag, yLag, springs):
    t = current_time
    springs[:N_BODY, 3] = DS1 + RATE * t * (DS2 - DS1)
    n_tail = springs.shape[0] - N_BODY
    if n_tail > 0:
        rl1, rl2 = RL_PHASE1[:n_tail], RL_PHASE2[:n_tail]
        springs[N_BODY:, 3] = rl1 + RATE * t * (rl2 - rl1)
    return springs
''' % (fmt([r[0] for r in rows]), fmt([r[1] for r in rows]))

for demo, txt in sorted(FILES.items()):
    p = os.path.join(ROOT, "demo_%03d" % demo, "ib2d_input", "update_Springs.py")
    assert not os.path.exists(p), p
    with open(p, "w") as fh:
        fh.write(txt)
    print("wrote", p, len(txt))
