"""Line-by-line port of the example's ``update_Springs.m``
(``Example_Jellyfish_Swimming/Failed_Attempts/Jellyfish_Spring_RL_Interpolation``).

pyIB2d's driver imports this module from the run directory (once per step, when
``update_springs`` is on) and calls

    springs_info = update_Springs(dt, current_time, xLag, yLag, springs_info)

with ``springs_info`` the (N_springs, 5) array ``[leader, follower, k, resting
length, alpha]`` and the *full-step* marker positions ``xLag``/``yLag``.

This is one of the authors' own ``Failed_Attempts/``.  It is the only
jellyfish variant with ``Lx = Ly = 1`` (a small square box).  The .m file
linearly interpolates every resting length from a "phase 1" to a "phase 2"
value over 0.2 s (``rate = 5``, i.e. ``RL = RL1 + 5 t (RL2 - RL1)``): the first
``Nbody = 106`` springs of the body use the Lagrangian spacings
``ds1 = 3.90625e-3`` -> ``ds2 = 3.342919e-3`` and the remaining springs use the
hand-tabulated 41x2 table below (MATLAB ``give_Resting_Lengths()``; only the
first rows are used, this geometry has just 17 springs past ``Nbody``).

MATLAB's 1-based indexing is translated to 0-based here; the identifiers
(1..N) are rebased by the reader (``ib2d_io.load_example`` auto-detects the
1-based numbering of this matIB2d example).
"""
import numpy as np

N_BODY = 106          # MATLAB Nbody: springs 1..106 use the ds interpolation
DS1 = 3.906250e-03    # MATLAB ds1 (phase-1 Lagrangian spacing)
DS2 = 3.342919e-03    # MATLAB ds2 (phase-2 Lagrangian spacing)
RATE = 5.0            # MATLAB 5*t: phases are 0.2 s apart
# MATLAB give_Resting_Lengths(): column 1 = phase 1, column 2 = phase 2
RL_PHASE1 = np.array([
    1.500000000000000e-01, 1.499267410002400e-01, 1.497068558575940e-01,
    1.493400206755370e-01, 1.488256974102940e-01, 1.481631365647090e-01,
    1.473513809960920e-01, 1.463886435715710e-01, 1.452740173653540e-01,
    1.440051528201540e-01, 1.425809534648290e-01, 1.409985443486210e-01,
    1.392556110620670e-01, 1.373485411896260e-01, 1.352744923184720e-01,
    1.330305169623240e-01, 1.306121687179200e-01, 1.280146004665640e-01,
    1.252341807455070e-01, 1.222656148322300e-01, 1.191033156575130e-01,
    1.157414514244010e-01, 1.121739994482140e-01, 1.083928187853590e-01,
    1.043935288440010e-01, 1.001656537548680e-01, 9.570265971856901e-02,
    9.099823702198900e-02, 8.604170105579199e-02, 8.082714047499900e-02,
    7.534928075486000e-02, 6.959854942222500e-02, 6.357370627912599e-02,
    5.727494134872200e-02, 5.070405027453000e-02, 4.387011940451200e-02,
    3.678730907709700e-02, 2.947790922672900e-02, 2.197542958631300e-02,
    1.431896613339300e-02, 6.558552552061000e-03])
RL_PHASE2 = np.array([
    7.500000000000000e-02, 7.497317043354500e-02, 7.489262401721300e-02,
    7.475818772418499e-02, 7.456972742982300e-02, 7.432676392445101e-02,
    7.402878152799100e-02, 7.367542142727400e-02, 7.326575266725800e-02,
    7.279892760636900e-02, 7.227418390174400e-02, 7.169030908812100e-02,
    7.104591276217000e-02, 7.033992922511401e-02, 6.957046963209600e-02,
    6.873540331745601e-02, 6.783297712288899e-02, 6.686070937819499e-02,
    6.581658631676900e-02, 6.469707842565800e-02, 6.349906786259200e-02,
    6.221844444780800e-02, 6.085077286398300e-02, 5.939131123548200e-02,
    5.783503332700000e-02, 5.617474946719600e-02, 5.440274784896800e-02,
    5.251083771212000e-02, 5.048721235819100e-02, 4.832036949743100e-02,
    4.599484164414300e-02, 4.349312099459900e-02, 4.079323575766200e-02,
    3.786597450600500e-02, 3.467691824701000e-02, 3.117990859268300e-02,
    2.731653789234300e-02, 2.300934935040300e-02, 1.816770778902700e-02,
    1.269459850757400e-02, 6.558010776142000e-03])


def update_Springs(dt, current_time, xLag, yLag, springs):
    t = current_time
    springs[:N_BODY, 3] = DS1 + RATE * t * (DS2 - DS1)
    n_tail = springs.shape[0] - N_BODY
    if n_tail > 0:
        rl1, rl2 = RL_PHASE1[:n_tail], RL_PHASE2[:n_tail]
        springs[N_BODY:, 3] = rl1 + RATE * t * (rl2 - rl1)
    return springs
