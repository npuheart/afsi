import os
ROOT = "/Users/pengfei.ma/afsi/afsic/demo/ib2d"
TMPL = '''"""Line-by-line port of the example's ``update_Springs.m``
(``Examples_Education/Convergence/Jellyfish/Simulation_Skeletons/{re}/{res}``).

pyIB2d's driver imports this module from the run directory (once per step, when
``update_springs`` is on) and calls

    springs_info = update_Springs(dt, current_time, xLag, yLag, springs_info)

with ``springs_info`` the (N_springs, 5) array ``[leader, follower, k, resting
length, alpha]`` and the *full-step* marker positions ``xLag``/``yLag``.  The .m
file makes the jellyfish bell pulsate by re-setting the resting lengths of the
springs of the bell margin (MATLAB rows ``{first}:end`` of the {nsp}-spring
{ladder} structure) every step:

    RL_i = |cos(freq t pi)|,   freq = 1  ->  period 1 s,

``|cos|`` gives two contractions per period, so the bell pulses twice per
second.

MATLAB's 1-based indexing is translated to 0-based here; the identifiers
(1..N) are rebased by the reader (``ib2d_io.load_example`` auto-detects the
1-based numbering of this matIB2d example).
"""
import numpy as np

FREQ = 1.0        # contraction frequency: RL = |cos(freq t pi)|
FIRST_ROW = {first}    # MATLAB springs_info({first}:end, 4)


def update_Springs(dt, current_time, xLag, yLag, springs):
    springs[FIRST_ROW - 1:, 3] = abs(np.cos(FREQ * current_time * np.pi))
    return springs
'''
# demo -> (re, res, first_row, n_springs)
D = {
 560: ("Re37pt5", "Res_96_120x36", 59, 65),
 561: ("Re37pt5", "Res_768_960x288", 477, 535),
 562: ("Re37pt5", "Res_512_640x192", 319, 357),
 563: ("Re37pt5", "Res_48_60x18", 29, 31),
 564: ("Re37pt5", "Res_384_480x144", 239, 267),
 565: ("Re37pt5", "Res_256_320x96", 159, 177),
 566: ("Re150", "Res_192_240x72", 119, 132),
 567: ("Re150", "Res_64_80x24", 39, 42),
 568: ("Re150", "Res_32_40x12", 19, 20),
 569: ("Re150", "Res_128_160x48", 79, 87),
 633: ("Re150", "Res_1024_1280x384", 637, 715),
 634: ("Re150", "Res_96_120x36", 59, 65),
 635: ("Re150", "Res_768_960x288", 477, 535),
 636: ("Re150", "Res_512_640x192", 319, 357),
 637: ("Re150", "Res_48_60x18", 29, 31),
 638: ("Re150", "Res_384_480x144", 239, 267),
 639: ("Re150", "Res_256_320x96", 159, 177),
}
for demo, (re, res, first, nsp) in D.items():
    txt = TMPL.format(re=re, res=res, first=first, nsp=nsp, ladder="jelly" + res.split("_")[1])
    p = os.path.join(ROOT, "demo_%03d" % demo, "ib2d_input", "update_Springs.py")
    assert not os.path.exists(p), p
    open(p, "w").write(txt)
    print("wrote", p)
