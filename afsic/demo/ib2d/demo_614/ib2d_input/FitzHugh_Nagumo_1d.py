"""Line-by-line port of the example's ``FitzHugh_Nagumo_1d.m``
(``Example_HeartTube/128x128_Electromechanical_Pumping_w_Ca_Dynamics``,
    the higher-resolution variant: 310 markers / 463 springs, ``Lx x Ly =
    5 x 2.5``, ``dt = 1e-5``, ``numFHN = 1``, ``numPtsAlongTube = 155``).

The example solves a 1-d FitzHugh-Nagumo reaction-diffusion model

    dv/dt = D v_xx + v (v - a) (v - 1) - w + I(t)
    dw/dt = eps (v - gamma w)

*once*, a-priori, before the immersed-boundary loop, on its own 1-d grid
(``numPtsAlongTube`` IBM points resolved with ``factor = 4`` finer spacing),
and returns the membrane potential sampled at the IBM tube points together
with the range of *spring rows* the driver then modulates.

Difference from the plain ``Electromechanical_Pumping`` example: the applied
current ``I(t)`` is not a prescribed pulse train but the **calcium signal**
(``Iapp_from_Calcium_Dynamics``): ``I(t) = 0.5 * Caf(t)`` in the injection
window ``j = floor(i1 N) .. floor(i2 N)``, where ``Caf`` comes from the
example's ``Calcium_Dynamics.m`` (``Calcium_Dynamics(Nsteps, T_final)`` with
``Nsteps = length(tVec)``, i.e. the IBM-step count, so ``dt_Ca = dt_IBM /
numFHN`` to round-off).  The ``Iapp`` state is rebuilt from scratch on every
step in that variant (unlike the persistent ``IIapp`` of the prescribed-pulse
version), which is reproduced here.

The MATLAB original runs nested loops; this port keeps the same arithmetic but
vectorises the spatial axis (the time loop stays explicit, exactly as the .m).
Everything the original computes is reproduced, including:

* the **non-periodic** Laplacian (the .m's periodic lines are commented out:
  ``DD(1) = (v(2) - 2 v(1) + 0)/dx^2``, ``DD(end) = (0 - 2 v(end) + v(end-1))/dx^2``),
* the storage cadence: one IMF frame every ``numFHN = 10`` model steps, the
  first row being the initial condition (FitzHugh-Nagumo is scaled by 2500, and
  the ``t = t + dt`` accumulation is kept so that the stimulus windows switch on
  the same steps as in MATLAB).

Deviations from the .m, both without physical effect:

* only the membrane potential ``vStore`` is stored (``wStore`` and the unused
  ``IIappStore`` are written by the original but never read);
* the storage array is sized to the true number of stores instead of relying on
  MATLAB's silent auto-grow by one row (the .m allocates ``NT_IBM`` rows but
  writes ``NT_IBM + 1`` -- see the readme).

``dt_IBM``/``T_final`` are given in IBM units; the .m rescales them by 2500
(the model is run in a scaled time unit).  Reducing ``T_final`` only truncates
the horizon here: the calcium stimulus is periodic in *physical* time
(``period = 0.5`` s), so its timing does not depend on ``T_final``.
"""
import numpy as np

from Calcium_Dynamics import Calcium_Dynamics

# --- IBM geometry of this example (see HeartTube.m) ------------------------
NUM_PTS_ALONG_TUBE = 155   # .m: numPtsAlongTube
IND_START = 309            # .m: ind_Start -- first modulated spring row
IND_END = 463              # .m: ind_End   -- last modulated spring row

# --- model parameters (.m) -------------------------------------------------
D = 10.0          # diffusion coefficient
A_THRESH = 0.3    # threshold potential
GAMMA = 1.0       # resetting rate
EPS = 0.001       # blocking strength

FACTOR = 4        # FHN resolution relative to the IBM tube resolution
L_DOMAIN = 500.0  # length of the FHN domain
NUM_FHN = 1       # .m: numFHN -- FHN steps per IBM step
I1, I2 = 0.25, 0.3   # .m: i1, i2 -- current injection window (fraction of N)


def FitzHugh_Nagumo_1d(dt_IBM, T_final):
    """Port of ``FitzHugh_Nagumo_1d.m``.

    Returns ``(v_INTERP, ind_Start, ind_End)``: the membrane potential sampled
    at the IBM tube points, shape ``(num_steps + 1, numPtsAlongTube)`` with
    ``row k`` the state at ``t = (k - 1) * dt_IBM`` (IBM step ``k``, 1-based,
    which is the row the driver reads via its ``ePhys_Ct`` counter), and the
    spring rows the driver overwrites with ``(16 v)^4``.
    """
    T_final_ibm = T_final
    dt_IBM = dt_IBM * 2500.0
    T_final = T_final * 2500.0

    N = FACTOR * NUM_PTS_ALONG_TUBE - 1     # # of spatial steps
    dx = L_DOMAIN / N
    dt = dt_IBM / NUM_FHN
    nt = int(round(T_final / dt))           # .m: NT
    nt_ibm = int(round(T_final / dt_IBM))   # .m: NT_IBM

    nsteps = nt + 1                         # .m: length(0:dt:T_final)
    n_stores = (nsteps - 1) // NUM_FHN      # mod(k, numFHN) == 0 for k = 1..nsteps-1
    # the .m allocates nt_ibm rows and (due to the off-by-one described in the
    # docstring) ends up with n_stores + 1 used rows.  Only the (every FACTOR-th)
    # values the .m's ``compute_IBM_Potential_At_IBM_Lag_Pts`` extracts are
    # stored, which is the subset everything downstream uses.
    v_store = np.zeros((max(n_stores + 1, nt_ibm + 1), NUM_PTS_ALONG_TUBE))

    v = np.zeros(N + 1)
    w = np.zeros(N + 1)
    j_lo = int(np.floor(I1 * N)) - 1        # MATLAB j = floor(i1 N) (1-based)
    j_hi = int(np.floor(I2 * N))            # .. floor(i2 N), inclusive -> slice end
    v_store[0] = v[::FACTOR]
    store = 1
    t = 0.0

    # the .m solves the calcium dynamics a-priori, with the IBM-scale horizon
    # (``Calcium_Dynamics(Nsteps, T_final_Ca)``, T_final_Ca unscaled)
    _Ca, Caf = Calcium_Dynamics(nsteps, T_final_ibm)

    inv_dx2 = 1.0 / dx ** 2
    for k in range(1, nsteps):              # .m: i = 2..Nsteps
        t = t + dt                          # accumulated, exactly as the .m
        # --- Laplacian, non-periodic (the .m's periodic version is commented out)
        dd = np.empty(N + 1)
        dd[1:-1] = v[2:] - 2.0 * v[1:-1] + v[:-2]
        dd[0] = v[1] - 2.0 * v[0]
        dd[-1] = -2.0 * v[-1] + v[-2]
        dd *= inv_dx2
        # --- applied current from the calcium signal
        #     (.m: IIapp = zeros(1,N+1); IIapp(j) = 0.5*Caf(i) for the window)
        app = np.zeros(N + 1)
        app[j_lo:j_hi] = 0.5 * Caf[k + 1]   # 1-based model step i = k + 1
        # --- forward Euler
        v_new = v + dt * (D * dd - v * (v - A_THRESH) * (v - 1.0) - w + app)
        w = w + dt * (EPS * (v - GAMMA * w))
        v = v_new
        # --- store every numFHN-th model step (only the IBM tube points) ----
        if k % NUM_FHN == 0:
            v_store[store] = v[::FACTOR]
            store += 1

    return v_store, IND_START, IND_END


def example_params(example_dir):
    """``(dt, Tfinal)`` of the example's ``input2d``, overridable by the
    environment variables ``IB2D_DT`` / ``IB2D_TFINAL`` (used by
    ``run_reference.py`` when it reduces ``T`` or ``dt``)."""
    import os
    import re
    dt = tend = None
    with open(os.path.join(example_dir, "input2d")) as fh:
        for line in fh:
            line = line.split("%", 1)[0]
            m = re.match(r"\s*dt\s*=\s*([0-9.eE+-]+)", line)
            if m:
                dt = float(m.group(1))
            m = re.match(r"\s*Tfinal\s*=\s*([0-9.eE+-]+)", line)
            if m:
                tend = float(m.group(1))
    if os.environ.get("IB2D_DT"):
        dt = float(os.environ["IB2D_DT"])
    if os.environ.get("IB2D_TFINAL"):
        tend = float(os.environ["IB2D_TFINAL"])
    return dt, tend
