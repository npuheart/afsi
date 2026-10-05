"""Line-by-line port of the example's ``Calcium_Dynamics.m``
(``Example_HeartTube/Electromechanical_Pumping_w_Ca_Dynamics``).

Model (Williams et al. 2008, law of mass action, dimensionless):

    dCa /dt = (k4 Caf - k3 Ca)(1 - Caf) + k1 (C - Ca - Caf)          [stimulus ON]
    dCa /dt = (k4 Caf - k3 Ca)(1 - Caf) + k2 Ca (C - S - Ca - Caf)   [stimulus OFF]
    dCaf/dt = -(k4 Caf - k3 Ca)(1 - Caf)

integrated with forward Euler; ``Ca`` is the free calcium concentration and
``Caf`` the calcium bound to the contractile filaments.  The stimulus is a
square wave of length ``act_period`` every ``period`` seconds:

    stimulus ON  <=>  rem(dt * i, period) < act_period,

with ``dt = t_Final / Nsteps`` (``i`` the 1-based step index).  This is what the
example's ``FitzHugh_Nagumo_1d.m`` uses in place of the prescribed current pulse
(``Iapp_from_Calcium_Dynamics``: ``Iapp = 0.5 * Caf`` in the injection window).

The .m indexes ``Ca(i+1)`` on the last step, so MATLAB grows both arrays to
``Nsteps + 1`` entries; that is reproduced here (only the first ``Nsteps`` are
ever read, and only ``Caf`` is used downstream).
"""
import numpy as np

K1, K2, K3, K4 = 9.0, 50.0, 40.0, 19.4    # binding / release rates (1/s)
C_TOTAL = 200.0                            # total dimensionless Ca2+
S_TOTAL = 200.0                            # total SR binding sites
PERIOD = 0.5                               # period of the Ca wave (s)
ACT_PERIOD = 0.005                         # stimulus length (s)


def Calcium_Dynamics(Nsteps, t_Final):
    """Port of ``Calcium_Dynamics.m``: returns ``(Ca, Caf)`` of length
    ``Nsteps + 1`` (MATLAB's auto-grown size)."""
    dt = t_Final / Nsteps
    Ca = np.zeros(Nsteps + 1)
    Caf = np.zeros(Nsteps + 1)
    for i in range(2, Nsteps + 1):          # .m: i = 2..Nsteps (1-based)
        j = i - 1                            # previous step, 0-based
        uptake_ca = K2 * Ca[j] * (C_TOTAL - S_TOTAL - Ca[j] - Caf[j])
        if np.remainder(dt * i, PERIOD) < ACT_PERIOD:
            uptake_ca = uptake_ca + K1 * (C_TOTAL - Ca[j] - Caf[j])
        rhs_ca = K4 * Caf[j] - K3 * Ca[j] * (1.0 - Caf[j]) + uptake_ca
        rhs_caf = -K4 * Caf[j] + K3 * Ca[j] * (1.0 - Caf[j])
        Ca[i] = Ca[j] + rhs_ca * dt
        Caf[i] = Caf[j] + rhs_caf * dt
    return Ca, Caf
