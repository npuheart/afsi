"""Python port of the example's own ``please_Form_Boussinesq_Forcing_Terms.m``.

    fBouss_X = exp_Coeff * g * gravity_Info[1] * mat
    fBouss_Y = exp_Coeff * g * gravity_Info[2] * mat

where ``mat`` is the 0/1 mask of the Eulerian nodes listed in
``boussinesq.mesh`` (here: the whole grid, so ``mat = ones((Ny, Nx))``).

Two differences from the *pyIB2d* module shipped with
``pyIB2d/Examples/Rayleigh_Taylor_Instability`` (see demo_577), both faithful
to the MATLAB original:

* ``g = 9.81`` here, while the pyIB2d copy hard-codes ``g = 0.981`` -- an
  upstream inconsistency of a factor of 10 (demo_577 uses pyIB2d's value
  because that is the module the pyIB2d example ships);
* the indices in ``boussinesq.mesh`` are MATLAB 1-based, so they are shifted
  by -1 here (pyIB2d's data files are 0-based and its module does not shift).

``IBM_Driver.py`` imports this module from the run directory and calls it once
before the time loop; ``main.py`` does the same, so both codes use one
implementation.
"""
import numpy as np

G = 9.81      # gravitational constant (MATLAB original); pyIB2d uses 0.981


def get_Inds_Please():
    """Read ``boussinesq.mesh`` -> 0-based (xInds, yInds) Eulerian indices."""
    with open("boussinesq.mesh") as f:
        xInds, yInds = np.loadtxt(f, unpack=True)
    return xInds.astype(int) - 1, yInds.astype(int) - 1


def please_Form_Boussinesq_Forcing_Terms(exp_Coeff, Nx, Ny, gravity_Info):
    """(fBouss_X, fBouss_Y) on the (Ny, Nx) Eulerian grid."""
    mat = np.zeros((Ny, Nx))
    xInds, yInds = get_Inds_Please()
    mat[yInds, xInds] = 1

    fBouss_X = exp_Coeff * G * gravity_Info[1] * mat
    fBouss_Y = exp_Coeff * G * gravity_Info[2] * mat
    return fBouss_X, fBouss_Y
