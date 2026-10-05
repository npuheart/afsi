"""Concentration source/sink term of IB2d -- line-by-line port of

    matIB2d/IBM_Blackbox/please_Find_Source_For_Concentration.m
    matIB2d/IBM_Blackbox/please_Update_Adv_Diff_Concentration_Source.m
      (``perform_Upwind_source`` + its private ``give_Necessary_Derivative``)

**Why this module exists.**  Every example of
``Examples/Example_Concentration_Dynamics`` (and both seminar ``*Corals*``
examples) runs with ``source > 0`` in the ``Concentration_Info`` block, i.e. the
immersed boundary emits (``source = 1``, ``2``) or absorbs (``source = 3``) the
scalar.  The *pyIB2d* reference driver has no counterpart at all: its
``read_In_Concentration_Info`` reads only ``kDiff`` + the field, and its main
loop calls ``please_Update_Adv_Diff_Concentration`` unconditionally -- the
``source`` / ``k_source`` / ``c_inf`` lines of ``input2d`` are silently ignored
(upstream gap, reported in the demo ``readme.md``).  Without this module the
reference would solve plain advection-diffusion and, with the ``C0 == 0``
shipped by the three ``Rubberband_*`` examples, would stay identically zero.

``run_reference.py`` therefore splices the two MATLAB routines into the pyIB2d
driver:

    please_Find_Source_For_Concentration(...)        -- marker value CL, source
        density fs(s), spreading ``Fs = delta_Y * diag(fs*ds) * delta_X``;
    please_Update_Adv_Diff_Concentration_Source(...) -- the *un-split* upwind
        advection-diffusion step ``C += dt (k lap C - u.C + Fs)`` that matIB2d
        uses on its ``source > 0`` branch (``Adv_flag = 0``; the 3rd-order WENO
        branch needs ``WENO_3O.m`` and is implemented by neither code).

Only the periodic 4-point Peskin kernel is used, through pyIB2d's own
``Supp.give_1D_NonZero_Delta_Indices`` / ``Supp.give_Delta_Kernel`` /
``Supp.DD`` (the helpers the driver itself uses, i.e. the grid transfer already
verified against AFSI in demo_476/477).  The ``flag_Geo_Connect`` branch is not
used: pyIB2d has no ``.geo_connect`` reader, so the examples' ``*.geo_connect``
files are inert here and ``fsds = fs * ds`` with ``ds = Lx/(2 Nx)`` (Peskin's
constant spacing, documented in the readme).
"""
import numpy as np

import Supp


def _grid(grid_Info):
    """The 9 entries of MATLAB's ``grid_Info``."""
    return (int(grid_Info[0]), int(grid_Info[1]), float(grid_Info[2]),
            float(grid_Info[3]), float(grid_Info[4]), float(grid_Info[5]),
            int(grid_Info[6]), int(grid_Info[7]), float(grid_Info[8]))


def _periodic_distance(a, b, L):
    d = np.abs(a - b)
    return np.minimum(d, L - d)


def _nonzero_delta_indices(xLag, yLag, Nx, Ny, dx, dy, supp):
    """MATLAB ``give_NonZero_Delta_Indices_XY``: the (Nb, supp^2) matrices of
    *paired* Eulerian indices (all supp x supp combinations per marker)."""
    xAux = Supp.give_1D_NonZero_Delta_Indices(xLag, Nx, dx, supp).astype(int)
    yAux = Supp.give_1D_NonZero_Delta_Indices(yLag, Ny, dy, supp).astype(int)
    return np.tile(xAux, (1, supp)), np.repeat(yAux, supp, axis=1)


def _delta_function_entries(xLag, yLag, grid_Info):
    """The non-zero entries of ``delta_X (Nb x Nx)`` / ``delta_Y (Ny x Nb)``.

    This is MATLAB's nested helper
    ``give_Me_Delta_Function_Approximations_For_Force_Calc``: one *1-D* stencil
    per marker and direction (``supp`` nodes each), stored as the full matrices
    that the spreading ``Fs = delta_Y * diag(fs*ds) * delta_X`` multiplies.

    Returns ``(iX, iY, phiX, phiY)`` with ``iX, iY`` of shape (Nb, supp) and
    ``phiX`` of shape (Nb, supp), ``phiY`` of shape (supp, Nb) -- the MATLAB
    layout (``indX``, ``indY``, ``delta_1D_x``, ``delta_1D_y``).
    """
    Nx, Ny, Lx, Ly, dx, dy, supp, Nb, ds = _grid(grid_Info)
    xs = np.arange(Nx) * dx
    ys = np.arange(Ny) * dy
    iX = Supp.give_1D_NonZero_Delta_Indices(xLag, Nx, dx, supp).astype(int)
    iY = Supp.give_1D_NonZero_Delta_Indices(yLag, Ny, dy, supp).astype(int)
    # x(indX) - xLag(ind_Lag), y(indY) - yLag(ind_Lag') -- both periodic
    distX = _periodic_distance(xs[iX], xLag[:, None], Lx)
    distY = _periodic_distance(ys[iY].T, yLag[None, :], Ly)
    phiX = Supp.give_Delta_Kernel(np.ascontiguousarray(distX), dx)
    phiY = Supp.give_Delta_Kernel(np.ascontiguousarray(distY), dy)
    assert iX.shape == (Nb, supp) and iY.shape == (Nb, supp)
    return iX, iY, phiX, phiY


def _lagrangian_concentration(C, xLag, yLag, grid_Info):
    """MATLAB ``give_Lagrangian_Concentration``: the concentration *at the
    markers*,

        C_Lag(i) = dx dy sum_j C(yInd(i,j), xInd(i,j)) dX(i,j) dY(i,j)

    summed over the supp^2 paired index combinations of marker ``i``.
    """
    Nx, Ny, Lx, Ly, dx, dy, supp, Nb, ds = _grid(grid_Info)
    xs = np.arange(Nx) * dx
    ys = np.arange(Ny) * dy
    xInds, yInds = _nonzero_delta_indices(xLag, yLag, Nx, Ny, dx, dy, supp)
    distX = _periodic_distance(xs[xInds], xLag[:, None], Lx)
    distY = _periodic_distance(ys[yInds], yLag[:, None], Ly)
    dX = Supp.give_Delta_Kernel(np.ascontiguousarray(distX), dx)
    dY = Supp.give_Delta_Kernel(np.ascontiguousarray(distY), dy)
    return dx * dy * np.sum(C[yInds, xInds] * dX * dY, axis=1)


def please_Find_Source_For_Concentration(dt, current_time, xLag, yLag, x, y,
                                        grid_Info, model_Info, k, C, ci,
                                        flag_Geo_Connect=0, geo_Connect_MAT=None):
    """Source/sink density ``Fs`` on the Eulerian grid (MATLAB signature).

    ``model_Info``: 1 = constant (``fs = k``), 2 = limited (``fs = k(ci - CL)``),
    3 = reaction/sink (``fs = k CL``).  ``dt`` / ``current_time`` / ``x`` / ``y``
    are kept for signature compatibility (MATLAB does not use them either,
    apart from its debug dump).
    """
    assert not flag_Geo_Connect, \
        "the trapezoid-rule (.geo_connect) branch is not ported (pyIB2d has no "\
        ".geo_connect reader; both codes use Peskin's constant ds)"
    Nx, Ny, Lx, Ly, dx, dy, supp, Nb, ds = _grid(grid_Info)
    xLag = np.mod(np.asarray(xLag, dtype=float), Lx)
    yLag = np.mod(np.asarray(yLag, dtype=float), Ly)

    CL = _lagrangian_concentration(C, xLag, yLag, grid_Info)

    model_Info = int(model_Info)
    if model_Info == 1:
        fs = np.zeros(Nb) + k
    elif model_Info == 2:
        fs = k * (ci - CL)
    elif model_Info == 3:
        fs = k * CL
    else:
        raise ValueError(f"Invalid source model! ({model_Info})")

    # delta_X (Nb, Nx) / delta_Y (Ny, Nb) of the nested MATLAB helper
    iX, iY, phiX, phiY = _delta_function_entries(xLag, yLag, grid_Info)
    deltaX = np.zeros((Nb, Nx))
    deltaY = np.zeros((Ny, Nb))
    rows = np.arange(Nb)
    for j in range(supp):
        deltaX[rows, iX[:, j]] = phiX[:, j]
        deltaY[iY[:, j], rows] = phiY[j, :]

    # Fs = delta_Y * diag(fs * ds) * delta_X  (Peskin constant ds)
    return deltaY @ (deltaX * (fs * ds)[:, None])


def give_Necessary_Derivative(C, dz, uZ, string):
    """Upwind 1st derivative of the *source* time-stepper.

    It differs from ``Supp.give_Necessary_Derivative``: MATLAB's copy inside
    ``please_Update_Adv_Diff_Concentration_Source.m`` loops ``for j = 2:lenX-1``
    and therefore *does* fill the second-to-last row/column (0-based
    ``1 .. n-2``), whereas pyIB2d's standalone helper stops at
    ``range(1, n-2)`` and leaves that line zero.  Reproduced verbatim from the
    MATLAB file (that difference is real -- it is why these source examples are
    not bit-comparable with the ``source = 0`` ones).
    """
    C_z = np.zeros(C.shape)
    signs = np.sign(uZ)
    n = C.shape[0] if string == 'y' else C.shape[1]
    if string == 'x':
        C_z[:, 0] = np.where(signs[:, 0] <= 0, (C[:, 1] - C[:, 0]) / dz,
                             (C[:, 0] - C[:, n - 1]) / dz)
        C_z[:, n - 1] = np.where(signs[:, n - 1] <= 0,
                                 (C[:, 0] - C[:, n - 1]) / dz,
                                 (C[:, n - 1] - C[:, n - 2]) / dz)
        j = np.arange(1, n - 1)
        C_z[:, j] = np.where(signs[:, j] <= 0, (C[:, j + 1] - C[:, j]) / dz,
                             (C[:, j] - C[:, j - 1]) / dz)
    elif string == 'y':
        C_z[0, :] = np.where(signs[0, :] <= 0, (C[1, :] - C[0, :]) / dz,
                             (C[0, :] - C[n - 1, :]) / dz)
        C_z[n - 1, :] = np.where(signs[n - 1, :] <= 0,
                                 (C[0, :] - C[n - 1, :]) / dz,
                                 (C[n - 1, :] - C[n - 2, :]) / dz)
        i = np.arange(1, n - 1)
        C_z[i, :] = np.where(signs[i, :] <= 0, (C[i + 1, :] - C[i, :]) / dz,
                             (C[i, :] - C[i - 1, :]) / dz)
    else:
        raise ValueError("Need to specify which desired derivative, x or y.")
    return C_z


def please_Update_Adv_Diff_Concentration_Source(C, dt, dx, dy, uX, uY, k, f,
                                                Lx, Ly, Adv_flag=0):
    """MATLAB ``please_Update_Adv_Diff_Concentration_Source`` (``Adv_flag = 0``).

        C = perform_Upwind_source(C, dt, dx, dy, uX, uY, k, f)

    with the *un-split* update ``C += dt (k (Cxx + Cyy) - (uX Cx + uY Cy) + f)``
    and periodic central second differences (``DD``).
    """
    if int(Adv_flag) != 0:
        raise NotImplementedError(
            "Adv_flag = 1 is the 3rd-order WENO advection (WENO_3O.m), which "
            "neither pyIB2d nor the AFSI port implements")
    Cx = give_Necessary_Derivative(C, dx, uX, 'x')
    Cxx = Supp.DD(C, dx, 'x')
    Cy = give_Necessary_Derivative(C, dy, uY, 'y')
    Cyy = Supp.DD(C, dy, 'y')
    return C + dt * (k * (Cxx + Cyy) - (uX * Cx + uY * Cy) + f)
