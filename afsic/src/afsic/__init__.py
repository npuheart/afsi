"""AFSI — Automated Fluid-Structure Interaction Solver."""

__version__ = "0.0.1"

from petsc4py import PETSc


from .afsic_ext import add, __doc__
from .afsic_ext import send_email, EmailInfo
from .afsic_ext import IBMesh, IBMesh3D, IBInterpolation, IBInterpolation3D, assign_fibers_function


from .euler.IPCSSolver import IPCSSolver
from .euler.ChorinSolver import ChorinSolver

from .common.utilities import (TimeManager, swanlab_init, swanlab_upload,
                               unique_filename, get_project_name, log, pressure_waveform)
from .common.boundaries import (tag_boundaries, rectangle_boundaries, box_boundaries,
                                 MARKER_LEFT, MARKER_RIGHT, MARKER_BOTTOM, MARKER_TOP,
                                 MARKER_FRONT, MARKER_BACK)
from .common.bcs import (UpVelocity2D, UpVelocity3D, TurekInlet, TurekInlet3D,
                          SinusoidalInlet, PipeInlet3D)

__all__ = [
    # C++ extension
    "add", "send_email", "EmailInfo",
    "IBMesh", "IBMesh3D", "IBInterpolation", "IBInterpolation3D",
    "assign_fibers_function",
    # Solvers
    "IPCSSolver", "ChorinSolver",
    # Utilities
    "TimeManager", "swanlab_init", "swanlab_upload",
    "unique_filename", "get_project_name", "log", "pressure_waveform",
    # Boundaries
    "tag_boundaries", "rectangle_boundaries", "box_boundaries",
    "MARKER_LEFT", "MARKER_RIGHT", "MARKER_BOTTOM", "MARKER_TOP",
    "MARKER_FRONT", "MARKER_BACK",
    # BCs
    "UpVelocity2D", "UpVelocity3D", "TurekInlet", "TurekInlet3D",
    "SinusoidalInlet", "PipeInlet3D",
]
