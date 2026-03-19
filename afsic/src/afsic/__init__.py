from petsc4py import PETSc


from .afsic_ext import add
from .afsic_ext import send_email, EmailInfo
from .afsic_ext import coupling, IBMesh, IBMesh3D, IBInterpolation, IBInterpolation3D, assign_fibers_function

from .euler.IPCSSolver import IPCSSolver
from .euler.ChorinSolver import ChorinSolver

from .common.utilities import TimeManager, swanlab_init, swanlab_upload, unique_filename, get_project_name, log
