from .afsic_ext import add, __doc__
from .afsic_ext import send_email, EmailInfo
from .afsic_ext import coupling, IBMesh, IBInterpolation

from .euler.IPCSSolver import IPCSSolver
from .euler.ChorinSolver import ChorinSolver

from .common.utilities import TimeManager, swanlab_init, swanlab_upload, unique_filename