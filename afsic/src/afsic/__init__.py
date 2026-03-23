from .afsic_ext import send_email, EmailInfo
from .afsic_ext import coupling
from .euler import ChorinSolver
from .euler import define_poisson_problem
# , IPCSSolver

__all__ = ["send_email", "EmailInfo", "coupling", "ChorinSolver", "define_poisson_problem"]
