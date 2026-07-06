# -----------------------------------------------------------------------------
# 版权所有 (c) 2025 保留所有权利。
#
# 本文件隶属于 Poromechanics Solver 项目，主要开发者为：
#   - 马鹏飞：mapengfei@mail.nwpu.edu.cn
#   - 王璇：wangxuan2022@mail.nwpu.edu.cn
#
# 本软件仅供内部使用和学术研究之用。未经明确许可，严禁重新分发、修改或用于商业用途。
# 详细授权条款请参阅：https://www.pengfeima.cn/license-strict/
# -----------------------------------------------------------------------------


"""neo-Hookean material model"""


from dolfinx import default_scalar_type
import ufl
from dolfinx import fem
from mpi4py import MPI

__all__ = ['FRHMaterial'] 

class FRHMaterial:
    def __init__(self, **params) :
        params = params or {}
        self._parameters = self.default_parameters()
        self._parameters.update(params)

    @staticmethod
    def default_parameters() :
        p = { 'nu' : 0.4,
              'E'  : 5.6e5}
        return p

    # Implement the strain energy calculation
    def strain_energy(self, domain, F) :
        # Parameters
        params = self._parameters
        J = ufl.det(F)
        C = F.T*F
        F_bar = J**(-1/2)*F
        C_bar = F_bar.T*F_bar
        I1_bar = ufl.tr(C_bar)
        I4_bar = ufl.dot(params["f1"], C_bar*params["f1"])
        # Ic = ufl.variable(ufl.tr(C))
        # J = ufl.variable(ufl.det(F))
        # psi = (mu_s / 2) * (Ic - 3) - mu_s * ufl.ln(J) + (lambda_s / 2) * (ufl.ln(J))**2
        psi = 0.5*params["C0"]*(I1_bar-3) + params["C1"]*(ufl.exp(I4_bar-1) - I4_bar) + 0.5*params["kappa_s"]*(0.5*(J*J-1)-ufl.ln(J))
        return psi

    def first_piola_kirchhoff_stress(self, domain, u, p=None) :
        d = len(u)
        I = ufl.variable(ufl.Identity(d))
        F = ufl.variable(I + ufl.grad(u))
        return ufl.diff(self.strain_energy(domain, F), F)

    def first_piola_kirchhoff_stress_v1(self, domain, coords, p=None) :
        F = ufl.variable(ufl.grad(coords))
        return ufl.diff(self.strain_energy(domain, F), F)