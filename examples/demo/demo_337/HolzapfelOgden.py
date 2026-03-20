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



""" Holzapfel & Ogden material """


from dolfinx import default_scalar_type
import ufl
from dolfinx import fem
from mpi4py import MPI

__all__ = ['HolzapfelOgdenMaterial'] 

class HolzapfelOgdenMaterial:
    def __init__(self, **params) :
        params = params or {}
        self._parameters = self.default_parameters()
        self._parameters.update(params)

    @staticmethod
    def default_parameters() :
        p = {   "a" : 2244.87,
                "b" : 1.6215,
                "a_f" : 24267.0,
                "b_f" : 1.8268,
                "a_s" : 5562.38,
                "b_s" : 0.7746,
                "a_fs" : 3905.16,
                "b_fs" : 1.695,
                "kappa" : 1.0e5,
                "f0" : None,
                "s0" : None,
                "tension" : None
                }
        return p

    # Implement the strain energy calculation
    def strain_energy(self, domain, F) :
        # Parameters
        params = self._parameters
        a = default_scalar_type(params["a"])
        b = default_scalar_type(params["b"])
        a_f = default_scalar_type(params["a_f"])
        b_f = default_scalar_type(params["b_f"])
        a_s = default_scalar_type(params["a_s"])
        b_s = default_scalar_type(params["b_s"])
        a_fs = default_scalar_type(params["a_fs"])
        b_fs = default_scalar_type(params["b_fs"])
        f0 = params["f0"]
        s0 = params["s0"]
        # f0 = ufl.as_vector((1, 0, 0))
        # s0 = ufl.as_vector((0, 1, 0))
        # ufl.as_vector((0, 0, 1)),
        
        # Invariants
        C = ufl.variable(F.T * F)
        I1 = ufl.variable(ufl.tr(C))

        I_4f = ufl.max_value(ufl.inner(f0, C*f0), 1.0)
        I_4s = ufl.max_value(ufl.inner(s0, C*s0), 1.0)
        I_8fs = ufl.inner(f0, C*s0)

        # Strain energy
        W = a/2.0/b*ufl.exp(b*(I1-3))
        W += a_f/2.0/b_f*(ufl.exp(b_f*(I_4f-1.0)*(I_4f-1.0))-1.0)
        W += a_s/2.0/b_s*(ufl.exp(b_s*(I_4s-1.0)*(I_4s-1.0))-1.0)
        W += a_fs/2.0/b_fs*(ufl.exp(b_fs*I_8fs*I_8fs)-1.0)

        return W

    def active_contraction(self, F):
        params = self._parameters
        f0 = params["f0"]
        tension = params["tension"]
        
        C = F.T*F
        I_4f = ufl.inner(f0, C*f0)
        P = tension*(  1.0+4.9*(ufl.sqrt(I_4f)-1.0)  )*F*ufl.outer(f0,f0)
        return P


    def first_piola_kirchhoff_stress_v1(self, domain, coords, p=None) :
        params = self._parameters
        a = default_scalar_type(params["a"])
        b = default_scalar_type(params["b"])
        kappa = default_scalar_type(params["kappa"])
        F = ufl.variable(ufl.grad(coords))
        # nearly incompressible
        C = ufl.variable(F.T * F)
        I1 = ufl.variable(ufl.tr(C))
        I3 = ufl.variable(ufl.det(C))
        P = - a*ufl.exp(b*(I1-3))*ufl.inv(F).T + kappa*ufl.ln(I3)*ufl.inv(F).T
        # active contraction
        # P_a = self.active_contraction(F)
        return ufl.diff(self.strain_energy(domain, F), F) + P # + P_a
        # return F-ufl.inv(F).T+ kappa*ufl.ln(I3)*ufl.inv(F).T


