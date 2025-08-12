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

__all__ = ['GuccioneMaterial'] 

class GuccioneMaterial:
    def __init__(self, **params) :
        params = params or {}
        self._parameters = self.default_parameters()
        self._parameters.update(params)

    @staticmethod
    def default_parameters() :
        p = {   "C" : 1e5,
                "bf" : 1.0,
                "bt" : 1.0,
                "bfs" : 1.0,
                "f0" : ufl.as_vector((1, 0, 0)),
                "s0" : ufl.as_vector((0, 1, 0)),
                "n0" : ufl.as_vector((0, 0, 1)),
                "tension" : None,
                "deviatoric" : False,
                "contraction" : False,
                }
        return p

    def strain_energy(self, domain, F) :
        params = self._parameters
        kappa = params["kappa"]
        J = ufl.det(F)
        return 0.5 * params["C"] * (ufl.exp(self._Q(F)) - 1.0) + kappa * ufl.ln(J)**2

    # def active_contraction(self, F):
    #     params = self._parameters
    #     f0 = params["f0"]
    #     tension = params["tension"]
        
    #     C = F.T*F
    #     I_4f = ufl.inner(f0, C*f0)
    #     P = tension*(  1.0+4.9*(ufl.sqrt(I_4f)-1.0)  )*F*ufl.outer(f0,f0)
    #     return P

    def _Q(self, F):
        params = self._parameters
        f0 = params["f0"]
        s0 = params["s0"]
        n0 = params["n0"]
        bt = params["bt"]
        bf = params["bf"]
        bfs = params["bfs"]
        kappa = params["kappa"]

        C = F.T * F
        J = ufl.det(F)
        dim = C.ufl_shape[0]
        if params["deviatoric"]:
            Jm23 = pow(J, -1.0 / dim)
            C *= Jm23
        E = 0.5 * (C - ufl.Identity(dim))

        E11, E12, E13 = (
            ufl.inner(E * f0, f0),
            ufl.inner(E * f0, s0),
            ufl.inner(E * f0, n0),
        )
        _, E22, E23 = (
            ufl.inner(E * s0, f0),
            ufl.inner(E * s0, s0),
            ufl.inner(E * s0, n0),
        )
        _, _, E33 = (
            ufl.inner(E * n0, f0),
            ufl.inner(E * n0, s0),
            ufl.inner(E * n0, n0),
        )

        return (
            bf * E11**2 + bt * (E22**2 + E33**2 + 2 * E23**2) + bfs * (2 * E12**2 + 2 * E13**2)
        )

    def first_piola_kirchhoff_stress_v1(self, domain, coords, p=None) :
        F = ufl.variable(ufl.grad(coords))
        return ufl.diff(self.strain_energy(domain, F), F) 



